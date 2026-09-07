from view4.common import load_training_checkpoint
"""Run with torchrun; tiny encoders test real loss, DDP and checkpoint machinery."""
import argparse
import copy
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from view4.common import manifest, reset_training_rng, restore_rng, utcnow, write_json
from view4.model import decorrelation
from view4.runtime import SynchronizedScaler, assert_model_synchronized, save_checkpoint
from view4.upstream import bootstrap
from tests.helpers import tiny_batch, tiny_model


def run(args):
    bootstrap(args.nltk_data)
    torch.set_num_threads(1)
    cuda = args.cuda
    if cuda:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"])) if cuda else torch.device("cpu")
    dist.init_process_group("nccl" if cuda else "gloo")
    control = dist.new_group(backend="gloo")
    rank, size = dist.get_rank(), dist.get_world_size()
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=False)
        run_manifest = manifest(" ".join(sys.argv), [], output_dir, 19)
        write_json(output_dir / "run_manifest.json", run_manifest)
    dist.barrier()
    reset_training_rng(19)
    model = tiny_model("E3").to(device)
    model.encode_text.dropout.p = 0.
    for head in model.experts:
        torch.nn.init.normal_(head[-1].weight, std=.03)
    reference = copy.deepcopy(model)
    reference.use_gather = False
    wrapped = DDP(model, device_ids=[device.index] if cuda else None, find_unused_parameters=True)
    batch = tiny_batch(4 * size)
    # Last rank has no local valid pair, and some experts are unused on every rank.
    batch["image_id"][-3] = batch["image_id"][-4]
    batch["image_id"][-1] = batch["image_id"][-2]
    batch = {k: v.to(device) for k, v in batch.items()}
    local = {k: v[rank * 4:(rank + 1) * 4] for k, v in batch.items()}
    output = wrapped(local, .3, .01)
    output["loss_total"].backward()
    expected = reference(batch, .3, .01)
    expected["loss_total"].backward()
    for (name, actual), (name_b, target) in zip(model.named_parameters(), reference.named_parameters()):
        if actual.grad is None or target.grad is None:
            assert actual.grad is None and target.grad is None, name
        else:
            torch.testing.assert_close(actual.grad, target.grad, atol=1e-5, rtol=1e-4, msg=name)
    # All ranks empty: connected zero permits standalone backward and zero encoder gradient.
    model.zero_grad(set_to_none=True)
    empty = dict(local, person_id=torch.arange(rank * 4, (rank + 1) * 4, device=device))
    h = model.encode_image(empty["image"])
    zero, stats = decorrelation(model.route(h.detach(), empty["view_aug"]), empty)
    zero.backward()
    assert int(stats["ortho_pairs"]) == 0
    assert all(p.grad is None for p in model.visual.parameters())
    gpu_amp = "not_run_cpu_only"
    if not cuda:
        optimizer = torch.optim.SGD(model.parameters(), lr=.001)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            wrapped(local, .3, .01)["loss_total"].backward()
            optimizer.step()
            assert_model_synchronized(model, None, control)
    if cuda:
        optimizer = torch.optim.SGD(model.parameters(), lr=.001)
        scaler = SynchronizedScaler()
        # Isolate injected-overflow synchronization from synthetic-model scale calibration.
        # Production training still uses SynchronizedScaler's unchanged default scale.
        scaler.scaler = torch.cuda.amp.GradScaler(init_scale=1.)
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        for inject in (True, False):
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                output = wrapped(local, .3, 0.)
            scaler.backward(output["loss_total"])
            assert all(torch.isfinite(p.grad).all().item() for p in model.parameters()
                       if p.grad is not None), "Synthetic AMP gradients must be finite before injection"
            if inject and rank == 0:
                next(p for p in model.parameters() if p.grad is not None).grad.flatten()[0] = float("inf")
            skipped = scaler.step(optimizer)
            assert skipped == inject
            if inject:
                assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
            else:
                assert any(not torch.equal(v, before[k]) for k, v in model.state_dict().items())
            states = [None] * size
            dist.all_gather_object(states, scaler.state_dict(), group=control)
            assert all(state == states[0] for state in states)
            assert_model_synchronized(model, scaler, control)
        gpu_amp = "passed"
    assert_model_synchronized(model, None, control)
    generator = torch.Generator().manual_seed(123 + rank)
    save_checkpoint([output_dir / "tiny_rng_checkpoint.pth"], {"test_only": True}, generator, control)
    expected_random = torch.rand(8, device=device)
    state = load_training_checkpoint(output_dir / "tiny_rng_checkpoint.pth", map_location="cpu")
    assert len(state["rng_by_rank"]) == size
    restore_rng(state["rng_by_rank"][rank], generator)
    assert torch.equal(expected_random, torch.rand(8, device=device))
    if rank == 0:
        write_json(output_dir / "distributed_result.json", {"status": "passed", "world_size": size,
                   "device": str(device), "gradient_global_reference": "passed", "empty_pair_backward": "passed",
                   "rank_rng_roundtrip": "passed", "amp_single_rank_inf": gpu_amp,
                   "synthetic_amp_init_scale": 1. if cuda else None,
                   "production_scaler_default_unchanged": True})
        run_manifest.update(status="passed", ended_at=utcnow(), world_size=size,
                            amp_single_rank_inf=gpu_amp)
        write_json(output_dir / "run_manifest.json", run_manifest)
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--nltk-data", required=True)
    parser.add_argument("--cuda", action="store_true")
    run(parser.parse_args())

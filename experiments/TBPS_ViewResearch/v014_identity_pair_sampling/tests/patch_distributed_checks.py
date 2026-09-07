"""CPU/Gloo regression for the patched collective checkpoint transaction.

Example: torchrun --standalone --nproc_per_node=2 -m tests.patch_distributed_checks --output-dir /tmp/view4-gloo
Does not exercise CUDA, AMP or the complete CLIP training graph.
"""
import argparse
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist

from view4.common import load_training_checkpoint, read_json, write_json
from view4.runtime import save_checkpoint, resolve_best_checkpoint, atomic_torch_save, assert_model_synchronized


def run(output_dir):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", timeout=timedelta(seconds=25))
    rank = dist.get_rank()
    folder = Path(output_dir)
    if rank == 0:
        folder.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    try:
        def payload(epoch):
            return {"model":{"x":torch.tensor([float(epoch)])}, "best":{"epoch":epoch,"r1":70.+epoch},
                    "next_epoch":epoch+1, "next_step":0, "experiment":"E0", "run_kind":"formal", "seed":1}
        save_checkpoint([folder/"last.pth",folder/"best.pth"],payload(1),None,dist.group.WORLD)
        caught = torch.zeros(1)
        def fail_last(path, saved):
            if Path(path).name == "last.pth":
                raise OSError("injected rank0 commit failure")
            atomic_torch_save(path,saved)
        with patch("view4.runtime.atomic_torch_save",side_effect=fail_last):
            try:
                save_checkpoint([folder/"last.pth",folder/"best.pth"],payload(2),None,dist.group.WORLD)
            except RuntimeError:
                caught.fill_(1)
        dist.all_reduce(caught)
        assert int(caught.item()) == dist.get_world_size()
        assert resolve_best_checkpoint(folder).name == "best_epoch_001.pth"
        save_checkpoint([folder/"last.pth",folder/"best.pth"],payload(2),None,dist.group.WORLD)
        saved = load_training_checkpoint(folder/"last.pth")
        assert len(saved["rng_by_rank"]) == dist.get_world_size()
        assert saved["best_checkpoint"] == "best_epoch_002.pth"
        torch.manual_seed(5)
        model=torch.nn.Linear(3,2)
        assert_model_synchronized(model,None,dist.group.WORLD)
        if rank==0:
            with torch.no_grad():
                model.weight.add_(1.)
        try:
            assert_model_synchronized(model,None,dist.group.WORLD)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Divergence was not detected")
        dist.barrier()
        if rank==0:
            write_json(folder/"result.json", {"status":"passed", "backend":"gloo", "world_size":dist.get_world_size(),
                       "collective_failure_broadcast":True,"old_commit_retained":True,"retry_commit":True,
                       "per_rank_rng_saved":True,"numeric_divergence_check":True,
                       "cuda_amp_full_training_tested":False})
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--output-dir",required=True)
    run(p.parse_args().output_dir)

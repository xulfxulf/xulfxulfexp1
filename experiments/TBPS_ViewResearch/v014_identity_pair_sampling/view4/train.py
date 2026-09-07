import json
import fcntl
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from .common import (ROOT, manifest, rank, read_json, load_training_checkpoint,
                     reset_training_rng, restore_rng, seed32, utcnow, write_json)
from . import __version__
from .validation import check_resume, check_e0_checkpoint
from .config import loss_weights, lr_value, official_config
from .data import PlannedTrainDataset, PlanBatchSampler, device_batch
from .evaluation import evaluate, expert_diagnostics
from .model import build_model, build_optimizer
from .prepare import verify_prepared
from .runtime import (SynchronizedScaler, assert_global_finite, assert_model_synchronized, disk_guard, init_training,
                      rank0_call, save_checkpoint, resolve_best_checkpoint)
from .upstream import bootstrap
from .audit_status import require_optimizer_updates


def telemetry(output, batch, device):
    values = torch.stack([output[key].detach().float() for key in ("loss_total", "nitc", "ritc", "retrieval")])
    dist.all_reduce(values)
    values /= dist.get_world_size()
    hist = torch.cat([torch.bincount(batch["view_" + suffix], minlength=4) for suffix in ("raw", "aug")])
    dist.all_reduce(hist)
    rho_parts = [torch.empty_like(output["rho"]) for _ in range(dist.get_world_size())]
    dist.all_gather(rho_parts, output["rho"].contiguous())
    rho = torch.cat(rho_parts).float()
    median, p95 = torch.quantile(rho, torch.tensor([.5, .95], device=device)).tolist()
    pair_count = int(output.get("ortho_pairs", torch.tensor(0)).item())
    pid_parts = [torch.empty_like(batch["person_id"]) for _ in range(dist.get_world_size())]
    dist.all_gather(pid_parts, batch["person_id"])
    return {**dict(zip(("loss_total", "nitc", "ritc", "retrieval"), values.tolist())),
            "distinct_pids": int(torch.cat(pid_parts).unique().numel()),
            "raw_views": hist[:4].tolist(), "aug_views": hist[4:].tolist(),
            "rho_median": median, "rho_p95": p95, "rho_alarm": median > .5 or p95 > 1.,
            "ortho_pairs": pair_count, "ortho_mean": float(output.get("ortho_mean", 0)),
            "zero_norm_pair_ratio": int(output["ortho_zero_pairs"]) / pair_count if pair_count else None,
            "near_zero_pair_ratio": int(output["ortho_near_zero_pairs"]) / pair_count if pair_count else None}


def train(cfg, experiment, seed, run_kind, output_dir, command, e0_checkpoint=None, resume=None, audit_steps=None):
    if experiment != "E0":
        raise ValueError("Use the research entry for CUHK portrait soft-view continuation")
    if audit_steps is not None and (audit_steps != 8 or resume):
        raise ValueError("E0 audit is eight fresh steps only")
    if cfg["epochs"] == 10 and experiment != "E0":
        raise ValueError("The 10-epoch profile is approved for E0 only; no E1-E3 launch")
    device, control = init_training(cfg)
    out = Path(output_dir).resolve()
    process_lock = None
    def acquire_training_lock():
        nonlocal process_lock
        process_lock = open(ROOT / ".training.lock", "a")
        fcntl.flock(process_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    rank0_call(acquire_training_lock, control)
    bootstrap(cfg["nltk_data"])
    if seed not in cfg["seeds"] or (experiment == "E0" and (seed != 1 or run_kind != ("audit" if audit_steps else "formal"))):
        raise ValueError("E0 is seed1 formal only; E1-E3 seeds are 1,2,3")
    if experiment != "E0" and not (e0_checkpoint or resume):
        raise ValueError("Explicit E0 best.pth is required")
    if e0_checkpoint and experiment == "E0":
        raise ValueError("E0 must initialize from OpenAI CLIP")
    inputs = rank0_call(lambda: verify_prepared(cfg), control)
    identity = {"experiment": experiment, "seed": seed, "run_kind": run_kind,
                "implementation_version": __version__, "world_size": cfg["world_size"]}
    e0_source = None
    if experiment != "E0" and not resume:
        e0_source = rank0_call(lambda: check_e0_checkpoint(e0_checkpoint, cfg, inputs), control)

    def create_record():
        if resume:
            if Path(resume).resolve().parent != out / "checkpoints":
                raise ValueError("Resume checkpoint must belong to this run directory")
            if (out / "result.json").exists():
                raise ValueError("Completed runs may not be resumed or retested automatically")
        elif out.exists():
            raise FileExistsError("Run directory already exists; no overwriting experiments")
        out.mkdir(parents=True, exist_ok=True)
        (out / "checkpoints").mkdir(exist_ok=True)
        disk_guard(out, 1800 * 1024**2, compact_best=cfg.get('compact_best', False))
        if not resume:
            write_json(out / "config.json", dict(cfg, experiment=experiment, seed=seed, run_kind=run_kind,
                       e0_checkpoint=str(e0_checkpoint), num_epochs=None if audit_steps else cfg["epochs"],
                       audit_steps=audit_steps))
            input_files = [Path(cfg["prepared_dir"]) / "catalog.json", cfg["clip_checkpoint"]]
            if e0_checkpoint:
                input_files.append(e0_checkpoint)
            write_json(out / "run_manifest.json", dict(manifest(command, input_files, out, seed),
                       **identity, e0_source=e0_source))
        return utcnow().replace(":", "").replace(".", "_")

    attempt = rank0_call(create_record, control)
    catalog = read_json(Path(cfg["prepared_dir"]) / "catalog.json")
    dataset = PlannedTrainDataset(catalog, cfg["dataset_root"])
    subset = read_json(Path(cfg["prepared_dir"]) / "expert_diagnostic_subset.json")
    official = official_config(cfg, device)
    reset_training_rng(seed)
    model = build_model(official, experiment, seed, dataset.num_train_ids,
                        e0_checkpoint=None if resume else e0_checkpoint, load_openai=not bool(resume))
    model.to(device)
    wrapped = DDP(model, device_ids=[device.index], find_unused_parameters=True)
    optimizer, optimizer_audit = build_optimizer(model, lr_value(0, experiment, cfg["epochs"], cfg["steps_per_epoch"]))
    scaler = SynchronizedScaler()
    generator = torch.Generator().manual_seed(seed32(seed, rank(), "loader"))
    epoch, start_step, best, history, epoch_stats = 1, 0, {"r1": float("-inf"), "epoch": None}, [], []
    reset_training_rng(seed)
    if resume:
        saved = load_training_checkpoint(resume)
        check_resume(saved, identity, cfg, inputs)
        e0_source = saved.get("e0_source")
        if e0_checkpoint is None:
            e0_checkpoint = saved.get("e0_checkpoint")
        if saved["best"]["epoch"] is not None:
            rank0_call(lambda: str(resolve_best_checkpoint(out / "checkpoints", saved, repair_alias=True)), control)
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        epoch, start_step = saved["next_epoch"], saved["next_step"]
        best, history, epoch_stats = saved["best"], saved["history"], saved["epoch_stats"]
        restore_rng(saved["rng_by_rank"][rank()], generator)
        del saved
    if rank() == 0:
        write_json(out / ("attempt_" + attempt + ".json"), {
            "implementation_version": __version__, "command": command,
            "resume": str(resume) if resume else None, "e0_source": e0_source,
            "started_at": utcnow(), "experiment": experiment, "seed": seed})
        write_json(out / "optimizer_groups.json", optimizer_audit)
    limit = cfg["epochs"] if run_kind == "formal" else 2
    samples = open(out / ("samples_rank%d_%s.jsonl" % (rank(), attempt)), "x", encoding="utf-8")
    log = open(out / ("train_attempt_%s.jsonl" % attempt), "x", encoding="utf-8") if rank() == 0 else None
    text_log = open(out / "train_log.txt", "a", encoding="utf-8") if rank() == 0 else None
    if text_log:
        text_log.write("ATTEMPT %s experiment=%s seed=%s kind=%s\n" % (attempt, experiment, seed, run_kind))
        text_log.flush()

    def checkpoint(next_epoch, next_step, paths):
        payload = dict(identity, model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                       next_epoch=next_epoch, next_step=next_step, best=best, history=history,
                       epoch_stats=epoch_stats, config=cfg, e0_checkpoint=str(e0_checkpoint), attempt=attempt,
                       input_sources=inputs, e0_source=e0_source)
        save_checkpoint(paths, payload, generator, control)

    try:
        while epoch <= limit:
            kind = "E0" if experiment == "E0" else "PK"
            plan_path = Path(cfg["prepared_dir"]) / "plans" / ("seed_%d" % seed) / ("%s_epoch_%d.npz" % (kind, epoch))
            with np.load(str(plan_path), allow_pickle=False) as archive:
                indices = archive["pair_indices"]
            batch_sampler = PlanBatchSampler(indices, seed, epoch, rank(), start_step)
            # Isolate iterator construction so resume prefetch cannot advance the saved loader RNG.
            iterator_rng = generator.get_state()
            loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=cfg["workers"],
                                pin_memory=True, generator=generator, persistent_workers=False)
            iterator = iter(loader)
            generator.set_state(iterator_rng)
            model.train()
            for step, host in enumerate(iterator, start=start_step):
                started = time.monotonic()
                # Small ID/seed trace on the host, never hash/copy full GPU images.
                sample = {"epoch": epoch, "step": step, "image_ids": host["image_id"].tolist(),
                          "raw": host["view_raw"].tolist(), "aug": host["view_aug"].tolist(),
                          "text_seed": seed32(seed, epoch, step, rank(), "text_batch")}
                samples.write(json.dumps(sample) + "\n")
                samples.flush()
                batch = device_batch(host, device, sample["text_seed"])
                global_step = (epoch - 1) * cfg["steps_per_epoch"] + step
                lr = lr_value(global_step, experiment, cfg["epochs"], cfg["steps_per_epoch"])
                for group in optimizer.param_groups:
                    group["lr"] = lr * group["ratio"]
                beta, lam = loss_weights(epoch, step, experiment, cfg["steps_per_epoch"])
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast():
                    output = wrapped(batch, beta, lam)
                assert_global_finite(output["loss_total"], "forward loss")
                scaler.backward(output["loss_total"])
                skipped = scaler.step(optimizer)
                metrics = telemetry(output, batch, device)
                metrics.update(epoch=epoch, step=step, lr_base=lr, beta=beta, lambda_o=lam,
                               skipped=skipped, scale=scaler.scaler.get_scale(), seconds=time.monotonic() - started,
                               peak_allocated_bytes=torch.cuda.max_memory_allocated())
                epoch_stats.append(metrics)
                if rank() == 0:
                    log.write(json.dumps(metrics, allow_nan=False) + "\n")
                    log.flush()
                    if step % 10 == 0 or step == cfg["steps_per_epoch"] - 1:
                        line = "Epoch[%d] Iter[%d/%d] loss=%.5f nitc=%.5f ritc=%.5f pairs=%d lambda=%.6f skip=%s" % (
                            epoch, step + 1, cfg["steps_per_epoch"], metrics["loss_total"], metrics["nitc"], metrics["ritc"],
                            metrics["ortho_pairs"], lam, skipped)
                        print(line, flush=True)
                        text_log.write(line + "\n")
                        text_log.flush()
                if audit_steps and step + 1 >= audit_steps:
                    actual_updates = require_optimizer_updates(epoch_stats)
                    sync = assert_model_synchronized(model, scaler, control)
                    if rank() == 0:
                        write_json(out / "audit_result.json", dict(status="passed", steps=audit_steps,
                                   telemetry=epoch_stats, synchronization=sync, actual_updates=actual_updates,
                                   test_evaluated=False))
                        print("AUDIT PASSED", flush=True)
                    return
                if (step + 1) % 50 == 0:
                    checkpoint(epoch, step + 1, [out / "checkpoints/last.pth"])
            for parameter in model.parameters():
                assert_global_finite(parameter.detach(), "model parameter")
            synchronized_stats = assert_model_synchronized(model, scaler, control)
            validation = rank0_call(lambda: evaluate(model, catalog["val"], cfg["dataset_root"], device), control)
            diagnostics = rank0_call(lambda: expert_diagnostics(model, catalog, subset, cfg["dataset_root"], device), control)
            improved = validation["r1"] > best["r1"]
            if improved:
                best = dict(validation, epoch=epoch)
            history.append({"epoch": epoch, "validation": validation, "expert_diagnostics": diagnostics,
                            "batches": len(epoch_stats), "skipped": sum(x["skipped"] for x in epoch_stats),
                            "synchronized_parameter_stats": synchronized_stats})
            if rank() == 0:
                write_json(out / ("epoch_%d.json" % epoch), {"summary": history[-1], "batches": epoch_stats})
                print("VALIDATION", epoch, validation, "BEST", best, flush=True)
                text_log.write("VALIDATION %d %s BEST %s\n" % (epoch, validation, best))
                text_log.flush()
            epoch_stats = []
            targets = [out / "checkpoints/last.pth"]
            if improved:
                targets.append(out / "checkpoints/best.pth")
            # Smoke epoch1 is retained for the E2/E3 equality gate; not an extra formal checkpoint.
            if run_kind == "smoke" and experiment in ("E2", "E3") and epoch == 1:
                targets.append(out / "checkpoints/epoch1_parity.pth")
            checkpoint(epoch + 1, 0, targets)
            epoch, start_step = epoch + 1, 0
        committed_best = rank0_call(lambda: str(resolve_best_checkpoint(out / "checkpoints", repair_alias=True)), control)
        best_saved = load_training_checkpoint(committed_best)
        model.load_state_dict(best_saved["model"], strict=True)
        del best_saved
        # Baseline measurement and research selection use validation only.
        testing = None
        result = {**identity, "status": "complete", "epochs": limit, "best_validation": best,
                  "test_at_best_validation": testing, "history": history, "ended_at": utcnow(),
                  "input_sources": inputs, "config": cfg, "e0_source": e0_source}
        if rank() == 0:
            write_json(out / "result.json", result)
            run = read_json(out / "run_manifest.json")
            run.update(status="complete", ended_at=utcnow())
            write_json(out / "run_manifest.json", run)
            print("COMPLETE", result["best_validation"], testing, flush=True)
    finally:
        samples.close()
        if log:
            log.close()
        if text_log:
            text_log.close()
        if process_lock:
            process_lock.close()
        if dist.is_initialized():
            dist.destroy_process_group()

import fcntl
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from research import VERSION
from research.data import enriched_catalog, SupportDataset, SupportSampler, to_device, validate_pk_plan
from research.evaluation import evaluate, expert_diagnostics
from research.model import build_model
from view4.common import (ROOT, file_record, manifest, rank, read_json, load_training_checkpoint,
                          reset_training_rng, restore_rng, seed32, utcnow, write_json)
from view4.config import official_config
from view4.model import build_optimizer
from view4.runtime import (SynchronizedScaler, assert_global_finite, assert_model_synchronized,
                           disk_guard, init_training, init_evaluation, rank0_call, save_checkpoint,
                           resolve_best_checkpoint)
from view4.upstream import bootstrap
from view4.validation import check_e0_checkpoint
from view4.audit_status import require_optimizer_updates
from view4.geometry import check_geometry


def load_config(path):
    cfg = read_json(path)
    check_geometry(cfg)
    if cfg["version"] != VERSION or cfg["dataset"] != "CUHK-PEDES":
        raise ValueError("This immutable version is the CUHK initial joint-training experiment only")
    if (cfg["world_size"], cfg["local_batch"], cfg["steps_per_epoch"], cfg["seed"], cfg["seeds"]) != (4, 80, 212, 1, [1]):
        raise ValueError("Fixed four-GPU single-seed global320 setup")
    if (cfg["main_sampler"], cfg["epochs"], cfg['P'], cfg['K']) != ('view_aware_PK2', 5, 160, 2):
        raise ValueError("Five epochs from OpenAI with existing view-aware P160 K2 plans required")
    if cfg['primary_metric'] != 'mAP_at_best_validation_R1':
        raise ValueError('Keep the fixed best-R1 checkpoint selection for paired mAP search')
    if cfg['ritc_eps'] != .0001:
        raise ValueError('V016 fixes the R-ITC target floor to 0.0001')
    for key in ("dataset_root", "annotation_root", "orientation", "prepared_dir", "clip_checkpoint", "initializer", "baseline_checkpoint", "nltk_data"):
        if not Path(cfg[key]).is_absolute():
            raise ValueError(key + " must be absolute")
    for key in ("alpha", "shared_weight", "decorrelation_weight", "base_lr_peak", "expert_lr_peak"):
        if not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError("Invalid method setting: " + key)
    if not 1 <= cfg["max_supports_per_rank"] <= cfg["local_batch"] or cfg["delta"] != 60:
        raise ValueError("Invalid support configuration")
    if (cfg['initialization'] != 'openai_clip' or
            Path(cfg['initializer']).resolve() != Path(cfg['clip_checkpoint']).resolve()):
        raise ValueError('Initializer must be OpenAI, not the measured E0 reference')
    if (cfg['base_lr_peak'], cfg['expert_lr_peak']) != (1e-4, 1e-3):
        raise ValueError('Fixed initial-training LR profile: base1e-4, expert1e-3')
    baseline = read_json(Path(cfg["baseline_checkpoint"]).parents[1] / "result.json")
    if (baseline["status"], baseline["experiment"], baseline["seed"], baseline["epochs"]) != ("complete", "E0", 1, 5):
        raise ValueError("Measure the completed CUHK portrait E0 before the experiment")
    if baseline["config"]["dataset"] != "CUHK-PEDES" or baseline["test_at_best_validation"] is not None:
        raise ValueError("Expected a validation-only CUHK portrait baseline")
    check_geometry(baseline['config'])
    cfg["baseline_validation"] = {k: baseline["best_validation"][k] for k in ("r1", "mAP")}
    return cfg


def schedule(cfg, epoch, step):
    n, total = cfg["steps_per_epoch"], cfg["epochs"] * cfg["steps_per_epoch"]
    index = (epoch - 1) * n + step
    if not 0 <= index < total:
        raise ValueError("Step outside configured training horizon")
    peak = cfg["base_lr_peak"]
    if index < n:
        lr = peak * (.01 + .99 * index / (n - 1))
    else:
        lr = peak * (.05 + .475 * (1 + math.cos(math.pi * (index - n) / (total - n))))
    beta = .5 * min(1., index / n)
    shared = cfg["shared_weight"] * min(1., index / n)
    decor = 0. if epoch == 1 else cfg["decorrelation_weight"] * (step / (n - 1) if epoch == 2 else 1.)
    return lr, beta, shared, decor


def telemetry(output, batch):
    keys = ("loss_total", "nitc", "ritc", "retrieval")
    values = torch.stack([output[k].detach().float() for k in keys])
    dist.all_reduce(values)
    values /= dist.get_world_size()
    gathered = [torch.empty_like(output["rho"]) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, output["rho"].contiguous())
    med, p95 = torch.quantile(torch.cat(gathered), torch.tensor([.5, .95], device=values.device)).tolist()
    views = torch.cat([torch.bincount(batch["view3_" + s], minlength=3) for s in ("raw", "aug")])
    dist.all_reduce(views)
    ids = [torch.empty_like(batch["person_id"]) for _ in range(dist.get_world_size())]
    dist.all_gather(ids, batch["person_id"])
    images = [torch.empty_like(batch['image_id']) for _ in range(dist.get_world_size())]
    dist.all_gather(images, batch['image_id'])
    all_ids, all_images = torch.cat(ids), torch.cat(images)
    _, pid_counts = all_ids.unique(return_counts=True)
    _, image_counts = all_images.unique(return_counts=True)
    positive_candidates = pid_counts.square().sum()
    same_image_candidates = image_counts.square().sum()
    count = int(output["support_pairs"])
    return dict(zip(keys, values.tolist()), shared_mean=float(output["shared_mean"]),
                decor_mean=float(output["decor_mean"]), support_pairs=count,
                zero_norm_pair_ratio=float(output["zero_residual_pairs"]) / count if count else None,
                distinct_pids=int(pid_counts.numel()), raw_views=views[:3].tolist(),
                same_pid_candidates_per_anchor=float(positive_candidates) / all_ids.numel(),
                cross_image_positive_candidates_per_anchor=float(positive_candidates-same_image_candidates) / all_ids.numel(),
                aug_views=views[3:].tolist(), rho_median=med, rho_p95=p95, rho_alarm=med > .5 or p95 > 1.)


def train(cfg, output_dir, command, resume=None, audit_steps=None):
    device, control = init_training(cfg)
    bootstrap(cfg["nltk_data"])
    out = Path(output_dir).resolve()
    lock = None

    def record():
        nonlocal lock
        lock = open(ROOT / ".research_training.lock", "a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not resume and out.exists():
            raise FileExistsError("Run already exists; preserve it and choose a new directory")
        if resume and ((out / "result.json").exists() or Path(resume).resolve().parent != out / "checkpoints"):
            raise ValueError("Cannot resume completed or foreign run")
        out.mkdir(parents=True, exist_ok=True)
        disk_guard(out, 1800 * 1024**2, compact_best=cfg.get('compact_best', False))
        if not resume:
            write_json(out / "config.json", cfg)
            write_json(out / "run_manifest.json", dict(manifest(command, [cfg["initializer"], cfg["baseline_checkpoint"], cfg["orientation"],
                       Path(cfg["prepared_dir"]) / "catalog.json"], out, cfg["seed"]),
                       experiment=VERSION, implementation_version=VERSION,
                       evaluation_split="val_only", orientation_calibrated=False))
        return utcnow().replace(":", "").replace(".", "_")

    attempt = rank0_call(record, control)
    catalog, sources = enriched_catalog(cfg)
    initial = rank0_call(lambda: dict(weights_origin='openai_clip', file=file_record(cfg['initializer']),
                        comparison_reference=check_e0_checkpoint(cfg['baseline_checkpoint'], cfg, sources),
                        reference_weights_loaded_for_training=False), control)
    dataset = SupportDataset(catalog, cfg["dataset_root"], cfg["delta"])
    subset = read_json(Path(cfg["prepared_dir"]) / "expert_diagnostic_subset.json")
    reset_training_rng(cfg["seed"])
    model = build_model(official_config(cfg, device), cfg, dataset.num_train_ids,
                        initializer=None if resume else cfg["initializer"]).to(device)
    wrapped = DDP(model, device_ids=[device.index], find_unused_parameters=False)
    optimizer, audit = build_optimizer(model, schedule(cfg, 1, 0)[0])
    ratio = cfg["expert_lr_peak"] / cfg["base_lr_peak"]
    for group, row in zip(optimizer.param_groups, audit):
        group["ratio"] = ratio if row["name"].startswith("experts.") or row["name"] == "expert_queries" else 1.
        row["ratio"] = group["ratio"]
    scaler = SynchronizedScaler()
    generator = torch.Generator().manual_seed(seed32(cfg["seed"], rank(), "loader"))
    epoch, start_step, best, history, batches = 1, 0, {"r1": -math.inf, "epoch": None}, [], []
    identity = dict(experiment=VERSION, seed=cfg["seed"], run_kind="audit" if audit_steps else "validation_search")
    reset_training_rng(cfg["seed"])
    if resume:
        saved = load_training_checkpoint(resume)
        if saved.get('checkpoint_role') == 'selection_only':
            raise ValueError('best.pth is selection-only; resume from last.pth')
        if saved["config"] != cfg or saved["input_sources"] != sources:
            raise ValueError("Changed configuration/input metadata: cannot resume")
        if any(saved[k] != v for k, v in identity.items()):
            raise ValueError("Resume identity mismatch")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        epoch, start_step = saved["next_epoch"], saved["next_step"]
        best, history, batches = saved["best"], saved["history"], saved["epoch_stats"]
        restore_rng(saved["rng_by_rank"][rank()], generator)
        del saved
    if rank() == 0:
        write_json(out / "optimizer_groups.json", audit)
        write_json(out / "support_coverage.json", {"train_images": len(dataset.images),
                   "with_crossview_support": sum(bool(v) for v in dataset.candidates),
                   "candidate_count": sum(len(v) for v in dataset.candidates), "train_only": True})
    trace = open(out / ("samples_rank%d_%s.jsonl" % (rank(), attempt)), "x", encoding="utf-8")
    log = open(out / ("telemetry_%s.jsonl" % attempt), "x", encoding="utf-8") if rank() == 0 else None
    text = open(out / "train_log.txt", "a", encoding="utf-8") if rank() == 0 else None

    def say(message):
        if rank() == 0:
            print(message, flush=True)
            text.write(message + "\n")
            text.flush()

    def checkpoint(next_epoch, next_step, improved=False):
        payload = dict(identity, model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                       next_epoch=next_epoch, next_step=next_step, best=best, history=history, epoch_stats=batches,
                       config=cfg, input_sources=sources, initializer_source=initial, implementation_version=VERSION)
        paths = [out / "checkpoints/last.pth"]
        if improved:
            paths.append(out / "checkpoints/best.pth")
        save_checkpoint(paths, payload, generator, control)

    try:
        say("START %s init=OpenAI seed=1 main=view-aware-P160-K2 val-only auxiliary_train_images_only" % VERSION)
        while epoch <= cfg["epochs"]:
            plan = Path(cfg["prepared_dir"]) / "plans/seed_1" / ("PK_epoch_%d.npz" % epoch)
            with np.load(str(plan), allow_pickle=False) as archive:
                indices = archive["pair_indices"]
            plan_audit = validate_pk_plan(indices, dataset, cfg['steps_per_epoch'], cfg['world_size'],
                                         cfg['local_batch'], cfg['P'], cfg['K'])
            if rank() == 0 and not (out / ('pk_plan_epoch_%d.json' % epoch)).exists():
                write_json(out / ('pk_plan_epoch_%d.json' % epoch), dict(plan_audit, source=file_record(plan)))
            sampler = SupportSampler(indices, cfg["seed"], epoch, rank(), dataset,
                                     cfg["max_supports_per_rank"], start_step)
            state = generator.get_state()
            loader = DataLoader(dataset, batch_sampler=sampler, num_workers=cfg["workers"],
                                pin_memory=True, generator=generator, persistent_workers=False)
            iterator = iter(loader)
            generator.set_state(state)
            model.train()
            for step, host in enumerate(iterator, start=start_step):
                started = time.monotonic()
                text_seed = seed32(cfg["seed"], epoch, step, rank(), "text_batch")
                trace.write(json.dumps({"epoch": epoch, "step": step, "image_ids": host["image_id"].tolist(),
                     "support_ids": host["support_image_id"].tolist(), "support_valid": host["support_valid"].tolist(),
                     "raw": host["view3_raw"].tolist(), "aug": host["view3_aug"].tolist(), "text_seed": text_seed}) + "\n")
                trace.flush()
                batch = to_device(host, device, text_seed)
                lr, beta, shared_weight, decor_weight = schedule(cfg, epoch, step)
                for group in optimizer.param_groups:
                    group["lr"] = lr * group["ratio"]
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast():
                    output = wrapped(batch, beta, shared_weight, decor_weight)
                assert_global_finite(output["loss_total"], "forward loss")
                scaler.backward(output["loss_total"])
                skipped = scaler.step(optimizer)
                stats = telemetry(output, batch)
                stats.update(epoch=epoch, step=step, lr=lr, beta=beta, shared_weight=shared_weight,
                             decorrelation_weight=decor_weight, skipped=skipped, scale=scaler.scaler.get_scale(),
                             seconds=time.monotonic() - started, peak_allocated_bytes=torch.cuda.max_memory_allocated())
                batches.append(stats)
                if log:
                    log.write(json.dumps(stats, allow_nan=False) + "\n")
                    log.flush()
                if step % 10 == 0 or step == cfg["steps_per_epoch"] - 1:
                    say("Epoch[%d] Iter[%d/%d] loss=%.5f ret=%.5f shared=%.5f decor=%.5f supports=%d pids=%d skip=%s" %
                        (epoch, step + 1, cfg["steps_per_epoch"], stats["loss_total"], stats["retrieval"],
                         stats["shared_mean"], stats["decor_mean"], stats["support_pairs"], stats["distinct_pids"], skipped))
                if audit_steps and step + 1 >= audit_steps:
                    actual_updates = require_optimizer_updates(batches)
                    synchronization = assert_model_synchronized(model, scaler, control)
                    if rank() == 0:
                        write_json(out / "audit_result.json", dict(status="passed", steps=audit_steps,
                                   telemetry=batches, synchronization=synchronization, actual_updates=actual_updates,
                                   test_evaluated=False))
                    say("AUDIT PASSED: full local batch, synchronized four-GPU updates")
                    return
                if (step + 1) % 50 == 0:
                    checkpoint(epoch, step + 1)
            for p in model.parameters():
                assert_global_finite(p.detach(), "model parameter")
            sync = assert_model_synchronized(model, scaler, control)
            val = rank0_call(lambda: evaluate(model, catalog["val"], cfg["dataset_root"], device), control)
            diagnosis = rank0_call(lambda: expert_diagnostics(model, catalog["train"], subset, cfg["dataset_root"], device), control)
            improved = val["r1"] > best["r1"]
            if improved:
                best = dict(val, epoch=epoch)
            history.append(dict(epoch=epoch, validation=val, expert_diagnostics=diagnosis,
                                batches=len(batches), skipped=sum(s["skipped"] for s in batches), synchronization=sync,
                                decorrelation_updates=sum(not s["skipped"] and s["decorrelation_weight"] > 0
                                    and s["zero_norm_pair_ratio"] is not None and s["zero_norm_pair_ratio"] < 1
                                    for s in batches)))
            if rank() == 0:
                write_json(out / ("epoch_%d.json" % epoch), dict(summary=history[-1], batches=batches))
            say("VALIDATION %d %s BEST %s" % (epoch, val, best))
            batches = []
            checkpoint(epoch + 1, 0, improved)
            epoch, start_step = epoch + 1, 0
        result = dict(identity, status="complete", epochs=cfg["epochs"], best_validation=best, history=history,
                      test_at_best_validation=None, ended_at=utcnow(), config=cfg, input_sources=sources,
                      initializer_source=initial, selection_rule="max validation R1; earliest epoch on ties")
        result["joint_validation_gain_pp"] = min(best[k] - cfg["baseline_validation"][k] for k in ("r1", "mAP"))
        result['primary_metric'] = cfg['primary_metric']
        result['primary_metric_value'] = best['mAP']
        result["selected_received_decorrelation"] = any(
            row["epoch"] <= best["epoch"] and row["decorrelation_updates"] > 0 for row in history)
        if rank() == 0:
            write_json(out / "result.json", result)
            run = read_json(out / "run_manifest.json")
            run.update(status="complete", ended_at=utcnow())
            write_json(out / "run_manifest.json", run)
        say("COMPLETE validation-only " + str(best))
    finally:
        trace.close()
        if log:
            log.close()
        if text:
            text.close()
        if lock:
            lock.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def verify_result(path):
    result = read_json(path)
    if result["status"] != "complete" or result["epochs"] != 5 or len(result["history"]) != 5:
        raise RuntimeError("Incomplete five-epoch experiment")
    best = max(result["history"], key=lambda x: x["validation"]["r1"])
    if best["epoch"] != result["best_validation"]["epoch"]:
        raise RuntimeError("Incorrect validation checkpoint selection")
    if result["test_at_best_validation"] is not None:
        raise RuntimeError("Test must not be evaluated in a validation-search training run")
    if (result['primary_metric'] != 'mAP_at_best_validation_R1' or
            result['primary_metric_value'] != result['best_validation']['mAP']):
        raise RuntimeError('Primary mAP must come from the selected best-R1 checkpoint')
    for row in result["history"]:
        if row["batches"] != 212 or not all(math.isfinite(row["validation"][k]) for k in ("r1", "mAP")):
            raise RuntimeError("Missing batches or nonfinite validation")
    print(json.dumps({"guard": "passed", "metric": result["primary_metric_value"],
                      "joint_validation_gain_pp": result['joint_validation_gain_pp'],
                      "validation": result["best_validation"]}))


def final_test(cfg, run_dir, selection_file, output_dir):
    out, run_dir = Path(output_dir).resolve(), Path(run_dir).resolve()
    if out.exists():
        raise FileExistsError("Do not overwrite a test evaluation")
    selection = read_json(selection_file)
    result = read_json(run_dir / "result.json")
    if not result.get("selected_received_decorrelation"):
        raise ValueError("Selected checkpoint has not received decorrelation updates; not a full-mechanism final candidate")
    if (selection["run_dir"] != str(run_dir) or selection["validation_epoch"] != result["best_validation"]["epoch"]
            or not selection.get("reason") or result["config"] != cfg):
        raise ValueError("Freeze a validation-based selection before opening test")
    marker = run_dir / "test_evaluation_started.json"
    if marker.exists():
        raise RuntimeError("This run has already entered final test; inspect existing records")
    device = init_evaluation("cuda:0")
    bootstrap(cfg["nltk_data"])
    catalog, sources = enriched_catalog(cfg)
    if result["input_sources"] != sources:
        raise ValueError("Input metadata changed since training")
    ckpt = resolve_best_checkpoint(run_dir / "checkpoints")
    model = build_model(official_config(cfg, device), cfg,
                        len({r["person_id"] for r in catalog["train"]})).to(device)
    saved = load_training_checkpoint(ckpt)
    model.load_state_dict(saved["model"], strict=True)
    del saved
    write_json(marker, dict(started_at=utcnow(), selection=selection, output_dir=str(out)))
    out.mkdir(parents=True)
    metrics = evaluate(model, catalog["test"], cfg["dataset_root"], device)
    target = {"r1": 75.94, "mAP": 67.56}
    margin = float(metrics['mAP']) - target['mAP']
    write_json(out / "result.json", dict(status="complete", test=metrics, selected_validation=result["best_validation"],
               rde_target=target, both_exceeded=all(round(metrics[k], 2) > target[k] for k in target),
               primary_metric='mAP', primary_test_margin_pp=margin,
               single_metric_margin_met=margin >= 1.0, required_margin_pp=1.0,
               selection=selection, ended_at=utcnow()))
    print("FINAL TEST", metrics, flush=True)

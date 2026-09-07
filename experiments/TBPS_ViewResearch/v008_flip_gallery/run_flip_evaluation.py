"""Validation-only gallery flip averaging; fixed checkpoints, no training."""
import argparse
import json
import os
import sys
import time
from pathlib import Path, PurePosixPath

import torch
import torch.nn.functional as F


def average_embeddings(original, mirrored):
    if original.shape != mirrored.shape or original.ndim != 2:
        raise ValueError("Expected matching [images,dim] feature matrices")
    fused = original.float() + mirrored.float()
    if not torch.isfinite(fused).all() or (fused.norm(dim=-1) <= 1e-12).any():
        raise FloatingPointError("Degenerate flip-averaged feature")
    return F.normalize(fused, dim=-1, eps=1e-12)


def validate_spec(spec):
    if (spec["version"], spec["dataset"], spec["split"], spec["seed"]) != (
            "v008_flip_gallery", "CUHK-PEDES", "val", 1):
        raise ValueError("This experiment is frozen to CUHK validation, seed 1")
    if (spec["image_batch_size"], spec["text_batch_size"], spec["plain_metric_tolerance"]) != (32, 256, 0.0001):
        raise ValueError("Do not change the plain-score reproduction conditions")
    if [row["name"] for row in spec["sources"]] != ["E0", "V004"]:
        raise ValueError("Both matched E0 and V004 inference controls are required")
    for row in spec["sources"]:
        if not PurePosixPath(row["checkpoint"]).is_absolute() or PurePosixPath(row["checkpoint"]).name != "best.pth":
            raise ValueError("Only the frozen validation-selected checkpoint is allowed")


def guard_plain(found, expected, tolerance):
    if any(abs(found[key] - expected[key]) > tolerance for key in ("r1", "mAP")):
        raise RuntimeError("Plain validation no longer reproduces the recorded checkpoint")


@torch.no_grad()
def export(model, data, records, device, spec):
    from torch.utils.data import DataLoader
    from text_utils.tokenizer import tokenize
    model.eval()
    original = torch.empty((len(data), model.embed_dim), device=device, dtype=torch.float32)
    mirrored = torch.empty_like(original)
    seen = torch.zeros(len(data), dtype=torch.int64)
    for batch in DataLoader(data, batch_size=spec["image_batch_size"], shuffle=False, num_workers=0):
        idx = batch["gallery_index"]
        if seen[idx].any() or [records[i]["image_id"] for i in idx.tolist()] != batch["image_id"].tolist():
            raise RuntimeError("Gallery alignment/duplication error")
        image, probabilities = batch["image"].to(device), batch["view_prob"].to(device)
        original[idx.to(device)] = model.image_embedding(image, probabilities).float()
        # F/S/B probabilities are invariant to a left/right horizontal mirror.
        mirrored[idx.to(device)] = model.image_embedding(image.flip(-1), probabilities).float()
        seen[idx] += 1
        if int(seen.sum()) % 512 == 0 or int(seen.sum()) == len(data):
            print("GALLERY %d/%d" % (int(seen.sum()), len(data)), flush=True)
    if not (seen == 1).all():
        raise RuntimeError("Incomplete gallery coverage")
    texts = []
    for start in range(0, len(data.query_texts), spec["text_batch_size"]):
        # No source image, PID, view or candidate metadata enters this path.
        tokens = tokenize(data.query_texts[start:start + spec["text_batch_size"]], context_length=77).to(device)
        texts.append(F.normalize(model.encode_text(tokens), dim=-1, eps=1e-12).float())
    texts = torch.cat(texts)
    for name, values in (("original", original), ("mirrored", mirrored), ("text", texts)):
        if not torch.isfinite(values).all() or not torch.allclose(
                values.norm(dim=-1), torch.ones(len(values), device=device), atol=1e-5):
            raise FloatingPointError("Invalid evaluation embeddings: " + name)
    return texts, original, average_embeddings(original, mirrored)


def run(args):
    import fcntl
    from research.runner import load_config
    from research.model import build_model
    from research.data import enriched_catalog, SoftGalleryDataset
    from view4.common import (file_record, load_training_checkpoint, manifest, read_json,
                              reset_training_rng, utcnow, write_json)
    from view4.config import official_config
    from view4.runtime import init_evaluation
    from view4.upstream import bootstrap
    from misc.eval import metric_eval

    spec = read_json(args.config)
    validate_spec(spec)
    source_config = Path(__file__).resolve().parent / spec["source_config"]
    cfg = load_config(source_config)
    bootstrap(cfg["nltk_data"])
    output = Path(args.output_dir)
    if not output.is_absolute():
        raise ValueError("Explicit absolute --output-dir is required")
    with open('/root/autodl-tmp/.tbps_research_gpu.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        device = init_evaluation("cuda:0")
        output.mkdir(parents=True, exist_ok=False)
        record = manifest(" ".join(sys.argv), [args.config, source_config, cfg["orientation"],
            Path(cfg["prepared_dir"])/"catalog.json", *[row["checkpoint"] for row in spec["sources"]]], output, 1)
        record.update(experiment=spec["version"], code_commit=os.environ.get("CODEX_EXPERIMENT_COMMIT"),
                      evaluation_split="val", optimizer_updates=0,
                      orientation_calibrated=False, test_evaluated=False)
        write_json(output/"run_manifest.json", record)
        write_json(output/"config.json", spec)
        try:
            catalog, _ = enriched_catalog(cfg)
            records = catalog["val"]
            train_ids = {r["person_id"] for r in catalog["train"]}
            # The prepared data adapter already checks original split identities.
            data = SoftGalleryDataset(records, cfg["dataset_root"])
            measurements = []
            for source in spec["sources"]:
                started = time.monotonic()
                before = file_record(source["checkpoint"])
                saved = load_training_checkpoint(source["checkpoint"])
                if (saved["experiment"] != source["experiment"] or saved["seed"] != 1 or
                        saved["best"]["epoch"] != source["best_epoch"] or
                        saved["next_epoch"] != source["best_epoch"] + 1 or saved["next_step"] != 0):
                    raise ValueError("Wrong source checkpoint or non-selected snapshot")
                for key in ("dataset_root", "annotation_root", "orientation", "prepared_dir", "clip_checkpoint"):
                    if Path(saved["config"][key]).resolve() != Path(cfg[key]).resolve():
                        raise ValueError("Source checkpoint input mismatch: " + key)
                received_dec = any(row.get("decorrelation_updates", 0) > 0 for row in saved["history"])
                if source["name"] == "V004" and not received_dec:
                    raise ValueError("Method checkpoint lacks actual decorrelation updates")
                reset_training_rng(1)
                model = build_model(official_config(cfg, device), cfg, len(train_ids))
                result = model.load_state_dict(saved["model"], strict=False)
                expected = ({k for k in model.state_dict() if k.startswith("experts.") or k == "expert_queries"}
                            if source["name"] == "E0" else set())
                if set(result.missing_keys) != expected or result.unexpected_keys:
                    raise RuntimeError("Source encoder/expert weights did not load completely")
                del saved
                model = model.to(device).eval()
                print("SOURCE " + source["name"], flush=True)
                texts, plain, flipped = export(model, data, records, device, spec)
                plain_scores = metric_eval(texts @ plain.t(), data.gallery_person_ids, data.query_person_ids)
                guard_plain(plain_scores, source["plain_validation"], spec["plain_metric_tolerance"])
                flip_scores = metric_eval(texts @ flipped.t(), data.gallery_person_ids, data.query_person_ids)
                if file_record(source["checkpoint"]) != before:
                    raise RuntimeError("Source checkpoint changed during inference")
                row = dict(name=source["name"], checkpoint=before, best_epoch=source["best_epoch"],
                           selected_received_decorrelation=received_dec, plain=plain_scores, flip_average=flip_scores,
                           seconds=time.monotonic()-started, images=len(records), texts=len(data.query_texts),
                           peak_allocated_bytes=torch.cuda.max_memory_allocated())
                measurements.append(row)
                write_json(output/(source["name"]+"_validation.json"), row)
                print(json.dumps(row), flush=True)
                del model, texts, plain, flipped
                torch.cuda.empty_cache()
            e0, method = measurements
            deltas = {}
            for name, reference in (("e0_plain", e0["plain"]), ("e0_flip", e0["flip_average"]),
                                    ("v004_plain", method["plain"])):
                deltas[name] = {key: method["flip_average"][key]-reference[key] for key in ("r1", "mAP")}
            result = dict(status="complete", experiment=spec["version"], split="val", test_evaluated=False,
                          optimizer_updates=0, checkpoint_writes=0, measurements=measurements, deltas=deltas,
                          joint_gain_vs_original_e0=min(deltas["e0_plain"].values()),
                          warning="Inference augmentation experiment; not a retraining or test result", ended_at=utcnow())
            write_json(output/"result.json", result)
            record.update(status="complete", ended_at=utcnow())
        except Exception as exc:
            record.update(status="failed", error=repr(exc), ended_at=utcnow())
            raise
        finally:
            write_json(output/"run_manifest.json", record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())

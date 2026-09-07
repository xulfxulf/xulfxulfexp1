import copy
import io
import json
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .common import (ROOT, capture_rng, manifest, load_training_checkpoint,
                     preserve_rng, read_json, restore_rng, utcnow, write_json, atomic_bytes)
from .runtime import configure_backends, init_evaluation
from . import __version__
from .validation import critical_config
from .config import official_config
from .data import GalleryDataset, preprocess_text
from .evaluation import feature_export
from .model import build_model
from .upstream import bootstrap, install_checkpoint_compatibility


def cpu_audit(cfg, output_dir, command):
    bootstrap(cfg["nltk_data"])
    out = Path(output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a new audit directory")
    out.mkdir(parents=True, exist_ok=True)
    run = manifest(command, [], out, 1)
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_*.py", top_level_dir=str(ROOT))
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    atomic_bytes(out / "unittest.log", stream.getvalue().encode("utf-8"))
    print(stream.getvalue(), flush=True)
    run.update(ended_at=utcnow(), status="passed" if result.wasSuccessful() else "failed",
               tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
               skipped=len(result.skipped), skipped_tests=[str(test) for test, _ in result.skipped])
    write_json(out / "run_manifest.json", run)
    if not result.wasSuccessful():
        raise RuntimeError("CPU audit failed; see unittest.log")


def backbone_audit(cfg, output_dir, command, device_name):
    bootstrap(cfg["nltk_data"])
    torch.set_num_threads(2)
    configure_backends()
    device = torch.device(device_name)
    if device.type == "cuda":
        device = init_evaluation(device_name)
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    run = manifest(command, [cfg["clip_checkpoint"]], out, 1)
    write_json(out / "run_manifest.json", run)
    cfg_ref = official_config(cfg, device)
    cfg_ref.model.use_gather = False
    model = build_model(cfg_ref, "E0", 1, 11003).to(device)
    catalog = read_json(Path(cfg["prepared_dir"]) / "catalog.json")
    sample = next(iter(DataLoader(GalleryDataset(catalog["val"][:1], cfg["dataset_root"]), batch_size=1)))
    image = sample["image"].to(device)
    captions = catalog["val"][0]["captions"][:1]
    # A real image and tokenizer, without training/backpropagation on the validation split.
    from model.tbps_model import CLIP
    from text_utils.tokenizer import tokenize
    model.eval()
    batch = {"image": image, "text_tokens": tokenize(captions).to(device),
             "person_id": torch.tensor([0], device=device), "image_id": sample["image_id"].to(device),
             "theta_raw": sample["theta_raw"].to(device), "theta_aug": sample["theta_raw"].to(device),
             "view_raw": sample["view_raw"].to(device), "view_aug": sample["view_raw"].to(device)}
    results = []
    with torch.no_grad():
        image_features = model.image_embedding(image, batch["view_raw"])
        text_features = F.normalize(model.encode_text(batch["text_tokens"]), dim=-1)
        for beta in (0., .5):
            state = capture_rng()
            ref = CLIP.forward(model, {"image": image, "aug1": image, "id": batch["person_id"],
                               "caption": list(captions), "caption_bt": list(captions)}, beta)
            restore_rng(state)
            batch["text_tokens"] = preprocess_text(captions, captions).to(device)
            new = model(batch, beta, 0.)
            for old_key, new_key in (("nitc_loss", "nitc"), ("ritc_loss", "ritc")):
                torch.testing.assert_close(ref[old_key], new[new_key], atol=1e-6, rtol=1e-5)
            results.append({"beta": beta, "nitc": float(new["nitc"]), "ritc": float(new["ritc"])})
    run.update(ended_at=utcnow(), status="passed", image_shape=list(image_features.shape),
               text_shape=list(text_features.shape), image_norm=image_features.norm(dim=-1).tolist(),
               text_norm=text_features.norm(dim=-1).tolist(), forward_equivalence=results,
               real_backbone_backward="not_run", validation_training=False)
    write_json(out / "run_manifest.json", run)


def compare_epoch1(a, b, output_dir):
    left, right = load_training_checkpoint(a, map_location="cpu"), load_training_checkpoint(b, map_location="cpu")
    if (left["experiment"], right["experiment"], left["next_epoch"], right["next_epoch"]) != ("E2", "E3", 2, 2):
        raise ValueError("Need E2/E3 epoch1 checkpoints")
    for key in ("seed", "world_size"):
        if left[key] != right[key]:
            raise RuntimeError("Parity metadata mismatch: " + key)
    if "config" in left and "config" in right:
        if critical_config(left["config"]) != critical_config(right["config"]):
            raise RuntimeError("Parity training settings differ")
    if left["scaler"] != right["scaler"] or left["history"][0]["skipped"] != right["history"][0]["skipped"]:
        raise RuntimeError("AMP behavior differs between E2 and E3 epoch1")
    if left["model"].keys() != right["model"].keys():
        raise RuntimeError("Model key mismatch")
    for name in left["model"]:
        torch.testing.assert_close(left["model"][name], right["model"][name], atol=1e-6, rtol=1e-5, msg=name)
    for rank in range(left["world_size"]):
        def sample_trace(checkpoint):
            folder = Path(checkpoint).parents[1]
            paths = sorted(folder.glob("samples_rank%d_*.jsonl" % rank))
            if not paths:  # Old run compatibility; do not require/check its hashes.
                paths = sorted(folder.glob("input_hashes_rank%d_*.jsonl" % rank))
            by_step = {}
            for path in paths:
                for line in path.read_text().splitlines():
                    if line:
                        record = json.loads(line)
                        if record["epoch"] == 1:
                            by_step[record["step"]] = {key: record[key] for key in ("image_ids", "raw", "aug")}
            if len(by_step) != 212:
                raise RuntimeError("Incomplete epoch1 sample trace")
            return by_step
        if sample_trace(a) != sample_trace(b):
            raise RuntimeError("E2/E3 epoch1 sample/augmentation metadata differ")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "parity.json", {"status": "passed", "implementation_version": __version__, "a": str(a), "b": str(b),
               "atol": 1e-6, "rtol": 1e-5, "sample_trace_equal": True, "tensor_bytes_checked": False})


def reference_validation(cfg, checkpoint, output_dir, device_name):
    bootstrap(cfg["nltk_data"])
    device = init_evaluation(device_name)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    saved = load_training_checkpoint(checkpoint, map_location="cpu")
    if saved["experiment"] != "E0":
        raise ValueError("Reference validation only applies to E0")
    model = build_model(official_config(cfg, device), "E0", 1, 11003, load_openai=False).to(device)
    model.load_state_dict(saved["model"], strict=True)
    del saved
    catalog = read_json(Path(cfg["prepared_dir"]) / "catalog.json")
    texts, images, dataset = feature_export(model, catalog["val"], cfg["dataset_root"], device)
    from misc.eval import test as original_test, metric_eval
    from misc.caption_dataset import ps_eval_dataset
    from view4.data import eval_transform
    original_dataset = ps_eval_dataset(cfg["annotation_root"], str(Path(cfg["dataset_root"]) / "imgs"),
                                       eval_transform(), "val", max_words=77)
    original_loader = DataLoader(original_dataset, batch_size=32, shuffle=False, num_workers=0)
    if original_dataset.text != dataset.query_texts or not torch.equal(original_dataset.img2person, dataset.gallery_person_ids):
        raise RuntimeError("Reference validation annotation order differs")
    # Capture the exact score matrix passed to the untouched reference metric function.
    import misc.eval as reference_module
    captured = {}
    original_metric = reference_module.metric_eval
    def capture(scores, imgids, txtids):
        captured["scores"] = scores.detach().cpu()
        return original_metric(scores, imgids, txtids)
    reference_module.metric_eval = capture
    try:
        official_metrics = original_test(model, original_loader, 77, device)
    finally:
        reference_module.metric_eval = original_metric
    scores = texts @ images.t()
    torch.testing.assert_close(scores.cpu(), captured["scores"], atol=1e-6, rtol=1e-6)
    actual = metric_eval(scores, dataset.gallery_person_ids, dataset.query_person_ids)
    if any(abs(actual[k] - official_metrics[k]) > 1e-5 for k in actual):
        raise RuntimeError("Reference/new validation metrics differ")
    write_json(out / "reference_validation.json", {"status": "passed", "implementation_version": __version__, "checkpoint": str(checkpoint),
               "new": actual, "official": official_metrics})


def gpu_equivalence(cfg, output_dir, command, device_name):
    device = init_evaluation(device_name)
    bootstrap(cfg["nltk_data"])
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    run = manifest(command, [cfg["clip_checkpoint"]], out, 1)
    write_json(out / "run_manifest.json", run)
    official = official_config(cfg, device)
    official.model.use_gather = False
    reference = build_model(official, "E0", 1, 11003).to(device)
    candidate = copy.deepcopy(reference)
    catalog = read_json(Path(cfg["prepared_dir"]) / "catalog.json")
    chosen, ids = [], set()
    for record in catalog["train"]:
        if record["original_pid"] not in ids:
            chosen.append(record)
            ids.add(record["original_pid"])
        if len(chosen) == 4:
            break
    data = next(iter(DataLoader(GalleryDataset(chosen, cfg["dataset_root"]), batch_size=4)))
    captions = [r["captions"][0] for r in chosen]
    bt = [r["captions_bt"][0] for r in chosen]
    batch = {"image": data["image"].to(device), "person_id": torch.arange(4, device=device),
             "image_id": data["image_id"].to(device), "view_raw": data["view_raw"].to(device),
             "view_aug": data["view_raw"].to(device), "theta_raw": data["theta_raw"].to(device),
             "theta_aug": data["theta_raw"].to(device)}
    from model.tbps_model import CLIP
    from .model import build_optimizer
    aopt, _ = build_optimizer(reference, 1e-6)
    bopt, _ = build_optimizer(candidate, 1e-6)
    checks = []
    for use_amp, beta in ((False, 0.), (False, .5), (True, .5)):
        aopt.zero_grad(set_to_none=True)
        bopt.zero_grad(set_to_none=True)
        state = capture_rng()
        with torch.cuda.amp.autocast(enabled=use_amp):
            old = CLIP.forward(reference, {"image": batch["image"], "aug1": batch["image"],
                    "caption": list(captions), "caption_bt": list(bt), "id": batch["person_id"]}, beta)
            old_loss = old["nitc_loss"] + old["ritc_loss"]
        old_loss.backward()
        end_rng = capture_rng()
        restore_rng(state)
        batch["text_tokens"] = preprocess_text(captions, bt).to(device)
        with torch.cuda.amp.autocast(enabled=use_amp):
            new = candidate(batch, beta, 0.)
        new["loss_total"].backward()
        atol, rtol = (1e-6, 1e-5) if not use_amp else (1e-4, 1e-3)
        torch.testing.assert_close(old_loss, new["loss_total"], atol=atol, rtol=rtol)
        if not torch.equal(end_rng["torch_cuda"], torch.cuda.get_rng_state()):
            raise RuntimeError("Reference/new CUDA RNG trajectories differ")
        for (name, a), (_, b) in zip(reference.named_parameters(), candidate.named_parameters()):
            if a.grad is None or b.grad is None:
                if a.grad is not None or b.grad is not None:
                    raise RuntimeError("Reference/new gradient presence differs: " + name)
            else:
                torch.testing.assert_close(a.grad, b.grad, atol=atol, rtol=max(rtol, 1e-4), msg=name)
        if not use_amp:
            aopt.step()
            bopt.step()
            for (name, a), (_, b) in zip(reference.named_parameters(), candidate.named_parameters()):
                torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5, msg=name)
        checks.append({"precision": "amp" if use_amp else "fp32", "beta": beta,
                       "loss": float(new["loss_total"]), "successful_update": not use_amp})
    run.update(status="passed", ended_at=utcnow(), real_training_images=4, checks=checks,
               fp32_updates=2, official_forward_loss_unchanged=True)
    write_json(out / "run_manifest.json", run)

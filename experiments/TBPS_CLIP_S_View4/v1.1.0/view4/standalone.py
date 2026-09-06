from pathlib import Path

import torch

from .common import manifest, read_json, utcnow, write_json, load_training_checkpoint
from .runtime import init_evaluation, resolve_best_checkpoint
from .config import official_config
from .evaluation import evaluate
from .model import build_model
from .prepare import verify_prepared


def standalone_evaluate(cfg, checkpoint, split, output_dir, device_name, command):
    device = init_evaluation(device_name)
    if split == "test" and Path(checkpoint).name != "best.pth":
        raise ValueError("Test only the validation-selected best.pth")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    verify_prepared(cfg)
    resolved = resolve_best_checkpoint(Path(checkpoint).parent) if Path(checkpoint).name == "best.pth" else Path(checkpoint)
    run = manifest(command, [resolved, Path(cfg["prepared_dir"]) / "catalog.json"], out)
    run["requested_checkpoint"] = str(checkpoint)
    write_json(out / "run_manifest.json", run)
    saved = load_training_checkpoint(resolved)
    if split == "test" and saved["best"]["epoch"] != saved["next_epoch"] - 1:
        raise ValueError("Checkpoint is not its own validation best epoch")
    if split == "test" and saved["run_kind"] != "formal":
        raise ValueError("Smoke checkpoints may not use the test set")
    model = build_model(official_config(cfg, device), saved["experiment"], saved["seed"], 11003, load_openai=False)
    model.load_state_dict(saved["model"], strict=True)
    model.to(device)
    del saved
    catalog = read_json(Path(cfg["prepared_dir"]) / "catalog.json")
    metrics = evaluate(model, catalog[split], cfg["dataset_root"], device)
    write_json(out / "metrics.json", {"split": split, "checkpoint": str(checkpoint),
                                     "resolved_checkpoint": str(resolved), "metrics": metrics})
    run.update(status="complete", ended_at=utcnow())
    write_json(out / "run_manifest.json", run)

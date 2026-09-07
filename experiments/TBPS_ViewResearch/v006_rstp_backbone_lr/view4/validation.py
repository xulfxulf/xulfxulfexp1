"""Small, explicit compatibility checks; code versions are not input hashes."""
from pathlib import Path

from .common import file_record, load_training_checkpoint, read_json

PATH_KEYS = ("dataset_root", "annotation_root", "orientation", "clip_checkpoint", "prepared_dir")
TRAIN_KEYS = ("world_size", "local_batch", "epochs", "steps_per_epoch", "P", "K",
              "alpha", "delta", "lambda_max", "seeds")


def critical_config(cfg):
    result = {key: cfg[key] for key in TRAIN_KEYS}
    result.update({key: str(Path(cfg[key]).resolve()) for key in PATH_KEYS})
    return result


def check_resume(saved, identity, cfg, inputs):
    for key in ("experiment", "seed", "run_kind", "world_size"):
        if saved.get(key) != identity[key]:
            raise ValueError("Resume metadata mismatch: " + key)
    if "config" not in saved or critical_config(saved["config"]) != critical_config(cfg):
        raise ValueError("Resume training settings/data paths changed; start a new experiment group")
    if saved.get("input_sources") is not None and saved["input_sources"] != inputs:
        raise ValueError("Resume input paths/file metadata changed")
    # Deliberately ignore legacy code/config/input SHA256 fields. A code bugfix
    # must not force retraining E0. State shapes and optimizer groups are checked
    # by their strict loaders; the current config is checked above.


def check_e0_checkpoint(path, cfg, inputs):
    path = Path(path).resolve()
    if path.name != "best.pth":
        raise ValueError("Continuation must use E0's validation-selected best.pth")
    result = read_json(path.parents[1] / "result.json")
    expected = ("E0", 1, "formal", "complete")
    found = tuple(result.get(k) for k in ("experiment", "seed", "run_kind", "status"))
    if found != expected or result.get("epochs") not in (5, 10):
        raise ValueError("E0 must have completed its 5- or 10-epoch S reproduction")
    saved = load_training_checkpoint(path)
    if (saved.get("experiment"), saved.get("seed"), saved.get("run_kind")) != expected[:3]:
        raise ValueError("E0 checkpoint metadata disagrees with its completed run")
    if saved.get("best", {}).get("epoch") != result["best_validation"]["epoch"]:
        raise ValueError("E0 checkpoint is not the completed run's selected best")
    if saved.get("next_epoch") != saved["best"]["epoch"] + 1 or saved.get("next_step") != 0:
        raise ValueError("E0 best.pth is not a validation-boundary snapshot")
    if saved.get("config") is None:
        raise ValueError("E0 checkpoint must include its original configuration")
    if saved["config"].get("epochs") != result["epochs"]:
        raise ValueError("E0 checkpoint and completed run disagree on training duration")
    for key in PATH_KEYS:
        if str(Path(saved["config"][key]).resolve()) != str(Path(cfg[key]).resolve()):
            raise ValueError("E0 and continuation data paths differ: " + key)
    if saved.get("input_sources") is not None and saved["input_sources"] != inputs:
        raise ValueError("E0 and continuation input metadata differ")
    return {"file": file_record(path), "best_epoch": saved["best"]["epoch"], "source_epochs": result["epochs"],
            "source_implementation_version": saved.get("implementation_version", "legacy")}

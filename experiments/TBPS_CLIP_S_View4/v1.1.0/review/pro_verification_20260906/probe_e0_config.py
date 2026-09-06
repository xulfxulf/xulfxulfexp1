"""Check E0 source-config rejection without using real model weights."""
import argparse
import json
import sys
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.project_root).resolve()))

    import torch
    from view4.common import manifest, utcnow, write_json
    from view4.config import load_config
    from view4.runtime import atomic_torch_save
    from view4.validation import check_e0_checkpoint, check_resume

    cfg = load_config(args.config)
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = manifest(sys.argv, [args.config], out)
    cases = []
    for key, value in (("local_batch", 40), ("steps_per_epoch", 106)):
        with tempfile.TemporaryDirectory(prefix="view4_e0_config_probe_") as temporary:
            checkpoint = Path(temporary) / "E0/checkpoints/best.pth"
            saved = {
                "experiment": "E0", "seed": 1, "run_kind": "formal", "world_size": 4,
                "model": {"probe": torch.zeros(1)}, "best": {"epoch": 2, "r1": 70.0},
                "next_epoch": 3, "next_step": 0,
                "config": dict(cfg, **{key: value}), "input_sources": {},
            }
            atomic_torch_save(checkpoint, saved)
            write_json(checkpoint.parents[1] / "result.json", {
                "experiment": "E0", "seed": 1, "run_kind": "formal",
                "status": "complete", "epochs": 5, "best_validation": {"epoch": 2},
            })
            case = {"changed_source_config_key": key, "source": value, "current": cfg[key]}
            for name, call in (
                ("e0_reuse", lambda: check_e0_checkpoint(checkpoint, cfg, {})),
                ("same_run_resume", lambda: check_resume(saved, saved, cfg, {})),
            ):
                try:
                    call()
                except (ValueError, RuntimeError) as exc:
                    case[name] = {"rejected": True, "error": str(exc)}
                else:
                    case[name] = {"rejected": False}
            cases.append(case)
    report.update(
        ended_at=utcnow(), status="gap_confirmed" if any(
            not c["e0_reuse"]["rejected"] for c in cases
        ) else "all_rejected", cases=cases, synthetic_weights_only=True,
        real_checkpoints_modified=False, training_started=False,
    )
    write_json(out / "e0_config_probe.json", report)
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()

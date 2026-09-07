from pathlib import Path

import numpy as np

from .common import atomic_bytes, read_json, write_json


def summarize_queue(queue_dir, output_dir):
    queue = Path(queue_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for path in sorted(queue.glob("*_formal/result.json")):
        result = read_json(path)
        rows.append({"experiment": result["experiment"], "seed": result["seed"],
                     "best_epoch": result["best_validation"]["epoch"],
                     "validation_r1": result["best_validation"]["r1"],
                     "test": result["test_at_best_validation"], "result_path": str(path)})
    aggregates, paired = {}, {}
    for experiment in ("E1", "E2", "E3"):
        group = [r for r in rows if r["experiment"] == experiment]
        if len(group) == 3:
            aggregates[experiment] = {metric: {"mean": float(np.mean([r["test"][metric] for r in group])),
                                      "sample_std": float(np.std([r["test"][metric] for r in group], ddof=1))}
                                     for metric in ("r1", "r5", "r10", "mAP")}
    for a, b in (("E2", "E1"), ("E3", "E2")):
        ga = {r["seed"]: r for r in rows if r["experiment"] == a}
        gb = {r["seed"]: r for r in rows if r["experiment"] == b}
        paired[a + "-" + b] = {s: ga[s]["test"]["r1"] - gb[s]["test"]["r1"] for s in ga.keys() & gb.keys()}
    write_json(out / "results.json", {"runs": rows, "aggregates": aggregates, "paired_R1_deltas": paired,
               "interpretation": "E0/E1 confounds additional epochs, sampler and unfreezing; E4 not run"})
    lines = ["# CUHK G0 Results", "", "Only the best validation R1 checkpoint is tested.", "",
             "| Experiment | Seed | Best Epoch | Val R1 | Test R1 | R5 | R10 | mAP |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        t = row["test"]
        lines.append("| %s | %d | %d | %.5f | %.5f | %.5f | %.5f | %.5f |" % (
            row["experiment"], row["seed"], row["best_epoch"], row["validation_r1"], t["r1"], t["r5"], t["r10"], t["mAP"]))
    lines += ["", "Sample SD uses ddof=1. No strict expert-subspace orthogonality claim."]
    atomic_bytes(out / "results.md", ("\n".join(lines) + "\n").encode())

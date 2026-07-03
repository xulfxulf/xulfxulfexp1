import argparse
import csv
import json
import re
from pathlib import Path


METRIC_RE = re.compile(
    r"\|\s*t2i\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|"
)
VAL_EPOCH_RE = re.compile(r"Validation Results - Epoch:\s*([0-9]+)")
BEST_RE = re.compile(r"best R1:\s*([0-9.]+)\s*at epoch\s*([0-9]+)")
RHO_RE = re.compile(
    r"Epoch\[(\d+)\] Iteration\[(\d+)/(\d+)\].*?"
    r"support_rho_mean:\s*([0-9.]+).*?"
    r"support_rho_zero_ratio:\s*([0-9.]+).*?"
    r"support_rho_mid_ratio:\s*([0-9.]+).*?"
    r"support_rho_one_ratio:\s*([0-9.]+)"
)

PURE_BASELINES = {
    "single_pure": {"best_epoch": 52, "R1": 57.719},
    "single_proj_pure": {"best_epoch": 45, "R1": 57.538},
    "split_pure": {"best_epoch": 52, "R1": 57.235},
}

SCHEME1_DEFAULTS = {
    "single_proj_bag": {
        "best_epoch": 59,
        "R1": 52.908,
        "R5": 72.922,
        "R10": 80.126,
        "mAP": 40.288,
        "mINP": 20.859,
    },
    "split_bag": {
        "best_epoch": 58,
        "R1": 55.108,
        "R5": 74.612,
        "R10": 81.362,
        "mAP": 41.843,
        "mINP": 21.716,
    },
}


def read_text(path):
    path = Path(path)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def parse_train_log(path):
    path = Path(path)
    text = read_text(path)
    if not text:
        return {
            "path": str(path),
            "exists": False,
            "complete": False,
            "validation_metrics": [],
            "best": None,
            "final": None,
            "rho": None,
            "errors": [],
        }

    lines = text.splitlines()
    metrics = []
    for idx, line in enumerate(lines):
        epoch_match = VAL_EPOCH_RE.search(line)
        if not epoch_match:
            continue
        block = "\n".join(lines[idx : idx + 12])
        metric_match = METRIC_RE.search(block)
        if not metric_match:
            continue
        values = list(map(float, metric_match.groups()))
        metrics.append(
            {
                "epoch": int(epoch_match.group(1)),
                "R1": values[0],
                "R5": values[1],
                "R10": values[2],
                "mAP": values[3],
                "mINP": values[4],
            }
        )

    best_match = None
    for match in BEST_RE.finditer(text):
        best_match = match
    best = None
    if best_match:
        best_epoch = int(best_match.group(2))
        best = next((row for row in metrics if row["epoch"] == best_epoch), None)
        if best is None:
            best = {"epoch": best_epoch, "R1": float(best_match.group(1))}
    elif metrics:
        best = max(metrics, key=lambda row: row["R1"])

    rho_rows = []
    for match in RHO_RE.finditer(text):
        rho_rows.append(
            {
                "epoch": int(match.group(1)),
                "iteration": int(match.group(2)),
                "total_iterations": int(match.group(3)),
                "rho_mean": float(match.group(4)),
                "rho_zero_ratio": float(match.group(5)),
                "rho_mid_ratio": float(match.group(6)),
                "rho_one_ratio": float(match.group(7)),
            }
        )

    error_lines = [
        line
        for line in lines
        if "Traceback" in line
        or "CUDA out of memory" in line
        or ("RuntimeError" in line and "INFO" not in line)
    ]
    return {
        "path": str(path),
        "exists": True,
        "complete": BEST_RE.search(text) is not None,
        "validation_metrics": metrics,
        "best": best,
        "final": metrics[-1] if metrics else None,
        "rho": rho_rows[-1] if rho_rows else None,
        "errors": error_lines[-10:],
    }


def find_mode_log(root, mode):
    root = Path(root)
    candidates = []
    patterns = [
        f"**/{mode}/train_log.txt",
        f"**/*{mode}*/train_log.txt",
    ]
    for pattern in patterns:
        candidates.extend(root.glob(pattern))
    candidates = [path for path in candidates if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_pre_analysis(path):
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def blocked_slot_distribution(support_conflict_csv, consistency_csv):
    support_path = Path(support_conflict_csv)
    if support_path.exists():
        blocked_types = {"soft_mismatch", "hard_contradiction", "unparsed"}
        counts = {}
        total = {}
        with support_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                slot = row.get("slot", "")
                ctype = row.get("conflict_type", "")
                if not slot:
                    continue
                total[slot] = total.get(slot, 0) + 1
                if ctype in blocked_types:
                    counts[slot] = counts.get(slot, 0) + 1
        dist = {}
        for slot in sorted(total):
            dist[slot] = {
                "blocked": counts.get(slot, 0),
                "total": total[slot],
                "ratio": counts.get(slot, 0) / total[slot] if total[slot] else 0.0,
            }
        return {
            "source": str(support_path),
            "exists": True,
            "blocked_definition": "support pair conflict_type in soft_mismatch/hard_contradiction/unparsed",
            "blocked_slots": dist,
        }

    path = Path(consistency_csv)
    if not path.exists():
        return {"source": str(path), "exists": False, "blocked_slots": {}}
    blocked_types = {"soft_mismatch", "hard_contradiction", "unparsed"}
    counts = {}
    total = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            slot = row.get("slot", "")
            ctype = row.get("consistency_type", "")
            if not slot:
                continue
            total[slot] = total.get(slot, 0) + 1
            if ctype in blocked_types:
                counts[slot] = counts.get(slot, 0) + 1
    dist = {}
    for slot in sorted(total):
        dist[slot] = {
            "blocked": counts.get(slot, 0),
            "total": total[slot],
            "ratio": counts.get(slot, 0) / total[slot] if total[slot] else 0.0,
        }
    return {
        "source": str(path),
        "exists": True,
        "blocked_definition": "image-level consistency_type in soft_mismatch/hard_contradiction/unparsed",
        "blocked_slots": dist,
    }


def metric_delta(a, b):
    if not a or not b:
        return None
    out = {}
    for key in ["R1", "R5", "R10", "mAP", "mINP"]:
        if key in a and key in b:
            out[key] = a[key] - b[key]
    return out


def format_metric(row, key):
    if not row or key not in row:
        return "NA"
    return f"{row[key]:.3f}"


def conclusion_level(results):
    single_base = results["modes"].get("single_proj_bag", {}).get("best")
    split_base = results["modes"].get("split_bag", {}).get("best")
    single_cons = results["modes"].get("single_proj_bag_consistency", {}).get("best")
    split_cons = results["modes"].get("split_bag_consistency", {}).get("best")
    if not single_cons or not split_cons:
        return "实验失败需复查"

    single_improves = single_cons["R1"] > single_base["R1"]
    split_improves = split_cons["R1"] > split_base["R1"]
    split_beats_single = split_cons["R1"] > single_cons["R1"]
    pure_best = max(item["R1"] for item in PURE_BASELINES.values())

    if split_cons["R1"] > pure_best:
        return "可进入下一阶段"
    if single_improves or split_improves:
        if split_beats_single:
            return "方向有效但未达主方法"
        return "方向有效但未达主方法"
    return "可靠度规则无效"


def build_summary(args):
    project_root = Path(args.project_root)
    pre_dir = project_root / "diagnostics" / "TAG-PEDES" / "scheme2_pre_analysis"
    pre = load_pre_analysis(pre_dir / "scheme2_pre_analysis_summary.json")
    blocked = blocked_slot_distribution(
        pre_dir / "support_conflict" / "support_conflict_pairs.csv",
        pre_dir / "support_conflict" / "intra_image_caption_consistency.csv"
    )

    mode_logs = {
        "single_proj_bag": project_root
        / "server_logs"
        / "4090"
        / "v16_scheme1_bag_20260702"
        / "single_proj_bag"
        / "train_log.txt",
        "split_bag": project_root
        / "server_logs"
        / "4090"
        / "v16_scheme1_bag_20260702"
        / "split_bag"
        / "train_log.txt",
    }
    scheme2_root = Path(args.scheme2_log_root) if args.scheme2_log_root else project_root
    for mode in ["single_proj_bag_consistency", "split_bag_consistency"]:
        found = find_mode_log(scheme2_root, mode)
        mode_logs[mode] = found if found else Path("__missing__") / mode / "train_log.txt"

    modes = {}
    for mode, log_path in mode_logs.items():
        parsed = parse_train_log(log_path)
        if mode in SCHEME1_DEFAULTS and not parsed["best"]:
            parsed["best"] = {"epoch": SCHEME1_DEFAULTS[mode]["best_epoch"], **SCHEME1_DEFAULTS[mode]}
            parsed["final"] = parsed["best"]
        modes[mode] = parsed

    results = {
        "summary_version": "scheme2_experiment_summary_v1",
        "project_root": str(project_root),
        "scheme2_log_root": str(scheme2_root),
        "pre_analysis": pre,
        "pure_baselines": PURE_BASELINES,
        "blocked_slot_distribution": blocked,
        "modes": modes,
        "deltas": {
            "single_proj_bag_consistency_vs_single_proj_bag": metric_delta(
                modes["single_proj_bag_consistency"].get("best"),
                modes["single_proj_bag"].get("best"),
            ),
            "split_bag_consistency_vs_split_bag": metric_delta(
                modes["split_bag_consistency"].get("best"),
                modes["split_bag"].get("best"),
            ),
            "split_bag_consistency_vs_single_proj_bag_consistency": metric_delta(
                modes["split_bag_consistency"].get("best"),
                modes["single_proj_bag_consistency"].get("best"),
            ),
        },
    }
    results["conclusion_level"] = conclusion_level(results)
    results["complete"] = all(
        modes[mode]["complete"]
        for mode in ["single_proj_bag_consistency", "split_bag_consistency"]
    )
    return results


def write_markdown(results, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# v16 Scheme-2 Experiment Summary",
        "",
        f"- complete: `{results['complete']}`",
        f"- conclusion level: `{results['conclusion_level']}`",
        f"- project root: `{results['project_root']}`",
        f"- scheme2 log root: `{results['scheme2_log_root']}`",
        "",
        "## Four-Mode Results",
        "",
        "| mode | complete | best epoch | best R1 | R5 | R10 | mAP | mINP | final epoch | final R1 | log |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for mode in [
        "single_proj_bag",
        "single_proj_bag_consistency",
        "split_bag",
        "split_bag_consistency",
    ]:
        row = results["modes"][mode]
        best = row.get("best")
        final = row.get("final")
        lines.append(
            "| {mode} | {complete} | {best_epoch} | {r1} | {r5} | {r10} | {map_} | {minp} | {final_epoch} | {final_r1} | `{log}` |".format(
                mode=mode,
                complete=row.get("complete"),
                best_epoch=best.get("epoch", "NA") if best else "NA",
                r1=format_metric(best, "R1"),
                r5=format_metric(best, "R5"),
                r10=format_metric(best, "R10"),
                map_=format_metric(best, "mAP"),
                minp=format_metric(best, "mINP"),
                final_epoch=final.get("epoch", "NA") if final else "NA",
                final_r1=format_metric(final, "R1"),
                log=row.get("path", ""),
            )
        )

    lines += ["", "## Reliability Statistics", ""]
    pre = results.get("pre_analysis", {})
    lines += [
        f"- mean support count: `{pre.get('mean_support_used', 'NA')}`",
        f"- anchors >=2 support ratio: `{pre.get('anchors_ge_2_support_ratio', 'NA')}`",
        f"- anchors with 3 support ratio: `{pre.get('anchors_with_3_support_ratio', 'NA')}`",
        "",
        "| mode | rho mean | rho=0 | rho mid | rho=1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode in ["single_proj_bag_consistency", "split_bag_consistency"]:
        rho = results["modes"][mode].get("rho") or {}
        lines.append(
            f"| {mode} | {rho.get('rho_mean', 'NA')} | {rho.get('rho_zero_ratio', 'NA')} | {rho.get('rho_mid_ratio', 'NA')} | {rho.get('rho_one_ratio', 'NA')} |"
        )

    lines += [
        "",
        "## Blocked Slot Distribution",
        "",
        "This distribution is derived from the scheme-2 pre-analysis support/conflict files; the current training log records rho ratios, not per-epoch blocked slot counts.",
        f"- source: `{results['blocked_slot_distribution'].get('source', 'NA')}`",
        f"- blocked definition: `{results['blocked_slot_distribution'].get('blocked_definition', 'NA')}`",
        "",
        "| slot | blocked | total | ratio |",
        "|---|---:|---:|---:|",
    ]
    for slot, row in results["blocked_slot_distribution"].get("blocked_slots", {}).items():
        lines.append(f"| {slot} | {row['blocked']} | {row['total']} | {row['ratio']:.4f} |")

    lines += ["", "## Deltas", ""]
    for name, delta in results["deltas"].items():
        if not delta:
            lines.append(f"- `{name}`: NA")
            continue
        compact = ", ".join(f"{key} {value:+.3f}" for key, value in delta.items())
        lines.append(f"- `{name}`: {compact}")

    lines += [
        "",
        "## Pure Baseline Check",
        "",
        "| baseline | best epoch | R1 |",
        "|---|---:|---:|",
    ]
    for name, row in results["pure_baselines"].items():
        lines.append(f"| {name} | {row['best_epoch']} | {row['R1']:.3f} |")

    lines += [
        "",
        "## Evidence Notes",
        "",
        "- Scheme-1 bag results are reused from the completed 20260702 logs.",
        "- Scheme-2 consistency results are parsed from synced train logs.",
        "- If `complete` is false, this file is a draft and must not be used as the final route decision.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--scheme2-log-root", default="")
    parser.add_argument(
        "--output-dir",
        default="diagnostics/TAG-PEDES/scheme2_pre_analysis",
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    results = build_summary(args)
    if not results["complete"] and not args.allow_incomplete:
        missing = [
            mode
            for mode in ["single_proj_bag_consistency", "split_bag_consistency"]
            if not results["modes"][mode]["complete"]
        ]
        raise SystemExit(
            "Incomplete scheme-2 formal runs: "
            + ", ".join(missing)
            + ". Use --allow-incomplete only for draft checks."
        )

    output_dir = Path(args.project_root) / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "scheme2_experiment_summary.json"
    md_path = output_dir / "scheme2_experiment_summary.md"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(results, md_path)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()

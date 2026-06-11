import argparse
import csv
import os
import re
from pathlib import Path


METRIC_RE = re.compile(
    r"\|\s*t2i\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|"
)
BEST_RE = re.compile(r"best R1:\s*([0-9.]+)\s*at epoch\s*([0-9]+)")
STAT_RE = re.compile(
    r"duplicate_ids=(\d+).*?negative_ordered_pairs=(\d+).*?"
    r"duplicate_images=(\d+).*?same_image_ordered_pairs=(\d+)"
)


def read_text(path):
    if not path or not Path(path).exists():
        return ""
    return Path(path).read_text(encoding="utf-8", errors="replace")


def last_t2i_metrics(path):
    matches = METRIC_RE.findall(read_text(path))
    if not matches:
        return ["", "", "", "", ""]
    return list(matches[-1])


def best_val(path):
    matches = BEST_RE.findall(read_text(path))
    if not matches:
        return "", ""
    return matches[-1]


def batch_stats(path):
    rows = [tuple(map(int, m)) for m in STAT_RE.findall(read_text(path))]
    if not rows:
        return {
            "stat_lines": 0,
            "duplicate_ids_avg": "",
            "duplicate_ids_max": "",
            "negative_pairs_avg": "",
            "duplicate_images_avg": "",
            "duplicate_images_max": "",
            "same_image_ordered_pairs_avg": "",
            "same_image_ordered_pairs_max": "",
        }

    def avg(index):
        return f"{sum(row[index] for row in rows) / len(rows):.3f}"

    def maxv(index):
        return str(max(row[index] for row in rows))

    return {
        "stat_lines": len(rows),
        "duplicate_ids_avg": avg(0),
        "duplicate_ids_max": maxv(0),
        "negative_pairs_avg": avg(1),
        "duplicate_images_avg": avg(2),
        "duplicate_images_max": maxv(2),
        "same_image_ordered_pairs_avg": avg(3),
        "same_image_ordered_pairs_max": maxv(3),
    }


def gpu_stats(path):
    if not path or not Path(path).exists():
        return {
            "gpu_mem_first_mb": "",
            "gpu_mem_max_mb": "",
            "gpu_mem_last_mb": "",
            "gpu_mem_growth_mb": "",
        }
    mems = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith("timestamp") or line.startswith("GPU_NOT_AVAILABLE"):
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            mems.append(float(parts[-2]))
        except ValueError:
            continue
    if not mems:
        return {
            "gpu_mem_first_mb": "",
            "gpu_mem_max_mb": "",
            "gpu_mem_last_mb": "",
            "gpu_mem_growth_mb": "",
        }
    return {
        "gpu_mem_first_mb": f"{mems[0]:.0f}",
        "gpu_mem_max_mb": f"{max(mems):.0f}",
        "gpu_mem_last_mb": f"{mems[-1]:.0f}",
        "gpu_mem_growth_mb": f"{mems[-1] - mems[0]:.0f}",
    }


def parse_config(path):
    data = {}
    for line in read_text(path).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip().strip("'\"")
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--gpu_csv", default="")
    parser.add_argument("--summary_file", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config = parse_config(run_dir / "configs.yaml")
    best_r1, best_epoch = best_val(run_dir / "train_log.txt")
    test_r1, test_r5, test_r10, test_map, test_minp = last_t2i_metrics(run_dir / "test_log.txt")

    row = {
        "dataset": config.get("dataset_name", ""),
        "mode": config.get("irra_light_mode", ""),
        "seed": config.get("seed", ""),
        "img_aug": config.get("img_aug", ""),
        "num_epoch": config.get("num_epoch", ""),
        "run_dir": str(run_dir),
        "best_val_r1": best_r1,
        "best_val_epoch": best_epoch,
        "test_r1": test_r1,
        "test_r5": test_r5,
        "test_r10": test_r10,
        "test_mAP": test_map,
        "test_mINP": test_minp,
    }
    row.update(batch_stats(run_dir / "train_log.txt"))
    row.update(gpu_stats(args.gpu_csv))

    summary = Path(args.summary_file)
    summary.parent.mkdir(parents=True, exist_ok=True)
    write_header = not summary.exists()
    with summary.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()), delimiter="\t")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"Wrote summary row to {summary}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Add one v16.6-mined different-PID image to each phrase teacher case."""

from __future__ import annotations

import argparse
import json
import os.path as op
import sys
from pathlib import Path

PROJECT_ROOT = op.abspath(op.join(op.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.mllm.phrase_teacher_common import read_jsonl, write_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description="Build v16.7 comparative cases")
    parser.add_argument("--v1660-cases", required=True)
    parser.add_argument("--hard-negatives", required=True)
    parser.add_argument("--output-file", required=True)
    return parser.parse_args()


def main():
    cli = parse_args()
    cases = read_jsonl(cli.v1660_cases)
    negatives = {
        int(row["record_index"]): row for row in read_jsonl(cli.hard_negatives)
    }
    output = []
    for case in cases:
        record_index = int(case["record_index"])
        if record_index not in negatives:
            raise RuntimeError(
                "Missing hard negative for record {}".format(record_index)
            )
        negative = negatives[record_index]
        if int(negative["anchor_pid"]) != int(case["pid"]):
            raise RuntimeError("Hard-negative anchor PID mismatch")
        if int(negative["hard_negative_pid"]) == int(case["pid"]):
            raise RuntimeError("Hard negative has the anchor PID")
        item = dict(case)
        item["case_version"] = "v16.7.0"
        item["case_type"] = "comparative"
        item["hard_negative"] = {
            "pid": int(negative["hard_negative_pid"]),
            "image_id": int(negative["hard_negative_image_id"]),
            "path": negative["hard_negative_path"],
            "score": float(negative["hard_negative_score"]),
        }
        output.append(item)
    write_jsonl(cli.output_file, output)
    summary = {
        "case_count": len(output),
        "phrase_count": sum(len(case["phrases"]) for case in output),
        "output": str(Path(cli.output_file).expanduser().resolve()),
    }
    with Path(cli.output_file).expanduser().resolve().with_suffix(
        ".summary.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Strictly merge v16.7 hard-negative labels and build comparative targets."""

from __future__ import annotations

import argparse
import json
import os.path as op
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = op.abspath(op.join(op.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.phrase_route_io import read_jsonl, write_jsonl
from tools.mllm.phrase_teacher_common import (
    comparative_raw_score,
    normalize_record_phrase_targets,
    strict_label,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Merge v16.7 comparative teachers")
    parser.add_argument("--v1660-labels", required=True)
    parser.add_argument("--comparative-cases", required=True)
    parser.add_argument("--qwen-forward", required=True)
    parser.add_argument("--qwen-reverse", required=True)
    parser.add_argument("--intern-forward", required=True)
    parser.add_argument("--intern-reverse", required=True)
    parser.add_argument("--output-labels", required=True)
    parser.add_argument("--output-summary", required=True)
    return parser.parse_args()


def load_tables(paths):
    return [
        {str(row["case_id"]): row for row in read_jsonl(path)} for path in paths
    ]


def negative_judgment(tables, case_id, phrase_id):
    values = []
    for table in tables:
        row = table.get(case_id)
        if not row or not bool(row.get("parsed_ok", False)):
            values.append(None)
            continue
        phrase = row.get("phrases", {}).get(phrase_id)
        values.append(None if phrase is None else phrase.get("hard_negative"))
    return strict_label(values)


def main():
    cli = parse_args()
    base_rows = read_jsonl(cli.v1660_labels)
    cases = read_jsonl(cli.comparative_cases)
    case_by_record = {int(case["record_index"]): case for case in cases}
    tables = load_tables(
        [
            cli.qwen_forward,
            cli.qwen_reverse,
            cli.intern_forward,
            cli.intern_reverse,
        ]
    )

    output = []
    label_counter = Counter()
    strict_agree = 0
    strict_total = 0
    supervised = 0
    for base in base_rows:
        record_index = int(base["record_index"])
        case = case_by_record.get(record_index)
        phrases = []
        for phrase in base["phrases"]:
            item = dict(phrase)
            if case is None or phrase.get("fallback", False):
                label = "unknown"
                agreement = False
            else:
                label, agreement = negative_judgment(
                    tables, str(case["case_id"]), str(phrase["phrase_id"])
                )
                strict_total += 1
                strict_agree += int(agreement)
            label_counter[label] += 1
            comparative = comparative_raw_score(
                float(item.get("propagation_raw_score", 0.0)), label
            )
            item.update(comparative)
            item["hard_negative_strict_agreement"] = bool(agreement)
            phrases.append(item)

        phrases, route_valid = normalize_record_phrase_targets(
            phrases, raw_field="comparative_raw_score"
        )
        supervised += int(route_valid)
        row = dict(base)
        row.update(
            {
                "experiment_version": "v16.7.0",
                "route_kind": "comparative",
                "comparative_teacher_case_present": case is not None,
                "comparative_teacher_case_id": (
                    None if case is None else str(case["case_id"])
                ),
                "route_supervision_valid": bool(route_valid),
                "hard_negative": (
                    None if case is None else dict(case["hard_negative"])
                ),
                "phrases": phrases,
            }
        )
        output.append(row)

    write_jsonl(cli.output_labels, output)
    summary = {
        "experiment_version": "v16.7.0",
        "route_kind": "comparative",
        "record_count": len(output),
        "comparative_teacher_case_count": sum(
            bool(row.get("comparative_teacher_case_present")) for row in output
        ),
        "supervised_record_count": supervised,
        "supervised_record_ratio": supervised / max(1, len(output)),
        "hard_negative_strict_agreement": strict_agree / max(1, strict_total),
        "hard_negative_labels": dict(label_counter),
        "output_labels": str(Path(cli.output_labels).expanduser().resolve()),
    }
    target = Path(cli.output_summary).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

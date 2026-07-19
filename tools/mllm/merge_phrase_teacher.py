#!/usr/bin/env python
"""Strictly merge two teachers x two support orders into v16.6.0 labels."""

from __future__ import annotations

import argparse
import json
import os.path as op
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = op.abspath(op.join(op.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.phrase_route_io import read_jsonl, write_jsonl
from tools.mllm.phrase_teacher_common import (
    normalize_record_phrase_targets,
    propagation_raw_score,
    strict_label,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Merge v16.6 phrase teachers")
    parser.add_argument("--train-spans", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--qwen-forward", required=True)
    parser.add_argument("--qwen-reverse", required=True)
    parser.add_argument("--intern-forward", required=True)
    parser.add_argument("--intern-reverse", required=True)
    parser.add_argument("--output-labels", required=True)
    parser.add_argument("--output-summary", required=True)
    return parser.parse_args()


def load_predictions(paths):
    output = []
    for path in paths:
        rows = {str(row["case_id"]): row for row in read_jsonl(path)}
        output.append(rows)
    return output


def judgment(rows, case_id, phrase_id, field, support_id=None):
    values = []
    for table in rows:
        row = table.get(case_id)
        if not row or not bool(row.get("parsed_ok", False)):
            values.append(None)
            continue
        phrase = row.get("phrases", {}).get(phrase_id)
        if not phrase:
            values.append(None)
            continue
        if field == "support":
            values.append(
                phrase.get("support_by_image_id", {}).get(str(int(support_id)))
            )
        else:
            values.append(phrase.get(field))
    return strict_label(values)


def main():
    cli = parse_args()
    span_rows = read_jsonl(cli.train_spans)
    cases = read_jsonl(cli.cases)
    case_by_record = {int(case["record_index"]): case for case in cases}
    predictions = load_predictions(
        [
            cli.qwen_forward,
            cli.qwen_reverse,
            cli.intern_forward,
            cli.intern_reverse,
        ]
    )

    merged_rows = []
    agreement_counter = Counter()
    label_counter = Counter()
    supervised_records = 0
    for span_row in span_rows:
        record_index = int(span_row["record_index"])
        case = case_by_record.get(record_index)
        phrase_results = []
        if case is None:
            for phrase in span_row["phrases"]:
                item = dict(phrase)
                item.update(
                    {
                        "anchor_label": "unknown",
                        "sibling_label": "unknown",
                        "support_labels": [],
                        "propagation_raw_score": 0.0,
                        "target_weight": 0.0,
                        "route_candidate": not bool(item.get("fallback", False)),
                    }
                )
                phrase_results.append(item)
        else:
            case_id = str(case["case_id"])
            case_phrase_ids = {str(value["phrase_id"]) for value in case["phrases"]}
            support_ids = [int(value["image_id"]) for value in case["supports"]]
            for phrase in span_row["phrases"]:
                item = dict(phrase)
                phrase_id = str(item["phrase_id"])
                if phrase_id not in case_phrase_ids:
                    item.update(
                        {
                            "anchor_label": "unknown",
                            "sibling_label": "unknown",
                            "support_labels": [],
                            "propagation_raw_score": 0.0,
                            "target_weight": 0.0,
                            "route_candidate": False,
                        }
                    )
                    phrase_results.append(item)
                    continue

                anchor, anchor_agree = judgment(
                    predictions, case_id, phrase_id, "anchor"
                )
                sibling, sibling_agree = judgment(
                    predictions, case_id, phrase_id, "sibling"
                )
                supports = []
                support_agreements = []
                for support_id in support_ids:
                    label, agree = judgment(
                        predictions,
                        case_id,
                        phrase_id,
                        "support",
                        support_id=support_id,
                    )
                    supports.append(label)
                    support_agreements.append(bool(agree))
                    label_counter["support_image_{}".format(label)] += 1
                agreement_counter["anchor_total"] += 1
                agreement_counter["anchor_agree"] += int(anchor_agree)
                agreement_counter["sibling_total"] += 1
                agreement_counter["sibling_agree"] += int(sibling_agree)
                agreement_counter["support_total"] += len(support_agreements)
                agreement_counter["support_agree"] += sum(support_agreements)
                label_counter["anchor_{}".format(anchor)] += 1
                label_counter["sibling_{}".format(sibling)] += 1

                score = propagation_raw_score(anchor, sibling, supports)
                item.update(
                    {
                        "anchor_label": anchor,
                        "sibling_label": sibling,
                        "support_labels": [
                            {
                                "image_id": support_id,
                                "label": label,
                                "strict_agreement": support_agreements[position],
                            }
                            for position, (support_id, label) in enumerate(
                                zip(support_ids, supports)
                            )
                        ],
                        "anchor_strict_agreement": bool(anchor_agree),
                        "sibling_strict_agreement": bool(sibling_agree),
                        **score,
                    }
                )
                phrase_results.append(item)

        phrase_results, route_valid = normalize_record_phrase_targets(
            phrase_results, raw_field="propagation_raw_score"
        )
        supervised_records += int(route_valid)
        merged = dict(span_row)
        merged.update(
            {
                "experiment_version": "v16.6.0",
                "route_kind": "propagation",
                "teacher_case_present": case is not None,
                "teacher_case_id": None if case is None else str(case["case_id"]),
                "route_supervision_valid": bool(route_valid),
                "phrases": phrase_results,
            }
        )
        merged_rows.append(merged)

    write_jsonl(cli.output_labels, merged_rows)
    summary = {
        "experiment_version": "v16.6.0",
        "route_kind": "propagation",
        "record_count": len(merged_rows),
        "teacher_case_count": sum(
            bool(row.get("teacher_case_present")) for row in merged_rows
        ),
        "supervised_record_count": supervised_records,
        "supervised_record_ratio": supervised_records / max(1, len(merged_rows)),
        "strict_agreement": {
            "anchor": agreement_counter["anchor_agree"]
            / max(1, agreement_counter["anchor_total"]),
            "sibling": agreement_counter["sibling_agree"]
            / max(1, agreement_counter["sibling_total"]),
            "support": agreement_counter["support_agree"]
            / max(1, agreement_counter["support_total"]),
        },
        "labels": dict(label_counter),
        "output_labels": str(Path(cli.output_labels).expanduser().resolve()),
    }
    target = Path(cli.output_summary).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

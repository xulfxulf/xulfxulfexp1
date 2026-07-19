#!/usr/bin/env python
"""Audit phrase teacher coverage and prepare a deterministic manual review CSV."""

from __future__ import annotations

import argparse
import csv
import json
import os.path as op
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = op.abspath(op.join(op.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.phrase_route_io import read_jsonl


CATEGORIES = ("upper", "lower", "shoes", "bag", "hat", "hair", "pose")


def parse_args():
    parser = argparse.ArgumentParser(description="Audit phrase teacher labels")
    parser.add_argument("--labels", required=True)
    parser.add_argument(
        "--cases",
        default="",
        help="teacher case JSONL used to restrict the audit to actually annotated rows",
    )
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--review-csv", required=True)
    parser.add_argument("--per-category", type=int, default=50)
    parser.add_argument(
        "--completed-review-csv",
        default="",
        help="optional filled review CSV; manual_label must be correct/incorrect/uncertain",
    )
    return parser.parse_args()


def main():
    cli = parse_args()
    rows = read_jsonl(cli.labels)
    case_record_indices = None
    if cli.cases:
        case_record_indices = {
            int(row["record_index"]) for row in read_jsonl(cli.cases)
        }
    phrase_count = 0
    unknown = 0
    support = 0
    contradiction = 0
    category_pool = defaultdict(list)
    target_entropy_values = []

    for row in rows:
        if case_record_indices is not None:
            teacher_case_present = int(row["record_index"]) in case_record_indices
        elif "comparative_teacher_case_present" in row:
            teacher_case_present = bool(row["comparative_teacher_case_present"])
        elif "teacher_case_present" in row:
            teacher_case_present = bool(row["teacher_case_present"])
        else:
            raise RuntimeError(
                "Cannot distinguish teacher-covered rows. Pass --cases or regenerate "
                "labels with teacher_case_present metadata."
            )
        if not teacher_case_present:
            continue
        weights = []
        for phrase in row.get("phrases", []):
            if phrase.get("fallback", False):
                continue
            phrase_count += 1
            labels = [item["label"] for item in phrase.get("support_labels", [])]
            unknown += sum(value == "unknown" for value in labels)
            support += sum(value == "support" for value in labels)
            contradiction += sum(value == "contradiction" for value in labels)
            weights.append(float(phrase.get("target_weight", 0.0)))
            category = phrase.get("category", "other")
            if category in CATEGORIES:
                category_pool[category].append(
                    {
                        "record_index": row.get("record_index"),
                        "image_id": row.get("image_id"),
                        "caption": row.get("caption"),
                        "phrase_id": phrase.get("phrase_id"),
                        "phrase": phrase.get("text"),
                        "category": category,
                        "anchor_label": phrase.get("anchor_label"),
                        "sibling_label": phrase.get("sibling_label"),
                        "support_labels": "|".join(labels),
                        "raw_score": phrase.get(
                            "comparative_raw_score",
                            phrase.get("propagation_raw_score", 0.0),
                        ),
                        "target_weight": phrase.get("target_weight", 0.0),
                        "manual_label": "",
                        "manual_comment": "",
                    }
                )
        positive = [value for value in weights if value > 0.0]
        if positive:
            import math

            target_entropy_values.append(
                -sum(value * math.log(max(value, 1e-8)) for value in positive)
            )

    review_rows = []
    for category in CATEGORIES:
        candidates = sorted(
            category_pool[category],
            key=lambda item: (
                int(item["record_index"]), str(item["phrase_id"])
            ),
        )
        review_rows.extend(candidates[: cli.per_category])

    review_path = Path(cli.review_csv).expanduser().resolve()
    review_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(review_rows[0].keys()) if review_rows else [
        "record_index",
        "image_id",
        "caption",
        "phrase_id",
        "phrase",
        "category",
        "anchor_label",
        "sibling_label",
        "support_labels",
        "raw_score",
        "target_weight",
        "manual_label",
        "manual_comment",
    ]
    with review_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(review_rows)

    manual_correct = 0
    manual_incorrect = 0
    manual_uncertain = 0
    if cli.completed_review_csv:
        with Path(cli.completed_review_csv).expanduser().resolve().open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            for row in csv.DictReader(handle):
                label = str(row.get("manual_label", "")).strip().lower()
                if label == "correct":
                    manual_correct += 1
                elif label == "incorrect":
                    manual_incorrect += 1
                elif label == "uncertain":
                    manual_uncertain += 1
                elif label:
                    raise RuntimeError(
                        "manual_label must be correct/incorrect/uncertain, got {!r}".format(
                            label
                        )
                    )
    manual_denominator = manual_correct + manual_incorrect
    manual_accuracy = (
        manual_correct / manual_denominator if manual_denominator else None
    )

    total_support_judgments = support + contradiction + unknown
    supervised = sum(bool(row.get("route_supervision_valid")) for row in rows)
    summary = {
        "experiment_version": rows[0].get("experiment_version") if rows else None,
        "route_kind": rows[0].get("route_kind") if rows else None,
        "record_count": len(rows),
        "audited_teacher_case_count": sum(
            1
            for row in rows
            if (
                int(row["record_index"]) in case_record_indices
                if case_record_indices is not None
                else bool(
                    row.get(
                        "comparative_teacher_case_present",
                        row.get("teacher_case_present", False),
                    )
                )
            )
        ),
        "supervised_record_count": supervised,
        "supervised_record_ratio": supervised / max(1, len(rows)),
        "phrase_count": phrase_count,
        "support_relation_ratio": support / max(1, total_support_judgments),
        "contradiction_relation_ratio": contradiction
        / max(1, total_support_judgments),
        "unknown_relation_ratio": unknown / max(1, total_support_judgments),
        "mean_teacher_target_entropy": (
            sum(target_entropy_values) / max(1, len(target_entropy_values))
        ),
        "manual_review_rows": len(review_rows),
        "manual_review_expected": cli.per_category * len(CATEGORIES),
        "manual_review_counts_by_category": {
            category: sum(row["category"] == category for row in review_rows)
            for category in CATEGORIES
        },
        "manual_review": {
            "correct": manual_correct,
            "incorrect": manual_incorrect,
            "uncertain": manual_uncertain,
            "accuracy_excluding_uncertain": manual_accuracy,
        },
        "quality_gate": {
            "manual_review_complete": len(review_rows)
            == cli.per_category * len(CATEGORIES),
            "unknown_ratio_le_0_70": unknown / max(1, total_support_judgments) <= 0.70,
            "all_three_labels_present": support > 0 and contradiction > 0 and unknown > 0,
            "manual_accuracy_ge_0_80": (
                None if manual_accuracy is None else manual_accuracy >= 0.80
            ),
        },
    }
    target = Path(cli.output_summary).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

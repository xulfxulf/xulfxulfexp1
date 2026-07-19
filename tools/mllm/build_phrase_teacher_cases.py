#!/usr/bin/env python
"""Build v16.6.0 multi-view phrase teacher cases from TAG-PEDES train."""

from __future__ import annotations

import argparse
import json
import os.path as op
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = op.abspath(op.join(op.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.hire_v2_identity_dataset import HIREV2IdentityDataset
from datasets.phrase_route_io import PhraseRouteTable
from datasets.tagpedes import TAGPEDES
from tools.mllm.phrase_teacher_common import stable_sha1, write_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description="Build v16.6 phrase teacher cases")
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--train-spans", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--support-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--teacher-support-epoch", type=int, default=0)
    parser.add_argument(
        "--per-category",
        type=int,
        default=0,
        help="0 builds all cases; positive value builds a deterministic quality subset",
    )
    return parser.parse_args()


def quality_subset(cases, per_category):
    if per_category <= 0:
        return cases
    categories = ("upper", "lower", "shoes", "bag", "hat", "hair", "pose")
    pools = {}
    for category in categories:
        pools[category] = [
            case
            for case in cases
            if category
            in {
                phrase.get("category", "other")
                for phrase in case.get("phrases", [])
            }
        ]

    # Assign each quality case to exactly one category.  Selecting one multi-category
    # case for several counters made ``--per-category 50`` produce fewer than the
    # documented 350 unique cases and invalidated the manual quality gate.
    selected = []
    selected_ids = set()
    counts = {category: 0 for category in categories}
    for category in sorted(categories, key=lambda value: (len(pools[value]), value)):
        for case in pools[category]:
            case_id = str(case["case_id"])
            if case_id in selected_ids:
                continue
            item = dict(case)
            item["quality_sampling_category"] = category
            selected.append(item)
            selected_ids.add(case_id)
            counts[category] += 1
            if counts[category] == per_category:
                break
        if counts[category] != per_category:
            raise RuntimeError(
                "Quality subset cannot provide {} unique cases for {!r}; got {}"
                .format(per_category, category, counts[category])
            )

    expected = int(per_category) * len(categories)
    if len(selected) != expected or len(selected_ids) != expected:
        raise RuntimeError(
            "Quality subset must contain {} unique cases, got {}"
            .format(expected, len(selected))
        )
    return selected


def main():
    cli = parse_args()
    dataset = TAGPEDES(root=cli.root_dir, verbose=False)
    span_table = PhraseRouteTable(
        cli.train_spans,
        split="train",
        expected_version="span-only",
        expected_route_kind="span-only",
    )
    support_dataset = HIREV2IdentityDataset(
        dataset.train,
        transform=None,
        text_length=77,
        support_size=cli.support_size,
        support_image_views=getattr(dataset, "train_image_views", None),
        seed=cli.seed,
    )
    support_dataset.set_epoch(cli.teacher_support_epoch)

    records_by_image = defaultdict(list)
    for index, (_pid, image_id, _path, _caption) in enumerate(dataset.train):
        records_by_image[int(image_id)].append(index)

    cases = []
    skipped_fallback_only = 0
    for index, (pid, image_id, image_path, caption) in enumerate(dataset.train):
        span_row = span_table.validate_caption(index, caption, int(image_id))
        phrases = [
            dict(phrase)
            for phrase in span_row["phrases"]
            if not bool(phrase.get("fallback", False))
        ]
        if not phrases:
            skipped_fallback_only += 1
            continue

        sibling_indices = [
            value for value in records_by_image[int(image_id)] if value != index
        ]
        sibling_caption = (
            dataset.train[sibling_indices[0]][3] if sibling_indices else ""
        )
        supports = []
        for support_index in support_dataset.support_indices_for(index):
            support_pid, support_image_id, support_path, _support_caption = dataset.train[
                support_index
            ]
            if int(support_pid) != int(pid):
                raise RuntimeError("Teacher support crossed a PID boundary")
            if int(support_image_id) == int(image_id):
                raise RuntimeError("Teacher support reused the anchor image")
            supports.append(
                {
                    "image_id": int(support_image_id),
                    "path": support_path,
                    "view": (
                        None
                        if getattr(dataset, "train_image_views", None) is None
                        else dataset.train_image_views[int(support_image_id)]
                    ),
                }
            )
        if len(supports) < 2:
            # The training model also masks groups with fewer than two supports.
            continue

        cases.append(
            {
                "case_version": "v16.6.0",
                "case_type": "propagation",
                "case_id": stable_sha1(
                    "v16.6.0", index, image_id, span_row["caption_sha1"]
                )[:24],
                "record_index": index,
                "pid": int(pid),
                "image_id": int(image_id),
                "caption": caption,
                "caption_sha1": span_row["caption_sha1"],
                "sibling_caption": sibling_caption,
                "anchor_image_path": image_path,
                "supports": supports,
                "phrases": phrases,
                "teacher_support_epoch": int(cli.teacher_support_epoch),
            }
        )

    cases = quality_subset(cases, cli.per_category)
    write_jsonl(cli.output_file, cases)
    summary = {
        "case_count": len(cases),
        "phrase_count": sum(len(case["phrases"]) for case in cases),
        "skipped_fallback_only": skipped_fallback_only,
        "per_category": cli.per_category,
        "quality_sampling_counts": {
            category: sum(
                case.get("quality_sampling_category") == category for case in cases
            )
            for category in ("upper", "lower", "shoes", "bag", "hat", "hair", "pose")
        },
        "support_size": cli.support_size,
        "teacher_support_epoch": cli.teacher_support_epoch,
        "output": str(Path(cli.output_file).expanduser().resolve()),
    }
    summary_path = Path(cli.output_file).expanduser().resolve().with_suffix(
        ".summary.json"
    )
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

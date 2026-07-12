#!/usr/bin/env python3
"""Convert frozen TAG-PEDES diagnostics into the three v16 fast3 input tables.

The script is deliberately conversion-only: it reads existing diagnostics and
never loads a retrieval model or re-mines candidates.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


SLOTS = ("upper", "lower", "shoes", "bag", "hat", "hair", "pose")
RELIABILITY = {
    "consistent": 1.0,
    "complementary": 1.0,
    "omission": 1.0,
    "soft_mismatch": 1.0,
    "hard_contradiction": 0.0,
    "unparsed": 0.0,
}
HARD_CONTRADICTION_TYPES = {
    "hard_contradiction",
    "hard-contradiction",
    "hard contradiction",
    "explicit_contradiction",
    "explicit-contradiction",
    "explicit contradiction",
}
ANCHOR_FIELD_ALIASES = (
    "anchor_index",
    "anchor_sample_id",
    "anchor_dataset_index",
)
BAD_CANDIDATE_FLAGS = (
    "known_duplicate_identity",
    "duplicate_identity",
    "is_duplicate_identity",
    "known_annotation_error",
    "annotation_error",
    "has_annotation_error",
)


class InputValidationError(RuntimeError):
    pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare validated v16 fast3 TAG-PEDES input tables from frozen diagnostics."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--consistency-csv", required=True)
    parser.add_argument("--support-conflict-csv", required=True)
    parser.add_argument("--similar-pid-csv", required=True)
    parser.add_argument(
        "--similar-score-csv",
        default="",
        help=(
            "Optional frozen relation_score_table.csv. Required when --similar-pid-csv "
            "is the native relation_pairs_with_hard.csv table rather than an aggregated pool."
        ),
    )
    parser.add_argument("--similar-score-mode", default="split_pure")
    parser.add_argument("--similar-score-head", default="id")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _read_csv(path):
    path = Path(path)
    if not path.is_file():
        raise InputValidationError(f"Input csv does not exist: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise InputValidationError(f"Input csv has no header: {path}")
        return list(reader), list(reader.fieldnames)


def _write_csv(path, fieldnames, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _as_int(value, field, row):
    try:
        return int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise InputValidationError(f"Invalid integer {field}={value!r} in row: {row}") from exc


def _as_bool(value, field, row):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no", ""}:
        return False
    raise InputValidationError(f"Invalid boolean {field}={value!r} in row: {row}")


def _find_column(fieldnames, choices, label, required=True):
    for choice in choices:
        if choice in fieldnames:
            return choice
    if required:
        raise InputValidationError(
            f"Missing {label}; expected one of {list(choices)}, found {fieldnames}"
        )
    return None


def _tag_train_annotations(dataset_root):
    dataset_dir = Path(dataset_root) / "TAG-PEDES"
    candidates = (
        dataset_dir / "anno_dir" / "train_reid.json",
        dataset_dir / "train_reid.json",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise InputValidationError(
        "TAG-PEDES train_reid.json is missing under "
        f"{dataset_dir}; checked {[str(path) for path in candidates]}"
    )


def load_tag_training_index(dataset_root):
    annotation_path = _tag_train_annotations(dataset_root)
    with annotation_path.open("r", encoding="utf-8") as handle:
        annotations = json.load(handle)
    if not isinstance(annotations, list):
        raise InputValidationError(f"TAG train annotation must be a list: {annotation_path}")

    original_to_train_pid = {}
    image_by_id = {}
    caption_by_index = {}
    next_pid = 0
    dataset_index = 0
    for image_id, annotation in enumerate(annotations):
        try:
            original_pid = int(annotation["id"])
            captions = annotation["captions"]
        except (KeyError, TypeError, ValueError) as exc:
            raise InputValidationError(
                f"Invalid TAG training annotation at image_id={image_id}: {annotation}"
            ) from exc
        if original_pid not in original_to_train_pid:
            original_to_train_pid[original_pid] = next_pid
            next_pid += 1
        train_pid = original_to_train_pid[original_pid]
        image_by_id[image_id] = {
            "pid": train_pid,
            "original_pid": original_pid,
        }
        if not isinstance(captions, list):
            raise InputValidationError(
                f"TAG captions must be a list at image_id={image_id}"
            )
        for caption in captions:
            caption_by_index[dataset_index] = {
                "pid": train_pid,
                "image_id": image_id,
                "caption": str(caption),
            }
            dataset_index += 1

    if not image_by_id or not caption_by_index:
        raise InputValidationError("TAG-PEDES training data is empty")
    return {
        "annotation_path": annotation_path,
        "image_by_id": image_by_id,
        "caption_by_index": caption_by_index,
        "original_to_train_pid": original_to_train_pid,
        "num_pids": next_pid,
    }


def prepare_reliability(rows, train_index, audit):
    image_by_id = train_index["image_by_id"]
    table = {}
    duplicate_rows = 0
    for row in rows:
        image_id = _as_int(row.get("image_id"), "image_id", row)
        slot = str(row.get("slot", "")).strip()
        consistency_type = str(row.get("consistency_type", "")).strip()
        if image_id not in image_by_id:
            raise InputValidationError(
                f"Consistency row refers to non-training image_id={image_id}"
            )
        if slot not in SLOTS:
            raise InputValidationError(f"Unsupported consistency slot={slot!r}")
        if consistency_type not in RELIABILITY:
            raise InputValidationError(
                f"Unsupported consistency_type={consistency_type!r} for image_id={image_id}"
            )
        key = (image_id, slot)
        previous = table.get(key)
        if previous is not None:
            if previous != consistency_type:
                raise InputValidationError(
                    f"Conflicting consistency rows for image_id={image_id}, slot={slot}"
                )
            duplicate_rows += 1
            continue
        table[key] = consistency_type

    missing = [
        {"image_id": image_id, "slot": slot}
        for image_id in sorted(image_by_id)
        for slot in SLOTS
        if (image_id, slot) not in table
    ]
    coverage = {
        slot: sum((image_id, slot) in table for image_id in image_by_id) / len(image_by_id)
        for slot in SLOTS
    }
    output_rows = [
        {
            "image_id": image_id,
            "slot": slot,
            "consistency_type": table[(image_id, slot)],
            "reliability": f"{RELIABILITY[table[(image_id, slot)]]:.1f}",
        }
        for image_id in sorted(image_by_id)
        for slot in SLOTS
        if (image_id, slot) in table
    ]
    audit.update({
        "theoretical_reliability_rows": len(image_by_id) * len(SLOTS),
        "actual_reliability_rows": len(output_rows),
        "slot_coverage": coverage,
        "missing_reliability_entries": missing,
        "reliability_duplicate_rows": duplicate_rows,
    })
    return output_rows, missing


def prepare_conflicts(rows, fieldnames, train_index, audit):
    anchor_field = _find_column(fieldnames, ANCHOR_FIELD_ALIASES, "anchor index")
    support_field = _find_column(
        fieldnames, ("support_image_id", "candidate_image_id"), "support image id"
    )
    direct_flag_field = "has_hard_contradiction" if "has_hard_contradiction" in fieldnames else None
    if direct_flag_field is None and "conflict_type" not in fieldnames:
        raise InputValidationError(
            "Support conflict csv requires has_hard_contradiction or conflict_type"
        )

    image_by_id = train_index["image_by_id"]
    caption_by_index = train_index["caption_by_index"]
    slots_by_relation = defaultdict(set)
    duplicate_rows = 0
    seen_rows = set()
    for row in rows:
        anchor_index = _as_int(row.get(anchor_field), anchor_field, row)
        support_image_id = _as_int(row.get(support_field), support_field, row)
        if anchor_index not in caption_by_index:
            raise InputValidationError(
                f"Cannot map {anchor_field}={anchor_index} to a stable training index"
            )
        if support_image_id not in image_by_id:
            raise InputValidationError(
                f"Support relation refers to non-training image_id={support_image_id}"
            )
        anchor = caption_by_index[anchor_index]
        if anchor["image_id"] == support_image_id:
            raise InputValidationError(
                f"Support relation reuses anchor image at anchor_index={anchor_index}"
            )
        if anchor["pid"] != image_by_id[support_image_id]["pid"]:
            raise InputValidationError(
                f"Support relation crosses PID at anchor_index={anchor_index}"
            )

        if direct_flag_field:
            has_hard = _as_bool(row.get(direct_flag_field), direct_flag_field, row)
            slots_text = str(row.get("contradict_slots", row.get("slot", ""))).strip()
        else:
            conflict_type = str(row.get("conflict_type", "")).strip().lower()
            has_hard = conflict_type in HARD_CONTRADICTION_TYPES
            slots_text = str(row.get("contradict_slots", row.get("slot", ""))).strip()
        if not has_hard:
            continue
        slots = tuple(slot.strip() for slot in slots_text.split("|") if slot.strip())
        if not slots:
            raise InputValidationError(
                f"Explicit hard conflict has no slot at anchor_index={anchor_index}"
            )
        if any(slot not in SLOTS for slot in slots):
            raise InputValidationError(
                f"Unsupported hard-conflict slots {slots} at anchor_index={anchor_index}"
            )
        row_key = (anchor_index, support_image_id, tuple(sorted(slots)))
        if row_key in seen_rows:
            duplicate_rows += 1
            continue
        seen_rows.add(row_key)
        slots_by_relation[(anchor_index, support_image_id)].update(slots)

    output_rows = [
        {
            "anchor_index": anchor_index,
            "support_image_id": support_image_id,
            "has_hard_contradiction": 1,
            "contradict_slots": "|".join(
                slot for slot in SLOTS if slot in slots_by_relation[(anchor_index, support_image_id)]
            ),
        }
        for anchor_index, support_image_id in sorted(slots_by_relation)
    ]
    conflict_anchors = {row["anchor_index"] for row in output_rows}
    audit.update({
        "hard_conflict_relation_count": len(output_rows),
        "hard_conflict_anchor_ratio": len(conflict_anchors) / len(caption_by_index),
        "conflict_duplicate_rows": duplicate_rows,
        "conflict_anchor_field": anchor_field,
    })
    return output_rows


def _map_pid(raw_value, original_value, original_to_train_pid, valid_train_pids, label, row):
    if original_value not in (None, ""):
        original_pid = _as_int(original_value, f"{label}_original_pid", row)
        if original_pid not in original_to_train_pid:
            raise InputValidationError(
                f"Unknown original {label}_pid={original_pid} in row: {row}"
            )
        return original_to_train_pid[original_pid]

    raw_pid = _as_int(raw_value, f"{label}_pid", row)
    as_train = raw_pid if raw_pid in valid_train_pids else None
    as_original = original_to_train_pid.get(raw_pid)
    if as_train is not None and as_original is not None and as_train != as_original:
        raise InputValidationError(
            f"Ambiguous {label}_pid={raw_pid}; provide {label}_original_pid explicitly"
        )
    mapped = as_train if as_train is not None else as_original
    if mapped is None:
        raise InputValidationError(f"Unknown {label}_pid={raw_pid} in row: {row}")
    return mapped


def prepare_hard_negatives(rows, fieldnames, train_index, audit):
    required = {
        "anchor_pid",
        "negative_pid",
        "negative_image_id",
        "rank",
        "trigger_caption_count",
        "trigger_image_count",
    }
    missing_fields = sorted(required - set(fieldnames))
    if missing_fields:
        raise InputValidationError(
            f"Similar-PID csv is missing mandatory fields: {missing_fields}"
        )

    image_by_id = train_index["image_by_id"]
    original_to_train_pid = train_index["original_to_train_pid"]
    valid_train_pids = {record["pid"] for record in image_by_id.values()}
    canonical = {}
    filtered = 0
    duplicate_rows = 0
    for row in rows:
        if any(
            field in fieldnames and _as_bool(row.get(field), field, row)
            for field in BAD_CANDIDATE_FLAGS
        ):
            filtered += 1
            continue
        negative_image_id = _as_int(
            row.get("negative_image_id"), "negative_image_id", row
        )
        if negative_image_id not in image_by_id:
            filtered += 1
            continue
        anchor_pid = _map_pid(
            row.get("anchor_pid"),
            row.get("anchor_original_pid"),
            original_to_train_pid,
            valid_train_pids,
            "anchor",
            row,
        )
        negative_pid = _map_pid(
            row.get("negative_pid"),
            row.get("negative_original_pid"),
            original_to_train_pid,
            valid_train_pids,
            "negative",
            row,
        )
        rank = _as_int(row.get("rank"), "rank", row)
        trigger_caption_count = _as_int(
            row.get("trigger_caption_count"), "trigger_caption_count", row
        )
        trigger_image_count = _as_int(
            row.get("trigger_image_count"), "trigger_image_count", row
        )
        if (
            anchor_pid == negative_pid
            or negative_pid != image_by_id[negative_image_id]["pid"]
            or rank < 1
            or trigger_caption_count < 3
            or trigger_image_count < 2
        ):
            filtered += 1
            continue
        key = (anchor_pid, negative_pid, negative_image_id)
        candidate = {
            "anchor_pid": anchor_pid,
            "negative_pid": negative_pid,
            "negative_image_id": negative_image_id,
            "rank": rank,
            "trigger_caption_count": trigger_caption_count,
            "trigger_image_count": trigger_image_count,
        }
        previous = canonical.get(key)
        if previous is not None:
            duplicate_rows += 1
            if (candidate["rank"], candidate["negative_image_id"]) < (
                previous["rank"], previous["negative_image_id"]
            ):
                canonical[key] = candidate
        else:
            canonical[key] = candidate

    output_rows = sorted(
        canonical.values(),
        key=lambda row: (row["anchor_pid"], row["rank"], row["negative_image_id"]),
    )
    anchors_with_candidate = {row["anchor_pid"] for row in output_rows}
    audit.update({
        "hard_negative_anchor_pid_ratio": (
            len(anchors_with_candidate) / len(valid_train_pids)
        ),
        "hard_negative_candidate_count": len(output_rows),
        "invalid_hard_negative_rows_filtered": filtered,
        "hard_negative_duplicate_rows": duplicate_rows,
    })
    return output_rows


def derive_hard_negatives_from_relation_pairs(
    relation_rows,
    relation_fields,
    score_rows,
    score_fields,
    train_index,
    score_mode,
    score_head,
):
    """Aggregate frozen train E-relation candidates without rerunning a model."""
    relation_required = {
        "pair_id",
        "split",
        "relation_type",
        "anchor_text_index",
        "anchor_image_index",
        "anchor_pid",
        "candidate_image_index",
        "candidate_pid",
    }
    missing_relation_fields = sorted(relation_required - set(relation_fields))
    if missing_relation_fields:
        raise InputValidationError(
            "Native relation-pair table is missing fields: "
            f"{missing_relation_fields}"
        )
    score_required = {"mode", "head", "pair_id", "score"}
    missing_score_fields = sorted(score_required - set(score_fields))
    if missing_score_fields:
        raise InputValidationError(
            f"Relation score table is missing fields: {missing_score_fields}"
        )

    scores = {}
    for row in score_rows:
        if row.get("mode") != score_mode or row.get("head") != score_head:
            continue
        try:
            scores[row["pair_id"]] = float(row["score"])
        except (TypeError, ValueError) as exc:
            raise InputValidationError(f"Invalid relation score row: {row}") from exc
    if not scores:
        raise InputValidationError(
            f"No frozen scores found for mode={score_mode!r}, head={score_head!r}"
        )

    image_by_id = train_index["image_by_id"]
    valid_train_pids = {record["pid"] for record in image_by_id.values()}
    grouped = defaultdict(lambda: {
        "caption_ids": set(),
        "image_ids": set(),
        "candidate_scores": defaultdict(list),
    })
    selected_rows = 0
    for row in relation_rows:
        if row.get("split") != "train":
            continue
        if row.get("relation_type") != "E_different_identity_high_similarity_candidate":
            continue
        pair_id = row.get("pair_id")
        if pair_id not in scores:
            raise InputValidationError(
                f"Missing frozen {score_mode}/{score_head} score for pair_id={pair_id}"
            )
        anchor_pid = _as_int(row.get("anchor_pid"), "anchor_pid", row)
        negative_pid = _as_int(row.get("candidate_pid"), "candidate_pid", row)
        anchor_text_index = _as_int(
            row.get("anchor_text_index"), "anchor_text_index", row
        )
        anchor_image_index = _as_int(
            row.get("anchor_image_index"), "anchor_image_index", row
        )
        negative_image_id = _as_int(
            row.get("candidate_image_index"), "candidate_image_index", row
        )
        if anchor_pid not in valid_train_pids or negative_pid not in valid_train_pids:
            raise InputValidationError(f"Native relation pair has non-training PID: {row}")
        if anchor_pid == negative_pid:
            raise InputValidationError(f"Native hard relation has equal PIDs: {row}")
        if negative_image_id not in image_by_id:
            raise InputValidationError(
                f"Native hard relation references non-training image_id={negative_image_id}"
            )
        if image_by_id[negative_image_id]["pid"] != negative_pid:
            raise InputValidationError(f"Native hard relation PID/image mismatch: {row}")
        key = (anchor_pid, negative_pid)
        grouped[key]["caption_ids"].add(anchor_text_index)
        grouped[key]["image_ids"].add(anchor_image_index)
        grouped[key]["candidate_scores"][negative_image_id].append(scores[pair_id])
        selected_rows += 1

    by_anchor = defaultdict(list)
    for (anchor_pid, negative_pid), values in grouped.items():
        trigger_caption_count = len(values["caption_ids"])
        trigger_image_count = len(values["image_ids"])
        if trigger_caption_count < 3 or trigger_image_count < 2:
            continue
        ranked_images = sorted(
            (
                (sum(scores_for_image) / len(scores_for_image), negative_image_id)
                for negative_image_id, scores_for_image in values["candidate_scores"].items()
            ),
            key=lambda item: (-item[0], item[1]),
        )
        mean_score, negative_image_id = ranked_images[0]
        by_anchor[anchor_pid].append({
            "anchor_pid": anchor_pid,
            "negative_pid": negative_pid,
            "negative_image_id": negative_image_id,
            "trigger_caption_count": trigger_caption_count,
            "trigger_image_count": trigger_image_count,
            "_mean_score": mean_score,
        })

    output_rows = []
    for anchor_pid in sorted(by_anchor):
        candidates = sorted(
            by_anchor[anchor_pid],
            key=lambda item: (-item["_mean_score"], item["negative_image_id"]),
        )
        for rank, candidate in enumerate(candidates, start=1):
            candidate = dict(candidate)
            candidate["rank"] = rank
            candidate.pop("_mean_score")
            output_rows.append(candidate)
    return output_rows, selected_rows


def write_audit(output_dir, audit):
    output_dir = Path(output_dir)
    json_path = output_dir / "fast3_input_audit.json"
    md_path = output_dir / "fast3_input_audit.md"
    json_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# v16 fast3 input audit\n\n")
        handle.write(f"- Training identities: {audit['training_identity_count']}\n")
        handle.write(f"- Training images: {audit['training_image_count']}\n")
        handle.write(f"- Theoretical reliability rows: {audit['theoretical_reliability_rows']}\n")
        handle.write(f"- Actual reliability rows: {audit['actual_reliability_rows']}\n")
        handle.write(f"- Explicit hard-conflict relations: {audit['hard_conflict_relation_count']}\n")
        handle.write(
            "- Anchors with an explicit hard-conflict support: "
            f"{audit['hard_conflict_anchor_ratio']:.4%}\n"
        )
        handle.write(
            "- PIDs with an eligible hard-negative candidate: "
            f"{audit['hard_negative_anchor_pid_ratio']:.4%}\n"
        )
        handle.write(
            "- Invalid hard-negative candidates filtered: "
            f"{audit['invalid_hard_negative_rows_filtered']}\n"
        )
        handle.write(
            "- Duplicate input rows: "
            f"{audit['duplicate_row_count']}\n"
        )
        handle.write(f"- Missing required fields: {audit['missing_field_count']}\n\n")
        handle.write("## Slot Coverage\n\n")
        handle.write("| slot | coverage |\n|---|---:|\n")
        for slot in SLOTS:
            handle.write(f"| {slot} | {audit['slot_coverage'][slot]:.2%} |\n")
        if audit["missing_reliability_entries"]:
            handle.write("\n## Missing Reliability Entries\n\n")
            for item in audit["missing_reliability_entries"]:
                handle.write(f"- image_id={item['image_id']}, slot={item['slot']}\n")
    return json_path, md_path


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_index = load_tag_training_index(args.dataset_root)
    consistency_rows, _consistency_fields = _read_csv(args.consistency_csv)
    conflict_rows, conflict_fields = _read_csv(args.support_conflict_csv)
    similar_rows, similar_fields = _read_csv(args.similar_pid_csv)

    audit = {
        "dataset_annotation": str(train_index["annotation_path"]),
        "consistency_csv": str(Path(args.consistency_csv).resolve()),
        "support_conflict_csv": str(Path(args.support_conflict_csv).resolve()),
        "similar_pid_csv": str(Path(args.similar_pid_csv).resolve()),
        "training_identity_count": train_index["num_pids"],
        "training_image_count": len(train_index["image_by_id"]),
        "training_caption_count": len(train_index["caption_by_index"]),
        "missing_field_count": 0,
    }
    native_relation_pair_fields = {
        "pair_id",
        "split",
        "relation_type",
        "anchor_text_index",
        "candidate_image_index",
        "candidate_pid",
    }
    if native_relation_pair_fields.issubset(similar_fields):
        if not args.similar_score_csv:
            raise InputValidationError(
                "Native relation_pairs_with_hard.csv requires --similar-score-csv "
                "to rank frozen high-similarity candidates."
            )
        score_rows, score_fields = _read_csv(args.similar_score_csv)
        similar_rows, native_relation_rows = derive_hard_negatives_from_relation_pairs(
            similar_rows,
            similar_fields,
            score_rows,
            score_fields,
            train_index,
            args.similar_score_mode,
            args.similar_score_head,
        )
        similar_fields = [
            "anchor_pid",
            "negative_pid",
            "negative_image_id",
            "rank",
            "trigger_caption_count",
            "trigger_image_count",
        ]
        audit.update({
            "similar_source_kind": "frozen_relation_pairs_with_hard",
            "similar_score_csv": str(Path(args.similar_score_csv).resolve()),
            "similar_score_mode": args.similar_score_mode,
            "similar_score_head": args.similar_score_head,
            "native_hard_relation_rows": native_relation_rows,
        })
    else:
        audit["similar_source_kind"] = "aggregated_similar_pid_csv"
    reliability_rows, missing_reliability = prepare_reliability(
        consistency_rows, train_index, audit
    )
    if missing_reliability:
        audit.update({
            "hard_conflict_relation_count": 0,
            "hard_conflict_anchor_ratio": 0.0,
            "conflict_duplicate_rows": 0,
            "hard_negative_anchor_pid_ratio": 0.0,
            "hard_negative_candidate_count": 0,
            "invalid_hard_negative_rows_filtered": 0,
            "hard_negative_duplicate_rows": 0,
            "duplicate_row_count": audit["reliability_duplicate_rows"],
        })
        write_audit(output_dir, audit)
        raise SystemExit(
            "Reliability coverage is incomplete; wrote fast3_input_audit with missing image-slot entries."
        )

    conflict_output = prepare_conflicts(conflict_rows, conflict_fields, train_index, audit)
    hard_negative_output = prepare_hard_negatives(
        similar_rows, similar_fields, train_index, audit
    )
    audit["duplicate_row_count"] = (
        audit["reliability_duplicate_rows"]
        + audit["conflict_duplicate_rows"]
        + audit["hard_negative_duplicate_rows"]
    )
    _write_csv(
        output_dir / "support_reliability_hard_only.csv",
        ["image_id", "slot", "consistency_type", "reliability"],
        reliability_rows,
    )
    _write_csv(
        output_dir / "support_hard_contradiction.csv",
        ["anchor_index", "support_image_id", "has_hard_contradiction", "contradict_slots"],
        conflict_output,
    )
    _write_csv(
        output_dir / "hard_negative_pool.csv",
        [
            "anchor_pid",
            "negative_pid",
            "negative_image_id",
            "rank",
            "trigger_caption_count",
            "trigger_image_count",
        ],
        hard_negative_output,
    )
    write_audit(output_dir, audit)
    print(f"Prepared v16 fast3 inputs in: {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except InputValidationError as exc:
        print(f"fast3 input validation failed: {exc}", file=sys.stderr)
        raise SystemExit(2)

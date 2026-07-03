import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


SLOTS = ["upper", "lower", "shoes", "bag", "hat", "hair", "pose"]

COLORS = {
    "black",
    "white",
    "red",
    "blue",
    "green",
    "yellow",
    "grey",
    "gray",
    "brown",
    "orange",
    "pink",
    "purple",
    "beige",
    "khaki",
    "dark",
    "light",
}

CATEGORY_TERMS = {
    "upper": {
        "shirt",
        "t-shirt",
        "tee",
        "top",
        "jacket",
        "coat",
        "hoodie",
        "sweater",
        "blouse",
        "vest",
        "uniform",
    },
    "lower": {
        "pants",
        "trousers",
        "jeans",
        "shorts",
        "skirt",
        "leggings",
    },
    "shoes": {
        "shoe",
        "shoes",
        "sneaker",
        "sneakers",
        "boot",
        "boots",
        "sandals",
        "footwear",
    },
    "bag": {
        "bag",
        "backpack",
        "handbag",
        "purse",
        "suitcase",
        "luggage",
        "cart",
    },
    "hat": {
        "hat",
        "cap",
        "helmet",
        "beanie",
    },
    "hair": {
        "hair",
        "ponytail",
        "bald",
        "braid",
        "bun",
    },
    "pose": {
        "walking",
        "standing",
        "sitting",
        "running",
        "riding",
        "pushing",
        "pulling",
        "dragging",
        "carrying",
        "holding",
    },
}

NEGATION_RE = re.compile(
    r"\b(no|not|without|does not|isn'?t|aren'?t|no visible|not carrying)\b",
    re.IGNORECASE,
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, rows):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def source_from_path(file_path):
    return Path(file_path).parts[0] if Path(file_path).parts else ""


def view_from_cam(cam_id):
    try:
        cam_id = int(cam_id)
    except (TypeError, ValueError):
        return "unknown"
    if cam_id == 0:
        return "aerial"
    if cam_id == 1:
        return "ground"
    return "unknown"


def normalize_text(text):
    text = text.lower().replace("grey", "gray")
    text = re.sub(r"([.!\"()*#:;~])", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def tokenize(text):
    text = normalize_text(text)
    text = re.sub(r"[^a-z0-9\- ]+", " ", text)
    return [t for t in text.split() if t]


def contains_negated_category(text, terms):
    text_l = text.lower()
    if not NEGATION_RE.search(text_l):
        return False
    return any(term in text_l for term in terms)


def extract_slot_value(caption, slot):
    text = normalize_text(caption)
    tokens = set(tokenize(text))
    terms = CATEGORY_TERMS[slot]
    term_hits = {t for t in terms if t in text or t in tokens}
    color_hits = COLORS & tokens
    negated = contains_negated_category(text, terms)
    if term_hits or negated:
        return {
            "terms": sorted(term_hits),
            "colors": sorted("gray" if c == "grey" else c for c in color_hits),
            "negated": bool(negated),
        }
    return None


def extract_slot_values(caption):
    return {slot: extract_slot_value(caption, slot) for slot in SLOTS}


def slot_value_to_text(value):
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def relation_type(anchor_value, support_value):
    if anchor_value is None and support_value is None:
        return "omission"
    if anchor_value is None:
        return "omission"
    if support_value is None:
        return "unknown"
    if bool(anchor_value["negated"]) != bool(support_value["negated"]):
        return "hard_contradiction"
    if anchor_value["negated"] and support_value["negated"]:
        return "support"

    a_terms = set(anchor_value["terms"])
    b_terms = set(support_value["terms"])
    a_colors = set(anchor_value["colors"])
    b_colors = set(support_value["colors"])

    if (a_terms and b_terms and a_terms & b_terms) or (a_colors and b_colors and a_colors & b_colors):
        return "support"
    if a_colors and b_colors and not (a_colors & b_colors):
        return "hard_contradiction"
    if a_terms and b_terms and not (a_terms & b_terms):
        return "soft_mismatch"
    return "unknown"


def consistency_type(values):
    present = [v for v in values if v is not None]
    if not present:
        return "omission"
    neg_states = {bool(v["negated"]) for v in present}
    if len(neg_states) > 1:
        return "hard_contradiction"
    if len(present) == 1:
        return "consistent"
    term_sets = [set(v["terms"]) for v in present if v["terms"]]
    color_sets = [set(v["colors"]) for v in present if v["colors"]]
    if color_sets and len(set().union(*color_sets)) > 1 and not set.intersection(*color_sets):
        return "hard_contradiction"
    if term_sets and len(set().union(*term_sets)) > 1 and not set.intersection(*term_sets):
        return "soft_mismatch"
    if len(present) < len(values):
        return "complementary"
    return "consistent"


def rotate(items, offset_seed):
    if not items:
        return items
    offset = offset_seed % len(items)
    return items[offset:] + items[:offset]


def build_train_samples(train_annos):
    person_id2idx = {}
    next_pid = 0
    images = []
    samples = []
    first_sample_by_pid_image = defaultdict(dict)

    for image_id, anno in enumerate(train_annos):
        person_id = int(anno["id"])
        if person_id not in person_id2idx:
            person_id2idx[person_id] = next_pid
            next_pid += 1
        pid = person_id2idx[person_id]
        file_path = anno["file_path"]
        view = view_from_cam(anno.get("cam_id"))
        image = {
            "image_id": image_id,
            "pid": pid,
            "original_pid": person_id,
            "file_path": file_path,
            "source": source_from_path(file_path),
            "cam_id": anno.get("cam_id", ""),
            "view": view,
            "captions": [normalize_text(c) for c in anno.get("captions", [])],
        }
        images.append(image)
        for caption_idx, caption in enumerate(image["captions"]):
            sample_id = len(samples)
            sample = {
                "sample_id": sample_id,
                "caption_id": f"train:{image_id}:{caption_idx}",
                "caption_idx": caption_idx,
                "pid": pid,
                "original_pid": person_id,
                "image_id": image_id,
                "file_path": file_path,
                "source": image["source"],
                "view": view,
                "caption": caption,
                "slot_values": extract_slot_values(caption),
            }
            first_sample_by_pid_image[pid].setdefault(image_id, sample_id)
            samples.append(sample)
    return images, samples, first_sample_by_pid_image


def build_support_indices(samples, first_sample_by_pid_image, support_size):
    support_indices = []
    view_by_image = {s["image_id"]: s["view"] for s in samples}
    for sample in samples:
        pid = int(sample["pid"])
        image_id = int(sample["image_id"])
        anchor_view = sample["view"]
        candidate_items = [
            (int(candidate_image_id), first_idx)
            for candidate_image_id, first_idx in sorted(first_sample_by_pid_image[pid].items())
            if int(candidate_image_id) != image_id
        ]
        if anchor_view != "unknown":
            cross_view_items = [
                (candidate_image_id, first_idx)
                for candidate_image_id, first_idx in candidate_items
                if view_by_image.get(candidate_image_id, "unknown") != "unknown"
                and view_by_image.get(candidate_image_id, "unknown") != anchor_view
            ]
            cross_set = set(cross_view_items)
            same_view_items = [
                (candidate_image_id, first_idx)
                for candidate_image_id, first_idx in candidate_items
                if (candidate_image_id, first_idx) not in cross_set
            ]
            candidate_items = rotate(cross_view_items, sample["sample_id"]) + rotate(same_view_items, sample["sample_id"])
        else:
            candidate_items = rotate(candidate_items, sample["sample_id"])
        support_indices.append([first_idx for _candidate_image_id, first_idx in candidate_items[:support_size]])
    return support_indices


def load_quality_map(path):
    by_file = {}
    for row in read_csv_rows(path):
        if row.get("split") != "train":
            continue
        by_file[row.get("file_path", "")] = row
    return by_file


def float_or_blank(value):
    if value is None:
        return ""
    try:
        if math.isnan(float(value)):
            return ""
    except (TypeError, ValueError):
        return value
    return value


def add_quality(samples, quality_by_file):
    for sample in samples:
        q = quality_by_file.get(sample["file_path"], {})
        sample["sharpness"] = q.get("sharpness", "")
        sample["quality_layer"] = q.get("quality_layer", "")
        sample["low_quality_aerial"] = sample["view"] == "aerial" and q.get("quality_layer") == "low_quality"


def support_usage(samples, support_indices, output_dir):
    rows = []
    support_source_counter = Counter()
    support_view_counter = Counter()
    used_counts = Counter()
    for sample, indices in zip(samples, support_indices):
        supports = [samples[i] for i in indices]
        same_view = sum(s["view"] == sample["view"] for s in supports)
        cross_view = sum(s["view"] != sample["view"] for s in supports)
        aerial = sum(s["view"] == "aerial" for s in supports)
        ground = sum(s["view"] == "ground" for s in supports)
        low_aerial = sum(bool(s.get("low_quality_aerial")) for s in supports)
        for support in supports:
            support_source_counter[support["source"]] += 1
            support_view_counter[support["view"]] += 1
        used_counts[len(supports)] += 1
        rows.append({
            "anchor_sample_id": sample["sample_id"],
            "anchor_pid": sample["pid"],
            "anchor_image_id": sample["image_id"],
            "anchor_view": sample["view"],
            "anchor_source": sample["source"],
            "anchor_sharpness": float_or_blank(sample.get("sharpness")),
            "num_support_available": max(0, len(first_images_by_pid[sample["pid"]]) - 1),
            "num_support_used": len(supports),
            "num_same_view_support": same_view,
            "num_cross_view_support": cross_view,
            "num_aerial_support": aerial,
            "num_ground_support": ground,
            "num_low_quality_aerial_support": low_aerial,
            "support_image_ids": ";".join(str(s["image_id"]) for s in supports),
            "support_views": ";".join(str(s["view"]) for s in supports),
            "support_sources": ";".join(str(s["source"]) for s in supports),
            "support_sharpness_values": ";".join(str(float_or_blank(s.get("sharpness"))) for s in supports),
        })

    write_csv(Path(output_dir) / "support_usage" / "support_usage_per_anchor.csv", rows)

    total_anchors = len(rows)
    total_support = sum(r["num_support_used"] for r in rows)
    def ratio(n, d):
        return float(n / d) if d else 0.0

    mean_used = ratio(total_support, total_anchors)
    same_total = sum(r["num_same_view_support"] for r in rows)
    cross_total = sum(r["num_cross_view_support"] for r in rows)
    aerial_total = sum(r["num_aerial_support"] for r in rows)
    ground_total = sum(r["num_ground_support"] for r in rows)
    low_aerial_total = sum(r["num_low_quality_aerial_support"] for r in rows)

    lines = [
        "# Scheme-2 Pre-Analysis: Support Usage",
        "",
        f"- anchors: {total_anchors}",
        f"- mean num_support_used: {mean_used:.4f}",
        "",
        "## support_size Distribution",
        "",
        "| support used | anchors | ratio |",
        "|---:|---:|---:|",
    ]
    for k in range(0, 4):
        lines.append(f"| {k} | {used_counts.get(k, 0)} | {ratio(used_counts.get(k, 0), total_anchors):.4f} |")
    lines += [
        "",
        "## Support Composition",
        "",
        f"- same-view support ratio: {ratio(same_total, total_support):.4f}",
        f"- cross-view support ratio: {ratio(cross_total, total_support):.4f}",
        f"- aerial support ratio: {ratio(aerial_total, total_support):.4f}",
        f"- ground support ratio: {ratio(ground_total, total_support):.4f}",
        f"- low-quality aerial support ratio: {ratio(low_aerial_total, total_support):.4f}",
        "",
        "## Source Distribution",
        "",
        "| source | support count | ratio |",
        "|---|---:|---:|",
    ]
    for source, count in support_source_counter.most_common():
        lines.append(f"| {source} | {count} | {ratio(count, total_support):.4f} |")
    Path(output_dir, "support_usage", "support_usage_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def support_conflicts(samples, images, support_indices, output_dir):
    rows = []
    low_quality_by_sample = {int(s["sample_id"]): bool(s.get("low_quality_aerial")) for s in samples}
    for sample, indices in zip(samples, support_indices):
        for support_index in indices:
            support = samples[support_index]
            for slot in SLOTS:
                av = sample["slot_values"][slot]
                sv = support["slot_values"][slot]
                rows.append({
                    "anchor_sample_id": sample["sample_id"],
                    "support_sample_id": support["sample_id"],
                    "anchor_pid": sample["pid"],
                    "support_pid": support["pid"],
                    "anchor_image_id": sample["image_id"],
                    "support_image_id": support["image_id"],
                    "anchor_view": sample["view"],
                    "support_view": support["view"],
                    "anchor_source": sample["source"],
                    "support_source": support["source"],
                    "anchor_sharpness": float_or_blank(sample.get("sharpness")),
                    "support_sharpness": float_or_blank(support.get("sharpness")),
                    "slot": slot,
                    "anchor_slot_value": slot_value_to_text(av),
                    "support_slot_value": slot_value_to_text(sv),
                    "conflict_type": relation_type(av, sv),
                })

    write_csv(Path(output_dir) / "support_conflict" / "support_conflict_pairs.csv", rows)

    image_consistency_rows = []
    for image in images:
        captions = image["captions"]
        for slot in SLOTS:
            values = [extract_slot_value(caption, slot) for caption in captions]
            image_consistency_rows.append({
                "image_id": image["image_id"],
                "slot": slot,
                "consistency_type": consistency_type(values),
                "caption_values": " || ".join(slot_value_to_text(v) for v in values),
                "view": image["view"],
                "source": image["source"],
                "sharpness": "",
            })
    write_csv(Path(output_dir) / "support_conflict" / "intra_image_caption_consistency.csv", image_consistency_rows)

    total_by_slot = defaultdict(Counter)
    same_cross = {"same_view": Counter(), "cross_view": Counter()}
    low_quality_counter = Counter()
    source_counter = defaultdict(Counter)
    for row in rows:
        ctype = row["conflict_type"]
        total_by_slot[row["slot"]][ctype] += 1
        rel_view = "same_view" if row["anchor_view"] == row["support_view"] else "cross_view"
        same_cross[rel_view][ctype] += 1
        support_is_low_quality_aerial = row["support_view"] == "aerial" and low_quality_by_sample.get(int(row["support_sample_id"]), False)
        if support_is_low_quality_aerial:
            low_quality_counter[ctype] += 1
        source_counter[row["support_source"]][ctype] += 1

    def c_ratio(counter, key):
        denom = sum(counter.values())
        return float(counter.get(key, 0) / denom) if denom else 0.0

    lines = [
        "# Scheme-2 Pre-Analysis: Support Conflict",
        "",
        "Phrase evidence is a caption-text proxy, not visual verification.",
        "",
        "## Slot Conflict Distribution",
        "",
        "| slot | support | omission | soft_mismatch | hard_contradiction | unknown | unparsed | hard_contradiction_ratio | unknown_ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for slot in SLOTS:
        ctr = total_by_slot[slot]
        lines.append(
            f"| {slot} | {ctr.get('support', 0)} | {ctr.get('omission', 0)} | "
            f"{ctr.get('soft_mismatch', 0)} | {ctr.get('hard_contradiction', 0)} | "
            f"{ctr.get('unknown', 0)} | {ctr.get('unparsed', 0)} | "
            f"{c_ratio(ctr, 'hard_contradiction'):.4f} | {c_ratio(ctr, 'unknown'):.4f} |"
        )
    lines += [
        "",
        "## View Layer Risk",
        "",
        f"- same-view hard_contradiction ratio: {c_ratio(same_cross['same_view'], 'hard_contradiction'):.4f}",
        f"- cross-view hard_contradiction ratio: {c_ratio(same_cross['cross_view'], 'hard_contradiction'):.4f}",
        f"- low-quality aerial unknown ratio: {c_ratio(low_quality_counter, 'unknown'):.4f}",
        "",
        "## Source Hard-Contradiction Ratio",
        "",
        "| source | pairs | hard_contradiction_ratio | unknown_ratio |",
        "|---|---:|---:|---:|",
    ]
    for source, ctr in sorted(source_counter.items()):
        lines.append(f"| {source} | {sum(ctr.values())} | {c_ratio(ctr, 'hard_contradiction'):.4f} | {c_ratio(ctr, 'unknown'):.4f} |")
    Path(output_dir, "support_conflict", "support_conflict_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows, image_consistency_rows


def write_summary(output_dir, usage_rows, conflict_rows):
    total = len(usage_rows)
    support_enough = sum(1 for r in usage_rows if int(r["num_support_used"]) >= 2)
    hard_or_unknown = sum(1 for r in conflict_rows if r["conflict_type"] in {"hard_contradiction", "unknown", "soft_mismatch"})
    conflict_total = len(conflict_rows)
    mean_support = sum(int(r["num_support_used"]) for r in usage_rows) / total if total else 0.0
    risk_ratio = hard_or_unknown / conflict_total if conflict_total else 0.0

    lines = [
        "# Scheme-2 Pre-Analysis Summary",
        "",
        "This summary covers the local, checkpoint-free parts of the scheme-2 gate.",
        "",
        "| question | current evidence | status |",
        "|---|---|---|",
        f"| 1. actual support sufficient? | mean support used = {mean_support:.4f}; anchors with >=2 support = {support_enough / total if total else 0.0:.4f} | {'yes' if mean_support >= 2 else 'weak'} |",
        f"| 2. local conflict / unknown risk visible? | hard/unknown/soft proxy ratio = {risk_ratio:.4f} | {'yes' if risk_ratio > 0.1 else 'weak'} |",
        "| 3. support positive over-dominates set loss? | requires single_proj_bag/split_bag best checkpoints | missing evidence |",
        "| 4. split_bag has stronger head separation than split_pure? | requires split_bag and split_pure best checkpoints | missing evidence |",
        "| 5. scheme-2 training admission? | local data supports reliability-gating; checkpoint evidence still missing | provisional, not final |",
        "",
        "Generated files:",
        "",
        "- `support_usage/support_usage_per_anchor.csv`",
        "- `support_usage/support_usage_summary.md`",
        "- `support_conflict/support_conflict_pairs.csv`",
        "- `support_conflict/support_conflict_summary.md`",
        "- `support_conflict/intra_image_caption_consistency.csv`",
        "",
        "Next required step: run checkpoint-based contribution and head-analysis on the 4090 server, because the best checkpoints were not synced to the local machine.",
    ]
    Path(output_dir, "scheme2_pre_analysis_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    data = {
        "mean_support_used": mean_support,
        "anchors_ge_2_support_ratio": support_enough / total if total else 0.0,
        "hard_unknown_soft_ratio": risk_ratio,
        "checkpoint_required_items": [
            "support_positive_contribution",
            "split_bag_head_analysis",
        ],
    }
    Path(output_dir, "scheme2_pre_analysis_summary.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="v16 scheme-2 pre-analysis for TAG-PEDES support bags")
    parser.add_argument("--dataset-root", default="D:/002datasets/TAG-PEDES")
    parser.add_argument("--output-dir", default="D:/004SSH/IRRA_light_baseline/diagnostics/TAG-PEDES/scheme2_pre_analysis")
    parser.add_argument("--quality-csv", default="D:/004SSH/IRRA_light_baseline/diagnostics/TAG-PEDES/relation_model_diagnosis_v1/quality_view_analysis/quality_view_table.csv")
    parser.add_argument("--support-size", type=int, default=3)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_annos = read_json(dataset_root / "train_reid.json")
    images, samples, first_sample_by_pid_image = build_train_samples(train_annos)
    global first_images_by_pid
    first_images_by_pid = {pid: set(image_map.keys()) for pid, image_map in first_sample_by_pid_image.items()}

    quality = load_quality_map(args.quality_csv)
    add_quality(samples, quality)
    support_indices = build_support_indices(samples, first_sample_by_pid_image, args.support_size)

    usage_rows = support_usage(samples, support_indices, output_dir)
    conflict_rows, _ = support_conflicts(samples, images, support_indices, output_dir)
    write_summary(output_dir, usage_rows, conflict_rows)
    print(f"scheme-2 local pre-analysis written to {output_dir}")


if __name__ == "__main__":
    main()

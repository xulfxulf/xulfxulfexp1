import collections
from pathlib import Path

import numpy as np

from .common import file_record, manifest, read_json, utcnow, write_json
from .data import pre_caption, view_of
from .sampling import e0_plan, pk_plan

EXPECTED = {"train": (34054, 11003, 68126), "val": (3078, 1000, 6158), "test": (3074, 1000, 6156)}


def prepare(cfg, output_dir, command):
    out = Path(output_dir).resolve()
    if out != Path(cfg["prepared_dir"]).resolve():
        raise ValueError("prepare --output-dir must equal configured prepared_dir")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Prepared inputs are immutable; choose a new group/config")
    out.mkdir(parents=True, exist_ok=True)
    root = Path(cfg["dataset_root"]).resolve()
    inputs = [Path(cfg["annotation_root"]) / (s + "_reid.json") for s in EXPECTED]
    inputs += [Path(cfg["orientation"]), Path(cfg["clip_checkpoint"])]
    run = manifest(command, inputs, out)
    write_json(out / "run_manifest.json", run)
    orientation = read_json(cfg["orientation"])
    angles = {}
    for rec in orientation["images"]:
        key = rec["image_path"]
        if key in angles:
            raise ValueError("Duplicate orientation path: " + key)
        angles[key] = {k: rec[k] for k in ("status", "body_orientation_degrees", "pids", "splits")}
    del orientation
    catalog, seen_paths, split_pids = {}, set(), {}
    for split in EXPECTED:
        annotations = read_json(Path(cfg["annotation_root"]) / (split + "_reid.json"))
        records, pid_map = [], {}
        for ann in annotations:
            relative = "imgs/" + ann["file_path"].replace("\\", "/")
            path = (root / relative).resolve()
            if root not in path.parents or not path.is_file() or relative in seen_paths:
                raise ValueError("Invalid/duplicate/cross-split image: " + relative)
            seen_paths.add(relative)
            o = angles.get(relative)
            if not o or o["status"] != "ok" or split not in o["splits"] or ann["id"] not in o["pids"]:
                raise ValueError("Orientation identity/split/status mismatch: " + relative)
            original_theta = float(o["body_orientation_degrees"])
            view = view_of(original_theta)
            theta = original_theta % 360
            captions = [pre_caption(c, 77) for c in ann["captions"]]
            if not captions or not all(c.strip() for c in captions):
                raise ValueError("Empty captions")
            bt = [pre_caption(c, 77) for c in ann.get("captions_bt", [])]
            if split == "train" and (len(bt) != len(captions) or not all(c.strip() for c in bt)):
                raise ValueError("Training BT must be present, nonempty and paired")
            pid_map.setdefault(ann["id"], len(pid_map))
            records.append({"image_path": relative, "original_pid": ann["id"],
                            "person_id": pid_map[ann["id"]], "theta_raw": theta,
                            "view_raw": view, "captions": captions, "captions_bt": bt})
        actual = (len(records), len(pid_map), sum(len(r["captions"]) for r in records))
        if actual != EXPECTED[split]:
            raise ValueError("Unexpected CUHK counts: %s %s" % (split, actual))
        catalog[split], split_pids[split] = records, set(pid_map)
        print("validated", split, actual, flush=True)
    if any(split_pids[a] & split_pids[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Identity leakage across splits")
    image_ids = {path: i for i, path in enumerate(sorted(seen_paths))}
    for records in catalog.values():
        for rec in records:
            rec["image_id"] = image_ids[rec["image_path"]]
    write_json(out / "catalog.json", catalog)
    plan_stats = {}
    for seed in cfg["seeds"]:
        folder = out / "plans" / ("seed_%d" % seed)
        folder.mkdir(parents=True)
        for epoch in range(1, cfg["epochs"] + 1):
            pk, stats = pk_plan(catalog["train"], seed, epoch)
            for kind, plan in (("PK", pk), ("E0", e0_plan(EXPECTED["train"][2], epoch))):
                path = folder / ("%s_epoch_%d.npz" % (kind, epoch))
                np.savez_compressed(str(path), pair_indices=plan)
            plan_stats["seed_%d_epoch_%d" % (seed, epoch)] = stats
        print("prepared all", cfg["epochs"], "epochs, seed", seed, flush=True)
    write_json(out / "plan_statistics.json", plan_stats)
    rng = np.random.RandomState(20260905)
    subset = []
    for view in range(4):
        indices = [i for i, r in enumerate(catalog["train"]) if r["view_raw"] == view]
        subset.extend(rng.choice(indices, min(64, len(indices)), replace=False).tolist())
    write_json(out / "expert_diagnostic_subset.json", subset)
    write_json(out / "summary.json", {"counts": EXPECTED, "num_train_ids": 11003,
               "diagnostic_subset_size": len(subset), "all_images_verified": len(seen_paths),
               "train_pid_image_counts": dict(collections.Counter(collections.Counter(
                   r["person_id"] for r in catalog["train"]).values())),
               "dataset_modified": False})
    run.update(ended_at=utcnow(), status="passed")
    write_json(out / "run_manifest.json", run)
    write_json(out / "prepared_sources.json", source_records(cfg))


def source_paths(cfg):
    return {**{split: str((Path(cfg["annotation_root"]) / (split + "_reid.json")).resolve())
               for split in EXPECTED},
            "orientation": str(Path(cfg["orientation"]).resolve()),
            "clip_checkpoint": str(Path(cfg["clip_checkpoint"]).resolve())}


def source_records(cfg):
    return {"dataset_root": str(Path(cfg["dataset_root"]).resolve()),
            "files": {key: file_record(path) for key, path in source_paths(cfg).items()}}


def verify_prepared(cfg):
    """Validate the files that THIS config will use; do not scan image bytes.

    Source paths and inexpensive stat metadata detect stale preparations. Then
    compare annotation/angle semantics against the prepared catalog, and check
    image/plan existence. Old packaged preparations are supported by their
    manifest's source *paths*; legacy SHA fields are never checked.
    """
    root, dataset = Path(cfg["prepared_dir"]), Path(cfg["dataset_root"]).resolve()
    actual = source_records(cfg)  # raises immediately on missing configured files
    records_path = root / "prepared_sources.json"
    if records_path.is_file():
        saved = read_json(records_path)
        if saved != actual:
            raise RuntimeError("Prepared source paths/metadata do not match the current config. "
                               "Run prepare into a new prepared_dir; no hashes are required.")
    else:
        legacy = read_json(root / "run_manifest.json")
        entries = legacy.get("inputs", {})
        paths = set(entries) if isinstance(entries, dict) else {r["path"] for r in entries}
        if {str(Path(p).resolve()) for p in paths} != set(source_paths(cfg).values()):
            raise RuntimeError("Legacy prepared input paths differ from the current config; "
                               "run prepare into a new prepared_dir instead of checking old paths.")

    catalog = read_json(root / "catalog.json")
    orientation = read_json(cfg["orientation"])
    angles = {}
    for record in orientation["images"]:
        path = record["image_path"]
        if path in angles:
            raise ValueError("Duplicate orientation path: " + path)
        angles[path] = record
    all_paths, split_ids = set(), {}
    for split, expected in EXPECTED.items():
        annotations = read_json(source_paths(cfg)[split])
        rows = catalog.get(split, [])
        if len(rows) != len(annotations):
            raise RuntimeError("Prepared catalog/annotation size mismatch: " + split)
        pids, pid_map, caption_count = set(), {}, 0
        for row, ann in zip(rows, annotations):
            relative = "imgs/" + ann["file_path"].replace("\\", "/")
            pid = ann["id"]
            caps = [pre_caption(c, 77) for c in ann["captions"]]
            bt = [pre_caption(c, 77) for c in ann.get("captions_bt", [])]
            pid_map.setdefault(pid, len(pid_map))
            if (row["image_path"] != relative or row["original_pid"] != pid
                    or row["person_id"] != pid_map[pid] or row["captions"] != caps
                    or row.get("captions_bt", []) != bt):
                raise RuntimeError("Prepared annotation differs from current input: " + relative)
            if not caps or not all(caps) or (split == "train" and (len(bt) != len(caps) or not all(bt))):
                raise ValueError("Invalid caption/back-translation: " + relative)
            image = (dataset / relative).resolve()
            if dataset not in image.parents or not image.is_file() or relative in all_paths:
                raise ValueError("Missing/duplicate/invalid dataset image: " + relative)
            all_paths.add(relative)
            a = angles.get(relative)
            if (not a or a["status"] != "ok" or pid not in a["pids"] or split not in a["splits"]):
                raise ValueError("Orientation association mismatch: " + relative)
            theta = float(a["body_orientation_degrees"])
            view = view_of(theta)
            if row["theta_raw"] != theta % 360 or row["view_raw"] != view:
                raise RuntimeError("Prepared angles differ from current input: " + relative)
            pids.add(pid)
            caption_count += len(caps)
        if (len(rows), len(pids), caption_count) != tuple(expected):
            raise ValueError("Unexpected dataset counts: " + split)
        split_ids[split] = pids
    if any(split_ids[a] & split_ids[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Identity leakage across splits")
    expected_ids = {path: i for i, path in enumerate(sorted(all_paths))}
    for rows in catalog.values():
        if any(r["image_id"] != expected_ids[r["image_path"]] for r in rows):
            raise ValueError("Prepared image_id mapping is inconsistent")
    pair_count = sum(len(r["captions"]) for r in catalog["train"])
    shape = (cfg["steps_per_epoch"], cfg["world_size"], cfg["local_batch"])
    for seed in cfg["seeds"]:
        for epoch in range(1, cfg["epochs"] + 1):
            for kind in ("E0", "PK"):
                plan = root / "plans" / ("seed_%d" % seed) / ("%s_epoch_%d.npz" % (kind, epoch))
                with np.load(str(plan), allow_pickle=False) as archive:
                    values = archive["pair_indices"]
                    if (values.shape != shape or not np.issubdtype(values.dtype, np.integer)
                            or values.min() < 0 or values.max() >= pair_count):
                        raise ValueError("Invalid prepared sampling plan: " + str(plan))
    if not (root / "expert_diagnostic_subset.json").is_file():
        raise FileNotFoundError("Missing expert diagnostic subset")
    return actual

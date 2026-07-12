#!/usr/bin/env python3
"""Static and lightweight dynamic audit for the v16 fast3 modes."""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.bases import ImageTextDataset
from datasets.tagpedes import TAGPEDES
from model import objectives
from model.build import IRRA


FAST3_MODES = ("split_bag_safe", "split_bag_state", "split_bag_state_hn")


def parse_args():
    parser = argparse.ArgumentParser(description="Audit v16 fast3 code and optional TAG inputs.")
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--consistency-csv", default="")
    parser.add_argument("--support-relation-csv", default="")
    parser.add_argument("--hard-negative-csv", default="")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _loss_model():
    model = IRRA.__new__(IRRA)
    nn.Module.__init__(model)
    return model


def _check(condition, label, checks, detail=""):
    checks.append({"name": label, "passed": bool(condition), "detail": detail})


def _source_audit(repo_root, checks):
    options_text = (repo_root / "utils" / "options.py").read_text(encoding="utf-8")
    model_text = (repo_root / "model" / "build.py").read_text(encoding="utf-8")
    dataset_text = (repo_root / "datasets" / "bases.py").read_text(encoding="utf-8")
    for mode in FAST3_MODES:
        _check(mode in options_text, f"mode parser: {mode}", checks)
        _check(mode in model_text, f"split-head model flag: {mode}", checks)
    _check("self.irra_light_fast_bag" in model_text, "independent fast3 forward flag", checks)
    _check("if self.irra_light_fast_bag:" in model_text, "independent fast3 forward branch", checks)
    bag_start = model_text.index("self.irra_light_bag =")
    bag_end = model_text.index("self.irra_light_bag_consistency", bag_start)
    old_bag_definition = model_text[bag_start:bag_end]
    _check(
        not any(mode in old_bag_definition for mode in FAST3_MODES),
        "fast3 modes bypass legacy support-bag branch",
        checks,
    )
    _check("_caption_slots_strict" in dataset_text, "strict slot parser exists", checks)
    _check("support_image_only" in dataset_text, "image-only support path exists", checks)
    _check("_load_hard_negative_pool" in dataset_text, "hard-negative loader exists", checks)
    _check("_state_nontransitive_loss" in model_text, "state nontransitive loss exists", checks)
    _check("encode_image_heads" in model_text and "encode_text_heads" in model_text,
           "dual-head offline encoding helpers exist", checks)


def _loss_audit(checks):
    model = _loss_model()
    image_feats = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], requires_grad=True
    )
    text_feats = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], requires_grad=True
    )
    support_feats = torch.tensor(
        [
            [[1.0, 0.0], [0.5, 0.5]],
            [[0.0, 1.0], [0.5, 0.5]],
            [[0.7, 0.7], [0.5, 0.5]],
        ],
        requires_grad=True,
    )
    pids = torch.tensor([0, 1, 2])
    support_mask = torch.ones(3, 2, dtype=torch.bool)
    weights = torch.tensor([[1.0, 0.0], [1.0, 1.0], [1.0, 1.0]])
    identity_sdm_loss = objectives.compute_sdm(image_feats, text_feats, pids, torch.tensor(1.0))
    state_itc_loss = objectives.compute_itc(image_feats, text_feats, torch.tensor(1.0))
    identity_bag_loss, valid_rows, support_valid_rows, _ = model._support_bag_rank_loss(
        image_feats,
        text_feats,
        support_feats,
        support_mask,
        weights,
        pids,
        torch.tensor(1.0),
    )
    conflict_mask = torch.tensor([[True, False], [False, False], [True, False]])
    state_loss, conflict_rows = model._state_nontransitive_loss(
        image_feats,
        text_feats,
        support_feats,
        support_mask,
        conflict_mask,
        torch.tensor(1.0),
    )
    total = identity_sdm_loss + state_itc_loss + identity_bag_loss + state_loss
    total.backward()
    values = {
        "identity_sdm_loss": identity_sdm_loss,
        "state_itc_loss": state_itc_loss,
        "identity_bag_loss": identity_bag_loss,
        "state_nontransitive_loss": state_loss,
    }
    _check(all(torch.isfinite(value).item() for value in values.values()),
           "all new and baseline losses are finite", checks)
    _check(all(value.item() >= 0 for value in values.values()),
           "all new losses are non-negative", checks)
    _check(image_feats.grad is not None and text_feats.grad is not None,
           "fast3 loss bundle backpropagates", checks)
    _check(valid_rows.any().item() and support_valid_rows.any().item(),
           "synthetic batch has valid supports and negatives", checks)
    _check(conflict_rows.any().item(), "state loss uses explicit conflict anchors", checks)


def _missing_slot_error_audit(checks):
    dataset = [
        (0, 0, "unused-a.jpg", "black shirt"),
        (0, 1, "unused-b.jpg", "black shirt"),
    ]
    with tempfile.TemporaryDirectory() as directory:
        csv_path = Path(directory) / "incomplete.csv"
        csv_path.write_text(
            "image_id,slot,consistency_type,reliability\n"
            "0,upper,consistent,1.0\n"
            "1,upper,consistent,1.0\n",
            encoding="utf-8",
        )
        try:
            ImageTextDataset(
                dataset,
                support_size=1,
                support_image_views=[0, 1],
                support_consistency_csv=str(csv_path),
                support_selection_policy="balanced",
                support_reliability_rule="hard_only",
                support_image_only=True,
            )
        except RuntimeError as exc:
            _check("Missing image-slot reliability entries" in str(exc),
                   "missing reliability slot raises explicitly", checks)
        else:
            _check(False, "missing reliability slot raises explicitly", checks)


def _dataset_audit(args, checks):
    supplied = [
        args.dataset_root,
        args.consistency_csv,
        args.support_relation_csv,
        args.hard_negative_csv,
    ]
    if not any(supplied):
        _check(True, "TAG input audit skipped (no input paths supplied)", checks)
        return
    if not all(supplied):
        _check(False, "TAG input audit has complete paths", checks,
               "Pass all four input paths or none.")
        return

    dataset = TAGPEDES(root=args.dataset_root, verbose=False)
    train_set = ImageTextDataset(
        dataset.train,
        support_size=3,
        support_image_views=dataset.train_image_views,
        support_consistency_csv=args.consistency_csv,
        support_selection_policy="balanced",
        support_reliability_rule="hard_only",
        support_relation_csv=args.support_relation_csv,
        hard_negative_csv=args.hard_negative_csv,
        hard_negative_size=1,
        support_image_only=True,
    )
    support_ok = True
    conflict_bool = True
    hard_negative_ok = True
    for index, (pid, image_id, _path, _caption) in enumerate(dataset.train):
        support_indices = train_set.support_indices[index]
        support_records = [dataset.train[support_index] for support_index in support_indices]
        support_image_ids = [int(record[1]) for record in support_records]
        support_ok &= (
            all(int(record[0]) == int(pid) for record in support_records)
            and all(value != int(image_id) for value in support_image_ids)
            and len(support_image_ids) == len(set(support_image_ids))
        )
        for support_image_id in support_image_ids:
            relation = train_set.support_hard_contradictions.get(
                (index, support_image_id),
                {"has_hard_contradiction": False},
            )
            conflict_bool &= isinstance(relation["has_hard_contradiction"], bool)
        candidates = train_set.hard_negative_pool.get(int(pid), [])
        if candidates:
            candidate = candidates[index % len(candidates)]
            hard_negative_ok &= (
                candidate["negative_pid"] != int(pid)
                and candidate["negative_image_id"] in train_set.first_index_by_image_id
            )
    _check(support_ok, "support PID/image invariants", checks)
    _check(conflict_bool, "support conflict mask is boolean", checks)
    _check(hard_negative_ok, "hard-negative PID/image invariants", checks)


def _write_report(output_dir, checks):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    passed = all(check["passed"] for check in checks)
    payload = {"passed": passed, "checks": checks}
    json_path = output_dir / "fast3_audit.json"
    md_path = output_dir / "fast3_audit.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# v16 fast3 audit\n\n")
        handle.write(f"Overall result: {'PASS' if passed else 'FAIL'}\n\n")
        handle.write("| check | result | detail |\n|---|---|---|\n")
        for check in checks:
            result = "PASS" if check["passed"] else "FAIL"
            handle.write(f"| {check['name']} | {result} | {check['detail']} |\n")
    return passed, json_path, md_path


def main():
    args = parse_args()
    checks = []
    _source_audit(REPO_ROOT, checks)
    _loss_audit(checks)
    _missing_slot_error_audit(checks)
    _dataset_audit(args, checks)
    passed, json_path, md_path = _write_report(args.output_dir, checks)
    print(f"Saved: {json_path}")
    print(f"Saved: {md_path}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

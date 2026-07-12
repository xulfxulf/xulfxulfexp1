#!/usr/bin/env python3
"""Offline dual-head evaluation for v16 split_bag_state and split_bag_state_hn."""

import argparse
import json
import os
import os.path as osp
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from prettytable import PrettyTable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import build_dataloader
from model import build_model
from utils.checkpoint import Checkpointer
from utils.iotools import load_train_configs
from utils.metrics import rank


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate v16 fast3 identity/state heads.")
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--ckpt-file", default="")
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def _encode_heads(model, loader, kind):
    device = next(model.parameters()).device
    ids, identity_feats, state_feats = [], [], []
    model.eval()
    with torch.no_grad():
        for pid, data in loader:
            data = data.to(device)
            heads = (
                model.encode_text_heads(data)
                if kind == "text"
                else model.encode_image_heads(data)
            )
            ids.append(pid.view(-1).cpu())
            identity_feats.append(heads["identity"].cpu())
            state_feats.append(heads["state"].cpu())
    return {
        "ids": torch.cat(ids, dim=0),
        "identity": torch.cat(identity_feats, dim=0),
        "state": torch.cat(state_feats, dim=0),
    }


def _evaluate(name, similarity, qids, gids):
    cmc, m_ap, m_inp, _ = rank(
        similarity=similarity,
        q_pids=qids,
        g_pids=gids,
        max_rank=10,
        get_mAP=True,
    )
    return {
        "score": name,
        "R1": float(cmc[0]),
        "R5": float(cmc[4]),
        "R10": float(cmc[9]),
        "mAP": float(m_ap),
        "mINP": float(m_inp),
    }


def _format_table(rows):
    table = PrettyTable(["score", "R1", "R5", "R10", "mAP", "mINP"])
    for row in rows:
        table.add_row([row[key] for key in ("score", "R1", "R5", "R10", "mAP", "mINP")])
    for key in ("R1", "R5", "R10", "mAP", "mINP"):
        table.custom_format[key] = lambda _field, value: f"{value:.3f}"
    return str(table)


def _write_results(output_dir, args, rows):
    os.makedirs(output_dir, exist_ok=True)
    payload = {
        "config_file": osp.abspath(args.config_file),
        "checkpoint": osp.abspath(args.ckpt_file),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "fusion": "row-standardized 0.5 * identity + 0.5 * state",
        "results": rows,
    }
    json_path = osp.join(output_dir, "fast3_dual_head_metrics.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    md_path = osp.join(output_dir, "fast3_dual_head_metrics.md")
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("# v16 fast3 dual-head offline evaluation\n\n")
        handle.write(f"- Config: `{payload['config_file']}`\n")
        handle.write(f"- Checkpoint: `{payload['checkpoint']}`\n")
        handle.write("- Fusion: row-standardized equal-weight identity/state score\n\n")
        handle.write("| score | R1 | R5 | R10 | mAP | mINP |\n")
        handle.write("|---|---:|---:|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                f"| {row['score']} | {row['R1']:.3f} | {row['R5']:.3f} | "
                f"{row['R10']:.3f} | {row['mAP']:.3f} | {row['mINP']:.3f} |\n"
            )
    return json_path, md_path


def main():
    cli_args = parse_args()
    train_args = load_train_configs(cli_args.config_file)
    mode = getattr(train_args, "irra_light_mode", "")
    if mode not in {"split_bag_state", "split_bag_state_hn"}:
        raise RuntimeError(
            "Dual-head fast3 evaluation only supports split_bag_state or split_bag_state_hn, "
            f"got {mode!r}."
        )
    train_args.training = False
    train_args.distributed = False
    train_args.irra_light = True
    train_args.loss_names = "irra_light"
    ckpt_file = cli_args.ckpt_file or osp.join(train_args.output_dir, "best.pth")
    if not osp.isfile(ckpt_file):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_file}")
    cli_args.ckpt_file = ckpt_file
    output_dir = cli_args.output_dir or osp.join(train_args.output_dir, "fast3_dual_head_eval")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_img_loader, test_txt_loader, num_classes = build_dataloader(train_args)
    model = build_model(train_args, num_classes=num_classes)
    Checkpointer(model).load(f=ckpt_file)
    model.to(device)

    text = _encode_heads(model, test_txt_loader, "text")
    image = _encode_heads(model, test_img_loader, "image")
    q_identity = F.normalize(text["identity"].to(device), p=2, dim=1)
    g_identity = F.normalize(image["identity"].to(device), p=2, dim=1)
    q_state = F.normalize(text["state"].to(device), p=2, dim=1)
    g_state = F.normalize(image["state"].to(device), p=2, dim=1)
    identity_scores = q_identity @ g_identity.t()
    state_scores = q_state @ g_state.t()
    identity_standardized = (
        identity_scores - identity_scores.mean(dim=1, keepdim=True)
    ) / identity_scores.std(dim=1, keepdim=True).clamp_min(1e-6)
    state_standardized = (
        state_scores - state_scores.mean(dim=1, keepdim=True)
    ) / state_scores.std(dim=1, keepdim=True).clamp_min(1e-6)
    fused_scores = 0.5 * identity_standardized + 0.5 * state_standardized

    rows = [
        _evaluate("identity", identity_scores, text["ids"], image["ids"]),
        _evaluate("state", state_scores, text["ids"], image["ids"]),
        _evaluate("equal_weight_fusion", fused_scores, text["ids"], image["ids"]),
    ]
    json_path, md_path = _write_results(output_dir, cli_args, rows)
    print(_format_table(rows))
    print(f"Saved: {json_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()

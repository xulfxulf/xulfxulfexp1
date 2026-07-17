#!/usr/bin/env python
"""Offline component evaluation for HIRE-v2 v16.3.0."""

from __future__ import annotations

import argparse
import json
import os
import os.path as op
import sys
from typing import Dict

import torch
import torch.nn.functional as F
from prettytable import PrettyTable

PROJECT_ROOT = op.abspath(op.join(op.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets import build_dataloader
from model import build_model
from utils.checkpoint import Checkpointer
from utils.iotools import load_train_configs
from utils.logger import setup_logger
from utils.metrics import rank


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate HIRE-v2 v16.3.0 identity/state components"
    )
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--query-chunk", type=int, default=128)
    return parser.parse_args()


def unwrap(model):
    return model.module if hasattr(model, "module") else model


def chunked_similarity(query, gallery, device, query_chunk):
    query = F.normalize(query.float(), dim=-1)
    gallery = F.normalize(gallery.float(), dim=-1)
    gallery_device = gallery.to(device)
    output = torch.empty(
        query.shape[0],
        gallery.shape[0],
        dtype=torch.float32,
    )
    for start in range(0, query.shape[0], query_chunk):
        end = min(start + query_chunk, query.shape[0])
        output[start:end] = (
            query[start:end].to(device)
            @ gallery_device.t()
        ).float().cpu()
    return output


@torch.no_grad()
def collect(model, image_loader, text_loader):
    actual = unwrap(model)
    device = next(actual.parameters()).device
    text_keys = (
        "global",
        "local",
        "observation",
        "identity",
        "identity_final",
        "state_tokens",
        "state_mask",
        "state_weights",
    )
    image_keys = (
        "global",
        "local",
        "observation",
        "identity",
        "identity_final",
        "state_tokens",
        "state_mask",
    )
    text_parts: Dict[str, list] = {key: [] for key in text_keys}
    image_parts: Dict[str, list] = {key: [] for key in image_keys}
    qids, gids = [], []

    model.eval()
    for pid, token_ids in text_loader:
        encoded = actual.encode_text_state_retrieval(
            token_ids.to(device)
        )
        qids.append(pid.view(-1).cpu())
        for key in text_keys:
            value = encoded[key]
            if value.dtype == torch.bool:
                text_parts[key].append(value.bool().cpu())
            else:
                text_parts[key].append(value.float().cpu())

    for pid, images in image_loader:
        encoded = actual.encode_image_state_retrieval(
            images.to(device)
        )
        gids.append(pid.view(-1).cpu())
        for key in image_keys:
            value = encoded[key]
            if value.dtype == torch.bool:
                image_parts[key].append(value.bool().cpu())
            else:
                image_parts[key].append(value.float().cpu())

    text_repr = {
        key: torch.cat(values, dim=0)
        for key, values in text_parts.items()
    }
    image_repr = {
        key: torch.cat(values, dim=0)
        for key, values in image_parts.items()
    }
    return (
        text_repr,
        image_repr,
        torch.cat(qids),
        torch.cat(gids),
    )


def summarize(similarity, qids, gids):
    cmc, mean_ap, mean_inp, order = rank(
        similarity,
        qids,
        gids,
        max_rank=10,
        get_mAP=True,
    )
    top1 = order[:, 0].clone().cpu()
    del order
    return {
        "R1": float(cmc[0]),
        "R5": float(cmc[4]),
        "R10": float(cmc[9]),
        "mAP": float(mean_ap),
        "mINP": float(mean_inp),
        "top1": top1,
    }


def main():
    cli = parse_args()
    args = load_train_configs(cli.config_file)
    args.training = False
    logger = setup_logger(
        "HIRE-v2-v16.3-state-eval",
        save_dir=op.dirname(cli.config_file),
        if_train=False,
    )

    image_loader, text_loader, num_classes = build_dataloader(args)
    model = build_model(args, num_classes=num_classes)
    actual = unwrap(model)
    if not getattr(actual, "is_hire_v2_state_model", False):
        raise RuntimeError("config does not build the v16.3.0 state model")

    checkpoint = cli.checkpoint or op.join(
        args.output_dir,
        "best.pth",
    )
    if not op.isfile(checkpoint):
        raise FileNotFoundError(
            "missing checkpoint: {}".format(checkpoint)
        )
    Checkpointer(model).load(f=checkpoint)
    model.to("cuda")

    text_repr, image_repr, qids, gids = collect(
        model,
        image_loader,
        text_loader,
    )
    device = torch.device("cuda")
    results = {}
    rows = []

    for key in ("global", "local", "observation", "identity"):
        similarity = chunked_similarity(
            text_repr[key],
            image_repr[key],
            device=device,
            query_chunk=cli.query_chunk,
        )
        result = summarize(similarity, qids, gids)
        results[key] = {
            name: value
            for name, value in result.items()
            if name != "top1"
        }
        rows.append([
            key,
            results[key]["R1"],
            results[key]["R5"],
            results[key]["R10"],
            results[key]["mAP"],
            results[key]["mINP"],
        ])
        del similarity

    state_matrices = actual.compute_state_reranked_similarity(
        text_repr=text_repr,
        image_repr=image_repr,
        query_chunk=cli.query_chunk,
    )
    identity_result = summarize(
        state_matrices["identity_final"],
        qids,
        gids,
    )
    final_result = summarize(
        state_matrices["state_final"],
        qids,
        gids,
    )
    results["identity_final"] = {
        name: value
        for name, value in identity_result.items()
        if name != "top1"
    }
    results["state_final"] = {
        name: value
        for name, value in final_result.items()
        if name != "top1"
    }
    for key in ("identity_final", "state_final"):
        rows.append([
            key,
            results[key]["R1"],
            results[key]["R5"],
            results[key]["R10"],
            results[key]["mAP"],
            results[key]["mINP"],
        ])

    base_correct = gids[
        identity_result["top1"]
    ].eq(qids)
    final_correct = gids[
        final_result["top1"]
    ].eq(qids)
    fix = (~base_correct & final_correct).sum().item()
    broken = (base_correct & ~final_correct).sum().item()
    fix_break = {
        "fix_count": int(fix),
        "break_count": int(broken),
        "net_top1": int(fix - broken),
        "stable_correct_count": int(
            (base_correct & final_correct).sum()
        ),
        "stable_wrong_count": int(
            (~base_correct & ~final_correct).sum()
        ),
    }

    table = PrettyTable(
        ["component", "R1", "R5", "R10", "mAP", "mINP"]
    )
    for row in rows:
        table.add_row(row)
    for field in ("R1", "R5", "R10", "mAP", "mINP"):
        table.custom_format[field] = (
            lambda _field, value: "{:.3f}".format(value)
        )
    logger.info("\n" + str(table))
    logger.info(
        "State fix/break: fix=%d, break=%d, net=%d",
        fix_break["fix_count"],
        fix_break["break_count"],
        fix_break["net_top1"],
    )

    output_json = cli.output_json or op.join(
        op.dirname(cli.config_file),
        "hire_v2_identity_state_components.json",
    )
    payload = {
        "experiment_version": "v16.3.0",
        "config_file": os.path.abspath(cli.config_file),
        "checkpoint": os.path.abspath(checkpoint),
        "identity_gate": float(
            actual.identity_gate().detach().cpu()
        ),
        "state_gate": float(
            actual.state_gate().detach().cpu()
        ),
        "state_topk": int(actual.state_topk),
        "state_image_tokens": int(actual.state_image_tokens),
        "state_text_tokens": int(actual.state_text_tokens),
        "state_dim": int(actual.state_dim),
        "results": results,
        "identity_final_to_state_final": fix_break,
    }
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
        )
    logger.info("saved component results to %s", output_json)


if __name__ == "__main__":
    main()

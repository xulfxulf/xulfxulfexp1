#!/usr/bin/env python
"""Offline component evaluation for HIRE-v2 v16.4.0."""

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
        description="Evaluate HIRE-v2 v16.4.0 token-route components"
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
    keys = ("global", "local", "observation", "identity", "final")
    text_parts: Dict[str, list] = {key: [] for key in keys}
    image_parts: Dict[str, list] = {key: [] for key in keys}
    route_probability = []
    route_mask = []
    token_residual_norm = []
    qids, gids = [], []

    model.eval()
    for pid, token_ids in text_loader:
        encoded = actual.encode_text_retrieval(
            token_ids.to(device)
        )
        qids.append(pid.view(-1).cpu())
        for key in keys:
            text_parts[key].append(
                F.normalize(
                    encoded[key].float(),
                    dim=-1,
                ).cpu()
            )
        route_probability.append(
            encoded["token_route_probability"].float().cpu()
        )
        route_mask.append(
            encoded["token_route_mask"].bool().cpu()
        )
        token_residual_norm.append(
            encoded["identity_token_residual"]
            .float()
            .norm(dim=-1)
            .cpu()
        )

    for pid, images in image_loader:
        encoded = actual.encode_image_retrieval(
            images.to(device)
        )
        gids.append(pid.view(-1).cpu())
        for key in keys:
            image_parts[key].append(
                F.normalize(
                    encoded[key].float(),
                    dim=-1,
                ).cpu()
            )

    return {
        "text": {
            key: torch.cat(values, dim=0)
            for key, values in text_parts.items()
        },
        "image": {
            key: torch.cat(values, dim=0)
            for key, values in image_parts.items()
        },
        "qids": torch.cat(qids),
        "gids": torch.cat(gids),
        "route_probability": torch.cat(
            route_probability,
            dim=0,
        ),
        "route_mask": torch.cat(route_mask, dim=0),
        "token_residual_norm": torch.cat(
            token_residual_norm,
            dim=0,
        ),
    }


def summarize(similarity, qids, gids):
    cmc, mean_ap, mean_inp, order = rank(
        similarity,
        qids,
        gids,
        max_rank=10,
        get_mAP=True,
    )
    result = {
        "R1": float(cmc[0]),
        "R5": float(cmc[4]),
        "R10": float(cmc[9]),
        "mAP": float(mean_ap),
        "mINP": float(mean_inp),
    }
    return result, order[:, 0].clone().cpu()


def main():
    cli = parse_args()
    args = load_train_configs(cli.config_file)
    args.training = False
    logger = setup_logger(
        "HIRE-v2-v16.4-token-route-eval",
        save_dir=op.dirname(cli.config_file),
        if_train=False,
    )

    image_loader, text_loader, num_classes = build_dataloader(args)
    model = build_model(args, num_classes=num_classes)
    actual = unwrap(model)
    if not getattr(
        actual,
        "is_hire_v2_identity_token_route_model",
        False,
    ):
        raise RuntimeError(
            "config does not build the v16.4.0 token-route model"
        )

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

    collected = collect(
        model,
        image_loader,
        text_loader,
    )
    device = torch.device("cuda")
    results = {}
    top1_indices = {}
    rows = []
    for key in (
        "global",
        "local",
        "observation",
        "identity",
        "final",
    ):
        similarity = chunked_similarity(
            collected["text"][key],
            collected["image"][key],
            device=device,
            query_chunk=cli.query_chunk,
        )
        result, top1 = summarize(
            similarity,
            collected["qids"],
            collected["gids"],
        )
        results[key] = result
        if key in {"observation", "final"}:
            top1_indices[key] = top1
        rows.append([
            key,
            result["R1"],
            result["R5"],
            result["R10"],
            result["mAP"],
            result["mINP"],
        ])
        del similarity
        torch.cuda.empty_cache()

    observation_correct = collected["gids"][
        top1_indices["observation"]
    ].eq(collected["qids"])
    final_correct = collected["gids"][
        top1_indices["final"]
    ].eq(collected["qids"])
    fix_count = int((~observation_correct & final_correct).sum())
    break_count = int((observation_correct & ~final_correct).sum())
    fix_break = {
        "fix_count": fix_count,
        "break_count": break_count,
        "net_top1": fix_count - break_count,
        "stable_correct_count": int(
            (observation_correct & final_correct).sum()
        ),
        "stable_wrong_count": int(
            (~observation_correct & ~final_correct).sum()
        ),
    }

    mask = collected["route_mask"]
    probability = collected["route_probability"]
    valid_probability = probability[mask]
    if valid_probability.numel() == 0:
        route_statistics = {
            "valid_token_count": 0,
            "mean": None,
            "std": None,
            "high_ratio": None,
            "entropy": None,
            "per_text_residual_norm_mean": float(
                collected["token_residual_norm"].mean()
            ),
        }
    else:
        p = valid_probability.clamp(1e-6, 1.0 - 1e-6)
        entropy = -(p * p.log() + (1.0 - p) * (1.0 - p).log())
        route_statistics = {
            "valid_token_count": int(valid_probability.numel()),
            "mean": float(valid_probability.mean()),
            "std": float(
                valid_probability.std(unbiased=False)
            ),
            "high_ratio": float(
                valid_probability.gt(0.5).float().mean()
            ),
            "entropy": float(entropy.mean()),
            "per_text_residual_norm_mean": float(
                collected["token_residual_norm"].mean()
            ),
            "per_text_residual_norm_std": float(
                collected["token_residual_norm"].std(
                    unbiased=False
                )
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
        "Token route: mean=%s, std=%s, high_ratio=%s, residual_norm=%.6f",
        route_statistics["mean"],
        route_statistics["std"],
        route_statistics["high_ratio"],
        route_statistics["per_text_residual_norm_mean"],
    )
    logger.info(
        "Identity residual fix/break: fix=%d, break=%d, net=%d",
        fix_break["fix_count"],
        fix_break["break_count"],
        fix_break["net_top1"],
    )

    output_json = cli.output_json or op.join(
        op.dirname(cli.config_file),
        "hire_v2_identity_token_route_components.json",
    )
    payload = {
        "experiment_version": "v16.4.0",
        "config_file": os.path.abspath(cli.config_file),
        "checkpoint": os.path.abspath(checkpoint),
        "identity_gate": float(
            actual.identity_gate().detach().cpu()
        ),
        "results": results,
        "route_statistics": route_statistics,
        "observation_to_final": fix_break,
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

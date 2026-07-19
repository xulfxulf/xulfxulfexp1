#!/usr/bin/env python
"""Offline component evaluation for v16.6.0 and v16.7.0."""

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
    parser = argparse.ArgumentParser(description="Evaluate phrase-route components")
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
    output = torch.empty(query.shape[0], gallery.shape[0], dtype=torch.float32)
    for start in range(0, query.shape[0], query_chunk):
        end = min(start + query_chunk, query.shape[0])
        output[start:end] = (
            query[start:end].to(device) @ gallery_device.t()
        ).float().cpu()
    return output


@torch.no_grad()
def collect(model, image_loader, text_loader):
    actual = unwrap(model)
    device = next(actual.parameters()).device
    keys = ("global", "local", "observation", "identity", "final")
    text_parts: Dict[str, list] = {key: [] for key in keys}
    image_parts: Dict[str, list] = {key: [] for key in keys}
    phrase_probability = []
    phrase_valid = []
    phrase_residual_norm = []
    qids, gids = [], []

    model.eval()
    for batch in text_loader:
        if not isinstance(batch, dict):
            raise RuntimeError("Phrase-route evaluator requires PhraseTextDataset")
        encoded = actual.encode_text_retrieval(
            batch["caption_ids"].to(device),
            phrase_token_mask=batch["phrase_token_mask"].to(device),
            phrase_valid_mask=batch["phrase_valid_mask"].to(device),
        )
        qids.append(batch["pids"].view(-1).cpu())
        for key in keys:
            text_parts[key].append(F.normalize(encoded[key].float(), dim=-1).cpu())
        phrase_probability.append(encoded["phrase_probability"].float().cpu())
        phrase_valid.append(encoded["phrase_valid"].bool().cpu())
        phrase_residual_norm.append(
            encoded["phrase_identity_residual"].float().norm(dim=-1).cpu()
        )

    for pid, images in image_loader:
        encoded = actual.encode_image_retrieval(images.to(device))
        gids.append(pid.view(-1).cpu())
        for key in keys:
            image_parts[key].append(F.normalize(encoded[key].float(), dim=-1).cpu())

    return {
        "text": {key: torch.cat(values, dim=0) for key, values in text_parts.items()},
        "image": {key: torch.cat(values, dim=0) for key, values in image_parts.items()},
        "qids": torch.cat(qids),
        "gids": torch.cat(gids),
        "phrase_probability": torch.cat(phrase_probability, dim=0),
        "phrase_valid": torch.cat(phrase_valid, dim=0),
        "phrase_residual_norm": torch.cat(phrase_residual_norm, dim=0),
    }


def summarize(similarity, qids, gids):
    cmc, mean_ap, mean_inp, _ = rank(
        similarity, qids, gids, max_rank=10, get_mAP=True
    )
    return {
        "R1": float(cmc[0]),
        "R5": float(cmc[4]),
        "R10": float(cmc[9]),
        "mAP": float(mean_ap),
        "mINP": float(mean_inp),
    }


def main():
    cli = parse_args()
    args = load_train_configs(cli.config_file)
    args.training = False
    logger = setup_logger(
        "HIRE-v2-phrase-route-eval",
        save_dir=op.dirname(cli.config_file),
        if_train=False,
    )
    image_loader, text_loader, num_classes = build_dataloader(args)
    model = build_model(args, num_classes=num_classes)
    actual = unwrap(model)
    if not getattr(actual, "is_hire_v2_phrase_route_model", False):
        raise RuntimeError("Config does not build a phrase-route model")

    checkpoint = cli.checkpoint or op.join(args.output_dir, "best.pth")
    if not op.isfile(checkpoint):
        raise FileNotFoundError(checkpoint)
    Checkpointer(model).load(f=checkpoint)
    model.to("cuda")

    collected = collect(model, image_loader, text_loader)
    results = {}
    rows = []
    for key in ("global", "local", "observation", "identity", "final"):
        similarity = chunked_similarity(
            collected["text"][key],
            collected["image"][key],
            device=torch.device("cuda"),
            query_chunk=cli.query_chunk,
        )
        result = summarize(similarity, collected["qids"], collected["gids"])
        results[key] = result
        rows.append(
            [
                key,
                result["R1"],
                result["R5"],
                result["R10"],
                result["mAP"],
                result["mINP"],
            ]
        )
        del similarity
        torch.cuda.empty_cache()

    mask = collected["phrase_valid"]
    probabilities = collected["phrase_probability"]
    valid_values = probabilities[mask]
    if valid_values.numel():
        p = valid_values.clamp_min(1e-8)
        route_statistics = {
            "valid_phrase_count": int(valid_values.numel()),
            "probability_mean": float(valid_values.mean()),
            "probability_std": float(valid_values.std(unbiased=False)),
            "probability_max_mean": float(probabilities.max(dim=1).values.mean()),
            "student_entropy_mean": float(
                (-(probabilities.clamp_min(1e-8).log() * probabilities) * mask).sum(dim=1).mean()
            ),
            "phrase_count_mean": float(mask.float().sum(dim=1).mean()),
            "identity_phrase_residual_norm_mean": float(
                collected["phrase_residual_norm"].mean()
            ),
        }
    else:
        route_statistics = {
            "valid_phrase_count": 0,
            "probability_mean": None,
            "probability_std": None,
            "probability_max_mean": None,
            "student_entropy_mean": None,
            "phrase_count_mean": 0.0,
            "identity_phrase_residual_norm_mean": float(
                collected["phrase_residual_norm"].mean()
            ),
        }

    table = PrettyTable(["component", "R1", "R5", "R10", "mAP", "mINP"])
    for row in rows:
        table.add_row(row)
    for field in ("R1", "R5", "R10", "mAP", "mINP"):
        table.custom_format[field] = lambda _field, value: "{:.3f}".format(value)
    logger.info("\n" + str(table))
    logger.info("Phrase route statistics: %s", route_statistics)

    output_json = cli.output_json or op.join(
        op.dirname(cli.config_file), "hire_v2_identity_phrase_route_components.json"
    )
    payload = {
        "experiment_version": actual.hire_v2_experiment_version,
        "phrase_route_mode": actual.phrase_route_mode,
        "config_file": os.path.abspath(cli.config_file),
        "checkpoint": os.path.abspath(checkpoint),
        "identity_gate": float(actual.identity_gate().detach().cpu()),
        "results": results,
        "route_statistics": route_statistics,
    }
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    logger.info("saved component results to %s", output_json)


if __name__ == "__main__":
    main()

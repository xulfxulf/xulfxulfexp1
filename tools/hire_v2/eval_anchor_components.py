#!/usr/bin/env python
"""Offline component evaluation for a trained HIRE-v2 anchor checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import os.path as op
from typing import Dict

import torch
import torch.nn.functional as F
from prettytable import PrettyTable

from datasets import build_dataloader
from model import build_model
from utils.checkpoint import Checkpointer
from utils.iotools import load_train_configs
from utils.logger import setup_logger
from utils.metrics import rank


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate HIRE-v2 anchor components")
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--query-chunk", type=int, default=128)
    parser.add_argument("--gallery-chunk", type=int, default=512)
    return parser.parse_args()


def unwrap(model):
    return model.module if hasattr(model, "module") else model


def collect_representations(model, image_loader, text_loader):
    actual = unwrap(model)
    device = next(actual.parameters()).device
    text_parts: Dict[str, list] = {"global": [], "local": [], "observation": []}
    image_parts: Dict[str, list] = {"global": [], "local": [], "observation": []}
    query_pids, gallery_pids = [], []

    model.eval()
    for pid, token_ids in text_loader:
        token_ids = token_ids.to(device)
        with torch.no_grad():
            encoded = actual.encode_text_retrieval(token_ids)
        query_pids.append(pid.view(-1).cpu())
        for key in text_parts:
            text_parts[key].append(F.normalize(encoded[key].float(), dim=-1).cpu())

    for pid, images in image_loader:
        images = images.to(device)
        with torch.no_grad():
            encoded = actual.encode_image_retrieval(images)
        gallery_pids.append(pid.view(-1).cpu())
        for key in image_parts:
            image_parts[key].append(F.normalize(encoded[key].float(), dim=-1).cpu())

    text_repr = {key: torch.cat(values, dim=0) for key, values in text_parts.items()}
    image_repr = {key: torch.cat(values, dim=0) for key, values in image_parts.items()}
    return text_repr, image_repr, torch.cat(query_pids), torch.cat(gallery_pids)


def chunked_similarity(query, gallery, device, query_chunk, gallery_chunk):
    output = torch.empty(query.shape[0], gallery.shape[0], dtype=torch.float32)
    for q_start in range(0, query.shape[0], query_chunk):
        q_end = min(q_start + query_chunk, query.shape[0])
        q = query[q_start:q_end].to(device)
        rows = []
        for g_start in range(0, gallery.shape[0], gallery_chunk):
            g_end = min(g_start + gallery_chunk, gallery.shape[0])
            g = gallery[g_start:g_end].to(device)
            rows.append((q @ g.t()).float().cpu())
        output[q_start:q_end] = torch.cat(rows, dim=1)
    return output


def main():
    cli = parse_args()
    args = load_train_configs(cli.config_file)
    args.training = False
    logger = setup_logger("HIRE-v2-anchor-eval", save_dir=op.dirname(cli.config_file), if_train=False)

    image_loader, text_loader, num_classes = build_dataloader(args)
    model = build_model(args, num_classes=num_classes)
    actual = unwrap(model)
    if not getattr(actual, "is_hire_v2_anchor_model", False):
        raise RuntimeError("the config does not build a HIRE-v2 anchor model")

    checkpoint = cli.checkpoint or op.join(args.output_dir, "best.pth")
    if not op.isfile(checkpoint):
        raise FileNotFoundError("missing checkpoint: {}".format(checkpoint))
    Checkpointer(model).load(f=checkpoint)
    model.to("cuda")

    text_repr, image_repr, query_pids, gallery_pids = collect_representations(
        model, image_loader, text_loader
    )
    rows = []
    results = {}
    for key in ("global", "local", "observation"):
        similarity = chunked_similarity(
            text_repr[key],
            image_repr[key],
            device=torch.device("cuda"),
            query_chunk=cli.query_chunk,
            gallery_chunk=cli.gallery_chunk,
        )
        cmc, mean_ap, mean_inp, _ = rank(
            similarity,
            query_pids,
            gallery_pids,
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
        results[key] = result
        rows.append([key, result["R1"], result["R5"], result["R10"], result["mAP"], result["mINP"]])
        del similarity
        torch.cuda.empty_cache()

    table = PrettyTable(["component", "R1", "R5", "R10", "mAP", "mINP"])
    for row in rows:
        table.add_row(row)
    for field in ("R1", "R5", "R10", "mAP", "mINP"):
        table.custom_format[field] = lambda _field, value: "{:.3f}".format(value)
    logger.info("\n" + str(table))

    output_json = cli.output_json or op.join(op.dirname(cli.config_file), "hire_v2_anchor_components.json")
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "config_file": os.path.abspath(cli.config_file),
                "checkpoint": os.path.abspath(checkpoint),
                "results": results,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    logger.info("saved component results to %s", output_json)


if __name__ == "__main__":
    main()

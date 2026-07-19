#!/usr/bin/env python
"""Mine one highest-scoring different-PID train image per v16.6 train text."""

from __future__ import annotations

import argparse
import json
import os.path as op
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = op.abspath(op.join(op.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.build import build_transforms
from datasets.bases import tokenize
from datasets.phrase_route_io import PhraseRouteTable, phrase_record_to_tensors, write_jsonl
from datasets.tagpedes import TAGPEDES
from model import build_model
from utils.checkpoint import Checkpointer
from utils.iotools import load_train_configs, read_image
from utils.simple_tokenizer import SimpleTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Mine v16.7 train hard negatives")
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--v1660-train-labels", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--image-batch-size", type=int, default=128)
    parser.add_argument("--text-batch-size", type=int, default=512)
    parser.add_argument("--query-chunk", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


class UniqueTrainImageDataset(Dataset):
    def __init__(self, records, transform):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = read_image(record["path"])
        if self.transform is not None:
            image = self.transform(image)
        return {
            "row": int(index),
            "pid": int(record["pid"]),
            "image_id": int(record["image_id"]),
            "image": image,
        }


class TrainPhraseTextDataset(Dataset):
    def __init__(self, records, label_file, text_length):
        self.records = records
        self.text_length = int(text_length)
        self.table = PhraseRouteTable(
            label_file,
            split="train",
            expected_version="v16.6.0",
            expected_route_kind="propagation",
        )
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        pid, image_id, _path, caption = self.records[index]
        row = self.table.validate_caption(index, caption, int(image_id))
        tensors = phrase_record_to_tensors(row, self.text_length)
        return {
            "record_index": int(index),
            "pid": int(pid),
            "image_id": int(image_id),
            "caption_ids": tokenize(
                caption,
                tokenizer=self.tokenizer,
                text_length=self.text_length,
                truncate=True,
            ),
            "phrase_token_mask": tensors["phrase_token_mask"],
            "phrase_valid_mask": tensors["phrase_valid_mask"],
        }


@torch.no_grad()
def main():
    cli = parse_args()
    args = load_train_configs(cli.config_file)
    args.training = False
    if getattr(args, "hire_v2_mode", "") != "identity_phrase_route":
        raise RuntimeError("Hard-negative mining requires a v16.6.0 config")
    dataset = TAGPEDES(root=args.root_dir, verbose=False)
    model = build_model(args, num_classes=len(dataset.train_id_container))
    actual = model.module if hasattr(model, "module") else model
    if not getattr(actual, "is_hire_v2_phrase_route_model", False):
        raise RuntimeError("Config does not build a phrase-route model")
    Checkpointer(model).load(f=cli.checkpoint)
    device = torch.device("cuda")
    model.to(device).eval()

    unique_images = []
    seen = set()
    for pid, image_id, path, _caption in dataset.train:
        image_id = int(image_id)
        if image_id in seen:
            continue
        seen.add(image_id)
        unique_images.append(
            {"pid": int(pid), "image_id": image_id, "path": path}
        )

    transform = build_transforms(args.img_size, False, False)
    image_loader = DataLoader(
        UniqueTrainImageDataset(unique_images, transform),
        batch_size=cli.image_batch_size,
        shuffle=False,
        num_workers=cli.num_workers,
        pin_memory=True,
    )
    image_features = []
    image_pids = []
    image_ids = []
    for batch in image_loader:
        encoded = actual.encode_image_retrieval(
            batch["image"].to(device, non_blocking=True)
        )
        image_features.append(encoded["final"].float().cpu())
        image_pids.append(batch["pid"].view(-1).cpu())
        image_ids.append(batch["image_id"].view(-1).cpu())
    image_features = torch.cat(image_features, dim=0)
    image_features = torch.nn.functional.normalize(image_features, dim=-1)
    image_pids = torch.cat(image_pids)
    image_ids = torch.cat(image_ids)

    text_loader = DataLoader(
        TrainPhraseTextDataset(
            dataset.train, cli.v1660_train_labels, args.text_length
        ),
        batch_size=cli.text_batch_size,
        shuffle=False,
        num_workers=cli.num_workers,
        pin_memory=True,
    )
    text_features = []
    text_pids = []
    record_indices = []
    anchor_image_ids = []
    for batch in text_loader:
        encoded = actual.encode_text_retrieval(
            batch["caption_ids"].to(device, non_blocking=True),
            phrase_token_mask=batch["phrase_token_mask"].to(
                device, non_blocking=True
            ),
            phrase_valid_mask=batch["phrase_valid_mask"].to(
                device, non_blocking=True
            ),
        )
        text_features.append(encoded["final"].float().cpu())
        text_pids.append(batch["pid"].view(-1).cpu())
        record_indices.append(batch["record_index"].view(-1).cpu())
        anchor_image_ids.append(batch["image_id"].view(-1).cpu())
    text_features = torch.nn.functional.normalize(
        torch.cat(text_features, dim=0), dim=-1
    )
    text_pids = torch.cat(text_pids)
    record_indices = torch.cat(record_indices)
    anchor_image_ids = torch.cat(anchor_image_ids)

    gallery = image_features.to(device)
    gallery_pids = image_pids.to(device)
    output = []
    for start in range(0, text_features.shape[0], cli.query_chunk):
        end = min(start + cli.query_chunk, text_features.shape[0])
        query = text_features[start:end].to(device)
        scores = query @ gallery.t()
        same_pid = text_pids[start:end].to(device)[:, None].eq(
            gallery_pids[None, :]
        )
        scores = scores.masked_fill(same_pid, torch.finfo(scores.dtype).min)
        best_score, best_row = scores.max(dim=1)
        for offset in range(end - start):
            row = int(best_row[offset].cpu())
            record_index = int(record_indices[start + offset])
            output.append(
                {
                    "experiment_version": "v16.7.0",
                    "record_index": record_index,
                    "anchor_pid": int(text_pids[start + offset]),
                    "anchor_image_id": int(anchor_image_ids[start + offset]),
                    "hard_negative_pid": int(image_pids[row]),
                    "hard_negative_image_id": int(image_ids[row]),
                    "hard_negative_path": unique_images[row]["path"],
                    "hard_negative_score": float(best_score[offset].cpu()),
                    "source_checkpoint": str(
                        Path(cli.checkpoint).expanduser().resolve()
                    ),
                }
            )

    output.sort(key=lambda row: row["record_index"])
    write_jsonl(cli.output_file, output)
    summary = {
        "record_count": len(output),
        "unique_negative_pids": len(
            {row["hard_negative_pid"] for row in output}
        ),
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

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""v16.2.0 identity mechanism audit without training.

This script performs two independent audits on a trained HIRE-v2 identity
checkpoint:

1. Trusted-intersection audit on TAG-PEDES train:
   - precompute every unique train-image identity mean and variance;
   - rebuild the exact epoch-dependent, leave-one support sets;
   - compare the full heterogeneity-aware trusted intersection against
     variance-only fusion and simple mean fusion;
   - measure whether uncertainty/heterogeneity actually changes support weights;
   - evaluate strict paired-group retrieval for all three group constructions.

2. Fix/break audit on TAG-PEDES test:
   - compare v16.2 observation vs v16.2 final to isolate identity residual impact;
   - optionally compare v16.1 observation vs v16.2 final to measure net method impact;
   - export query-level fixes, breaks, rank/AP/mINP changes and metadata.

No optimizer, scheduler, backward pass, model update or checkpoint write is used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import os.path as op
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = op.abspath(op.join(op.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.build import build_dataloader, build_transforms  # noqa: E402
from datasets.hire_v2_identity_dataset import HIREV2IdentityDataset  # noqa: E402
from datasets.tagpedes import TAGPEDES  # noqa: E402
from datasets.bases import tokenize  # noqa: E402
from model import build_model  # noqa: E402
from model.hire_v2_identity_components import (  # noqa: E402
    heterogeneity_aware_identity_intersection,
)
from utils.checkpoint import Checkpointer  # noqa: E402
from utils.iotools import load_train_configs, read_image  # noqa: E402
from utils.simple_tokenizer import SimpleTokenizer  # noqa: E402


_EPS = 1e-8


def parse_int_list(value: str) -> List[int]:
    values = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        parsed = int(item)
        if parsed < 0:
            raise argparse.ArgumentTypeError("epoch values must be non-negative")
        values.append(parsed)
    if not values:
        raise argparse.ArgumentTypeError("at least one epoch is required")
    return sorted(set(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit v16.2.0 trusted intersection and identity-residual fix/break behavior"
    )
    parser.add_argument("--config-file", required=True, help="v16.2.0 configs.yaml")
    parser.add_argument("--checkpoint", required=True, help="v16.2.0 best.pth")
    parser.add_argument(
        "--anchor-config-file",
        default="",
        help="optional v16.1.0 configs.yaml for net fix/break comparison",
    )
    parser.add_argument(
        "--anchor-checkpoint",
        default="",
        help="optional v16.1.0 best.pth; required with --anchor-config-file",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="independent output directory; existing files are not silently overwritten",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--image-batch-size", type=int, default=128)
    parser.add_argument("--text-batch-size", type=int, default=512)
    parser.add_argument("--query-chunk", type=int, default=128)
    parser.add_argument("--gallery-chunk", type=int, default=1024)
    parser.add_argument(
        "--support-epochs",
        type=parse_int_list,
        default=parse_int_list("0,15,30,45,54,59"),
        help="epochs used for support-set structural audit",
    )
    parser.add_argument(
        "--retrieval-epochs",
        type=parse_int_list,
        default=parse_int_list("54"),
        help="support epochs on which strict paired-group retrieval is computed",
    )
    parser.add_argument(
        "--max-train-queries",
        type=int,
        default=0,
        help="0 means all train captions; positive value uses a deterministic prefix",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow writing into an existing non-empty output directory",
    )
    return parser.parse_args()


def ensure_file(path: str, name: str) -> str:
    path = op.abspath(op.expanduser(path))
    if not op.isfile(path):
        raise FileNotFoundError("{} does not exist: {}".format(name, path))
    return path


def ensure_output_dir(path: str, overwrite: bool) -> Path:
    output = Path(path).expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise RuntimeError(
            "output directory is non-empty; choose a new path or pass --overwrite: {}".format(output)
        )
    output.mkdir(parents=True, exist_ok=True)
    return output


def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def to_jsonable(value):
    if torch.is_tensor(value):
        if value.numel() == 1:
            return float(value.detach().cpu())
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: Iterable[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def percentile(values: torch.Tensor, q: float) -> float:
    values = values.detach().float().view(-1).cpu()
    if values.numel() == 0:
        return float("nan")
    values, _ = torch.sort(values)
    index = int(round((values.numel() - 1) * float(q)))
    index = max(0, min(index, values.numel() - 1))
    return float(values[index])


def tensor_summary(values: torch.Tensor) -> dict:
    values = values.detach().float().view(-1).cpu()
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return {
            "count": int(values.numel()),
            "finite_count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p05": None,
            "median": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": int(values.numel()),
        "finite_count": int(finite.numel()),
        "mean": float(finite.mean()),
        "std": float(finite.std(unbiased=False)),
        "min": float(finite.min()),
        "p05": percentile(finite, 0.05),
        "median": percentile(finite, 0.50),
        "p95": percentile(finite, 0.95),
        "max": float(finite.max()),
    }


def masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    mask_f = mask.to(dtype=values.dtype)
    while mask_f.ndim < values.ndim:
        mask_f = mask_f.unsqueeze(-1)
    numerator = (values * mask_f).sum(dim=dim)
    denominator = mask_f.sum(dim=dim).clamp_min(1.0)
    return numerator / denominator


def masked_std(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    mean = masked_mean(values, mask, dim=dim)
    expanded = mean.unsqueeze(dim)
    mask_f = mask.to(dtype=values.dtype)
    while mask_f.ndim < values.ndim:
        mask_f = mask_f.unsqueeze(-1)
    variance = ((values - expanded).pow(2) * mask_f).sum(dim=dim)
    count = mask_f.sum(dim=dim).clamp_min(1.0)
    return torch.sqrt(variance / count + _EPS)


def source_from_path(path: str) -> str:
    normalized = str(path).replace("\\", "/").strip("/")
    parts = [item for item in normalized.split("/") if item]
    if not parts:
        return ""
    if len(parts) == 1:
        stem = op.splitext(parts[0])[0]
        return stem.split("_")[0]
    return parts[0]


class UniqueTrainImageDataset(Dataset):
    def __init__(self, records: List[dict], transform):
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


class TrainCaptionDataset(Dataset):
    def __init__(self, records: List[dict], text_length: int):
        self.records = records
        self.text_length = int(text_length)
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        token_ids = tokenize(
            record["caption"],
            tokenizer=self.tokenizer,
            text_length=self.text_length,
            truncate=True,
        )
        return {
            "row": int(index),
            "pid": int(record["pid"]),
            "image_id": int(record["image_id"]),
            "token_ids": token_ids,
        }


def dict_collate(batch: List[dict]) -> dict:
    keys = batch[0].keys()
    result = {}
    for key in keys:
        values = [item[key] for item in batch]
        if torch.is_tensor(values[0]):
            result[key] = torch.stack(values)
        else:
            result[key] = torch.tensor(values, dtype=torch.long)
    return result


def load_model(
    config_file: str,
    checkpoint: str,
    device: torch.device,
    expected_mode: str,
):
    args = load_train_configs(config_file)
    args.training = False
    dataset = TAGPEDES(root=args.root_dir, verbose=False)
    model = build_model(args, num_classes=len(dataset.train_id_container))
    actual = model.module if hasattr(model, "module") else model
    if getattr(args, "hire_v2_mode", "") != expected_mode:
        raise RuntimeError(
            "expected HIRE-v2 mode {}, got {}".format(expected_mode, getattr(args, "hire_v2_mode", ""))
        )
    Checkpointer(model).load(f=checkpoint)
    model.to(device)
    model.eval()
    return args, dataset, model, actual


def build_train_records(dataset: TAGPEDES) -> Tuple[List[dict], List[dict], Dict[int, int]]:
    unique_images = []
    captions = []
    image_id_to_row = {}
    image_id_to_first_caption = {}
    for caption_index, (pid, image_id, path, caption) in enumerate(dataset.train):
        pid = int(pid)
        image_id = int(image_id)
        captions.append(
            {
                "caption_index": caption_index,
                "pid": pid,
                "image_id": image_id,
                "path": path,
                "caption": caption,
            }
        )
        if image_id not in image_id_to_row:
            row = len(unique_images)
            image_id_to_row[image_id] = row
            image_id_to_first_caption[image_id] = caption_index
            view = None
            if hasattr(dataset, "train_image_views") and image_id < len(dataset.train_image_views):
                view = dataset.train_image_views[image_id]
            unique_images.append(
                {
                    "row": row,
                    "pid": pid,
                    "image_id": image_id,
                    "path": path,
                    "view": None if view is None else int(view),
                    "source": source_from_path(path),
                    "first_caption_index": caption_index,
                }
            )
    return unique_images, captions, image_id_to_row


@torch.no_grad()
def encode_train_images(
    actual,
    records: List[dict],
    transform,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> dict:
    loader = DataLoader(
        UniqueTrainImageDataset(records, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=dict_collate,
    )
    outputs = {
        "observation": [],
        "identity": [],
        "variance": [],
        "pid": [],
        "image_id": [],
    }
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        encoded = actual.encode_image_retrieval(images)
        observation = encoded["observation"].float()
        identity = encoded["identity"].float()
        variance = actual.image_uncertainty(observation.detach()).float()
        outputs["observation"].append(observation.cpu())
        outputs["identity"].append(identity.cpu())
        outputs["variance"].append(variance.cpu())
        outputs["pid"].append(batch["pid"].cpu())
        outputs["image_id"].append(batch["image_id"].cpu())
    return {key: torch.cat(parts, dim=0) for key, parts in outputs.items()}


@torch.no_grad()
def encode_train_texts(
    actual,
    records: List[dict],
    text_length: int,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    max_queries: int,
) -> dict:
    if max_queries > 0:
        records = records[:max_queries]
    loader = DataLoader(
        TrainCaptionDataset(records, text_length),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=dict_collate,
    )
    outputs = {"identity": [], "pid": [], "image_id": [], "row": []}
    for batch in loader:
        token_ids = batch["token_ids"].to(device, non_blocking=True)
        encoded = actual.encode_text_retrieval(token_ids)
        outputs["identity"].append(encoded["identity"].float().cpu())
        outputs["pid"].append(batch["pid"].cpu())
        outputs["image_id"].append(batch["image_id"].cpu())
        outputs["row"].append(batch["row"].cpu())
    return {key: torch.cat(parts, dim=0) for key, parts in outputs.items()}


def build_support_index_matrix(
    support_dataset: HIREV2IdentityDataset,
    unique_images: List[dict],
    image_id_to_row: Dict[int, int],
    epoch: int,
    support_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, List[List[int]]]:
    support_dataset.set_epoch(epoch)
    index_matrix = torch.zeros((len(unique_images), support_size), dtype=torch.long)
    mask = torch.zeros((len(unique_images), support_size), dtype=torch.bool)
    support_image_ids: List[List[int]] = []
    for row, record in enumerate(unique_images):
        caption_index = int(record["first_caption_index"])
        dataset_indices = support_dataset.support_indices_for(caption_index)
        row_ids = []
        for position, dataset_index in enumerate(dataset_indices[:support_size]):
            support_pid, support_image_id, _path, _caption = support_dataset.dataset[dataset_index]
            support_pid = int(support_pid)
            support_image_id = int(support_image_id)
            if support_pid != int(record["pid"]):
                raise RuntimeError("support PID boundary violation in audit")
            if support_image_id == int(record["image_id"]):
                raise RuntimeError("support reused anchor image in audit")
            if support_image_id in row_ids:
                raise RuntimeError("support image duplicated in audit")
            if support_image_id not in image_id_to_row:
                raise RuntimeError("support image missing from unique-image cache")
            index_matrix[row, position] = int(image_id_to_row[support_image_id])
            mask[row, position] = True
            row_ids.append(support_image_id)
        support_image_ids.append(row_ids)
    return index_matrix, mask, support_image_ids


@torch.no_grad()
def construct_group_variants(
    image_identity: torch.Tensor,
    image_variance: torch.Tensor,
    support_rows: torch.Tensor,
    support_mask: torch.Tensor,
    device: torch.device,
    batch_size: int = 1024,
) -> dict:
    all_outputs = defaultdict(list)
    for start in range(0, support_rows.shape[0], batch_size):
        end = min(start + batch_size, support_rows.shape[0])
        rows = support_rows[start:end].to(device)
        mask = support_mask[start:end].to(device)
        means = image_identity.index_select(0, rows.reshape(-1).cpu()).reshape(
            rows.shape[0], rows.shape[1], -1
        ).to(device)
        variances = image_variance.index_select(0, rows.reshape(-1).cpu()).reshape(
            rows.shape[0], rows.shape[1], -1
        ).to(device)

        trusted = heterogeneity_aware_identity_intersection(
            means, variances, mask, min_supports=2
        )
        mask_f = mask.to(means.dtype).unsqueeze(-1)
        count = mask_f.sum(dim=1).clamp_min(1.0)
        simple_raw = (means * mask_f).sum(dim=1) / count
        simple = F.normalize(simple_raw, dim=-1)

        precision0 = mask_f / (variances + 1e-6)
        variance_raw = (precision0 * means).sum(dim=1) / precision0.sum(dim=1).clamp_min(1e-6)
        variance_only = F.normalize(variance_raw, dim=-1)

        valid = trusted["valid"]
        simple = torch.where(valid.unsqueeze(-1), simple, torch.zeros_like(simple))
        variance_only = torch.where(
            valid.unsqueeze(-1), variance_only, torch.zeros_like(variance_only)
        )

        support_scalar_variance = variances.mean(dim=-1)
        support_scalar_variance_std = masked_std(
            support_scalar_variance, mask, dim=1
        )
        support_scalar_variance_mean = masked_mean(
            support_scalar_variance, mask, dim=1
        )
        support_scalar_variance_cv = (
            support_scalar_variance_std / support_scalar_variance_mean.clamp_min(1e-8)
        )

        precision_full = trusted["precision"]
        support_scalar_precision = precision_full.mean(dim=-1)
        support_scalar_precision_std = masked_std(
            support_scalar_precision, mask, dim=1
        )
        support_scalar_precision_mean = masked_mean(
            support_scalar_precision, mask, dim=1
        )
        support_scalar_precision_cv = (
            support_scalar_precision_std / support_scalar_precision_mean.clamp_min(1e-8)
        )

        precision_mean_dim = masked_mean(precision_full, mask, dim=1)
        precision_std_dim = masked_std(precision_full, mask, dim=1)
        precision_dim_cv = (
            precision_std_dim / precision_mean_dim.clamp_min(1e-8)
        ).mean(dim=-1)

        mean_variance_dim = masked_mean(variances, mask, dim=1)
        tau_share = (
            trusted["tau2"] / (mean_variance_dim + trusted["tau2"] + 1e-8)
        ).mean(dim=-1)

        all_outputs["trusted"].append(trusted["mean"].cpu())
        all_outputs["variance_only"].append(variance_only.cpu())
        all_outputs["simple"].append(simple.cpu())
        all_outputs["valid"].append(valid.cpu())
        all_outputs["count"].append(trusted["count"].cpu())
        all_outputs["tau2"].append(trusted["tau2"].cpu())
        all_outputs["support_scalar_variance_cv"].append(
            support_scalar_variance_cv.cpu()
        )
        all_outputs["support_scalar_precision_cv"].append(
            support_scalar_precision_cv.cpu()
        )
        all_outputs["precision_dim_cv"].append(precision_dim_cv.cpu())
        all_outputs["tau_share"].append(tau_share.cpu())
        all_outputs["trusted_simple_cos"].append(
            (trusted["mean"] * simple).sum(dim=-1).cpu()
        )
        all_outputs["trusted_variance_cos"].append(
            (trusted["mean"] * variance_only).sum(dim=-1).cpu()
        )
        all_outputs["variance_simple_cos"].append(
            (variance_only * simple).sum(dim=-1).cpu()
        )
    return {key: torch.cat(parts, dim=0) for key, parts in all_outputs.items()}


@torch.no_grad()
def paired_group_ranks(
    queries: torch.Tensor,
    query_pids: torch.Tensor,
    query_image_ids: torch.Tensor,
    groups: torch.Tensor,
    group_pids: torch.Tensor,
    group_image_ids: torch.Tensor,
    group_valid: torch.Tensor,
    image_id_to_group_row: Dict[int, int],
    device: torch.device,
    query_chunk: int,
    gallery_chunk: int,
) -> Tuple[dict, List[dict]]:
    valid_query_rows = []
    own_group_rows = []
    for query_row, image_id in enumerate(query_image_ids.tolist()):
        group_row = image_id_to_group_row[int(image_id)]
        if bool(group_valid[group_row]):
            valid_query_rows.append(query_row)
            own_group_rows.append(group_row)
    if not valid_query_rows:
        raise RuntimeError("no valid paired identity groups")

    valid_query_rows_t = torch.tensor(valid_query_rows, dtype=torch.long)
    own_group_rows_t = torch.tensor(own_group_rows, dtype=torch.long)
    q_all = queries.index_select(0, valid_query_rows_t)
    qpid_all = query_pids.index_select(0, valid_query_rows_t)
    qimage_all = query_image_ids.index_select(0, valid_query_rows_t)

    ranks = torch.empty(q_all.shape[0], dtype=torch.long)
    positive_scores = torch.empty(q_all.shape[0], dtype=torch.float32)
    hardest_negative_scores = torch.empty(q_all.shape[0], dtype=torch.float32)

    valid_group_indices = group_valid.nonzero(as_tuple=False).view(-1)
    group_pids_valid = group_pids.index_select(0, valid_group_indices).to(device)
    group_rows_valid = valid_group_indices.to(device)
    groups_valid = groups.index_select(0, valid_group_indices).to(device)

    for q_start in range(0, q_all.shape[0], query_chunk):
        q_end = min(q_start + query_chunk, q_all.shape[0])
        q = q_all[q_start:q_end].to(device)
        qpid = qpid_all[q_start:q_end].to(device)
        own_rows = own_group_rows_t[q_start:q_end]
        own_group = groups.index_select(0, own_rows).to(device)
        positive = (q * own_group).sum(dim=-1)
        better_count = torch.zeros(q.shape[0], dtype=torch.long, device=device)
        hardest = torch.full(
            (q.shape[0],), -float("inf"), dtype=torch.float32, device=device
        )

        for g_start in range(0, groups_valid.shape[0], gallery_chunk):
            g_end = min(g_start + gallery_chunk, groups_valid.shape[0])
            candidate = groups_valid[g_start:g_end]
            scores = q @ candidate.t()
            candidate_pids = group_pids_valid[g_start:g_end]
            candidate_rows = group_rows_valid[g_start:g_end]
            negative_mask = candidate_pids.unsqueeze(0).ne(qpid.unsqueeze(1))
            better = scores.gt(positive.unsqueeze(1))
            ties = scores.eq(positive.unsqueeze(1))
            tie_before = candidate_rows.unsqueeze(0).lt(
                own_rows.to(device).unsqueeze(1)
            )
            better_count += (
                (better | (ties & tie_before)) & negative_mask
            ).sum(dim=1)
            masked_scores = scores.masked_fill(~negative_mask, -float("inf"))
            hardest = torch.maximum(hardest, masked_scores.max(dim=1).values)

        ranks[q_start:q_end] = (better_count + 1).cpu()
        positive_scores[q_start:q_end] = positive.cpu()
        hardest_negative_scores[q_start:q_end] = hardest.cpu()

    reciprocal = 1.0 / ranks.float()
    margins = positive_scores - hardest_negative_scores
    summary = {
        "query_count": int(ranks.numel()),
        "R1": float((ranks <= 1).float().mean() * 100.0),
        "R5": float((ranks <= 5).float().mean() * 100.0),
        "R10": float((ranks <= 10).float().mean() * 100.0),
        "MRR": float(reciprocal.mean() * 100.0),
        "mean_rank": float(ranks.float().mean()),
        "median_rank": percentile(ranks.float(), 0.5),
        "mean_positive_score": float(positive_scores.mean()),
        "mean_hardest_negative_score": float(hardest_negative_scores.mean()),
        "mean_margin": float(margins.mean()),
        "positive_margin_ratio": float((margins > 0).float().mean()),
    }
    rows = []
    for local_index, query_row in enumerate(valid_query_rows):
        rows.append(
            {
                "query_row": int(query_row),
                "query_pid": int(qpid_all[local_index]),
                "query_image_id": int(qimage_all[local_index]),
                "paired_group_image_id": int(
                    group_image_ids[own_group_rows_t[local_index]]
                ),
                "rank": int(ranks[local_index]),
                "positive_score": float(positive_scores[local_index]),
                "hardest_negative_score": float(hardest_negative_scores[local_index]),
                "margin": float(margins[local_index]),
            }
        )
    return summary, rows


def build_test_metadata(dataset: TAGPEDES) -> Tuple[List[dict], List[dict]]:
    query_meta = []
    gallery_meta = []
    query_index = 0
    for gallery_index, annotation in enumerate(dataset.test_annos):
        path = annotation["file_path"]
        pid = int(annotation["id"])
        view = annotation.get("cam_id", None)
        gallery_meta.append(
            {
                "gallery_index": gallery_index,
                "pid": pid,
                "path": path,
                "view": "" if view is None else int(view),
                "source": source_from_path(path),
            }
        )
        for caption_index, _caption in enumerate(annotation["captions"]):
            query_meta.append(
                {
                    "query_index": query_index,
                    "pid": pid,
                    "source_image_index": gallery_index,
                    "source_image_path": path,
                    "view": "" if view is None else int(view),
                    "source": source_from_path(path),
                    "caption_in_image": caption_index,
                }
            )
            query_index += 1
    return query_meta, gallery_meta


@torch.no_grad()
def collect_test_components(
    model,
    actual,
    args,
    device: torch.device,
    num_workers: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    args.training = False
    args.num_workers = int(num_workers)
    image_loader, text_loader, _num_classes = build_dataloader(args)
    text_parts = defaultdict(list)
    image_parts = defaultdict(list)
    qids = []
    gids = []

    for pid, token_ids in text_loader:
        token_ids = token_ids.to(device, non_blocking=True)
        encoded = actual.encode_text_retrieval(token_ids)
        qids.append(pid.view(-1).cpu())
        for key, value in encoded.items():
            if torch.is_tensor(value) and value.ndim == 2:
                text_parts[key].append(F.normalize(value.float(), dim=-1).cpu())

    for pid, images in image_loader:
        images = images.to(device, non_blocking=True)
        encoded = actual.encode_image_retrieval(images)
        gids.append(pid.view(-1).cpu())
        for key, value in encoded.items():
            if torch.is_tensor(value) and value.ndim == 2:
                image_parts[key].append(F.normalize(value.float(), dim=-1).cpu())

    text_repr = {key: torch.cat(parts, dim=0) for key, parts in text_parts.items()}
    image_repr = {key: torch.cat(parts, dim=0) for key, parts in image_parts.items()}
    return text_repr, image_repr, torch.cat(qids), torch.cat(gids)


def row_metrics_from_scores(
    scores: torch.Tensor,
    query_pids: torch.Tensor,
    gallery_pids: torch.Tensor,
) -> dict:
    order = torch.argsort(scores, dim=1, descending=True)
    ordered_pids = gallery_pids[order]
    matches = ordered_pids.eq(query_pids.view(-1, 1))
    num_rel = matches.sum(dim=1)
    if (num_rel == 0).any():
        raise RuntimeError("a test query has no relevant gallery image")
    cumulative = matches.cumsum(dim=1)
    first_rank = matches.float().argmax(dim=1) + 1
    positions = torch.arange(
        1, scores.shape[1] + 1, dtype=torch.float32
    ).view(1, -1)
    precision = cumulative.float() / positions
    ap = (precision * matches.float()).sum(dim=1) / num_rel.float()
    reverse = torch.flip(matches, dims=[1]).float().argmax(dim=1)
    last_rank = scores.shape[1] - reverse
    inp = num_rel.float() / last_rank.float()
    return {
        "order": order,
        "top1_index": order[:, 0],
        "top1_pid": ordered_pids[:, 0],
        "correct": matches[:, 0],
        "best_positive_rank": first_rank,
        "ap": ap,
        "inp": inp,
        "top10": order[:, :10],
    }


def topk_overlap(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    overlaps = []
    for row in range(a.shape[0]):
        left = set(a[row].tolist())
        right = set(b[row].tolist())
        overlaps.append(len(left.intersection(right)) / float(a.shape[1]))
    return torch.tensor(overlaps, dtype=torch.float32)


@torch.no_grad()
def compare_score_sources(
    base_text: torch.Tensor,
    base_image: torch.Tensor,
    candidate_text: torch.Tensor,
    candidate_image: torch.Tensor,
    query_pids: torch.Tensor,
    gallery_pids: torch.Tensor,
    query_meta: List[dict],
    gallery_meta: List[dict],
    device: torch.device,
    query_chunk: int,
    comparison_name: str,
    identity_text: Optional[torch.Tensor] = None,
    identity_image: Optional[torch.Tensor] = None,
    identity_gate_value: Optional[float] = None,
) -> Tuple[dict, List[dict]]:
    all_rows = []
    counts = defaultdict(int)
    rank_delta_values = []
    ap_delta_values = []
    inp_delta_values = []
    overlap_values = []
    base_image_device = base_image.to(device)
    candidate_image_device = candidate_image.to(device)
    identity_image_device = identity_image.to(device) if identity_image is not None else None

    for start in range(0, base_text.shape[0], query_chunk):
        end = min(start + query_chunk, base_text.shape[0])
        base_scores = base_text[start:end].to(device) @ base_image_device.t()
        candidate_scores = (
            candidate_text[start:end].to(device) @ candidate_image_device.t()
        )
        qpid = query_pids[start:end]
        base_metrics = row_metrics_from_scores(base_scores.cpu(), qpid, gallery_pids)
        candidate_metrics = row_metrics_from_scores(
            candidate_scores.cpu(), qpid, gallery_pids
        )
        overlap = topk_overlap(base_metrics["top10"], candidate_metrics["top10"])

        identity_delta = None
        if identity_text is not None and identity_image is not None:
            identity_scores = (
                identity_text[start:end].to(device) @ identity_image_device.t()
            ).cpu()
            identity_delta = identity_scores - base_scores.cpu()

        for offset in range(end - start):
            query_index = start + offset
            base_correct = bool(base_metrics["correct"][offset])
            candidate_correct = bool(candidate_metrics["correct"][offset])
            if (not base_correct) and candidate_correct:
                category = "fix"
            elif base_correct and (not candidate_correct):
                category = "break"
            elif base_correct and candidate_correct:
                category = "stable_correct"
            else:
                category = "stable_wrong"
            counts[category] += 1

            base_rank = int(base_metrics["best_positive_rank"][offset])
            candidate_rank = int(candidate_metrics["best_positive_rank"][offset])
            rank_delta = base_rank - candidate_rank
            if rank_delta > 0:
                counts["rank_improved"] += 1
            elif rank_delta < 0:
                counts["rank_worsened"] += 1
            else:
                counts["rank_same"] += 1

            base_ap = float(base_metrics["ap"][offset])
            candidate_ap = float(candidate_metrics["ap"][offset])
            base_inp = float(base_metrics["inp"][offset])
            candidate_inp = float(candidate_metrics["inp"][offset])
            ap_delta = candidate_ap - base_ap
            inp_delta = candidate_inp - base_inp
            overlap_value = float(overlap[offset])
            rank_delta_values.append(rank_delta)
            ap_delta_values.append(ap_delta)
            inp_delta_values.append(inp_delta)
            overlap_values.append(overlap_value)

            base_top1 = int(base_metrics["top1_index"][offset])
            candidate_top1 = int(candidate_metrics["top1_index"][offset])
            meta = query_meta[query_index]
            base_gallery = gallery_meta[base_top1]
            candidate_gallery = gallery_meta[candidate_top1]
            row = {
                "comparison": comparison_name,
                "query_index": query_index,
                "query_pid": int(query_pids[query_index]),
                "query_source_image_index": meta["source_image_index"],
                "query_source_image_path": meta["source_image_path"],
                "query_view": meta["view"],
                "query_source": meta["source"],
                "caption_in_image": meta["caption_in_image"],
                "category": category,
                "base_correct": int(base_correct),
                "candidate_correct": int(candidate_correct),
                "base_top1_index": base_top1,
                "base_top1_pid": int(gallery_pids[base_top1]),
                "base_top1_path": base_gallery["path"],
                "base_top1_view": base_gallery["view"],
                "base_top1_source": base_gallery["source"],
                "candidate_top1_index": candidate_top1,
                "candidate_top1_pid": int(gallery_pids[candidate_top1]),
                "candidate_top1_path": candidate_gallery["path"],
                "candidate_top1_view": candidate_gallery["view"],
                "candidate_top1_source": candidate_gallery["source"],
                "base_best_positive_rank": base_rank,
                "candidate_best_positive_rank": candidate_rank,
                "rank_improvement": rank_delta,
                "base_ap": base_ap,
                "candidate_ap": candidate_ap,
                "ap_delta": ap_delta,
                "base_inp": base_inp,
                "candidate_inp": candidate_inp,
                "inp_delta": inp_delta,
                "top10_overlap": overlap_value,
                "base_top1_score": float(base_scores[offset, base_top1]),
                "candidate_top1_score": float(
                    candidate_scores[offset, candidate_top1]
                ),
            }
            if identity_delta is not None:
                source_index = int(meta["source_image_index"])
                raw_base = float(identity_delta[offset, base_top1])
                raw_candidate = float(identity_delta[offset, candidate_top1])
                raw_source = float(identity_delta[offset, source_index])
                gate_value = 1.0 if identity_gate_value is None else float(identity_gate_value)
                row.update(
                    {
                        "identity_residual_on_base_top1": raw_base,
                        "identity_residual_on_candidate_top1": raw_candidate,
                        "identity_residual_on_source_image": raw_source,
                        "identity_contribution_on_base_top1": gate_value * raw_base,
                        "identity_contribution_on_candidate_top1": gate_value * raw_candidate,
                        "identity_contribution_on_source_image": gate_value * raw_source,
                    }
                )
            all_rows.append(row)

    query_count = len(all_rows)
    summary = {
        "comparison": comparison_name,
        "query_count": query_count,
        "fix_count": counts["fix"],
        "break_count": counts["break"],
        "net_top1": counts["fix"] - counts["break"],
        "fix_ratio": counts["fix"] / float(query_count),
        "break_ratio": counts["break"] / float(query_count),
        "stable_correct_count": counts["stable_correct"],
        "stable_wrong_count": counts["stable_wrong"],
        "rank_improved_count": counts["rank_improved"],
        "rank_worsened_count": counts["rank_worsened"],
        "rank_same_count": counts["rank_same"],
        "mean_rank_improvement": float(
            torch.tensor(rank_delta_values, dtype=torch.float32).mean()
        ),
        "median_rank_improvement": percentile(
            torch.tensor(rank_delta_values, dtype=torch.float32), 0.5
        ),
        "mean_ap_delta": float(
            torch.tensor(ap_delta_values, dtype=torch.float32).mean()
        ),
        "mean_inp_delta": float(
            torch.tensor(inp_delta_values, dtype=torch.float32).mean()
        ),
        "mean_top10_overlap": float(
            torch.tensor(overlap_values, dtype=torch.float32).mean()
        ),
    }

    by_view = defaultdict(lambda: defaultdict(int))
    by_source = defaultdict(lambda: defaultdict(int))
    for row in all_rows:
        by_view[str(row["query_view"])][row["category"]] += 1
        by_source[str(row["query_source"])][row["category"]] += 1
    summary["by_query_view"] = {key: dict(value) for key, value in by_view.items()}
    summary["by_query_source"] = {
        key: dict(value) for key, value in by_source.items()
    }
    return summary, all_rows


def metrics_from_embeddings(
    text: torch.Tensor,
    image: torch.Tensor,
    query_pids: torch.Tensor,
    gallery_pids: torch.Tensor,
    device: torch.device,
    query_chunk: int,
) -> dict:
    totals = defaultdict(float)
    query_count = text.shape[0]
    image_device = image.to(device)
    for start in range(0, query_count, query_chunk):
        end = min(start + query_chunk, query_count)
        scores = text[start:end].to(device) @ image_device.t()
        metrics = row_metrics_from_scores(
            scores.cpu(), query_pids[start:end], gallery_pids
        )
        totals["r1"] += float(metrics["correct"].float().sum())
        totals["r5"] += float(
            (metrics["best_positive_rank"] <= 5).float().sum()
        )
        totals["r10"] += float(
            (metrics["best_positive_rank"] <= 10).float().sum()
        )
        totals["ap"] += float(metrics["ap"].sum())
        totals["inp"] += float(metrics["inp"].sum())
    return {
        "R1": totals["r1"] / query_count * 100.0,
        "R5": totals["r5"] / query_count * 100.0,
        "R10": totals["r10"] / query_count * 100.0,
        "mAP": totals["ap"] / query_count * 100.0,
        "mINP": totals["inp"] / query_count * 100.0,
    }


def trusted_intersection_audit(
    args_cli: argparse.Namespace,
    train_args,
    dataset: TAGPEDES,
    actual,
    device: torch.device,
    output_dir: Path,
) -> dict:
    audit_dir = output_dir / "trusted_intersection"
    audit_dir.mkdir(parents=True, exist_ok=True)
    transform = build_transforms(train_args.img_size, False, False)
    unique_images, caption_records, image_id_to_row = build_train_records(dataset)

    image_cache = encode_train_images(
        actual,
        unique_images,
        transform,
        device,
        args_cli.image_batch_size,
        args_cli.num_workers,
    )
    text_cache = encode_train_texts(
        actual,
        caption_records,
        train_args.text_length,
        device,
        args_cli.text_batch_size,
        args_cli.num_workers,
        args_cli.max_train_queries,
    )
    support_dataset = HIREV2IdentityDataset(
        dataset.train,
        transform=transform,
        text_length=train_args.text_length,
        support_size=int(train_args.hire_v2_support_size),
        support_image_views=getattr(dataset, "train_image_views", None),
        seed=int(train_args.seed),
    )

    uncertainty_weight = actual.image_uncertainty.proj.weight.detach().float().cpu()
    uncertainty_bias = actual.image_uncertainty.proj.bias.detach().float().cpu()
    global_uncertainty = {
        "variance_all_images": tensor_summary(image_cache["variance"]),
        "per_image_variance_mean": tensor_summary(
            image_cache["variance"].mean(dim=-1)
        ),
        "per_image_variance_std": tensor_summary(
            image_cache["variance"].std(dim=-1, unbiased=False)
        ),
        "uncertainty_weight_norm": float(uncertainty_weight.norm()),
        "uncertainty_bias_norm": float(uncertainty_bias.norm()),
        "uncertainty_weight_abs_max": float(uncertainty_weight.abs().max()),
        "uncertainty_bias_abs_max": float(uncertainty_bias.abs().max()),
    }
    write_json(audit_dir / "uncertainty_head_summary.json", global_uncertainty)

    image_id_to_group_row = {
        int(record["image_id"]): row for row, record in enumerate(unique_images)
    }
    epoch_summaries = {}
    retrieval_summaries = {}

    group_csv_fields = [
        "support_epoch",
        "group_row",
        "pid",
        "anchor_image_id",
        "anchor_view",
        "anchor_source",
        "support_count",
        "support_image_ids",
        "valid",
        "trusted_simple_cos",
        "trusted_variance_cos",
        "variance_simple_cos",
        "support_scalar_variance_cv",
        "support_scalar_precision_cv",
        "precision_dim_cv",
        "tau_share",
        "tau2_mean",
    ]

    for epoch in args_cli.support_epochs:
        support_rows, support_mask, support_image_ids = build_support_index_matrix(
            support_dataset,
            unique_images,
            image_id_to_row,
            epoch,
            int(train_args.hire_v2_support_size),
        )
        groups = construct_group_variants(
            image_cache["identity"],
            image_cache["variance"],
            support_rows,
            support_mask,
            device,
        )
        valid = groups["valid"]
        valid_count = int(valid.sum())
        summary = {
            "support_epoch": epoch,
            "group_count": len(unique_images),
            "valid_group_count": valid_count,
            "valid_group_ratio": valid_count / float(len(unique_images)),
            "support_count": tensor_summary(groups["count"].float()),
            "trusted_simple_cos": tensor_summary(groups["trusted_simple_cos"][valid]),
            "trusted_variance_cos": tensor_summary(
                groups["trusted_variance_cos"][valid]
            ),
            "variance_simple_cos": tensor_summary(
                groups["variance_simple_cos"][valid]
            ),
            "support_scalar_variance_cv": tensor_summary(
                groups["support_scalar_variance_cv"][valid]
            ),
            "support_scalar_precision_cv": tensor_summary(
                groups["support_scalar_precision_cv"][valid]
            ),
            "precision_dim_cv": tensor_summary(groups["precision_dim_cv"][valid]),
            "tau_share": tensor_summary(groups["tau_share"][valid]),
            "tau2": tensor_summary(groups["tau2"][valid]),
            "near_simple_ratio_cos_ge_0_999": float(
                (groups["trusted_simple_cos"][valid] >= 0.999).float().mean()
            ),
            "near_simple_ratio_cos_ge_0_9995": float(
                (groups["trusted_simple_cos"][valid] >= 0.9995).float().mean()
            ),
            "near_simple_ratio_cos_ge_0_9999": float(
                (groups["trusted_simple_cos"][valid] >= 0.9999).float().mean()
            ),
        }
        epoch_summaries[str(epoch)] = summary
        epoch_dir = audit_dir / "epoch_{:03d}".format(epoch)
        write_json(epoch_dir / "group_summary.json", summary)

        rows = []
        for group_row, record in enumerate(unique_images):
            rows.append(
                {
                    "support_epoch": epoch,
                    "group_row": group_row,
                    "pid": record["pid"],
                    "anchor_image_id": record["image_id"],
                    "anchor_view": record["view"],
                    "anchor_source": record["source"],
                    "support_count": int(groups["count"][group_row]),
                    "support_image_ids": "|".join(
                        str(value) for value in support_image_ids[group_row]
                    ),
                    "valid": int(groups["valid"][group_row]),
                    "trusted_simple_cos": float(
                        groups["trusted_simple_cos"][group_row]
                    ),
                    "trusted_variance_cos": float(
                        groups["trusted_variance_cos"][group_row]
                    ),
                    "variance_simple_cos": float(
                        groups["variance_simple_cos"][group_row]
                    ),
                    "support_scalar_variance_cv": float(
                        groups["support_scalar_variance_cv"][group_row]
                    ),
                    "support_scalar_precision_cv": float(
                        groups["support_scalar_precision_cv"][group_row]
                    ),
                    "precision_dim_cv": float(
                        groups["precision_dim_cv"][group_row]
                    ),
                    "tau_share": float(groups["tau_share"][group_row]),
                    "tau2_mean": float(groups["tau2"][group_row].mean()),
                }
            )
        write_csv(epoch_dir / "group_per_image.csv", rows, group_csv_fields)

        if epoch in args_cli.retrieval_epochs:
            method_rows = {}
            method_summaries = {}
            for method in ("simple", "variance_only", "trusted"):
                method_summary, per_query = paired_group_ranks(
                    text_cache["identity"],
                    text_cache["pid"],
                    text_cache["image_id"],
                    groups[method],
                    image_cache["pid"],
                    image_cache["image_id"],
                    groups["valid"],
                    image_id_to_group_row,
                    device,
                    args_cli.query_chunk,
                    args_cli.gallery_chunk,
                )
                method_summaries[method] = method_summary
                for row in per_query:
                    key = int(row["query_row"])
                    merged = method_rows.setdefault(
                        key,
                        {
                            "query_row": row["query_row"],
                            "query_pid": row["query_pid"],
                            "query_image_id": row["query_image_id"],
                            "paired_group_image_id": row["paired_group_image_id"],
                        },
                    )
                    for field in (
                        "rank",
                        "positive_score",
                        "hardest_negative_score",
                        "margin",
                    ):
                        merged["{}_{}".format(method, field)] = row[field]
            retrieval_summaries[str(epoch)] = method_summaries
            write_json(
                epoch_dir / "paired_group_retrieval_summary.json",
                method_summaries,
            )
            retrieval_fields = [
                "query_row",
                "query_pid",
                "query_image_id",
                "paired_group_image_id",
            ]
            for method in ("simple", "variance_only", "trusted"):
                retrieval_fields.extend(
                    [
                        "{}_rank".format(method),
                        "{}_positive_score".format(method),
                        "{}_hardest_negative_score".format(method),
                        "{}_margin".format(method),
                    ]
                )
            write_csv(
                epoch_dir / "paired_group_retrieval_per_query.csv",
                [method_rows[key] for key in sorted(method_rows)],
                retrieval_fields,
            )

    final = {
        "unique_train_image_count": len(unique_images),
        "train_caption_query_count": int(text_cache["identity"].shape[0]),
        "support_epochs": args_cli.support_epochs,
        "retrieval_epochs": args_cli.retrieval_epochs,
        "uncertainty_head": global_uncertainty,
        "epoch_summaries": epoch_summaries,
        "paired_group_retrieval": retrieval_summaries,
    }
    write_json(audit_dir / "trusted_intersection_audit.json", final)
    return final


def fix_break_audit(
    args_cli: argparse.Namespace,
    identity_args,
    identity_dataset: TAGPEDES,
    identity_model,
    identity_actual,
    device: torch.device,
    output_dir: Path,
) -> dict:
    audit_dir = output_dir / "fix_break"
    audit_dir.mkdir(parents=True, exist_ok=True)

    identity_text, identity_image, qids, gids = collect_test_components(
        identity_model,
        identity_actual,
        identity_args,
        device,
        args_cli.num_workers,
    )
    required = {"observation", "identity", "final"}
    missing = required - set(identity_text.keys())
    if missing:
        raise RuntimeError(
            "v16.2 model did not return required text components: {}".format(
                sorted(missing)
            )
        )
    missing = required - set(identity_image.keys())
    if missing:
        raise RuntimeError(
            "v16.2 model did not return required image components: {}".format(
                sorted(missing)
            )
        )

    query_meta, gallery_meta = build_test_metadata(identity_dataset)
    if len(query_meta) != qids.numel() or len(gallery_meta) != gids.numel():
        raise RuntimeError("test metadata order/size does not match data loaders")

    component_metrics = {}
    for name in ("observation", "identity", "final"):
        component_metrics[name] = metrics_from_embeddings(
            identity_text[name],
            identity_image[name],
            qids,
            gids,
            device,
            args_cli.query_chunk,
        )

    comparisons = {}
    identity_gate_value = float(identity_actual.identity_gate().detach().cpu())
    summary, rows = compare_score_sources(
        identity_text["observation"],
        identity_image["observation"],
        identity_text["final"],
        identity_image["final"],
        qids,
        gids,
        query_meta,
        gallery_meta,
        device,
        args_cli.query_chunk,
        "v16.2_observation_vs_v16.2_final",
        identity_text=identity_text["identity"],
        identity_image=identity_image["identity"],
        identity_gate_value=identity_gate_value,
    )
    comparisons[summary["comparison"]] = summary
    export_comparison(audit_dir, summary["comparison"], summary, rows)

    if args_cli.anchor_config_file or args_cli.anchor_checkpoint:
        if not (args_cli.anchor_config_file and args_cli.anchor_checkpoint):
            raise ValueError(
                "--anchor-config-file and --anchor-checkpoint must be provided together"
            )
        anchor_config = ensure_file(args_cli.anchor_config_file, "anchor config")
        anchor_checkpoint = ensure_file(
            args_cli.anchor_checkpoint, "anchor checkpoint"
        )
        anchor_args, anchor_dataset, anchor_model, anchor_actual = load_model(
            anchor_config, anchor_checkpoint, device, expected_mode="anchor"
        )
        anchor_text, anchor_image, anchor_qids, anchor_gids = collect_test_components(
            anchor_model,
            anchor_actual,
            anchor_args,
            device,
            args_cli.num_workers,
        )
        if not torch.equal(anchor_qids, qids) or not torch.equal(anchor_gids, gids):
            raise RuntimeError("v16.1 and v16.2 test orders differ")
        component_metrics["v16.1_observation"] = metrics_from_embeddings(
            anchor_text["observation"],
            anchor_image["observation"],
            qids,
            gids,
            device,
            args_cli.query_chunk,
        )
        summary_anchor, rows_anchor = compare_score_sources(
            anchor_text["observation"],
            anchor_image["observation"],
            identity_text["final"],
            identity_image["final"],
            qids,
            gids,
            query_meta,
            gallery_meta,
            device,
            args_cli.query_chunk,
            "v16.1_observation_vs_v16.2_final",
        )
        comparisons[summary_anchor["comparison"]] = summary_anchor
        export_comparison(
            audit_dir,
            summary_anchor["comparison"],
            summary_anchor,
            rows_anchor,
        )
        del anchor_model
        torch.cuda.empty_cache()

    final = {
        "identity_gate": identity_gate_value,
        "component_metrics": component_metrics,
        "comparisons": comparisons,
    }
    write_json(audit_dir / "fix_break_audit.json", final)
    return final


def export_comparison(
    audit_dir: Path,
    name: str,
    summary: dict,
    rows: List[dict],
) -> None:
    comparison_dir = audit_dir / name
    comparison_dir.mkdir(parents=True, exist_ok=True)
    write_json(comparison_dir / "summary.json", summary)
    if not rows:
        return
    fields = list(rows[0].keys())
    write_csv(comparison_dir / "all_query_deltas.csv", rows, fields)
    write_csv(
        comparison_dir / "fix_cases.csv",
        [row for row in rows if row["category"] == "fix"],
        fields,
    )
    write_csv(
        comparison_dir / "break_cases.csv",
        [row for row in rows if row["category"] == "break"],
        fields,
    )
    write_csv(
        comparison_dir / "rank_improved_cases.csv",
        [row for row in rows if float(row["rank_improvement"]) > 0],
        fields,
    )
    write_csv(
        comparison_dir / "rank_worsened_cases.csv",
        [row for row in rows if float(row["rank_improvement"]) < 0],
        fields,
    )


def build_final_markdown(
    output_dir: Path,
    trusted: dict,
    fix_break: dict,
    manifest: dict,
) -> None:
    lines = []
    lines.append("# v16.2.0 身份机制无需训练审计")
    lines.append("")
    lines.append("## 运行信息")
    lines.append("")
    lines.append("- 配置：`{}`".format(manifest["config_file"]))
    lines.append("- 检查点：`{}`".format(manifest["checkpoint"]))
    lines.append("- 支持轮次：`{}`".format(", ".join(map(str, trusted["support_epochs"]))))
    lines.append("- 组检索轮次：`{}`".format(", ".join(map(str, trusted["retrieval_epochs"]))))
    lines.append("")
    lines.append("## 可信交集")
    lines.append("")
    variance = trusted["uncertainty_head"]["variance_all_images"]
    lines.append(
        "- 全训练图身份方差：均值 `{:.6f}`，标准差 `{:.6f}`。".format(
            variance["mean"], variance["std"]
        )
    )
    lines.append(
        "- 方差头权重范数：`{:.6f}`；偏置范数：`{:.6f}`。".format(
            trusted["uncertainty_head"]["uncertainty_weight_norm"],
            trusted["uncertainty_head"]["uncertainty_bias_norm"],
        )
    )
    for epoch, summary in trusted["epoch_summaries"].items():
        cos_summary = summary["trusted_simple_cos"]
        precision_cv = summary["precision_dim_cv"]
        tau_share = summary["tau_share"]
        lines.append("")
        lines.append("### 支持轮次 {}".format(epoch))
        lines.append("")
        lines.append(
            "- 有效身份组比例：`{:.4%}`。".format(summary["valid_group_ratio"])
        )
        lines.append(
            "- 可信交集与简单均值余弦：均值 `{:.6f}`，中位数 `{:.6f}`。".format(
                cos_summary["mean"], cos_summary["median"]
            )
        )
        lines.append(
            "- 余弦不低于 0.9995 的身份组比例：`{:.4%}`。".format(
                summary["near_simple_ratio_cos_ge_0_9995"]
            )
        )
        lines.append(
            "- 逐维有效精度变异系数：均值 `{:.6f}`，中位数 `{:.6f}`。".format(
                precision_cv["mean"], precision_cv["median"]
            )
        )
        lines.append(
            "- 异质性占总不确定性的比例：均值 `{:.6%}`，中位数 `{:.6%}`。".format(
                tau_share["mean"], tau_share["median"]
            )
        )
        if epoch in trusted["paired_group_retrieval"]:
            methods = trusted["paired_group_retrieval"][epoch]
            lines.append("")
            lines.append("| 组构造 | R1 | R5 | R10 | MRR | 平均排名 | 平均间隔 |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|")
            for method in ("simple", "variance_only", "trusted"):
                value = methods[method]
                lines.append(
                    "| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.6f} |".format(
                        method,
                        value["R1"],
                        value["R5"],
                        value["R10"],
                        value["MRR"],
                        value["mean_rank"],
                        value["mean_margin"],
                    )
                )

    lines.append("")
    lines.append("## 修复与破坏")
    lines.append("")
    lines.append("| 对比 | 修复 | 破坏 | 净修复 | 排名改善 | 排名恶化 | 平均 AP 变化 | 平均 mINP 变化 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, summary in fix_break["comparisons"].items():
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {:.6f} | {:.6f} |".format(
                name,
                summary["fix_count"],
                summary["break_count"],
                summary["net_top1"],
                summary["rank_improved_count"],
                summary["rank_worsened_count"],
                summary["mean_ap_delta"],
                summary["mean_inp_delta"],
            )
        )

    lines.append("")
    lines.append("## 自动判读规则")
    lines.append("")
    lines.append(
        "1. 若可信交集与简单均值的中位余弦大于 0.9995，且严格组检索 R1 差值小于 0.05，当前概率权重可视为基本未启动。"
    )
    lines.append(
        "2. 若有效精度变异系数接近零，说明三个支持图几乎等权；此时收益更可能来自严格留一组监督和共享身份映射。"
    )
    lines.append(
        "3. 若（v16.2 观测→最终）的修复数明显大于破坏数，身份残差本身有效；若（v16.1→v16.2 最终）净修复很小，则主要问题是 v16.2 训练削弱了观测锚点。"
    )
    lines.append("")
    with open(output_dir / "v162_identity_audit_report.md", "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")



def automatic_findings(trusted: dict, fix_break: dict) -> dict:
    findings = {}
    retrieval_epochs = trusted.get("retrieval_epochs", [])
    if retrieval_epochs:
        epoch_key = str(retrieval_epochs[0])
        epoch_summary = trusted["epoch_summaries"].get(epoch_key, {})
        retrieval = trusted["paired_group_retrieval"].get(epoch_key, {})
        if epoch_summary and retrieval:
            cosine_median = epoch_summary["trusted_simple_cos"]["median"]
            precision_cv_median = epoch_summary["precision_dim_cv"]["median"]
            r1_gap = retrieval["trusted"]["R1"] - retrieval["simple"]["R1"]
            findings["trusted_intersection"] = {
                "support_epoch": int(epoch_key),
                "trusted_simple_cosine_median": cosine_median,
                "precision_dim_cv_median": precision_cv_median,
                "trusted_minus_simple_group_R1": r1_gap,
                "effectively_simple_mean": bool(
                    cosine_median is not None
                    and cosine_median > 0.9995
                    and abs(r1_gap) < 0.05
                ),
            }

    internal = fix_break.get("comparisons", {}).get(
        "v16.2_observation_vs_v16.2_final"
    )
    external = fix_break.get("comparisons", {}).get(
        "v16.1_observation_vs_v16.2_final"
    )
    if internal:
        findings["identity_residual"] = {
            "internal_net_top1": internal["net_top1"],
            "internal_fix_count": internal["fix_count"],
            "internal_break_count": internal["break_count"],
            "net_positive": bool(internal["net_top1"] > 0),
        }
    if internal and external:
        findings["anchor_protection"] = {
            "v16.1_to_v16.2_final_net_top1": external["net_top1"],
            "internal_identity_residual_net_top1": internal["net_top1"],
            "anchor_degradation_suspected": bool(
                internal["net_top1"] > 0
                and external["net_top1"] < internal["net_top1"]
            ),
        }
    return findings


def main() -> None:
    cli = parse_args()
    started = time.time()
    config_file = ensure_file(cli.config_file, "v16.2 config")
    checkpoint = ensure_file(cli.checkpoint, "v16.2 checkpoint")
    output_dir = ensure_output_dir(cli.output_dir, cli.overwrite)
    missing_retrieval_epochs = sorted(set(cli.retrieval_epochs) - set(cli.support_epochs))
    if missing_retrieval_epochs:
        raise ValueError(
            "retrieval epochs must also appear in support epochs: {}".format(
                missing_retrieval_epochs
            )
        )
    device = torch.device(cli.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is not available")

    train_args, dataset, model, actual = load_model(
        config_file, checkpoint, device, expected_mode="identity"
    )
    if not getattr(actual, "is_hire_v2_identity_model", False):
        raise RuntimeError("checkpoint/config did not build HIRE-v2 identity model")

    manifest = {
        "script": op.abspath(__file__),
        "config_file": config_file,
        "checkpoint": checkpoint,
        "config_sha256": sha256_file(config_file),
        "checkpoint_sha256": sha256_file(checkpoint),
        "anchor_config_file": op.abspath(cli.anchor_config_file)
        if cli.anchor_config_file
        else "",
        "anchor_checkpoint": op.abspath(cli.anchor_checkpoint)
        if cli.anchor_checkpoint
        else "",
        "device": str(device),
        "support_epochs": cli.support_epochs,
        "retrieval_epochs": cli.retrieval_epochs,
        "max_train_queries": cli.max_train_queries,
        "training_or_backward_used": False,
    }
    write_json(output_dir / "audit_manifest.json", manifest)

    trusted = trusted_intersection_audit(
        cli, train_args, dataset, actual, device, output_dir
    )
    fix_break = fix_break_audit(
        cli, train_args, dataset, model, actual, device, output_dir
    )
    findings = automatic_findings(trusted, fix_break)
    combined = {
        "manifest": manifest,
        "trusted_intersection": trusted,
        "fix_break": fix_break,
        "automatic_findings": findings,
        "elapsed_seconds": time.time() - started,
    }
    write_json(output_dir / "automatic_findings.json", findings)
    write_json(output_dir / "v162_identity_audit_report.json", combined)
    build_final_markdown(output_dir, trusted, fix_break, manifest)
    print("Audit completed: {}".format(output_dir))
    print("Report: {}".format(output_dir / "v162_identity_audit_report.md"))


if __name__ == "__main__":
    main()

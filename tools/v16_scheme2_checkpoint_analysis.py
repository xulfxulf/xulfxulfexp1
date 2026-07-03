import argparse
import csv
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.bases import ImageDataset, ImageTextDataset, TextDataset  # noqa: E402
from datasets.build import build_transforms, collate  # noqa: E402
from datasets.tagpedes import TAGPEDES  # noqa: E402
from model import build_model  # noqa: E402
from utils.iotools import load_train_configs  # noqa: E402


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


def strip_module(state):
    if not state:
        return state
    if all(k.startswith("module.") for k in state):
        return {k[len("module."):]: v for k, v in state.items()}
    return state


def load_model(config_path, checkpoint_path, mode=None, device="cuda"):
    args = load_train_configs(config_path)
    if mode is not None:
        args.irra_light_mode = mode
    args.training = True
    args.distributed = False
    dataset = TAGPEDES(root=args.root_dir, verbose=False)
    model = build_model(args, len(dataset.train_id_container))
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = strip_module(ckpt.get("model", ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if len(missing) > 20 or len(unexpected) > 20:
        print(f"warning: many load differences for {checkpoint_path}: missing={len(missing)} unexpected={len(unexpected)}")
    model.to(device)
    model.eval()
    return args, dataset, model


def build_train_loader(args, dataset):
    transform = build_transforms(img_size=args.img_size, aug=False, is_train=False)
    train_set = ImageTextDataset(
        dataset.train,
        transform,
        text_length=args.text_length,
        support_size=args.irra_light_support_size,
        support_image_views=getattr(dataset, "train_image_views", None),
    )
    return DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(4, int(args.num_workers)),
        collate_fn=collate,
    )


def build_eval_loaders(args, dataset):
    transform = build_transforms(img_size=args.img_size, is_train=False)
    ds = dataset.test
    img_set = ImageDataset(ds["image_pids"], ds["img_paths"], transform)
    txt_set = TextDataset(ds["caption_pids"], ds["captions"], text_length=args.text_length)
    img_loader = DataLoader(img_set, batch_size=args.test_batch_size, shuffle=False, num_workers=min(4, int(args.num_workers)))
    txt_loader = DataLoader(txt_set, batch_size=args.test_batch_size, shuffle=False, num_workers=min(4, int(args.num_workers)))
    return img_loader, txt_loader


def train_image_views(dataset):
    views = {}
    for _idx, (_pid, image_id, _path, _caption) in enumerate(dataset.train):
        image_id = int(image_id)
        if image_id not in views:
            raw = None
            if getattr(dataset, "train_image_views", None) is not None and image_id < len(dataset.train_image_views):
                raw = dataset.train_image_views[image_id]
            views[image_id] = "aerial" if raw == 0 else "ground" if raw == 1 else "unknown"
    return views


def load_conflict_counts(path):
    hard = defaultdict(int)
    unknown = defaultdict(int)
    for row in read_csv_rows(path):
        sid = int(row["anchor_sample_id"])
        ctype = row.get("conflict_type", "")
        if ctype == "hard_contradiction":
            hard[sid] += 1
        if ctype == "unknown":
            unknown[sid] += 1
    return hard, unknown


def projected_features(model, image_feats, text_feats, caption_ids):
    i = image_feats[:, 0, :].float()
    t = text_feats[torch.arange(text_feats.shape[0], device=caption_ids.device), caption_ids.argmax(dim=-1)].float()
    if model.irra_light_split:
        return (
            F.normalize(model.identity_head(i), dim=-1),
            F.normalize(model.identity_head(t), dim=-1),
            F.normalize(model.state_head(i), dim=-1),
            F.normalize(model.state_head(t), dim=-1),
        )
    if model.irra_light_single_proj:
        x_i = F.normalize(model.single_head(i), dim=-1)
        x_t = F.normalize(model.single_head(t), dim=-1)
        return x_i, x_t, x_i, x_t
    i = F.normalize(i, dim=-1)
    t = F.normalize(t, dim=-1)
    return i, t, i, t


@torch.no_grad()
def support_contribution(mode_name, config_path, checkpoint_path, conflict_csv, output_dir, device="cuda"):
    args, dataset, model = load_model(config_path, checkpoint_path, mode=mode_name, device=device)
    loader = build_train_loader(args, dataset)
    views = train_image_views(dataset)
    hard_counts, unknown_counts = load_conflict_counts(conflict_csv) if conflict_csv else (defaultdict(int), defaultdict(int))
    rows = []
    support_ratios = []
    max_ratios = []
    low_quality_ratios = []
    hard_ratios = []

    sample_offset = 0
    logit_scale = model.logit_scale.to(device)
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        image_feats, text_feats = model.base_model(batch["images"], batch["caption_ids"])
        identity_i, identity_t, _state_i, _state_t = projected_features(model, image_feats, text_feats, batch["caption_ids"])
        support_i, _support_t, support_mask = model._encode_support_bag(batch)

        source_logits = torch.sum(identity_t * identity_i, dim=-1) * logit_scale
        support_logits = torch.sum(identity_t.unsqueeze(1) * support_i, dim=-1) * logit_scale
        support_exp = torch.exp(support_logits.float()) * support_mask.float()
        source_exp = torch.exp(source_logits.float())
        pos_sum = source_exp + support_exp.sum(dim=1)
        ratio = support_exp.sum(dim=1) / pos_sum.clamp_min(1e-12)
        max_ratio = support_exp.max(dim=1).values / pos_sum.clamp_min(1e-12)

        support_image_ids = batch["support_image_ids"].detach().cpu().numpy()
        anchor_image_ids = batch["image_ids"].detach().cpu().numpy()
        support_mask_np = support_mask.detach().cpu().numpy()
        for local_idx in range(ratio.shape[0]):
            sid = sample_offset + local_idx
            used_supports = [int(x) for x, m in zip(support_image_ids[local_idx].tolist(), support_mask_np[local_idx].tolist()) if m]
            anchor_view = views.get(int(anchor_image_ids[local_idx]), "unknown")
            support_views = [views.get(x, "unknown") for x in used_supports]
            has_cross = any(v != anchor_view for v in support_views)
            # The detailed low-quality flag is read from the conflict table in local summaries;
            # here we keep the field structural for downstream joins.
            row = {
                "anchor_sample_id": sid,
                "mode": mode_name,
                "support_positive_ratio": float(ratio[local_idx].item()),
                "max_single_support_ratio": float(max_ratio[local_idx].item()),
                "num_support_used": int(sum(support_mask_np[local_idx].tolist())),
                "has_cross_view_support": int(has_cross),
                "has_low_quality_aerial_support": "",
                "num_hard_contradiction_support": int(hard_counts.get(sid, 0)),
                "num_unknown_support": int(unknown_counts.get(sid, 0)),
            }
            rows.append(row)
            support_ratios.append(row["support_positive_ratio"])
            max_ratios.append(row["max_single_support_ratio"])
            if row["num_hard_contradiction_support"] > 0:
                hard_ratios.append(row["support_positive_ratio"])
        sample_offset += ratio.shape[0]
        if sample_offset % 6400 == 0:
            print(f"{mode_name} contribution processed {sample_offset}/{len(dataset.train)}")

    out = Path(output_dir) / "support_positive_contribution"
    write_csv(out / f"support_contribution_per_anchor_{mode_name}.csv", rows)

    def q(values, quant):
        return float(np.quantile(np.asarray(values, dtype=np.float64), quant)) if values else None

    summary = {
        "mode": mode_name,
        "anchors": len(rows),
        "support_positive_ratio_mean": float(np.mean(support_ratios)) if support_ratios else None,
        "support_positive_ratio_median": q(support_ratios, 0.5),
        "support_positive_ratio_p90": q(support_ratios, 0.9),
        "max_single_support_ratio_p90": q(max_ratios, 0.9),
        "hard_contradiction_support_ratio_mean": float(np.mean(hard_ratios)) if hard_ratios else None,
        "low_quality_aerial_support_ratio_mean": float(np.mean(low_quality_ratios)) if low_quality_ratios else None,
    }
    return summary


@torch.no_grad()
def extract_split_features(config_path, checkpoint_path, mode_name, output_dir, device="cuda"):
    args, dataset, model = load_model(config_path, checkpoint_path, mode=mode_name, device=device)
    img_loader, txt_loader = build_eval_loaders(args, dataset)
    qids, gids = [], []
    q_id_feats, q_state_feats = [], []
    g_id_feats, g_state_feats = [], []

    for pid, caption in txt_loader:
        caption = caption.to(device)
        text_feats = model.base_model.encode_text(caption)
        t = text_feats[torch.arange(text_feats.shape[0], device=device), caption.argmax(dim=-1)].float()
        q_id_feats.append(F.normalize(model.identity_head(t), dim=-1).cpu())
        q_state_feats.append(F.normalize(model.state_head(t), dim=-1).cpu())
        qids.append(pid.view(-1).cpu())

    for pid, img in img_loader:
        img = img.to(device)
        image_feats = model.base_model.encode_image(img)
        i = image_feats[:, 0, :].float()
        g_id_feats.append(F.normalize(model.identity_head(i), dim=-1).cpu())
        g_state_feats.append(F.normalize(model.state_head(i), dim=-1).cpu())
        gids.append(pid.view(-1).cpu())

    return {
        "q_id": torch.cat(q_id_feats, 0),
        "q_state": torch.cat(q_state_feats, 0),
        "g_id": torch.cat(g_id_feats, 0),
        "g_state": torch.cat(g_state_feats, 0),
        "qids": torch.cat(qids, 0),
        "gids": torch.cat(gids, 0),
    }


def rankdata_torch(x):
    order = torch.argsort(x)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(x.numel(), dtype=torch.float32)
    return ranks


def analyze_split_heads(mode_name, feats, output_dir, device="cuda", spearman_samples=200000, seed=20260702):
    q_id = feats["q_id"].to(device)
    q_state = feats["q_state"].to(device)
    g_id = feats["g_id"].to(device)
    g_state = feats["g_state"].to(device)
    qids = feats["qids"]
    gids = feats["gids"]

    n_q, n_g = q_id.shape[0], g_id.shape[0]
    chunk = 256
    sum_x = sum_y = sum_x2 = sum_y2 = sum_xy = 0.0
    n = 0
    top_same = 0
    overlap_sums = {5: 0.0, 10: 0.0, 50: 0.0}
    fix = break_count = both_correct = both_wrong = 0
    id_correct = state_correct = 0

    for start in range(0, n_q, chunk):
        end = min(start + chunk, n_q)
        s_id = q_id[start:end] @ g_id.t()
        s_state = q_state[start:end] @ g_state.t()
        x = s_id.float()
        y = s_state.float()
        sum_x += float(x.sum().item())
        sum_y += float(y.sum().item())
        sum_x2 += float((x * x).sum().item())
        sum_y2 += float((y * y).sum().item())
        sum_xy += float((x * y).sum().item())
        n += x.numel()

        id_top = torch.topk(s_id, k=50, dim=1).indices.cpu()
        st_top = torch.topk(s_state, k=50, dim=1).indices.cpu()
        top_same += int((id_top[:, 0] == st_top[:, 0]).sum().item())
        for local in range(id_top.shape[0]):
            qpid = int(qids[start + local].item())
            id_ok = int(gids[int(id_top[local, 0])].item()) == qpid
            st_ok = int(gids[int(st_top[local, 0])].item()) == qpid
            id_correct += int(id_ok)
            state_correct += int(st_ok)
            fix += int((not id_ok) and st_ok)
            break_count += int(id_ok and (not st_ok))
            both_correct += int(id_ok and st_ok)
            both_wrong += int((not id_ok) and (not st_ok))
            for k in overlap_sums:
                overlap_sums[k] += len(set(id_top[local, :k].tolist()) & set(st_top[local, :k].tolist())) / k

    mean_x = sum_x / n
    mean_y = sum_y / n
    cov = sum_xy / n - mean_x * mean_y
    var_x = sum_x2 / n - mean_x * mean_x
    var_y = sum_y2 / n - mean_y * mean_y
    pearson = cov / math.sqrt(max(var_x, 1e-12) * max(var_y, 1e-12))

    rng = random.Random(seed)
    sample_q = torch.tensor([rng.randrange(n_q) for _ in range(min(spearman_samples, n))], device=device)
    sample_g = torch.tensor([rng.randrange(n_g) for _ in range(sample_q.numel())], device=device)
    sx = (q_id[sample_q] * g_id[sample_g]).sum(dim=1).cpu()
    sy = (q_state[sample_q] * g_state[sample_g]).sum(dim=1).cpu()
    rx = rankdata_torch(sx)
    ry = rankdata_torch(sy)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    spearman = float(
        ((rx * ry).mean() / (rx.pow(2).mean().sqrt() * ry.pow(2).mean().sqrt()).clamp_min(1e-12)).item()
    )

    row = {
        "mode": mode_name,
        "queries": n_q,
        "gallery": n_g,
        "pearson_full_matrix": pearson,
        "spearman_sample": spearman,
        "spearman_sample_pairs": int(sample_q.numel()),
        "top1_same_rate": top_same / n_q if n_q else 0.0,
        "top5_overlap": overlap_sums[5] / n_q if n_q else 0.0,
        "top10_overlap": overlap_sums[10] / n_q if n_q else 0.0,
        "top50_overlap": overlap_sums[50] / n_q if n_q else 0.0,
        "state_fixes_identity_errors": fix,
        "state_breaks_identity_correct": break_count,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "identity_top1_acc": id_correct / n_q * 100 if n_q else 0.0,
        "state_top1_acc": state_correct / n_q * 100 if n_q else 0.0,
    }
    return row


def update_pre_summary(output_dir, contribution_summaries, head_rows):
    out = Path(output_dir)
    lines = [
        "# Scheme-2 Checkpoint Analysis Summary",
        "",
        "## Support Positive Contribution",
        "",
        "| mode | anchors | mean support positive ratio | median | p90 | max single p90 | hard-conflict mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in contribution_summaries:
        hard_ratio = s["hard_contradiction_support_ratio_mean"]
        hard_ratio_text = "" if hard_ratio is None else f"{hard_ratio:.4f}"
        lines.append(
            f"| {s['mode']} | {s['anchors']} | {s['support_positive_ratio_mean']:.4f} | "
            f"{s['support_positive_ratio_median']:.4f} | {s['support_positive_ratio_p90']:.4f} | "
            f"{s['max_single_support_ratio_p90']:.4f} | "
            f"{hard_ratio_text} |"
        )
    lines += [
        "",
        "## Split Head Analysis",
        "",
        "| mode | Pearson | Spearman sample | top1 same | top5 overlap | top10 overlap | top50 overlap | state fixes | state breaks | id top1 | state top1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in head_rows:
        lines.append(
            f"| {r['mode']} | {r['pearson_full_matrix']:.4f} | {r['spearman_sample']:.4f} | "
            f"{r['top1_same_rate']:.4f} | {r['top5_overlap']:.4f} | {r['top10_overlap']:.4f} | "
            f"{r['top50_overlap']:.4f} | {r['state_fixes_identity_errors']} | "
            f"{r['state_breaks_identity_correct']} | {r['identity_top1_acc']:.3f} | {r['state_top1_acc']:.3f} |"
        )
    lines += [
        "",
        "Spearman is computed on a fixed random sample of matrix entries to avoid sorting the full text-image score matrix.",
        "Relation-stratified rows are not generated in this script; use the existing relation-pair diagnosis if a per-relation audit is needed.",
    ]
    (out / "support_positive_contribution" / "support_contribution_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "split_bag_head_analysis" / "split_bag_head_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Checkpoint-based scheme-2 pre-analysis")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--single-proj-bag-config", required=True)
    parser.add_argument("--single-proj-bag-ckpt", required=True)
    parser.add_argument("--split-bag-config", required=True)
    parser.add_argument("--split-bag-ckpt", required=True)
    parser.add_argument("--split-pure-config", required=True)
    parser.add_argument("--split-pure-ckpt", required=True)
    parser.add_argument("--support-conflict-csv", default="")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out = Path(args.output_dir)
    (out / "support_positive_contribution").mkdir(parents=True, exist_ok=True)
    (out / "split_bag_head_analysis").mkdir(parents=True, exist_ok=True)

    contribution_summaries = [
        support_contribution("single_proj_bag", args.single_proj_bag_config, args.single_proj_bag_ckpt, args.support_conflict_csv, out, args.device),
        support_contribution("split_bag", args.split_bag_config, args.split_bag_ckpt, args.support_conflict_csv, out, args.device),
    ]
    write_csv(out / "support_positive_contribution" / "support_contribution_summary.csv", contribution_summaries)

    head_rows = []
    for mode, config, ckpt in [
        ("split_pure", args.split_pure_config, args.split_pure_ckpt),
        ("split_bag", args.split_bag_config, args.split_bag_ckpt),
    ]:
        feats = extract_split_features(config, ckpt, mode, out, args.device)
        head_rows.append(analyze_split_heads(mode, feats, out, args.device))
    write_csv(out / "split_bag_head_analysis" / "split_bag_head_correlation.csv", head_rows)
    update_pre_summary(out, contribution_summaries, head_rows)
    print(f"checkpoint scheme-2 analysis written to {out}")


if __name__ == "__main__":
    main()

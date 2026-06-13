import argparse
import json
import os
import os.path as op
import time

import torch
import torch.nn.functional as F
from prettytable import PrettyTable

from datasets import build_dataloader
from model import build_model
from utils.checkpoint import Checkpointer
from utils.iotools import load_train_configs
from utils.metrics import rank


def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline score fusion diagnostic for IRRA-light split_pure"
    )
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--ckpt_file", default="")
    parser.add_argument("--output_dir", default="")
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=[0.05, 0.1, 0.2],
        help="State-score weights for S_id + lambda * S_state.",
    )
    parser.add_argument(
        "--save_scores",
        action="store_true",
        help="Save full score matrices as fp16 .pt files. Disabled by default.",
    )
    return parser.parse_args()


def encode_split_heads(model, loader, kind):
    model.eval()
    device = next(model.parameters()).device
    ids, id_feats, state_feats = [], [], []

    with torch.no_grad():
        for pid, data in loader:
            data = data.to(device)
            if kind == "text":
                feats = model.base_model.encode_text(data)
                index = torch.arange(feats.shape[0], device=feats.device)
                pooled = feats[index, data.argmax(dim=-1)].float()
            elif kind == "image":
                feats = model.base_model.encode_image(data)
                pooled = feats[:, 0, :].float()
            else:
                raise ValueError(f"Unsupported kind: {kind}")

            id_feat = model._project_light_head(model.identity_head, pooled)
            state_feat = model._project_light_head(model.state_head, pooled)
            ids.append(pid.view(-1).cpu())
            id_feats.append(id_feat.cpu())
            state_feats.append(state_feat.cpu())

    return {
        "ids": torch.cat(ids, 0),
        "id": torch.cat(id_feats, 0),
        "state": torch.cat(state_feats, 0),
    }


def eval_similarity(name, similarity, qids, gids):
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


def format_table(rows):
    table = PrettyTable(["score", "R1", "R5", "R10", "mAP", "mINP"])
    for row in rows:
        table.add_row(
            [
                row["score"],
                row["R1"],
                row["R5"],
                row["R10"],
                row["mAP"],
                row["mINP"],
            ]
        )
    for key in ["R1", "R5", "R10", "mAP", "mINP"]:
        table.custom_format[key] = lambda f, v: f"{v:.3f}"
    return str(table)


def write_outputs(output_dir, config_file, ckpt_file, rows, args, score_paths):
    os.makedirs(output_dir, exist_ok=True)
    payload = {
        "config_file": config_file,
        "ckpt_file": ckpt_file,
        "lambdas": args.lambdas,
        "save_scores": args.save_scores,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "results": rows,
        "score_paths": score_paths,
        "note": "This is an offline diagnostic result log, not a training run.",
    }
    json_path = op.join(output_dir, "score_fusion_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    md_path = op.join(output_dir, "score_fusion_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# IRRA-light Split Pure Offline Score Fusion Diagnostic\n\n")
        f.write("This file records offline evaluation results only. It does not rank or compare modes.\n\n")
        f.write(f"- Config: `{config_file}`\n")
        f.write(f"- Checkpoint: `{ckpt_file}`\n")
        f.write("- Dataset split: TAG-PEDES test split from the saved config\n")
        f.write("- Scores: `S_id`, `S_state`, and `S_id + lambda * S_state`\n")
        f.write("- Headline result field: best/offline R1 for each score variant\n\n")
        f.write("| score | R1 | R5 | R10 | mAP | mINP |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['score']} | {row['R1']:.3f} | {row['R5']:.3f} | "
                f"{row['R10']:.3f} | {row['mAP']:.3f} | {row['mINP']:.3f} |\n"
            )
        if score_paths:
            f.write("\n## Saved Score Matrices\n\n")
            for name, path in score_paths.items():
                f.write(f"- `{name}`: `{path}`\n")
    return json_path, md_path


def main():
    cli_args = parse_args()
    train_args = load_train_configs(cli_args.config_file)
    train_args.training = False
    train_args.distributed = False
    train_args.irra_light = True
    train_args.irra_light_mode = "split_pure"
    train_args.loss_names = "irra_light"

    ckpt_file = cli_args.ckpt_file or op.join(train_args.output_dir, "best.pth")
    if not op.exists(ckpt_file):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_file}")

    output_dir = cli_args.output_dir or op.join(
        train_args.output_dir,
        "offline_score_fusion_" + time.strftime("%Y%m%d_%H%M%S", time.localtime()),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_img_loader, test_txt_loader, num_classes = build_dataloader(train_args)
    model = build_model(train_args, num_classes=num_classes)
    if not getattr(model, "irra_light_split", False):
        raise RuntimeError("This diagnostic requires an IRRA-light split_pure model.")
    Checkpointer(model).load(f=ckpt_file)
    model.to(device)
    model.eval()

    text = encode_split_heads(model, test_txt_loader, "text")
    image = encode_split_heads(model, test_img_loader, "image")

    q_id = F.normalize(text["id"].to(device), p=2, dim=1)
    g_id = F.normalize(image["id"].to(device), p=2, dim=1)
    q_state = F.normalize(text["state"].to(device), p=2, dim=1)
    g_state = F.normalize(image["state"].to(device), p=2, dim=1)

    s_id = q_id @ g_id.t()
    s_state = q_state @ g_state.t()

    qids = text["ids"]
    gids = image["ids"]
    rows = [eval_similarity("S_id", s_id, qids, gids)]
    rows.append(eval_similarity("S_state", s_state, qids, gids))
    for value in cli_args.lambdas:
        rows.append(
            eval_similarity(
                f"S_id + {value:g}*S_state",
                s_id + value * s_state,
                qids,
                gids,
            )
        )

    score_paths = {}
    if cli_args.save_scores:
        os.makedirs(output_dir, exist_ok=True)
        score_paths["S_id"] = op.join(output_dir, "S_id_fp16.pt")
        score_paths["S_state"] = op.join(output_dir, "S_state_fp16.pt")
        torch.save(s_id.half().cpu(), score_paths["S_id"])
        torch.save(s_state.half().cpu(), score_paths["S_state"])
        for value in cli_args.lambdas:
            name = f"S_id_plus_{value:g}_S_state".replace(".", "p")
            path = op.join(output_dir, f"{name}_fp16.pt")
            torch.save((s_id + value * s_state).half().cpu(), path)
            score_paths[f"S_id + {value:g}*S_state"] = path

    json_path, md_path = write_outputs(
        output_dir, cli_args.config_file, ckpt_file, rows, cli_args, score_paths
    )
    print(format_table(rows))
    print(f"Saved metrics: {json_path}")
    print(f"Saved summary: {md_path}")


if __name__ == "__main__":
    main()

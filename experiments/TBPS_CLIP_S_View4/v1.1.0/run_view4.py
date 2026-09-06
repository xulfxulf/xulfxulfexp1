#!/usr/bin/env python3
import argparse
import os
import shlex
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from view4.common import ROOT
from view4.config import load_config
from view4.upstream import bootstrap


def main():
    parser = argparse.ArgumentParser(description="TBPS-CLIP-S View4 G0; no action launches training implicitly")
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("prepare", "train", "evaluate", "audit", "queue"):
        item = sub.add_parser(action)
        item.add_argument("--config", required=True)
        item.add_argument("--output-dir", required=True)
        if action == "train":
            item.add_argument("--experiment", choices=("E0", "E1", "E2", "E3"), required=True)
            item.add_argument("--seed", type=int, choices=(1, 2, 3), required=True)
            item.add_argument("--run-kind", choices=("smoke", "formal"), required=True)
            item.add_argument("--e0-checkpoint")
            item.add_argument("--resume")
        elif action == "evaluate":
            item.add_argument("--checkpoint", required=True)
            item.add_argument("--split", choices=("val", "test"), required=True)
            item.add_argument("--device", default="cuda:0")
        elif action == "audit":
            item.add_argument("--suite", choices=("cpu", "backbone", "gpu-equivalence", "reference-val", "parity"), required=True)
            item.add_argument("--device", default="cpu")
            item.add_argument("--checkpoint")
            item.add_argument("--other-checkpoint")
        elif action == "queue":
            item.add_argument("--execute", action="store_true")
            item.add_argument("--review-approval", help="Optional approved=true JSON; no code hash required")
            item.add_argument("--resume", action="store_true", help="Resume this queue; skip completed runs")
    args = parser.parse_args()
    if not Path(args.output_dir).is_absolute():
        parser.error("--output-dir must be absolute")
    cfg = load_config(args.config)
    bootstrap(cfg["nltk_data"])
    command = shlex.join([sys.executable] + sys.argv)
    if args.action == "prepare":
        from view4.prepare import prepare
        prepare(cfg, args.output_dir, command)
    elif args.action == "train":
        from view4.train import train
        train(cfg, args.experiment, args.seed, args.run_kind, args.output_dir, command,
              e0_checkpoint=args.e0_checkpoint, resume=args.resume)
    elif args.action == "audit":
        from view4 import audit
        if args.suite == "cpu":
            audit.cpu_audit(cfg, args.output_dir, command)
        elif args.suite == "backbone":
            audit.backbone_audit(cfg, args.output_dir, command, args.device)
        elif args.suite == "gpu-equivalence":
            audit.gpu_equivalence(cfg, args.output_dir, command, args.device)
        elif args.suite == "reference-val":
            if not args.checkpoint:
                parser.error("reference-val requires --checkpoint")
            audit.reference_validation(cfg, args.checkpoint, args.output_dir, args.device)
        else:
            if not args.checkpoint or not args.other_checkpoint:
                parser.error("parity requires two checkpoints")
            audit.compare_epoch1(args.checkpoint, args.other_checkpoint, args.output_dir)
    elif args.action == "queue":
        from view4.queue import run_queue
        run_queue(cfg, Path(args.config).resolve(), args.output_dir, args.execute, args.review_approval, args.resume)
    elif args.action == "evaluate":
        from view4.standalone import standalone_evaluate
        standalone_evaluate(cfg, args.checkpoint, args.split, args.output_dir, args.device, command)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Build deterministic phrase spans for TAG-PEDES train/test captions."""

from __future__ import annotations

import argparse
import json
import os.path as op
import sys
from pathlib import Path

PROJECT_ROOT = op.abspath(op.join(op.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.phrase_route_io import normalized_caption_sha1, write_jsonl
from datasets.tagpedes import TAGPEDES
from utils.simple_tokenizer import SimpleTokenizer
from tools.mllm.phrase_extraction import build_phrase_entries


def parse_args():
    parser = argparse.ArgumentParser(description="Build TAG-PEDES phrase spans")
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument("--text-length", type=int, default=77)
    parser.add_argument(
        "--splits", nargs="+", default=["train", "test"], choices=["train", "test"]
    )
    return parser.parse_args()


def load_nlp(model_name: str):
    try:
        import spacy
    except ImportError as exc:
        raise RuntimeError(
            "spaCy is required. Install it in the offline MLLM environment."
        ) from exc
    try:
        return spacy.load(model_name)
    except OSError as exc:
        raise RuntimeError(
            "spaCy model {!r} is unavailable. Run: python -m spacy download {}".format(
                model_name, model_name
            )
        ) from exc


def main():
    cli = parse_args()
    output_dir = Path(cli.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = TAGPEDES(root=cli.root_dir, verbose=False)
    nlp = load_nlp(cli.spacy_model)
    tokenizer = SimpleTokenizer()
    summary = {}

    for split in cli.splits:
        rows = []
        category_counts = {}
        if split == "train":
            iterable = [
                {
                    "index": index,
                    "pid": int(pid),
                    "image_id": int(image_id),
                    "path": path,
                    "caption": caption,
                }
                for index, (pid, image_id, path, caption) in enumerate(dataset.train)
            ]
        else:
            iterable = []
            caption_index = 0
            for image_index, annotation in enumerate(dataset.test_annos):
                for caption in annotation["captions"]:
                    from datasets.tagpedes import pre_caption
                    iterable.append(
                        {
                            "index": caption_index,
                            "pid": int(annotation["id"]),
                            "image_id": image_index,
                            "path": op.join(dataset.img_dir, annotation["file_path"]),
                            "caption": pre_caption(caption),
                        }
                    )
                    caption_index += 1

        for record in iterable:
            phrases = build_phrase_entries(
                caption=record["caption"],
                nlp=nlp,
                tokenizer=tokenizer,
                text_length=cli.text_length,
                split=split,
                index=record["index"],
            )
            for phrase in phrases:
                category = phrase.get("category", "other")
                category_counts[category] = category_counts.get(category, 0) + 1
            row = {
                "experiment_version": "span-only",
                "route_kind": "span-only",
                "split": split,
                ("record_index" if split == "train" else "caption_index"): record[
                    "index"
                ],
                "pid": record["pid"],
                "image_id": record["image_id"],
                "image_path": record["path"],
                "caption": record["caption"],
                "caption_sha1": normalized_caption_sha1(record["caption"]),
                "phrases": phrases,
                "route_supervision_valid": False,
            }
            rows.append(row)

        target = output_dir / "phrase_spans_{}.jsonl".format(split)
        write_jsonl(str(target), rows)
        summary[split] = {
            "row_count": len(rows),
            "phrase_count": sum(len(row["phrases"]) for row in rows),
            "category_counts": category_counts,
            "output": str(target),
        }

    with (output_dir / "phrase_span_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

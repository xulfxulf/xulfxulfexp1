"""Pure JSONL and tensor utilities for HIRE-v2 phrase routing.

This module intentionally has no dependency on the repository dataset classes.
It is used by training/evaluation datasets, offline teacher tools, audits, and
unit tests.  One fixed-size phrase axis of ``text_length`` is used, so phrase
routing introduces no new maximum-phrase hyperparameter and never truncates a
valid phrase list before the CLIP text limit.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

import torch


SUPPORTED_VERSIONS = {"v16.6.0", "v16.7.0", "span-only"}
SUPPORTED_ROUTE_KINDS = {"propagation", "comparative", "span-only"}


def normalized_caption_sha1(caption: str) -> str:
    text = " ".join(str(caption).strip().split())
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def read_jsonl(path: str) -> List[dict]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError("JSONL file does not exist: {}".format(source))
    rows: List[dict] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "Invalid JSONL at {}:{}".format(source, line_number)
                ) from exc
            if not isinstance(value, dict):
                raise RuntimeError(
                    "JSONL row must be an object at {}:{}".format(
                        source, line_number
                    )
                )
            rows.append(value)
    return rows


def write_jsonl(path: str, rows: Iterable[Mapping]) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


@dataclass(frozen=True)
class PhraseRecordKey:
    split: str
    index: int


class PhraseRouteTable:
    """Validated phrase span/teacher records indexed by split-local index."""

    def __init__(
        self,
        path: str,
        split: str,
        expected_version: Optional[str] = None,
        expected_route_kind: Optional[str] = None,
    ):
        self.path = str(Path(path).expanduser().resolve())
        self.split = str(split)
        self.expected_version = expected_version
        self.expected_route_kind = expected_route_kind
        self.by_index: Dict[int, dict] = {}

        for row in read_jsonl(self.path):
            row_split = str(row.get("split", ""))
            if row_split != self.split:
                continue
            version = str(row.get("experiment_version", "span-only"))
            route_kind = str(row.get("route_kind", "span-only"))
            if version not in SUPPORTED_VERSIONS:
                raise RuntimeError(
                    "Unsupported phrase record version {!r} in {}".format(
                        version, self.path
                    )
                )
            if route_kind not in SUPPORTED_ROUTE_KINDS:
                raise RuntimeError(
                    "Unsupported route_kind {!r} in {}".format(
                        route_kind, self.path
                    )
                )
            if expected_version and version != expected_version:
                raise RuntimeError(
                    "Expected phrase version {}, got {} in {}".format(
                        expected_version, version, self.path
                    )
                )
            if expected_route_kind and route_kind != expected_route_kind:
                raise RuntimeError(
                    "Expected route_kind {}, got {} in {}".format(
                        expected_route_kind, route_kind, self.path
                    )
                )

            index_field = "record_index" if self.split == "train" else "caption_index"
            if index_field not in row:
                raise RuntimeError(
                    "Missing {} in phrase row from {}".format(
                        index_field, self.path
                    )
                )
            index = int(row[index_field])
            if index in self.by_index:
                raise RuntimeError(
                    "Duplicate phrase row for split={} index={} in {}".format(
                        self.split, index, self.path
                    )
                )
            self._validate_row(row)
            self.by_index[index] = row

        if not self.by_index:
            raise RuntimeError(
                "No rows for split={!r} in {}".format(self.split, self.path)
            )

    @staticmethod
    def _validate_row(row: dict) -> None:
        required = {"split", "caption", "caption_sha1", "phrases"}
        missing = required - set(row)
        if missing:
            raise RuntimeError(
                "Phrase row is missing fields: {}".format(sorted(missing))
            )
        if normalized_caption_sha1(row["caption"]) != str(row["caption_sha1"]):
            raise RuntimeError("caption_sha1 does not match the stored caption")
        if not isinstance(row["phrases"], list):
            raise RuntimeError("phrases must be a list")
        seen_phrase_ids = set()
        for phrase in row["phrases"]:
            if not isinstance(phrase, dict):
                raise RuntimeError("phrase entry must be an object")
            for field in ("phrase_id", "text", "token_positions"):
                if field not in phrase:
                    raise RuntimeError("phrase is missing field {!r}".format(field))
            phrase_id = str(phrase["phrase_id"])
            if phrase_id in seen_phrase_ids:
                raise RuntimeError("duplicate phrase_id {}".format(phrase_id))
            seen_phrase_ids.add(phrase_id)
            positions = [int(value) for value in phrase["token_positions"]]
            if not positions:
                raise RuntimeError("phrase {} has no token positions".format(phrase_id))
            if min(positions) < 1:
                raise RuntimeError("phrase token positions must exclude SOS")
            if len(positions) != len(set(positions)):
                raise RuntimeError("phrase token positions contain duplicates")

        route_valid = bool(row.get("route_supervision_valid", False))
        if route_valid:
            weights = [
                float(phrase.get("target_weight", 0.0))
                for phrase in row["phrases"]
                if bool(phrase.get("route_candidate", True))
            ]
            if len([value for value in weights if value > 0.0]) < 2:
                raise RuntimeError(
                    "route_supervision_valid requires at least two positive phrases"
                )
            if abs(sum(weights) - 1.0) > 1e-4:
                raise RuntimeError(
                    "valid route target must sum to one, got {}".format(sum(weights))
                )

    def validate_caption(
        self,
        index: int,
        caption: str,
        image_id: Optional[int] = None,
    ) -> dict:
        if int(index) not in self.by_index:
            raise KeyError(
                "Missing phrase record split={} index={} in {}".format(
                    self.split, index, self.path
                )
            )
        row = self.by_index[int(index)]
        expected_sha1 = normalized_caption_sha1(caption)
        if expected_sha1 != str(row["caption_sha1"]):
            raise RuntimeError(
                "Caption mismatch for split={} index={}: expected {}, got {}".format(
                    self.split, index, row["caption_sha1"], expected_sha1
                )
            )
        if image_id is not None and "image_id" in row:
            if int(row["image_id"]) != int(image_id):
                raise RuntimeError(
                    "image_id mismatch for split={} index={}".format(
                        self.split, index
                    )
                )
        return row


def phrase_record_to_tensors(
    row: Mapping,
    text_length: int,
) -> Dict[str, torch.Tensor]:
    """Convert one phrase JSON record to fixed-shape model tensors.

    The phrase axis is exactly ``text_length``.  This derives capacity from the
    existing CLIP context length and adds no new method hyperparameter.
    """
    length = int(text_length)
    if length < 3:
        raise ValueError("text_length must be at least three")

    token_mask = torch.zeros(length, length, dtype=torch.bool)
    phrase_valid = torch.zeros(length, dtype=torch.bool)
    route_target = torch.zeros(length, dtype=torch.float32)
    raw_score = torch.zeros(length, dtype=torch.float32)
    phrase_category = torch.full((length,), -1, dtype=torch.long)

    category_to_id = {
        "upper": 0,
        "lower": 1,
        "shoes": 2,
        "bag": 3,
        "hat": 4,
        "hair": 5,
        "pose": 6,
        "other": 7,
        "fallback": 8,
    }

    phrases = list(row.get("phrases", []))
    if len(phrases) > length:
        raise RuntimeError(
            "Phrase count {} exceeds derived capacity {}".format(
                len(phrases), length
            )
        )

    for phrase_index, phrase in enumerate(phrases):
        positions = sorted({int(value) for value in phrase["token_positions"]})
        positions = [value for value in positions if 0 < value < length - 1]
        if not positions:
            continue
        token_mask[phrase_index, positions] = True
        phrase_valid[phrase_index] = True
        route_target[phrase_index] = float(phrase.get("target_weight", 0.0))
        raw_score[phrase_index] = float(
            phrase.get(
                "comparative_raw_score",
                phrase.get("propagation_raw_score", 0.0),
            )
        )
        phrase_category[phrase_index] = int(
            category_to_id.get(str(phrase.get("category", "other")), 7)
        )

    route_supervision = bool(row.get("route_supervision_valid", False))
    if route_supervision:
        valid_target = route_target[phrase_valid]
        if int((valid_target > 0).sum()) < 2:
            raise RuntimeError(
                "Route-supervised row has fewer than two positive target phrases"
            )
        total = float(valid_target.sum())
        if abs(total - 1.0) > 1e-4:
            raise RuntimeError(
                "Route-supervised target does not sum to one: {}".format(total)
            )

    return {
        "phrase_token_mask": token_mask,
        "phrase_valid_mask": phrase_valid,
        "phrase_route_target": route_target,
        "phrase_teacher_raw_score": raw_score,
        "phrase_category": phrase_category,
        "phrase_route_supervision": torch.tensor(
            route_supervision, dtype=torch.bool
        ),
        "phrase_count": torch.tensor(int(phrase_valid.sum()), dtype=torch.long),
    }

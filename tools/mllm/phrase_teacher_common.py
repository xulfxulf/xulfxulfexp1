"""Shared case, prompt, parsing, and target logic for v16.6.0/v16.7.0."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


LABELS = ("support", "contradiction", "unknown")
DISCRIMINATION_FACTOR = {
    "support": 0.0,
    "unknown": 0.5,
    "contradiction": 1.0,
}


def stable_sha1(*parts: object) -> str:
    text = "\u241f".join(str(part) for part in parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def read_jsonl(path: str) -> List[dict]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    rows = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "Invalid JSONL {}:{}".format(source, line_number)
                ) from exc
            if not isinstance(value, dict):
                raise RuntimeError("JSONL rows must be objects")
            rows.append(value)
    return rows


def write_jsonl(path: str, rows: Iterable[Mapping]) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def extract_json_object(text: str) -> dict:
    value = str(text).strip()
    try:
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", value, flags=re.S)
    if fenced:
        parsed = json.loads(fenced.group(1))
        if isinstance(parsed, dict):
            return parsed
    start = value.find("{")
    end = value.rfind("}")
    if start >= 0 and end > start:
        parsed = json.loads(value[start : end + 1])
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("No JSON object found in teacher output")


def normalize_label(value: object) -> str:
    label = str(value).strip().lower()
    aliases = {
        "supported": "support",
        "yes": "support",
        "true": "support",
        "conflict": "contradiction",
        "contradict": "contradiction",
        "no": "contradiction",
        "false": "contradiction",
        "uncertain": "unknown",
        "not_visible": "unknown",
        "not visible": "unknown",
        "unobservable": "unknown",
    }
    label = aliases.get(label, label)
    if label not in LABELS:
        raise ValueError("Unsupported teacher label: {!r}".format(value))
    return label


def ordered_case_images(case: Mapping, order: str) -> List[dict]:
    if order not in {"forward", "reverse"}:
        raise ValueError("order must be forward or reverse")
    supports = list(case.get("supports", []))
    if order == "reverse":
        supports = list(reversed(supports))
    images = [
        {
            "role": "anchor",
            "image_id": int(case["image_id"]),
            "path": case["anchor_image_path"],
        }
    ]
    for support in supports:
        images.append(
            {
                "role": "support_{}".format(int(support["image_id"])),
                "image_id": int(support["image_id"]),
                "path": support["path"],
            }
        )
    if case.get("hard_negative"):
        negative = case["hard_negative"]
        images.append(
            {
                "role": "hard_negative",
                "image_id": int(negative["image_id"]),
                "path": negative["path"],
            }
        )
    return images


SYSTEM_PROMPT = """You are a strict visual-relation annotator for person retrieval.
You must judge only observable evidence. Never infer hidden details from identity,
common sense, dataset names, camera type, or another image. Use exactly three
labels: support, contradiction, unknown. Use unknown when the detail is not
visible, too small, blurred, occluded, ambiguous, or only partially inferable.
Return JSON only and follow the requested schema exactly."""


def build_teacher_prompt(case: Mapping, order: str) -> str:
    images = ordered_case_images(case, order)
    image_lines = []
    for position, image in enumerate(images, start=1):
        image_lines.append(
            "image_{} role={} image_id={}".format(
                position, image["role"], image["image_id"]
            )
        )
    phrase_lines = []
    for phrase in case.get("phrases", []):
        phrase_lines.append(
            '- phrase_id="{}" text="{}"'.format(
                phrase["phrase_id"],
                str(phrase["text"]).replace('"', "'"),
            )
        )

    hard_negative_instruction = ""
    if case.get("hard_negative"):
        hard_negative_instruction = (
            "\nFor the image whose role is hard_negative, judge the same phrase "
            "against that image independently."
        )

    return """The anchor caption is:
{caption}

The second human caption for the same anchor image is:
{sibling}

Images are supplied in this exact order:
{images}

Judge these visual phrases:
{phrases}

For every phrase, output:
1. anchor: relation between the phrase and the anchor image.
2. sibling: relation between the phrase and the second caption. Use support when
   the second caption explicitly agrees, contradiction only for an explicit
   incompatible statement, and unknown for omission or ambiguity.
3. support_by_image_id: one label for every supplied support image, keyed by its
   numeric image_id.{hard_negative}

Do not use person identity, PID, camera labels, or image order as evidence.
Do not convert omission into contradiction.

Return exactly this JSON shape:
{{
  "case_id": "{case_id}",
  "phrases": [
    {{
      "phrase_id": "...",
      "anchor": "support|contradiction|unknown",
      "sibling": "support|contradiction|unknown",
      "support_by_image_id": {{"123": "support|contradiction|unknown"}}{negative_schema}
    }}
  ]
}}
""".format(
        caption=case["caption"],
        sibling=case.get("sibling_caption", ""),
        images="\n".join(image_lines),
        phrases="\n".join(phrase_lines),
        hard_negative=hard_negative_instruction,
        case_id=case["case_id"],
        negative_schema=(
            ',\n      "hard_negative": "support|contradiction|unknown"'
            if case.get("hard_negative")
            else ""
        ),
    )


def validate_teacher_payload(case: Mapping, payload: Mapping) -> Dict[str, dict]:
    if str(payload.get("case_id", "")) != str(case["case_id"]):
        raise ValueError("Teacher case_id does not match input case")
    phrase_by_id = {
        str(item["phrase_id"]): item for item in payload.get("phrases", [])
    }
    expected_ids = [str(item["phrase_id"]) for item in case.get("phrases", [])]
    if set(phrase_by_id) != set(expected_ids):
        raise ValueError("Teacher phrase IDs do not match case phrase IDs")

    support_ids = {str(int(item["image_id"])) for item in case.get("supports", [])}
    normalized: Dict[str, dict] = {}
    for phrase_id in expected_ids:
        item = phrase_by_id[phrase_id]
        support_map = item.get("support_by_image_id", {})
        if set(str(key) for key in support_map) != support_ids:
            raise ValueError(
                "Teacher support image IDs do not match case for {}".format(phrase_id)
            )
        result = {
            "anchor": normalize_label(item.get("anchor")),
            "sibling": normalize_label(item.get("sibling")),
            "support_by_image_id": {
                str(key): normalize_label(value)
                for key, value in support_map.items()
            },
        }
        if case.get("hard_negative"):
            result["hard_negative"] = normalize_label(item.get("hard_negative"))
        normalized[phrase_id] = result
    return normalized


def strict_label(values: Sequence[Optional[str]]) -> Tuple[str, bool]:
    if len(values) != 4:
        raise ValueError("Strict merge requires exactly four judgments")
    if any(value is None for value in values):
        return "unknown", False
    normalized = [normalize_label(value) for value in values]
    agreement = len(set(normalized)) == 1
    return (normalized[0] if agreement else "unknown"), agreement


def propagation_raw_score(
    anchor_label: str,
    sibling_label: str,
    support_labels: Sequence[str],
) -> Dict[str, float]:
    support_labels = [normalize_label(value) for value in support_labels]
    if not support_labels:
        raise ValueError("At least one support label is required")
    anchor_reliable = (
        normalize_label(anchor_label) == "support"
        and normalize_label(sibling_label) != "contradiction"
    )
    count = float(len(support_labels))
    n_support = float(sum(value == "support" for value in support_labels))
    n_contradiction = float(
        sum(value == "contradiction" for value in support_labels)
    )
    n_unknown = float(sum(value == "unknown" for value in support_labels))
    raw = (
        (1.0 if anchor_reliable else 0.0)
        * (n_support / count)
        * (1.0 - n_contradiction / count)
    )
    return {
        "anchor_reliable": float(anchor_reliable),
        "n_support": n_support,
        "n_contradiction": n_contradiction,
        "n_unknown": n_unknown,
        "support_count": count,
        "propagation_raw_score": raw,
    }


def comparative_raw_score(
    propagation_score: float,
    hard_negative_label: str,
) -> Dict[str, float]:
    label = normalize_label(hard_negative_label)
    factor = float(DISCRIMINATION_FACTOR[label])
    return {
        "hard_negative_label": label,
        "discrimination_factor": factor,
        "comparative_raw_score": float(propagation_score) * factor,
    }


def normalize_record_phrase_targets(
    phrases: List[dict],
    raw_field: str,
) -> Tuple[List[dict], bool]:
    positive_count = sum(float(item.get(raw_field, 0.0)) > 0.0 for item in phrases)
    total = sum(max(0.0, float(item.get(raw_field, 0.0))) for item in phrases)
    valid = positive_count >= 2 and total > 0.0
    output = []
    for phrase in phrases:
        item = dict(phrase)
        raw = max(0.0, float(item.get(raw_field, 0.0)))
        item["target_weight"] = raw / total if valid else 0.0
        item["route_candidate"] = not bool(item.get("fallback", False))
        output.append(item)
    return output, valid

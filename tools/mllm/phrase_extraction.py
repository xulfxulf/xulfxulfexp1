"""Deterministic semantic-phrase extraction and CLIP token-span mapping."""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence, Tuple

from utils.simple_tokenizer import SimpleTokenizer
from .phrase_teacher_common import stable_sha1


CATEGORY_TERMS = {
    "upper": {
        "shirt", "t-shirt", "tee", "top", "jacket", "coat", "hoodie",
        "sweater", "blouse", "vest", "uniform", "jersey", "dress",
    },
    "lower": {
        "pants", "trousers", "jeans", "shorts", "skirt", "leggings",
        "slacks", "bottoms",
    },
    "shoes": {
        "shoe", "shoes", "sneaker", "sneakers", "boot", "boots",
        "sandals", "footwear", "heels",
    },
    "bag": {
        "bag", "backpack", "handbag", "purse", "suitcase", "luggage",
        "cart", "briefcase", "satchel",
    },
    "hat": {"hat", "cap", "helmet", "beanie", "hood"},
    "hair": {
        "hair", "ponytail", "bald", "braid", "braids", "bun", "beard",
        "mustache", "moustache",
    },
    "pose": {
        "walking", "standing", "sitting", "running", "riding", "pushing",
        "pulling", "dragging", "carrying", "holding", "looking", "facing",
        "turning", "bending", "crossing", "moving", "wearing",
    },
}

COLOR_TERMS = {
    "black", "white", "gray", "grey", "red", "blue", "green", "yellow",
    "brown", "pink", "purple", "orange", "beige", "tan", "navy", "dark",
    "light", "multicolored", "colourful", "colorful",
}

GENERIC_SUBJECTS = {
    "person", "man", "woman", "male", "female", "pedestrian", "individual",
    "people", "someone", "guy", "lady",
}

VISUAL_MODIFIERS = COLOR_TERMS | {
    "long", "short", "sleeved", "sleeveless", "striped", "plaid", "printed",
    "patterned", "large", "small", "tall", "thin", "heavy", "young", "old",
    "no", "not", "without", "with", "wearing", "carrying", "holding",
}


def _normalize_phrase_text(text: str) -> str:
    return " ".join(str(text).strip(" ,.;:").split())


def _category_for_tokens(tokens: Sequence[str]) -> str:
    token_set = {str(token).lower() for token in tokens}
    joined = " ".join(tokens).lower()
    for category, terms in CATEGORY_TERMS.items():
        if any(term in token_set or term in joined for term in terms):
            return category
    return "other"


def _contains_visual_content(tokens: Sequence[str]) -> bool:
    lowered = [str(token).lower() for token in tokens]
    token_set = set(lowered)
    if any(token in COLOR_TERMS for token in token_set):
        return True
    if any(
        term in token_set or term in " ".join(lowered)
        for terms in CATEGORY_TERMS.values()
        for term in terms
    ):
        return True
    if any(token in VISUAL_MODIFIERS for token in token_set):
        return True
    return False


def _trim_span(doc, start: int, end: int) -> Tuple[int, int]:
    while start < end and doc[start].pos_ in {"DET", "PRON"}:
        start += 1
    while end > start and doc[end - 1].is_punct:
        end -= 1
    return start, end


def extract_visual_phrases(caption: str, nlp) -> List[dict]:
    """Extract non-overlapping visual noun/action phrases with fixed rules."""
    doc = nlp(str(caption))
    candidates: List[dict] = []

    for chunk in doc.noun_chunks:
        start, end = _trim_span(doc, chunk.start, chunk.end)
        if start >= end:
            continue
        words = [doc[index].text.lower() for index in range(start, end)]
        lemmas = [doc[index].lemma_.lower() for index in range(start, end)]
        if set(lemmas).issubset(GENERIC_SUBJECTS) and not _contains_visual_content(words):
            continue
        if not _contains_visual_content(words + lemmas):
            continue
        span = doc[start:end]
        text = _normalize_phrase_text(span.text)
        if not text:
            continue
        candidates.append(
            {
                "char_start": int(span.start_char),
                "char_end": int(span.end_char),
                "text": text,
                "category": _category_for_tokens(words + lemmas),
                "source": "noun_chunk",
            }
        )

    for token in doc:
        lemma = token.lemma_.lower()
        if lemma not in CATEGORY_TERMS["pose"] and token.text.lower() not in CATEGORY_TERMS["pose"]:
            continue
        indices = {token.i}
        for child in token.children:
            if child.dep_ in {
                "aux", "auxpass", "neg", "prt", "dobj", "obj", "prep", "advmod",
            }:
                indices.update(item.i for item in child.subtree)
        start = max(0, min(indices))
        end = min(len(doc), max(indices) + 1)
        start, end = _trim_span(doc, start, end)
        if start >= end:
            continue
        span = doc[start:end]
        text = _normalize_phrase_text(span.text)
        if text:
            candidates.append(
                {
                    "char_start": int(span.start_char),
                    "char_end": int(span.end_char),
                    "text": text,
                    "category": "pose",
                    "source": "action",
                }
            )

    for token in doc:
        lemma = token.lemma_.lower()
        if lemma in CATEGORY_TERMS["hair"] or token.text.lower() in CATEGORY_TERMS["hair"]:
            start = token.i
            for child in token.children:
                if child.dep_ in {"amod", "neg", "compound"}:
                    start = min(start, child.i)
            span = doc[start : token.i + 1]
            text = _normalize_phrase_text(span.text)
            if text:
                candidates.append(
                    {
                        "char_start": int(span.start_char),
                        "char_end": int(span.end_char),
                        "text": text,
                        "category": "hair",
                        "source": "hair_rule",
                    }
                )

    # Prefer longer complete spans, then earlier spans.  Remove overlaps so each
    # phrase owns one unambiguous token range.
    candidates.sort(
        key=lambda item: (
            -(int(item["char_end"]) - int(item["char_start"])),
            int(item["char_start"]),
            str(item["text"]),
        )
    )
    selected: List[dict] = []
    occupied: List[Tuple[int, int]] = []
    for candidate in candidates:
        start = int(candidate["char_start"])
        end = int(candidate["char_end"])
        if any(not (end <= left or start >= right) for left, right in occupied):
            continue
        selected.append(candidate)
        occupied.append((start, end))

    selected.sort(key=lambda item: (item["char_start"], item["char_end"]))
    if not selected:
        stripped = str(caption).strip()
        if stripped:
            leading = len(str(caption)) - len(str(caption).lstrip())
            selected = [
                {
                    "char_start": leading,
                    "char_end": leading + len(stripped),
                    "text": stripped,
                    "category": "fallback",
                    "source": "fallback",
                    "fallback": True,
                }
            ]
    return selected


def phrase_token_positions(
    caption: str,
    char_start: int,
    char_end: int,
    tokenizer: SimpleTokenizer,
    text_length: int = 77,
) -> List[int]:
    """Map a word-boundary character span to CLIP token positions.

    The phrase extractor emits word-boundary spans.  Prefix tokenization is
    therefore stable for the OpenAI CLIP BPE used by the repository.
    """
    caption = str(caption)
    start = int(char_start)
    end = int(char_end)
    if not (0 <= start < end <= len(caption)):
        raise ValueError("Invalid phrase character span")
    prefix_count = len(tokenizer.encode(caption[:start]))
    through_count = len(tokenizer.encode(caption[:end]))
    token_start = 1 + prefix_count
    token_end = 1 + through_count
    token_end = min(token_end, int(text_length) - 1)
    token_start = min(token_start, token_end)
    return list(range(token_start, token_end))


def build_phrase_entries(
    caption: str,
    nlp,
    tokenizer: SimpleTokenizer,
    text_length: int = 77,
    split: str = "train",
    index: int = 0,
) -> List[dict]:
    phrases = extract_visual_phrases(caption, nlp)
    output = []
    for phrase_index, phrase in enumerate(phrases):
        positions = phrase_token_positions(
            caption=caption,
            char_start=phrase["char_start"],
            char_end=phrase["char_end"],
            tokenizer=tokenizer,
            text_length=text_length,
        )
        if not positions:
            continue
        item = dict(phrase)
        item["phrase_id"] = stable_sha1(
            split,
            index,
            phrase_index,
            item["char_start"],
            item["char_end"],
            item["text"],
        )[:20]
        item["token_positions"] = positions
        item.setdefault("fallback", False)
        output.append(item)
    return output

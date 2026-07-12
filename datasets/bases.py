from typing import List
from torch.utils.data import Dataset
import os.path as osp
import logging
import torch
from utils.iotools import read_image
from utils.simple_tokenizer import SimpleTokenizer
from prettytable import PrettyTable
import random
import regex as re
import copy
from collections import defaultdict
import csv


class BaseDataset(object):
    """
    Base class of text to image reid dataset
    """
    logger = logging.getLogger("IRRA.dataset")

    def show_dataset_info(self):
        num_train_pids, num_train_imgs, num_train_captions = len(
            self.train_id_container), len(self.train_annos), len(self.train)
        num_test_pids, num_test_imgs, num_test_captions = len(
            self.test_id_container), len(self.test_annos), len(
                self.test['captions'])
        num_val_pids, num_val_imgs, num_val_captions = len(
            self.val_id_container), len(self.val_annos), len(
                self.val['captions'])

        # TODO use prettytable print comand line table

        self.logger.info(f"{self.__class__.__name__} Dataset statistics:")
        table = PrettyTable(['subset', 'ids', 'images', 'captions'])
        table.add_row(
            ['train', num_train_pids, num_train_imgs, num_train_captions])
        table.add_row(
            ['test', num_test_pids, num_test_imgs, num_test_captions])
        table.add_row(['val', num_val_pids, num_val_imgs, num_val_captions])
        self.logger.info('\n' + str(table))


SLOT_TERMS = {
    "upper": {
        "shirt", "t-shirt", "tee", "top", "jacket", "coat", "hoodie",
        "sweater", "blouse", "vest", "uniform",
    },
    "lower": {"pants", "trousers", "jeans", "shorts", "skirt", "leggings"},
    "shoes": {"shoe", "shoes", "sneaker", "sneakers", "boot", "boots", "sandals", "footwear"},
    "bag": {"bag", "backpack", "handbag", "purse", "suitcase", "luggage", "cart"},
    "hat": {"hat", "cap", "helmet", "beanie"},
    "hair": {"hair", "ponytail", "bald", "braid", "bun"},
    "pose": {"walking", "standing", "sitting", "running", "riding", "pushing", "pulling", "dragging", "carrying", "holding"},
}


RELIABLE_CONSISTENCY_TYPES = {"consistent", "complementary", "omission"}
STRICT_SLOT_NAMES = tuple(SLOT_TERMS.keys())


def _normalize_caption_tokens(caption):
    text = str(caption).lower().replace("grey", "gray")
    text = re.sub(r"[^a-z0-9\- ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text, set(text.split())


def _caption_slots_strict(caption):
    """Return v16 fast3 slots using whole-token and whole-phrase matching only."""
    text, tokens = _normalize_caption_tokens(caption)
    padded_text = f" {text} "
    slots = []
    for slot, terms in SLOT_TERMS.items():
        for term in terms:
            if " " in term:
                matched = f" {term} " in padded_text
            else:
                matched = term in tokens
            if matched:
                slots.append(slot)
                break
    return slots


def _caption_slots(caption):
    text = caption.lower().replace("grey", "gray")
    tokens = set(re.sub(r"[^a-z0-9\- ]+", " ", text).split())
    slots = []
    for slot, terms in SLOT_TERMS.items():
        if any(term in text or term in tokens for term in terms):
            slots.append(slot)
    return slots


def _load_image_slot_reliability(path, rule="scheme2", required_image_ids=None):
    if not path:
        return {}
    if not osp.exists(path):
        raise RuntimeError(f"support consistency csv is not available: {path}")

    reliable_types_by_rule = {
        "scheme2": {"consistent", "complementary", "omission"},
        "hard_only": {"consistent", "complementary", "omission", "soft_mismatch"},
    }
    if rule not in reliable_types_by_rule:
        raise ValueError(f"Unsupported support reliability rule: {rule}")

    # Keep scheme-2/legacy loading byte-for-byte compatible in behavior. The
    # strict schema and completeness checks are reserved for v16 fast3.
    if rule == "scheme2" and required_image_ids is None:
        reliability = defaultdict(dict)
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    image_id = int(row["image_id"])
                except (KeyError, TypeError, ValueError):
                    continue
                slot = row.get("slot", "")
                consistency_type = row.get("consistency_type", "")
                if slot:
                    reliability[image_id][slot] = (
                        1.0 if consistency_type in RELIABLE_CONSISTENCY_TYPES else 0.0
                    )
        return reliability

    reliability = defaultdict(dict)
    seen = set()
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_fields = {"image_id", "slot", "consistency_type"}
        missing_fields = required_fields - set(reader.fieldnames or [])
        if missing_fields:
            raise RuntimeError(
                f"support consistency csv is missing fields: {sorted(missing_fields)}"
            )
        for row in reader:
            try:
                image_id = int(row["image_id"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(f"Invalid image_id in support consistency csv: {row}") from exc
            slot = str(row.get("slot", "")).strip()
            consistency_type = str(row.get("consistency_type", "")).strip()
            if slot not in STRICT_SLOT_NAMES:
                raise RuntimeError(f"Unsupported reliability slot {slot!r} for image_id={image_id}")
            if consistency_type not in {
                "consistent", "complementary", "omission", "soft_mismatch",
                "hard_contradiction", "unparsed",
            }:
                raise RuntimeError(
                    f"Unsupported consistency_type {consistency_type!r} for image_id={image_id}, slot={slot}"
                )
            key = (image_id, slot)
            if key in seen:
                raise RuntimeError(f"Duplicate image-slot reliability row: image_id={image_id}, slot={slot}")
            seen.add(key)
            value = 1.0 if consistency_type in reliable_types_by_rule[rule] else 0.0
            if row.get("reliability", "") not in ("", None):
                try:
                    supplied_value = float(row["reliability"])
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"Invalid reliability value for image_id={image_id}, slot={slot}"
                    ) from exc
                if abs(supplied_value - value) > 1e-6:
                    raise RuntimeError(
                        "Reliability mapping mismatch for "
                        f"image_id={image_id}, slot={slot}: supplied={supplied_value}, expected={value}"
                    )
            reliability[image_id][slot] = value

    if required_image_ids is not None:
        missing = [
            (int(image_id), slot)
            for image_id in sorted({int(value) for value in required_image_ids})
            for slot in STRICT_SLOT_NAMES
            if slot not in reliability.get(int(image_id), {})
        ]
        if missing:
            missing_text = ", ".join(f"{image_id}:{slot}" for image_id, slot in missing)
            raise RuntimeError(
                "Missing image-slot reliability entries for fast3: " + missing_text
            )
    return reliability


def _parse_bool(value, field_name, row):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no", ""}:
        return False
    raise RuntimeError(f"Invalid boolean {field_name}={value!r} in row: {row}")


def _load_support_hard_contradictions(path):
    if not path:
        return {}
    if not osp.exists(path):
        raise RuntimeError(f"support relation csv is not available: {path}")

    relations = {}
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_fields = {
            "anchor_index",
            "support_image_id",
            "has_hard_contradiction",
            "contradict_slots",
        }
        missing_fields = required_fields - set(reader.fieldnames or [])
        if missing_fields:
            raise RuntimeError(
                f"support relation csv is missing fields: {sorted(missing_fields)}"
            )
        for row in reader:
            try:
                anchor_index = int(row["anchor_index"])
                support_image_id = int(row["support_image_id"])
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"Invalid support relation row: {row}") from exc
            has_conflict = _parse_bool(
                row.get("has_hard_contradiction"), "has_hard_contradiction", row
            )
            slots = tuple(
                slot.strip()
                for slot in str(row.get("contradict_slots", "")).split("|")
                if slot.strip()
            )
            if any(slot not in STRICT_SLOT_NAMES for slot in slots):
                raise RuntimeError(
                    f"Unsupported contradiction slot for anchor_index={anchor_index}: {slots}"
                )
            if has_conflict and not slots:
                raise RuntimeError(
                    f"Hard contradiction requires contradict_slots for anchor_index={anchor_index}"
                )
            if not has_conflict and slots:
                raise RuntimeError(
                    f"Non-conflict row cannot declare slots for anchor_index={anchor_index}"
                )
            key = (anchor_index, support_image_id)
            value = {
                "has_hard_contradiction": has_conflict,
                "contradict_slots": slots,
            }
            if key in relations and relations[key] != value:
                raise RuntimeError(f"Conflicting support relation rows for {key}")
            relations[key] = value
    return relations


def _load_hard_negative_pool(path, first_index_by_image_id, pid_by_image_id):
    if not path:
        return {}
    if not osp.exists(path):
        raise RuntimeError(f"hard negative csv is not available: {path}")

    pool = defaultdict(list)
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_fields = {
            "anchor_pid",
            "negative_pid",
            "negative_image_id",
            "rank",
            "trigger_caption_count",
            "trigger_image_count",
        }
        missing_fields = required_fields - set(reader.fieldnames or [])
        if missing_fields:
            raise RuntimeError(
                f"hard negative csv is missing fields: {sorted(missing_fields)}"
            )
        for row in reader:
            try:
                anchor_pid = int(row["anchor_pid"])
                negative_pid = int(row["negative_pid"])
                negative_image_id = int(row["negative_image_id"])
                rank = int(row["rank"])
                trigger_caption_count = int(row["trigger_caption_count"])
                trigger_image_count = int(row["trigger_image_count"])
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"Invalid hard negative row: {row}") from exc
            if anchor_pid == negative_pid:
                raise RuntimeError(f"Hard negative must have different PID: {row}")
            if trigger_caption_count < 3 or trigger_image_count < 2:
                raise RuntimeError(
                    "Hard negative candidate does not meet persistent identity-level criteria: "
                    f"{row}"
                )
            if negative_image_id not in first_index_by_image_id:
                raise RuntimeError(
                    f"Hard negative image_id is absent from the training set: {negative_image_id}"
                )
            if pid_by_image_id[negative_image_id] != negative_pid:
                raise RuntimeError(
                    "Hard negative PID/image mismatch: "
                    f"image_id={negative_image_id}, expected={pid_by_image_id[negative_image_id]}, "
                    f"csv={negative_pid}"
                )
            pool[anchor_pid].append({
                "negative_pid": negative_pid,
                "negative_image_id": negative_image_id,
                "rank": rank,
            })

    for anchor_pid, candidates in pool.items():
        deduplicated = {}
        for candidate in candidates:
            key = (candidate["negative_pid"], candidate["negative_image_id"])
            previous = deduplicated.get(key)
            if previous is None or candidate["rank"] < previous["rank"]:
                deduplicated[key] = candidate
        pool[anchor_pid] = sorted(
            deduplicated.values(),
            key=lambda item: (item["rank"], item["negative_image_id"]),
        )
    return dict(pool)


def _select_balanced_supports(cross_view_items, same_view_items, support_size, offset_seed):
    """Select unique support images in deterministic cross/same-view alternation."""
    def rotate(items):
        if not items:
            return []
        offset = offset_seed % len(items)
        return list(items[offset:] + items[:offset])

    cross = rotate(cross_view_items)
    same = rotate(same_view_items)
    selected = []
    cross_index = 0
    same_index = 0
    prefer_cross = True
    while len(selected) < support_size and (
        cross_index < len(cross) or same_index < len(same)
    ):
        if prefer_cross and cross_index < len(cross):
            selected.append(cross[cross_index])
            cross_index += 1
        elif not prefer_cross and same_index < len(same):
            selected.append(same[same_index])
            same_index += 1
        elif cross_index < len(cross):
            selected.append(cross[cross_index])
            cross_index += 1
        else:
            selected.append(same[same_index])
            same_index += 1
        prefer_cross = not prefer_cross
    return selected


def tokenize(caption: str, tokenizer, text_length=77, truncate=True) -> torch.LongTensor:
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    tokens = [sot_token] + tokenizer.encode(caption) + [eot_token]

    result = torch.zeros(text_length, dtype=torch.long)
    if len(tokens) > text_length:
        if truncate:
            tokens = tokens[:text_length]
            tokens[-1] = eot_token
        else:
            raise RuntimeError(
                f"Input {caption} is too long for context length {text_length}"
            )
    result[:len(tokens)] = torch.tensor(tokens)
    return result


class ImageTextDataset(Dataset):
    def __init__(self,
                 dataset,
                 transform=None,
                 text_length: int = 77,
                 truncate: bool = True,
                 support_size: int = 0,
                 support_image_views=None,
                 support_consistency_csv: str = "",
                 support_selection_policy: str = "cross_first",
                 support_reliability_rule: str = "scheme2",
                 support_relation_csv: str = "",
                 hard_negative_csv: str = "",
                 hard_negative_size: int = 0,
                 support_image_only: bool = False):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.support_size = max(0, int(support_size))
        self.support_image_views = support_image_views
        self.support_consistency_csv = support_consistency_csv
        self.support_selection_policy = support_selection_policy
        self.support_reliability_rule = support_reliability_rule
        self.support_relation_csv = support_relation_csv
        self.hard_negative_csv = hard_negative_csv
        self.hard_negative_size = max(0, int(hard_negative_size))
        self.support_image_only = bool(support_image_only)
        if self.support_selection_policy not in {"cross_first", "balanced"}:
            raise ValueError(
                f"Unsupported support selection policy: {self.support_selection_policy}"
            )
        if self.hard_negative_size > 1:
            raise ValueError("v16 fast3 supports at most one hard negative per anchor")

        self.tokenizer = SimpleTokenizer()
        self.first_index_by_image_id, self.pid_by_image_id = self._build_image_index()
        required_image_ids = (
            self.first_index_by_image_id.keys() if self.support_image_only else None
        )
        self.support_reliability_by_image = _load_image_slot_reliability(
            support_consistency_csv,
            rule=self.support_reliability_rule,
            required_image_ids=required_image_ids,
        )
        self.support_hard_contradictions = _load_support_hard_contradictions(
            support_relation_csv
        )
        self.hard_negative_pool = _load_hard_negative_pool(
            hard_negative_csv,
            self.first_index_by_image_id,
            self.pid_by_image_id,
        )
        self._validate_fast3_relations()
        self.support_indices = self._build_support_indices() if self.support_size > 0 else None

    def _build_image_index(self):
        first_index_by_image_id = {}
        pid_by_image_id = {}
        for index, (pid, image_id, _img_path, _caption) in enumerate(self.dataset):
            image_id = int(image_id)
            pid = int(pid)
            first_index_by_image_id.setdefault(image_id, index)
            previous_pid = pid_by_image_id.setdefault(image_id, pid)
            if previous_pid != pid:
                raise RuntimeError(
                    f"Training image_id={image_id} appears under multiple PIDs: "
                    f"{previous_pid} and {pid}"
                )
        return first_index_by_image_id, pid_by_image_id

    def _validate_fast3_relations(self):
        if not self.support_image_only:
            return
        for (anchor_index, support_image_id), relation in self.support_hard_contradictions.items():
            if anchor_index < 0 or anchor_index >= len(self.dataset):
                raise RuntimeError(
                    f"support relation anchor_index is out of range: {anchor_index}"
                )
            if support_image_id not in self.first_index_by_image_id:
                raise RuntimeError(
                    f"support relation image_id is absent from training data: {support_image_id}"
                )
            anchor_pid, anchor_image_id, _anchor_path, _caption = self.dataset[anchor_index]
            if int(anchor_image_id) == int(support_image_id):
                raise RuntimeError(
                    f"support relation reuses anchor image: anchor_index={anchor_index}"
                )
            if int(anchor_pid) != self.pid_by_image_id[int(support_image_id)]:
                raise RuntimeError(
                    f"support relation crosses PID boundary: anchor_index={anchor_index}, "
                    f"support_image_id={support_image_id}"
                )
            if not isinstance(relation["has_hard_contradiction"], bool):
                raise RuntimeError("support conflict mask must be boolean")

    def _get_image_view(self, image_id):
        if self.support_image_views is None:
            return None
        try:
            image_id = int(image_id)
        except (TypeError, ValueError):
            return None
        if image_id < 0 or image_id >= len(self.support_image_views):
            return None
        view = self.support_image_views[image_id]
        return None if view is None else int(view)

    def _build_support_indices(self):
        first_index_by_pid_image = defaultdict(dict)
        for idx, (pid, image_id, _img_path, _caption) in enumerate(self.dataset):
            first_index_by_pid_image[int(pid)].setdefault(int(image_id), idx)

        def rotate(items, offset_seed):
            if not items:
                return items
            offset = offset_seed % len(items)
            return items[offset:] + items[:offset]

        support_indices = []
        for idx, (pid, image_id, _img_path, _caption) in enumerate(self.dataset):
            anchor_view = self._get_image_view(image_id)
            candidate_items = [
                (int(candidate_image_id), first_idx)
                for candidate_image_id, first_idx in sorted(first_index_by_pid_image[int(pid)].items())
                if int(candidate_image_id) != int(image_id)
            ]
            if self.support_selection_policy == "balanced":
                if anchor_view is None:
                    cross_view_items = []
                    same_view_items = candidate_items
                else:
                    cross_view_items = [
                        (candidate_image_id, first_idx)
                        for candidate_image_id, first_idx in candidate_items
                        if self._get_image_view(candidate_image_id) is not None
                        and self._get_image_view(candidate_image_id) != anchor_view
                    ]
                    same_view_items = [
                        (candidate_image_id, first_idx)
                        for candidate_image_id, first_idx in candidate_items
                        if (candidate_image_id, first_idx) not in cross_view_items
                    ]
                candidate_items = _select_balanced_supports(
                    cross_view_items,
                    same_view_items,
                    self.support_size,
                    idx,
                )
            elif anchor_view is not None:
                cross_view_items = [
                    (candidate_image_id, first_idx)
                    for candidate_image_id, first_idx in candidate_items
                    if self._get_image_view(candidate_image_id) is not None
                    and self._get_image_view(candidate_image_id) != anchor_view
                ]
                same_view_items = [
                    (candidate_image_id, first_idx)
                    for candidate_image_id, first_idx in candidate_items
                    if (candidate_image_id, first_idx) not in cross_view_items
                ]
                candidate_items = rotate(cross_view_items, idx) + rotate(same_view_items, idx)
            else:
                candidate_items = rotate(candidate_items, idx)
            candidates = [first_idx for _candidate_image_id, first_idx in candidate_items]
            support_indices.append(candidates[:self.support_size])
        return support_indices

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, image_id, img_path, caption = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)

        tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        ret = {
            'pids': pid,
            'image_ids': image_id,
            'images': img,
            'caption_ids': tokens,
        }

        if self.support_size > 0:
            support_images = []
            support_caption_ids = []
            support_masks = []
            support_pids = []
            support_image_ids = []
            support_reliability = []
            support_conflict_masks = []
            anchor_slots = (
                _caption_slots_strict(caption)
                if self.support_image_only else _caption_slots(caption)
            )

            indices = self.support_indices[index]
            for support_index in indices:
                support_pid, support_image_id, support_img_path, support_caption = self.dataset[support_index]
                support_img = read_image(support_img_path)
                if self.transform is not None:
                    support_img = self.transform(support_img)
                support_images.append(support_img)
                if not self.support_image_only:
                    support_tokens = tokenize(
                        support_caption,
                        tokenizer=self.tokenizer,
                        text_length=self.text_length,
                        truncate=self.truncate,
                    )
                    support_caption_ids.append(support_tokens)
                support_masks.append(1)
                support_pids.append(int(support_pid))
                support_image_ids.append(int(support_image_id))
                if self.support_image_only and anchor_slots:
                    slot_values = self.support_reliability_by_image.get(int(support_image_id), {})
                    missing_slots = [slot for slot in anchor_slots if slot not in slot_values]
                    if missing_slots:
                        raise RuntimeError(
                            "Missing queried support reliability slots for "
                            f"anchor_index={index}, support_image_id={support_image_id}: {missing_slots}"
                        )
                    rho = sum(float(slot_values[slot]) for slot in anchor_slots) / len(anchor_slots)
                elif anchor_slots and self.support_reliability_by_image:
                    slot_values = self.support_reliability_by_image.get(int(support_image_id), {})
                    rho = sum(float(slot_values.get(slot, 1.0)) for slot in anchor_slots) / len(anchor_slots)
                else:
                    rho = 1.0
                support_reliability.append(float(rho))
                relation = self.support_hard_contradictions.get(
                    (int(index), int(support_image_id)),
                    {"has_hard_contradiction": False, "contradict_slots": ()},
                )
                support_conflict_masks.append(
                    bool(relation["has_hard_contradiction"])
                    if self.support_image_only else False
                )

            while len(support_images) < self.support_size:
                support_images.append(img)
                if not self.support_image_only:
                    support_caption_ids.append(tokens)
                support_masks.append(0)
                support_pids.append(int(pid))
                support_image_ids.append(int(image_id))
                support_reliability.append(0.0)
                support_conflict_masks.append(False)

            ret.update({
                'support_images': torch.stack(support_images),
                'support_mask': torch.tensor(support_masks, dtype=torch.bool),
                'support_pids': torch.tensor(support_pids, dtype=torch.long),
                'support_image_ids': torch.tensor(support_image_ids, dtype=torch.long),
                'support_reliability': torch.tensor(support_reliability, dtype=torch.float32),
            })
            if self.support_image_only:
                ret['support_conflict_mask'] = torch.tensor(
                    support_conflict_masks, dtype=torch.bool
                )
            else:
                ret['support_caption_ids'] = torch.stack(support_caption_ids)

        if self.hard_negative_size > 0:
            candidates = self.hard_negative_pool.get(int(pid), [])
            selected = candidates[index % len(candidates)] if candidates else None
            if selected is None:
                hard_negative_img = img
                hard_negative_mask = False
                hard_negative_pid = -1
                hard_negative_image_id = -1
            else:
                hard_negative_pid = int(selected['negative_pid'])
                hard_negative_image_id = int(selected['negative_image_id'])
                if hard_negative_pid == int(pid):
                    raise RuntimeError(
                        f"Hard negative selected anchor PID at index={index}"
                    )
                hard_negative_index = self.first_index_by_image_id[hard_negative_image_id]
                _pid, _image_id, hard_negative_path, _caption = self.dataset[hard_negative_index]
                hard_negative_img = read_image(hard_negative_path)
                if self.transform is not None:
                    hard_negative_img = self.transform(hard_negative_img)
                hard_negative_mask = True
            ret.update({
                'hard_negative_image': hard_negative_img,
                'hard_negative_mask': torch.tensor(hard_negative_mask, dtype=torch.bool),
                'hard_negative_pid': torch.tensor(hard_negative_pid, dtype=torch.long),
                'hard_negative_image_id': torch.tensor(
                    hard_negative_image_id, dtype=torch.long
                ),
            })

        return ret


class ImageDataset(Dataset):
    def __init__(self, image_pids, img_paths, transform=None):
        self.image_pids = image_pids
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_pids)

    def __getitem__(self, index):
        pid, img_path = self.image_pids[index], self.img_paths[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return pid, img


class TextDataset(Dataset):
    def __init__(self,
                 caption_pids,
                 captions,
                 text_length: int = 77,
                 truncate: bool = True):
        self.caption_pids = caption_pids
        self.captions = captions
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.caption_pids)

    def __getitem__(self, index):
        pid, caption = self.caption_pids[index], self.captions[index]

        caption = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        return pid, caption


class ImageTextMLMDataset(Dataset):
    def __init__(self,
                 dataset,
                 transform=None,
                 text_length: int = 77,
                 truncate: bool = True):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate

        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, image_id, img_path, caption = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        
        caption_tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        mlm_tokens, mlm_labels = self._build_random_masked_tokens_and_labels(caption_tokens.cpu().numpy())

        ret = {
            'pids': pid,
            'image_ids': image_id,
            'images': img,
            'caption_ids': caption_tokens,
            'mlm_ids': mlm_tokens,
            'mlm_labels': mlm_labels
        }

        return ret

    def _build_random_masked_tokens_and_labels(self, tokens):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of int, tokenized sentence.
        :return: (list of int, list of int), masked tokens and related labels for MLM prediction
        """
        mask = self.tokenizer.encoder["<|mask|>"]
        token_range = list(range(1, len(self.tokenizer.encoder)-3)) # 1 ~ 49405
        
        labels = []
        for i, token in enumerate(tokens):
            if 0 < token < 49405:
                prob = random.random()
                # mask token with 15% probability
                if prob < 0.15:
                    prob /= 0.15

                    # 80% randomly change token to mask token
                    if prob < 0.8:
                        tokens[i] = mask

                    # 10% randomly change token to random token
                    elif prob < 0.9:
                        tokens[i] = random.choice(token_range)

                    # -> rest 10% randomly keep current token

                    # append current token to output (we will predict these later)
                    labels.append(token)
                else:
                    # no masking token (will be ignored by loss function later)
                    labels.append(0)
            else:
                labels.append(0)
        
        if all(l == 0 for l in labels):
            # at least mask 1
            labels[1] = tokens[1]
            tokens[1] = mask

        return torch.tensor(tokens), torch.tensor(labels)

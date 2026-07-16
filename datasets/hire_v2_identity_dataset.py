"""Training dataset for HIRE-v2 identity random-effect supervision.

The main batch remains ordinary random image-text pairs.  For each anchor this
wrapper additionally returns up to K same-PID, different-image support images.
Supports rotate deterministically across epochs and are used only to estimate a
latent identity intersection; no support caption is returned.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from utils.iotools import read_image
from utils.simple_tokenizer import SimpleTokenizer
from .bases import tokenize


SupportItem = Tuple[int, int]  # (image_id, first dataset index)


def _rotate(items: Sequence[SupportItem], offset_seed: int) -> List[SupportItem]:
    items = list(items)
    if not items:
        return []
    offset = int(offset_seed) % len(items)
    return items[offset:] + items[:offset]


def select_balanced_identity_supports(
    cross_view_items: Sequence[SupportItem],
    same_view_items: Sequence[SupportItem],
    support_size: int,
    offset_seed: int,
) -> List[SupportItem]:
    """Deterministically alternate cross-view and same-view unique images."""
    if support_size < 1:
        return []
    cross = _rotate(cross_view_items, offset_seed)
    same = _rotate(same_view_items, offset_seed // 7 + 1)
    selected: List[SupportItem] = []
    cross_index = 0
    same_index = 0
    prefer_cross = True
    while len(selected) < support_size and (
        cross_index < len(cross) or same_index < len(same)
    ):
        if prefer_cross and cross_index < len(cross):
            selected.append(cross[cross_index])
            cross_index += 1
        elif (not prefer_cross) and same_index < len(same):
            selected.append(same[same_index])
            same_index += 1
        elif cross_index < len(cross):
            selected.append(cross[cross_index])
            cross_index += 1
        elif same_index < len(same):
            selected.append(same[same_index])
            same_index += 1
        prefer_cross = not prefer_cross

    unique: List[SupportItem] = []
    seen = set()
    for item in selected:
        if item[0] not in seen:
            unique.append(item)
            seen.add(item[0])
    return unique[:support_size]


class HIREV2IdentityDataset(Dataset):
    """Random main samples plus dynamic same-ID support images."""

    def __init__(
        self,
        dataset,
        transform=None,
        text_length: int = 77,
        truncate: bool = True,
        support_size: int = 3,
        support_image_views: Optional[Sequence[Optional[int]]] = None,
        seed: int = 1,
    ):
        self.dataset = dataset
        self.transform = transform
        self.text_length = int(text_length)
        self.truncate = bool(truncate)
        self.support_size = int(support_size)
        self.support_image_views = support_image_views
        self.seed = int(seed)
        self.epoch = 0
        if self.support_size < 2:
            raise ValueError("HIRE-v2 identity requires support_size >= 2")

        self.tokenizer = SimpleTokenizer()
        self.first_index_by_pid_image: Dict[int, Dict[int, int]] = defaultdict(dict)
        self.pid_by_image_id: Dict[int, int] = {}
        for index, (pid, image_id, _path, _caption) in enumerate(self.dataset):
            pid = int(pid)
            image_id = int(image_id)
            self.first_index_by_pid_image[pid].setdefault(image_id, index)
            previous = self.pid_by_image_id.setdefault(image_id, pid)
            if previous != pid:
                raise RuntimeError(
                    "training image_id={} appears under PIDs {} and {}".format(
                        image_id, previous, pid
                    )
                )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.dataset)

    def _view(self, image_id: int) -> Optional[int]:
        if self.support_image_views is None:
            return None
        image_id = int(image_id)
        if image_id < 0 or image_id >= len(self.support_image_views):
            return None
        value = self.support_image_views[image_id]
        return None if value is None else int(value)

    def support_indices_for(self, index: int) -> List[int]:
        pid, anchor_image_id, _path, _caption = self.dataset[index]
        pid = int(pid)
        anchor_image_id = int(anchor_image_id)
        anchor_view = self._view(anchor_image_id)
        candidates = [
            (int(image_id), int(first_index))
            for image_id, first_index in sorted(
                self.first_index_by_pid_image[pid].items()
            )
            if int(image_id) != anchor_image_id
        ]

        if anchor_view is None:
            cross_view: List[SupportItem] = []
            same_view = candidates
        else:
            cross_view = [
                item
                for item in candidates
                if self._view(item[0]) is not None
                and self._view(item[0]) != anchor_view
            ]
            cross_ids = {item[0] for item in cross_view}
            same_view = [item for item in candidates if item[0] not in cross_ids]

        # Stable across workers and reproducible across runs, but changed by
        # epoch so a fixed anchor eventually observes more of its identity set.
        offset_seed = (
            self.seed * 1000003
            + self.epoch * 9176
            + anchor_image_id * 97
        )
        selected = select_balanced_identity_supports(
            cross_view,
            same_view,
            self.support_size,
            offset_seed,
        )
        return [first_index for _image_id, first_index in selected]

    def __getitem__(self, index: int):
        pid, image_id, image_path, caption = self.dataset[index]
        pid = int(pid)
        image_id = int(image_id)
        image = read_image(image_path)
        if self.transform is not None:
            image = self.transform(image)
        caption_ids = tokenize(
            caption,
            tokenizer=self.tokenizer,
            text_length=self.text_length,
            truncate=self.truncate,
        )

        support_images = []
        support_masks = []
        support_pids = []
        support_image_ids = []
        for support_index in self.support_indices_for(index):
            support_pid, support_image_id, support_path, _support_caption = (
                self.dataset[support_index]
            )
            support_pid = int(support_pid)
            support_image_id = int(support_image_id)
            if support_pid != pid:
                raise RuntimeError("identity support crosses a PID boundary")
            if support_image_id == image_id:
                raise RuntimeError("identity support reuses the anchor image")
            if support_image_id in support_image_ids:
                raise RuntimeError("identity support image is duplicated")
            support_image = read_image(support_path)
            if self.transform is not None:
                support_image = self.transform(support_image)
            support_images.append(support_image)
            support_masks.append(True)
            support_pids.append(support_pid)
            support_image_ids.append(support_image_id)

        support_count = len(support_images)
        while len(support_images) < self.support_size:
            # Padding never participates in the posterior because mask=False.
            support_images.append(image)
            support_masks.append(False)
            support_pids.append(pid)
            support_image_ids.append(image_id)

        return {
            "pids": pid,
            "image_ids": image_id,
            "images": image,
            "caption_ids": caption_ids,
            "support_images": torch.stack(support_images),
            "support_mask": torch.tensor(support_masks, dtype=torch.bool),
            "support_pids": torch.tensor(support_pids, dtype=torch.long),
            "support_image_ids": torch.tensor(support_image_ids, dtype=torch.long),
            "support_count": int(support_count),
        }

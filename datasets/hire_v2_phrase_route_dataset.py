"""Training and evaluation datasets for v16.6.0/v16.7.0 phrase routing."""

from __future__ import annotations

from typing import Optional, Sequence

from torch.utils.data import Dataset

from utils.simple_tokenizer import SimpleTokenizer
from .bases import tokenize
from .hire_v2_identity_dataset import HIREV2IdentityDataset
from .phrase_route_io import PhraseRouteTable, phrase_record_to_tensors


class HIREV2PhraseRouteDataset(HIREV2IdentityDataset):
    """v16.2.1 identity supports plus fixed offline phrase targets."""

    def __init__(
        self,
        dataset,
        transform=None,
        text_length: int = 77,
        truncate: bool = True,
        support_size: int = 3,
        support_image_views: Optional[Sequence[Optional[int]]] = None,
        seed: int = 1,
        phrase_label_file: str = "",
        expected_version: str = "v16.6.0",
        expected_route_kind: str = "propagation",
    ):
        super().__init__(
            dataset=dataset,
            transform=transform,
            text_length=text_length,
            truncate=truncate,
            support_size=support_size,
            support_image_views=support_image_views,
            seed=seed,
        )
        if not phrase_label_file:
            raise ValueError("phrase_label_file is required")
        self.phrase_table = PhraseRouteTable(
            phrase_label_file,
            split="train",
            expected_version=expected_version,
            expected_route_kind=expected_route_kind,
        )
        if len(self.phrase_table.by_index) != len(self.dataset):
            raise RuntimeError(
                "Training phrase-label coverage mismatch: labels={}, dataset={}".format(
                    len(self.phrase_table.by_index), len(self.dataset)
                )
            )

    def __getitem__(self, index: int):
        result = super().__getitem__(index)
        _pid, image_id, _path, caption = self.dataset[index]
        row = self.phrase_table.validate_caption(
            index=index,
            caption=caption,
            image_id=int(image_id),
        )
        result.update(
            phrase_record_to_tensors(row, text_length=self.text_length)
        )
        result["phrase_record_index"] = int(index)
        return result


class PhraseTextDataset(Dataset):
    """Evaluation text dataset carrying deterministic phrase spans only."""

    def __init__(
        self,
        caption_pids,
        captions,
        phrase_span_file: str,
        split: str,
        text_length: int = 77,
        truncate: bool = True,
    ):
        self.caption_pids = caption_pids
        self.captions = captions
        self.text_length = int(text_length)
        self.truncate = bool(truncate)
        self.tokenizer = SimpleTokenizer()
        self.phrase_table = PhraseRouteTable(
            phrase_span_file,
            split=split,
            expected_version="span-only",
            expected_route_kind="span-only",
        )
        if len(self.phrase_table.by_index) != len(self.caption_pids):
            raise RuntimeError(
                "Evaluation phrase-span coverage mismatch for {}: spans={}, captions={}".format(
                    split,
                    len(self.phrase_table.by_index),
                    len(self.caption_pids),
                )
            )

    def __len__(self):
        return len(self.caption_pids)

    def __getitem__(self, index):
        pid = int(self.caption_pids[index])
        caption = self.captions[index]
        token_ids = tokenize(
            caption,
            tokenizer=self.tokenizer,
            text_length=self.text_length,
            truncate=self.truncate,
        )
        row = self.phrase_table.validate_caption(index, caption)
        tensors = phrase_record_to_tensors(row, self.text_length)
        return {
            "pids": pid,
            "caption_ids": token_ids,
            "phrase_token_mask": tensors["phrase_token_mask"],
            "phrase_valid_mask": tensors["phrase_valid_mask"],
            "phrase_count": tensors["phrase_count"],
        }

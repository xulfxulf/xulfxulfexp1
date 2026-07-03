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


def _caption_slots(caption):
    text = caption.lower().replace("grey", "gray")
    tokens = set(re.sub(r"[^a-z0-9\- ]+", " ", text).split())
    slots = []
    for slot, terms in SLOT_TERMS.items():
        if any(term in text or term in tokens for term in terms):
            slots.append(slot)
    return slots


def _load_image_slot_reliability(path):
    if not path:
        return {}
    if not osp.exists(path):
        raise RuntimeError(f"support consistency csv is not available: {path}")
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
                reliability[image_id][slot] = 1.0 if consistency_type in RELIABLE_CONSISTENCY_TYPES else 0.0
    return reliability


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
                 support_consistency_csv: str = ""):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.support_size = max(0, int(support_size))
        self.support_image_views = support_image_views
        self.support_consistency_csv = support_consistency_csv
        self.support_reliability_by_image = _load_image_slot_reliability(support_consistency_csv)
        self.tokenizer = SimpleTokenizer()
        self.support_indices = self._build_support_indices() if self.support_size > 0 else None

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
            if anchor_view is not None:
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
            anchor_slots = _caption_slots(caption)

            indices = self.support_indices[index]
            for support_index in indices:
                support_pid, support_image_id, support_img_path, support_caption = self.dataset[support_index]
                support_img = read_image(support_img_path)
                if self.transform is not None:
                    support_img = self.transform(support_img)
                support_tokens = tokenize(
                    support_caption,
                    tokenizer=self.tokenizer,
                    text_length=self.text_length,
                    truncate=self.truncate,
                )
                support_images.append(support_img)
                support_caption_ids.append(support_tokens)
                support_masks.append(1)
                support_pids.append(int(support_pid))
                support_image_ids.append(int(support_image_id))
                if anchor_slots and self.support_reliability_by_image:
                    slot_values = self.support_reliability_by_image.get(int(support_image_id), {})
                    rho = sum(float(slot_values.get(slot, 1.0)) for slot in anchor_slots) / len(anchor_slots)
                else:
                    rho = 1.0
                support_reliability.append(float(rho))

            while len(support_images) < self.support_size:
                support_images.append(img)
                support_caption_ids.append(tokens)
                support_masks.append(0)
                support_pids.append(int(pid))
                support_image_ids.append(int(image_id))
                support_reliability.append(0.0)

            ret.update({
                'support_images': torch.stack(support_images),
                'support_caption_ids': torch.stack(support_caption_ids),
                'support_mask': torch.tensor(support_masks, dtype=torch.bool),
                'support_pids': torch.tensor(support_pids, dtype=torch.long),
                'support_image_ids': torch.tensor(support_image_ids, dtype=torch.long),
                'support_reliability': torch.tensor(support_reliability, dtype=torch.float32),
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

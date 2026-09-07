import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as TF

from .common import data_rng, seed32
from .upstream import bootstrap

bootstrap()
from misc.caption_dataset import pre_caption
from model.eda import EDA

VIEW_NAMES = ("F", "R", "B", "L")


def view_of(theta):
    theta = float(theta)
    if not math.isfinite(theta) or not 0 <= theta <= 360:
        raise ValueError("Angle must be finite and within [0,360]")
    theta %= 360
    if 150 <= theta <= 210:
        return 0
    if 210 < theta < 330:
        return 1
    if theta >= 330 or theta <= 30:
        return 2
    return 3


def circular_distance(a, b):
    diff = abs(float(a) - float(b)) % 360
    return min(diff, 360 - diff)


def flip(image, theta):
    return TF.hflip(image), (360.0 - theta) % 360.0


class TrainTransform:
    def __init__(self):
        self.resize = T.Resize((224, 224))
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.pool = [T.ColorJitter(.1, .1, .1, 0), T.RandomRotation(15),
                     T.RandomResizedCrop((224, 224), scale=(.9, 1.), antialias=True),
                     T.RandomGrayscale(), None, T.RandomErasing(scale=(.10, .20))]

    def __call__(self, image, theta):
        choices = np.random.choice(len(self.pool), 2, replace=True)
        image = self.to_tensor(self.resize(image))
        for choice in choices:
            if choice == 4:
                if torch.rand(1) < 0.5:
                    image, theta = flip(image, theta)
            else:
                image = self.pool[choice](image)
        return self.normalize(image), theta


def eval_transform():
    return T.Compose([T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
                      T.ToTensor(), T.Normalize([.485, .456, .406], [.229, .224, .225])])


def preprocess_text(captions, captions_bt, seed=None):
    from text_utils.tokenizer import tokenize
    def process():
        texts = list(captions)
        if len(texts) != len(captions_bt):
            raise ValueError("Caption/BT length mismatch")
        for i in range(len(texts)):
            if not texts[i].strip() or not captions_bt[i].strip():
                raise ValueError("Empty caption or back-translation")
            if random.random() < 0.1:
                texts[i] = captions_bt[i]
        # The upstream one-word return type is fixed here, not in the reference file.
        eda = EDA(stop_words=[])
        texts = [eda.random_deletion(text, .05) for text in texts]
        texts = [" ".join(text) if isinstance(text, list) else text for text in texts]
        return tokenize(texts, context_length=77)
    if seed is None:
        return process()
    with data_rng(seed):
        return process()


class PlannedTrainDataset(Dataset):
    def __init__(self, catalog, dataset_root):
        self.images = catalog["train"]
        self.pairs = [(i, j) for i, rec in enumerate(self.images) for j in range(len(rec["captions"]))]
        self.num_train_ids = len({rec["person_id"] for rec in self.images})
        self.root = Path(dataset_root)
        self.transform = TrainTransform()

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, item):
        pair_index, image_seed = item
        i, c = self.pairs[int(pair_index)]
        rec = self.images[i]
        with Image.open(self.root / rec["image_path"]) as opened:
            image = opened.convert("RGB")
        with data_rng(int(image_seed)):
            image, theta = self.transform(image, rec["theta_raw"])
        return {"image": image, "caption": rec["captions"][c], "caption_bt": rec["captions_bt"][c],
                "person_id": rec["person_id"], "image_id": rec["image_id"],
                "theta_raw": np.float32(rec["theta_raw"]), "view_raw": rec["view_raw"],
                "theta_aug": np.float32(theta), "view_aug": view_of(theta)}


class PlanBatchSampler:
    def __init__(self, indices, seed, epoch, rank, start_step=0):
        self.indices, self.seed, self.epoch, self.rank = indices, seed, epoch, rank
        self.start_step = start_step

    def __iter__(self):
        for step in range(self.start_step, len(self.indices)):
            local = self.indices[step, self.rank]
            yield [(int(pair), seed32(self.seed, self.epoch, step,
                                     self.rank * len(local) + j, "image"))
                   for j, pair in enumerate(local)]

    def __len__(self):
        return len(self.indices) - self.start_step


def device_batch(host, device, text_seed):
    batch = {k: host[k].to(device, non_blocking=True) for k in
             ("image", "person_id", "image_id", "theta_raw", "view_raw", "theta_aug", "view_aug")}
    batch["text_tokens"] = preprocess_text(host["caption"], host["caption_bt"], text_seed).to(device)
    return batch


class GalleryDataset(Dataset):
    def __init__(self, records, root):
        self.records, self.root, self.transform = records, Path(root), eval_transform()
        self.gallery_person_ids = torch.tensor([r["original_pid"] for r in records], dtype=torch.long)
        self.query_texts = [c for r in records for c in r["captions"]]
        self.query_person_ids = torch.tensor([r["original_pid"] for r in records for _ in r["captions"]])
        # Explicit aliases only for the untouched reference evaluator.
        self.text, self.img2person, self.txt2person = self.query_texts, self.gallery_person_ids, self.query_person_ids

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        rec = self.records[index]
        with Image.open(self.root / rec["image_path"]) as opened:
            image = self.transform(opened.convert("RGB"))
        return {"image": image, "gallery_index": index, "image_id": rec["image_id"],
                "theta_raw": np.float32(rec["theta_raw"]), "view_raw": rec["view_raw"]}

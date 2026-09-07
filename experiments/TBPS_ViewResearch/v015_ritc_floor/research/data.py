import collections
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from view4.common import data_rng, read_json, seed32
from view4.data import (PlannedTrainDataset, PlanBatchSampler, GalleryDataset,
                        circular_distance, device_batch)
from view4.prepare import verify_prepared

VIEW_NAMES = ("F", "S", "B")


def view3(theta):
    theta = float(theta) % 360
    if 150 <= theta <= 210:
        return 0
    if theta >= 330 or theta <= 30:
        return 2
    return 1


def aggregate_probabilities(probabilities):
    p = np.asarray(probabilities, dtype=np.float64)
    if p.shape != (72,) or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("Expected 72 finite, nonnegative OEFormer probabilities")
    if not np.isclose(p.sum(), 1., atol=1e-4):
        raise ValueError("OEFormer probabilities must sum to one")
    p = p / p.sum()
    result = np.bincount([view3(k * 5) for k in range(72)], weights=p, minlength=3)
    return result.astype(np.float32)


def enriched_catalog(cfg):
    sources = verify_prepared(cfg)
    catalog = read_json(Path(cfg["prepared_dir"]) / "catalog.json")
    orientation = read_json(cfg["orientation"])
    lookup = {r["image_path"]: r for r in orientation["images"]}
    if len(lookup) != len(orientation["images"]):
        raise ValueError("Duplicate orientation paths")
    for split, records in catalog.items():
        for row in records:
            source = lookup[row["image_path"]]
            if split not in source["splits"] or row["original_pid"] not in source["pids"]:
                raise ValueError("Orientation split/PID association failed")
            row["view_prob"] = aggregate_probabilities(source["probabilities"]).tolist()
            row["view3_raw"] = view3(row["theta_raw"])
    return catalog, sources


def support_candidates(records, delta=60):
    by_pid = collections.defaultdict(list)
    for i, rec in enumerate(records):
        by_pid[rec["person_id"]].append(i)
    return [[j for j in by_pid[a["person_id"]]
             if records[j]["image_id"] != a["image_id"]
             and records[j]["view3_raw"] != a["view3_raw"]
             and circular_distance(a["theta_raw"], records[j]["theta_raw"]) >= delta]
            for a in records]


def validate_pk_plan(indices, dataset, steps, world_size, local_batch, p, k=2):
    if (k != 2 or p % world_size or p * k != world_size * local_batch or
            indices.shape != (steps, world_size, local_batch) or
            not np.issubdtype(indices.dtype, np.integer) or indices.size == 0 or
            indices.min() < 0 or indices.max() >= len(dataset.pairs)):
        raise ValueError('Invalid whole-PID K2 plan shape or indices')
    pair_images = np.asarray([i for i, _ in dataset.pairs], dtype=np.int64)
    image_pids = np.asarray([r['person_id'] for r in dataset.images], dtype=np.int64)
    image_ids = np.asarray([r['image_id'] for r in dataset.images], dtype=np.int64)
    pid_images = collections.defaultdict(set)
    for row in dataset.images:
        pid_images[row['person_id']].add(row['image_id'])
    singleton_pairs = 0
    for batch in indices:
        selected = pair_images[batch]
        pids, images = image_pids[selected], image_ids[selected]
        identities, counts = np.unique(pids, return_counts=True)
        if len(identities) != p or not np.all(counts == k):
            raise ValueError('Global PK identity multiplicity mismatch')
        for local in pids:
            unique, counts = np.unique(local, return_counts=True)
            if len(unique) != p // world_size or not np.all(counts == k):
                raise ValueError('PID group is split across ranks')
        for pid in identities:
            if np.unique(images[pids == pid]).size != k:
                if len(pid_images[int(pid)]) != 1:
                    raise ValueError('Repeated image despite multiple training images')
                singleton_pairs += 1
    return dict(status='passed', batches=steps, identities_per_global_batch=p,
                images_per_identity=k, whole_pid_rank_allocation=True,
                repeated_single_image_pairs=singleton_pairs, train_only=True)


class SupportSampler(PlanBatchSampler):
    def __init__(self, indices, seed, epoch, rank, dataset, max_supports, start_step=0):
        super().__init__(indices, seed, epoch, rank, start_step)
        self.dataset, self.max_supports = dataset, max_supports

    def __iter__(self):
        for offset, ordinary in enumerate(super().__iter__()):
            step = self.start_step + offset
            rng = np.random.RandomState(seed32(self.seed, self.epoch, step, self.rank, "support"))
            eligible = [j for j, (pair, _) in enumerate(ordinary)
                        if self.dataset.candidates[self.dataset.pairs[pair][0]]]
            selected = set(rng.permutation(eligible)[:self.max_supports].tolist())
            batch = []
            for j, (pair, image_seed) in enumerate(ordinary):
                choices = self.dataset.candidates[self.dataset.pairs[pair][0]]
                support = int(rng.choice(choices)) if j in selected else -1
                batch.append((pair, image_seed, support,
                              seed32(self.seed, self.epoch, step, self.rank, j, "support_image")))
            yield batch


class SupportDataset(PlannedTrainDataset):
    def __init__(self, catalog, root, delta=60):
        super().__init__(catalog, root)
        self.delta = delta
        self.candidates = support_candidates(self.images, delta)

    def __getitem__(self, item):
        pair, image_seed, support, support_seed = item
        batch = super().__getitem__((pair, image_seed))
        anchor = self.images[self.pairs[pair][0]]
        batch["view_prob"] = torch.tensor(anchor["view_prob"], dtype=torch.float32)
        # F/S/B probabilities are invariant to left-right horizontal mirroring.
        batch["view3_raw"] = anchor["view3_raw"]
        batch["view3_aug"] = view3(batch["theta_aug"])
        batch.update(support_image=torch.zeros_like(batch["image"]), support_valid=False,
                     support_prob=batch["view_prob"].clone(), support_image_id=-1,
                     support_person_id=-1, support_theta_raw=np.float32(0),
                     support_theta_aug=np.float32(0), support_view3_raw=-1, support_view3_aug=-1)
        if support < 0:
            return batch
        rec = self.images[support]
        if (rec["person_id"] != anchor["person_id"] or rec["image_id"] == anchor["image_id"]
                or support not in self.candidates[self.pairs[pair][0]]):
            raise ValueError("Invalid same-identity cross-image support")
        with Image.open(self.root / rec["image_path"]) as opened:
            image = opened.convert("RGB")
        with data_rng(support_seed):
            image, theta = self.transform(image, rec["theta_raw"])
        valid = (view3(theta) != batch["view3_aug"]
                 and circular_distance(theta, batch["theta_aug"]) >= self.delta)
        batch.update(support_image=image, support_valid=valid,
                     support_prob=torch.tensor(rec["view_prob"], dtype=torch.float32),
                     support_image_id=rec["image_id"], support_person_id=rec["person_id"],
                     support_theta_raw=np.float32(rec["theta_raw"]), support_theta_aug=np.float32(theta),
                     support_view3_raw=rec["view3_raw"], support_view3_aug=view3(theta))
        return batch


def to_device(host, device, text_seed):
    batch = device_batch(host, device, text_seed)
    for key in ("view_prob", "view3_raw", "view3_aug", "support_image", "support_valid",
                "support_prob", "support_image_id", "support_person_id", "support_theta_raw",
                "support_theta_aug", "support_view3_raw", "support_view3_aug"):
        batch[key] = host[key].to(device, non_blocking=True)
    return batch


class SoftGalleryDataset(GalleryDataset):
    def __getitem__(self, index):
        batch = super().__getitem__(index)
        batch["view_prob"] = torch.tensor(self.records[index]["view_prob"], dtype=torch.float32)
        return batch

import collections
import itertools

import numpy as np
import torch
from torch.utils.data import DistributedSampler

from .common import seed32
from .data import circular_distance


def e0_plan(pair_count, epoch, steps=212, world_size=4, local_batch=80):
    plan = np.empty((steps, world_size, local_batch), dtype=np.int64)
    for rank in range(world_size):
        sampler = DistributedSampler(range(pair_count), num_replicas=world_size, rank=rank,
                                     shuffle=True, seed=0, drop_last=False)
        sampler.set_epoch(epoch - 1)
        indices = list(sampler)[:steps * local_batch]
        if len(indices) != steps * local_batch:
            raise ValueError("Not enough caption-pairs for the fixed E0 schedule")
        plan[:, rank] = np.asarray(indices).reshape(steps, local_batch)
    return plan


def pk_plan(records, seed, epoch, steps=212, world_size=4, p=160, k=2, delta=60):
    if k != 2 or p % world_size:
        raise ValueError("This audited group implements K2 and whole-PID rank allocation")
    pid_images = collections.defaultdict(list)
    pair_offsets, offset = [], 0
    for i, rec in enumerate(records):
        pid_images[rec["person_id"]].append(i)
        pair_offsets.append(offset)
        offset += len(rec["captions"])
    pids = sorted(pid_images)
    if len(pids) < p:
        raise ValueError("Not enough distinct identities")
    candidates = {}
    for pid, indices in pid_images.items():
        pairs = list(itertools.combinations(indices, 2)) or [(indices[0], indices[0])]
        good = [(a, b) for a, b in pairs if records[a]["view_raw"] != records[b]["view_raw"]
                and circular_distance(records[a]["theta_raw"], records[b]["theta_raw"]) >= delta]
        candidates[pid] = good or pairs
    rng = np.random.RandomState(seed32(seed, epoch, "pid_plan"))
    pending = collections.deque()
    batches, stats = [], []
    for step in range(steps):
        selected, selected_set, deferred = [], set(), []
        while len(selected) < p:
            if not pending:
                pending.extend(rng.permutation(pids).tolist())
            pid = pending.popleft()
            if pid in selected_set:
                deferred.append(pid)
            else:
                selected.append(pid)
                selected_set.add(pid)
        pending.extendleft(reversed(deferred))
        counts, groups, valid = np.zeros(4, dtype=np.int64), [], 0
        for pid in selected:
            options = candidates[pid]
            costs = []
            for a, b in options:
                after = counts.copy()
                after[records[a]["view_raw"]] += 1
                after[records[b]["view_raw"]] += 1
                costs.append(int(after @ after))
            ties = np.flatnonzero(np.asarray(costs) == min(costs))
            a, b = options[int(rng.choice(ties))]
            rows = []
            for i in (a, b):
                counts[records[i]["view_raw"]] += 1
                rows.append(pair_offsets[i] + int(rng.randint(len(records[i]["captions"]))))
            if a != b and records[a]["view_raw"] != records[b]["view_raw"] and circular_distance(
                    records[a]["theta_raw"], records[b]["theta_raw"]) >= delta:
                valid += 1
            groups.append(rows)
        rng.shuffle(groups)
        batches.append(np.asarray(groups, dtype=np.int64).reshape(world_size, p * k // world_size))
        stats.append({"step": step, "raw_views": counts.tolist(), "raw_valid_pairs": valid,
                      "identities": p})
    return np.stack(batches), stats

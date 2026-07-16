"""Identity-only components for HIRE-v2 version two.

The version-two experiment keeps the HIRE-v2 anchored global/local observation
space intact and adds only a grouped identity random-effect path.  Same-PID
support images are not inserted as ordinary positives.  They estimate a
heterogeneity-aware trusted identity intersection used by a single text-to-image
identity-group auxiliary objective.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


_EPS = 1e-8


class SharedIdentityMean(nn.Module):
    """Shared image/text identity mapping initialized as an exact identity."""

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)
        nn.init.eye_(self.proj.weight)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(observation.float()), dim=-1)


class BoundedImageUncertainty(nn.Module):
    """Predict bounded per-dimension image uncertainty for support fusion.

    The input is detached by the caller.  Uncertainty therefore learns how much
    to trust an already anchored observation without changing the CLIP/RDE
    observation space merely to make variance prediction easier.
    """

    def __init__(
        self,
        dim: int,
        variance_min: float = 0.1,
        variance_max: float = 2.0,
    ):
        super().__init__()
        if not 0.0 < variance_min < variance_max:
            raise ValueError("variance bounds must satisfy 0 < min < max")
        self.proj = nn.Linear(dim, dim)
        self.variance_min = float(variance_min)
        self.variance_max = float(variance_max)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        probability = torch.sigmoid(self.proj(observation.float()))
        return self.variance_min + (
            self.variance_max - self.variance_min
        ) * probability


class BoundedResidualGate(nn.Module):
    """A scalar identity residual gate constrained to (0, 1)."""

    def __init__(self, initial_value: float = 0.1):
        super().__init__()
        if not 0.0 < initial_value < 1.0:
            raise ValueError("initial gate must be in (0, 1)")
        initial_logit = math.log(initial_value / (1.0 - initial_value))
        self.logit = nn.Parameter(torch.tensor(initial_logit, dtype=torch.float32))

    def forward(self) -> torch.Tensor:
        return torch.sigmoid(self.logit)


def heterogeneity_aware_identity_intersection(
    means: torch.Tensor,
    variances: torch.Tensor,
    mask: torch.Tensor,
    min_supports: int = 2,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """Fuse same-ID support observations into a trusted identity intersection.

    Args:
        means: [B, S, D] normalized support identity means.
        variances: [B, S, D] positive bounded support variances.
        mask: [B, S] valid support-image mask.
        min_supports: fixed minimum required to estimate within-group
            heterogeneity.  Version two uses two; this is not exposed as a
            training hyperparameter.

    Returns:
        mean: [B, D] normalized trusted identity intersection.
        raw_mean: [B, D] unnormalized precision-weighted mean.
        tau2: [B, D] empirical within-identity heterogeneity.
        precision: [B, S, D] effective precision after heterogeneity.
        valid: [B] rows containing at least min_supports valid supports.
        count: [B] valid support counts.
    """
    if means.ndim != 3 or means.shape != variances.shape:
        raise ValueError("means and variances must have identical [B,S,D] shape")
    if mask.shape != means.shape[:2]:
        raise ValueError("mask must have shape [B,S]")
    if min_supports < 2:
        raise ValueError("min_supports must be at least two")
    if (variances <= 0).any():
        raise ValueError("variances must be positive")

    mask_f = mask.to(dtype=means.dtype).unsqueeze(-1)
    count = mask.sum(dim=1)
    valid = count.ge(int(min_supports))

    precision0 = mask_f / (variances + eps)
    precision0_sum = precision0.sum(dim=1).clamp_min(eps)
    center0 = (precision0 * means).sum(dim=1) / precision0_sum

    # The initial center is a statistical target, not a shortcut the network may
    # move to artificially reduce heterogeneity.
    center_for_tau = center0.detach()
    tau2 = (
        precision0 * (means - center_for_tau.unsqueeze(1)).pow(2)
    ).sum(dim=1) / precision0_sum

    effective_precision = mask_f / (
        variances + tau2.unsqueeze(1) + eps
    )
    precision_sum = effective_precision.sum(dim=1).clamp_min(eps)
    raw_mean = (effective_precision * means).sum(dim=1) / precision_sum
    normalized_mean = F.normalize(raw_mean, dim=-1)

    # Keep invalid rows finite.  Callers must exclude them from group loss.
    normalized_mean = torch.where(
        valid.unsqueeze(-1),
        normalized_mean,
        torch.zeros_like(normalized_mean),
    )
    raw_mean = torch.where(
        valid.unsqueeze(-1),
        raw_mean,
        torch.zeros_like(raw_mean),
    )
    tau2 = torch.where(valid.unsqueeze(-1), tau2, torch.zeros_like(tau2))
    effective_precision = torch.where(
        valid.view(-1, 1, 1),
        effective_precision,
        torch.zeros_like(effective_precision),
    )

    return {
        "mean": normalized_mean,
        "raw_mean": raw_mean,
        "tau2": tau2.clamp_min(0.0),
        "precision": effective_precision,
        "valid": valid,
        "count": count,
    }


def paired_identity_group_nce(
    scores: torch.Tensor,
    pids: torch.Tensor,
    target_valid: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """Text-to-image-group NCE with strict per-query leave-one semantics.

    Column i is the support-image posterior created for query i and excludes
    query i's own image.  It is the only positive target for row i.  Other
    columns with the same PID are ignored rather than treated as additional
    positives because they may contain row i's original image in their support
    set.  Different-PID valid groups are negatives.
    """
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("scores must be square [B,B]")
    pids = pids.view(-1)
    target_valid = target_valid.view(-1).bool()
    if pids.numel() != scores.shape[0] or target_valid.numel() != scores.shape[1]:
        raise ValueError("PID/valid sizes do not match score matrix")

    batch_size = scores.shape[0]
    diagonal = torch.eye(batch_size, dtype=torch.bool, device=scores.device)
    different_pid = pids[:, None].ne(pids[None, :])
    allowed = (different_pid | diagonal) & target_valid.unsqueeze(0)
    has_negative = (different_pid & target_valid.unsqueeze(0)).any(dim=1)
    valid_rows = target_valid & has_negative

    if not valid_rows.any():
        return scores.sum() * 0.0

    logits = scores * logit_scale.to(device=scores.device, dtype=scores.dtype)
    negative = torch.finfo(logits.dtype).min
    logits = logits.masked_fill(~allowed, negative)
    # Invalid rows are skipped; keep them finite before log-softmax.
    logits = torch.where(valid_rows.unsqueeze(1), logits, torch.zeros_like(logits))
    log_probability = F.log_softmax(logits, dim=1)
    diagonal_log_probability = log_probability.diagonal()
    return -diagonal_log_probability[valid_rows].mean()


def sdm_from_similarity(
    similarity: torch.Tensor,
    pids: torch.Tensor,
    logit_scale: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Exact score-matrix form of the repository's SDM objective."""
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError("similarity must be square [B,B]")
    pids = pids.view(-1)
    labels = pids[:, None].eq(pids[None, :]).to(similarity.dtype)
    labels_distribution = labels / labels.sum(dim=1, keepdim=True).clamp_min(1.0)

    t2i_logits = similarity * logit_scale.to(
        device=similarity.device, dtype=similarity.dtype
    )
    i2t_logits = t2i_logits.t()

    t2i_prediction = F.softmax(t2i_logits, dim=1)
    i2t_prediction = F.softmax(i2t_logits, dim=1)
    t2i = t2i_prediction * (
        F.log_softmax(t2i_logits, dim=1)
        - torch.log(labels_distribution + epsilon)
    )
    i2t = i2t_prediction * (
        F.log_softmax(i2t_logits, dim=1)
        - torch.log(labels_distribution + epsilon)
    )
    return t2i.sum(dim=1).mean() + i2t.sum(dim=1).mean()


def itc_from_similarity(
    similarity: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """Exact score-matrix form of the repository's paired ITC objective."""
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError("similarity must be square [B,B]")
    logits_t2i = similarity * logit_scale.to(
        device=similarity.device, dtype=similarity.dtype
    )
    labels = torch.arange(similarity.shape[0], device=similarity.device)
    return 0.5 * (
        F.cross_entropy(logits_t2i, labels)
        + F.cross_entropy(logits_t2i.t(), labels)
    )


def identity_residual_score(
    observation_score: torch.Tensor,
    identity_score: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    """Anchor-preserving identity residual score used during training.

    The value is a convex mixture, but stop-gradient on the subtracted anchor
    keeps the full final-retrieval gradient on the strong observation path.
    Identity observations themselves are produced from detached anchored
    observations, so identity auxiliary objectives cannot rewrite the anchor.
    """
    return observation_score + gate * (
        identity_score - observation_score.detach()
    )


def build_identity_final_embedding(
    observation: torch.Tensor,
    identity: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    """Build a single retrieval embedding exactly matching inference score.

    dot([sqrt(1-a) g_t, sqrt(a) u_t],
        [sqrt(1-a) g_i, sqrt(a) u_i])
      = (1-a) <g_t,g_i> + a <u_t,u_i>.
    """
    gate = gate.to(device=observation.device, dtype=observation.dtype).clamp(0.0, 1.0)
    left = torch.sqrt((1.0 - gate).clamp_min(0.0)) * F.normalize(
        observation.float(), dim=-1
    )
    right = torch.sqrt(gate.clamp_min(0.0)) * F.normalize(
        identity.float(), dim=-1
    )
    return torch.cat([left, right], dim=-1)


def aggregate_identity_objectives(
    global_sdm: torch.Tensor,
    global_itc: torch.Tensor,
    local_sdm: torch.Tensor,
    local_itc: torch.Tensor,
    final_sdm: torch.Tensor,
    final_itc: torch.Tensor,
    group_nce: torch.Tensor,
    auxiliary_weight: float,
) -> Dict[str, torch.Tensor]:
    """Aggregate HIRE-v2 identity objective exactly as documented.

    L_anchor = 0.5 * (L_global + L_local)
    L_total  = L_anchor + L_final + lambda * L_group
    """
    if auxiliary_weight < 0.0:
        raise ValueError("auxiliary_weight must be non-negative")
    sdm_loss = 0.5 * (global_sdm + local_sdm) + final_sdm
    itc_loss = 0.5 * (global_itc + local_itc) + final_itc
    identity_group_loss = float(auxiliary_weight) * group_nce
    return {
        "sdm_loss": sdm_loss,
        "itc_loss": itc_loss,
        "identity_group_loss": identity_group_loss,
        "anchor_objective": 0.5 * (
            global_sdm + global_itc + local_sdm + local_itc
        ),
        "final_objective": final_sdm + final_itc,
    }

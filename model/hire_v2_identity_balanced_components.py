"""HIRE-v2 v16.2.1 identity-balanced components.

The v16.2.0 no-training audit established two facts:

1. The identity residual itself is useful.
2. The learned variance/heterogeneity weighting is numerically equivalent to a
   simple support mean, while the observation anchor is weakened during joint
   training.

v16.2.1 therefore keeps the strict leave-one group supervision and shared
identity residual, replaces the inactive probabilistic weighting with a
deterministic masked group consensus, and explicitly balances the observation
and final retrieval objectives.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from .hire_v2_identity_components import (
    BoundedResidualGate,
    SharedIdentityMean,
    build_identity_final_embedding,
    identity_residual_score,
    itc_from_similarity,
    paired_identity_group_nce,
    sdm_from_similarity,
)


_EPS = 1e-8
_OBSERVATION_MAIN_WEIGHT = 0.5
_FINAL_MAIN_WEIGHT = 0.5


def masked_identity_group_consensus(
    means: torch.Tensor,
    mask: torch.Tensor,
    min_supports: int = 2,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """Build a strict leave-one identity consensus by masked arithmetic mean.

    Args:
        means: normalized support identity observations, shape [B, S, D].
        mask: valid support positions, shape [B, S].
        min_supports: fixed minimum number of different support images.
        eps: numerical stability constant.

    Returns:
        mean:
            L2-normalized group consensus [B, D].
        raw_mean:
            Unnormalized masked support mean [B, D].
        dispersion:
            Mean squared support deviation per feature dimension [B, D].
            This is a diagnostic and is not used as a training weight.
        dispersion_scalar:
            Per-group mean dispersion [B].
        mean_support_cosine:
            Average cosine from valid supports to the normalized consensus [B].
        valid:
            Rows containing at least ``min_supports`` valid supports [B].
        count:
            Valid support count [B].

    The offline v16.2.0 audit showed that the full variance/heterogeneity
    intersection had median cosine 1.0 to the simple mean and changed strict
    group R1 by only 0.002542 points.  v16.2.1 therefore uses the empirically
    supported deterministic group consensus and does not claim learned
    uncertainty weighting.
    """
    if means.ndim != 3:
        raise ValueError("means must have shape [B,S,D]")
    if mask.shape != means.shape[:2]:
        raise ValueError("mask must have shape [B,S]")
    if min_supports < 2:
        raise ValueError("min_supports must be at least two")

    mask = mask.bool()
    mask_f = mask.to(dtype=means.dtype).unsqueeze(-1)
    count = mask.sum(dim=1)
    valid = count.ge(int(min_supports))
    denominator = mask_f.sum(dim=1).clamp_min(1.0)

    raw_mean = (means * mask_f).sum(dim=1) / denominator
    normalized_mean = F.normalize(raw_mean, dim=-1, eps=eps)

    centered = means - raw_mean.detach().unsqueeze(1)
    dispersion = (centered.pow(2) * mask_f).sum(dim=1) / denominator
    dispersion_scalar = dispersion.mean(dim=-1)

    support_cosine = (means * normalized_mean.unsqueeze(1)).sum(dim=-1)
    mean_support_cosine = (
        support_cosine * mask.to(dtype=support_cosine.dtype)
    ).sum(dim=1) / count.to(dtype=support_cosine.dtype).clamp_min(1.0)

    zero_mean = torch.zeros_like(normalized_mean)
    zero_raw = torch.zeros_like(raw_mean)
    zero_dispersion = torch.zeros_like(dispersion)
    zero_scalar = torch.zeros_like(dispersion_scalar)

    normalized_mean = torch.where(valid.unsqueeze(-1), normalized_mean, zero_mean)
    raw_mean = torch.where(valid.unsqueeze(-1), raw_mean, zero_raw)
    dispersion = torch.where(
        valid.unsqueeze(-1), dispersion, zero_dispersion
    )
    dispersion_scalar = torch.where(valid, dispersion_scalar, zero_scalar)
    mean_support_cosine = torch.where(valid, mean_support_cosine, zero_scalar)

    return {
        "mean": normalized_mean,
        "raw_mean": raw_mean,
        "dispersion": dispersion.clamp_min(0.0),
        "dispersion_scalar": dispersion_scalar.clamp_min(0.0),
        "mean_support_cosine": mean_support_cosine,
        "valid": valid,
        "count": count,
    }


def aggregate_identity_balanced_objectives(
    global_sdm: torch.Tensor,
    global_itc: torch.Tensor,
    local_sdm: torch.Tensor,
    local_itc: torch.Tensor,
    observation_sdm: torch.Tensor,
    observation_itc: torch.Tensor,
    final_sdm: torch.Tensor,
    final_itc: torch.Tensor,
    group_nce: torch.Tensor,
    auxiliary_weight: float,
) -> Dict[str, torch.Tensor]:
    """Aggregate the exact v16.2.1 objective.

    L_anchor = 0.5 * (L_global + L_local)
    L_main   = 0.5 * L_observation + 0.5 * L_final
    L_total  = L_anchor + L_main + lambda * L_group

    The observation/final weights are fixed design constants, not searched
    hyperparameters.  At initialization S_final == S_observation, hence
    L_main == L_observation and the retrieval objective exactly recovers the
    v16.1.0 anchor objective before adding the 0.1 group auxiliary.
    """
    if auxiliary_weight < 0.0:
        raise ValueError("auxiliary_weight must be non-negative")

    sdm_loss = (
        0.5 * (global_sdm + local_sdm)
        + _OBSERVATION_MAIN_WEIGHT * observation_sdm
        + _FINAL_MAIN_WEIGHT * final_sdm
    )
    itc_loss = (
        0.5 * (global_itc + local_itc)
        + _OBSERVATION_MAIN_WEIGHT * observation_itc
        + _FINAL_MAIN_WEIGHT * final_itc
    )
    identity_group_loss = float(auxiliary_weight) * group_nce

    anchor_objective = 0.5 * (
        global_sdm + global_itc + local_sdm + local_itc
    )
    observation_objective = observation_sdm + observation_itc
    final_objective = final_sdm + final_itc
    balanced_main_objective = (
        _OBSERVATION_MAIN_WEIGHT * observation_objective
        + _FINAL_MAIN_WEIGHT * final_objective
    )

    return {
        "sdm_loss": sdm_loss,
        "itc_loss": itc_loss,
        "identity_group_loss": identity_group_loss,
        "anchor_objective": anchor_objective,
        "observation_objective": observation_objective,
        "final_objective": final_objective,
        "balanced_main_objective": balanced_main_objective,
        "observation_main_weight": observation_sdm.new_tensor(
            _OBSERVATION_MAIN_WEIGHT
        ),
        "final_main_weight": final_sdm.new_tensor(_FINAL_MAIN_WEIGHT),
    }


__all__ = [
    "BoundedResidualGate",
    "SharedIdentityMean",
    "aggregate_identity_balanced_objectives",
    "build_identity_final_embedding",
    "identity_residual_score",
    "itc_from_similarity",
    "masked_identity_group_consensus",
    "paired_identity_group_nce",
    "sdm_from_similarity",
]

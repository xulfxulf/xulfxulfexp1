"""Shared phrase-routing components for HIRE-v2 v16.6.0 and v16.7.0.

Both versions use exactly the same trainable architecture.  They differ only in
which offline teacher distribution is loaded:

- v16.6.0: same-identity multi-view propagation distribution;
- v16.7.0: the same distribution multiplied by one strict hard-negative
  discrimination factor.

The complete observation remains the retrieval subject.  Phrase routing only
adds a zero-initialized residual inside the text identity branch.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hire_v2_identity_components import (
    build_identity_final_embedding,
    identity_residual_score,
    itc_from_similarity,
    paired_identity_group_nce,
    sdm_from_similarity,
)


_EPS = 1e-8
_OBSERVATION_MAIN_WEIGHT = 0.5
_FINAL_MAIN_WEIGHT = 0.5


def masked_phrase_softmax(
    logits: torch.Tensor,
    phrase_valid_mask: torch.Tensor,
) -> torch.Tensor:
    if logits.ndim != 2 or phrase_valid_mask.shape != logits.shape:
        raise ValueError("phrase logits and mask must both have shape [B,M]")
    mask = phrase_valid_mask.bool()
    minimum = torch.finfo(logits.dtype).min
    probability = F.softmax(logits.masked_fill(~mask, minimum), dim=1)
    probability = probability * mask.to(dtype=probability.dtype)
    denominator = probability.sum(dim=1, keepdim=True).clamp_min(_EPS)
    probability = probability / denominator
    empty = ~mask.any(dim=1, keepdim=True)
    return torch.where(empty, torch.zeros_like(probability), probability)


def phrase_attention_pool(
    text_tokens: torch.Tensor,
    token_ids: torch.Tensor,
    text_attention: torch.Tensor,
    phrase_token_mask: torch.Tensor,
    phrase_valid_mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Pool CLIP words into phrase vectors using EOT attention within spans."""
    if text_tokens.ndim != 3:
        raise ValueError("text_tokens must be [B,L,D]")
    if token_ids.shape != text_tokens.shape[:2]:
        raise ValueError("token_ids must match text token sequence")
    if text_attention.shape[:2] != token_ids.shape:
        raise ValueError("text attention must match token sequence")
    if phrase_token_mask.ndim != 3:
        raise ValueError("phrase_token_mask must be [B,M,L]")
    if phrase_token_mask.shape[0] != text_tokens.shape[0]:
        raise ValueError("phrase and text batch sizes differ")
    if phrase_token_mask.shape[2] != text_tokens.shape[1]:
        raise ValueError("phrase mask sequence length differs from text")
    if phrase_valid_mask.shape != phrase_token_mask.shape[:2]:
        raise ValueError("phrase_valid_mask must be [B,M]")

    batch_size, sequence_length, _dim = text_tokens.shape
    eot = token_ids.argmax(dim=-1)
    content_valid = token_ids.ne(0)
    content_valid[:, 0] = False
    content_valid[
        torch.arange(batch_size, device=token_ids.device),
        eot,
    ] = False

    eot_attention = text_attention[
        torch.arange(batch_size, device=token_ids.device),
        eot,
        :,
    ].detach().float()
    phrase_mask = (
        phrase_token_mask.bool()
        & content_valid[:, None, :]
        & phrase_valid_mask[:, :, None].bool()
    )
    phrase_has_token = phrase_mask.any(dim=2)
    phrase_valid = phrase_valid_mask.bool() & phrase_has_token

    expanded_attention = eot_attention[:, None, :].expand_as(
        phrase_mask
    )
    minimum = torch.finfo(expanded_attention.dtype).min
    masked_attention = expanded_attention.masked_fill(
        ~phrase_mask, minimum
    )
    token_weight = F.softmax(masked_attention, dim=2)
    token_weight = token_weight * phrase_mask.to(token_weight.dtype)
    token_weight = token_weight / token_weight.sum(
        dim=2, keepdim=True
    ).clamp_min(_EPS)
    token_weight = torch.where(
        phrase_valid[:, :, None],
        token_weight,
        torch.zeros_like(token_weight),
    )

    phrase_feature = torch.einsum(
        "bml,bld->bmd",
        token_weight,
        text_tokens.detach().float(),
    )
    phrase_feature = F.normalize(phrase_feature, dim=-1)
    phrase_feature = torch.where(
        phrase_valid[:, :, None],
        phrase_feature,
        torch.zeros_like(phrase_feature),
    )
    return {
        "features": phrase_feature,
        "valid": phrase_valid,
        "token_weight": token_weight,
        "eot_attention": eot_attention,
    }


class RelativePhraseRouter(nn.Module):
    """Predict one relative logit per phrase, then normalize within a caption."""

    def __init__(self, dim: int):
        super().__init__()
        hidden = max(1, int(dim) // 4)
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        # Uniform initial distribution over valid phrases.
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(
        self,
        phrase_features: torch.Tensor,
        text_observation: torch.Tensor,
        phrase_valid_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if phrase_features.ndim != 3:
            raise ValueError("phrase_features must be [B,M,D]")
        if text_observation.shape != (
            phrase_features.shape[0],
            phrase_features.shape[2],
        ):
            raise ValueError("text observation and phrase features differ")
        if phrase_valid_mask.shape != phrase_features.shape[:2]:
            raise ValueError("phrase mask and features differ")
        contextual = self.norm(
            phrase_features.detach()
            + text_observation.detach().float().unsqueeze(1)
        )
        hidden = F.gelu(self.fc1(contextual))
        logits = self.fc2(hidden).squeeze(-1)
        probability = masked_phrase_softmax(
            logits, phrase_valid_mask
        )
        return {"logits": logits, "probability": probability}


class ZeroInitializedPhraseIdentityResidual(nn.Module):
    """Add the routed phrase pool to the raw text identity vector."""

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(
        self,
        base_identity_raw: torch.Tensor,
        phrase_features: torch.Tensor,
        phrase_probability: torch.Tensor,
        phrase_valid_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if base_identity_raw.ndim != 2 or phrase_features.ndim != 3:
            raise ValueError("base identity must be [B,D], phrases [B,M,D]")
        if phrase_probability.shape != phrase_features.shape[:2]:
            raise ValueError("phrase probability and features differ")
        if phrase_valid_mask.shape != phrase_probability.shape:
            raise ValueError("phrase mask and probability differ")
        weight = (
            phrase_probability
            * phrase_valid_mask.to(phrase_probability.dtype)
        )
        weight = weight / weight.sum(
            dim=1, keepdim=True
        ).clamp_min(_EPS)
        pooled = (
            phrase_features.detach()
            * weight.unsqueeze(-1)
        ).sum(dim=1)
        residual = self.proj(pooled.float())
        identity = F.normalize(
            base_identity_raw.float() + residual,
            dim=-1,
        )
        return {
            "identity": identity,
            "pooled": pooled,
            "residual": residual,
            "weight": weight,
        }


def phrase_route_kl_divergence(
    teacher_target: torch.Tensor,
    student_probability: torch.Tensor,
    phrase_valid_mask: torch.Tensor,
    route_supervision_mask: torch.Tensor,
) -> torch.Tensor:
    if teacher_target.shape != student_probability.shape:
        raise ValueError("teacher and student distributions must match")
    if phrase_valid_mask.shape != teacher_target.shape:
        raise ValueError("phrase mask must match route distributions")
    if route_supervision_mask.view(-1).shape[0] != teacher_target.shape[0]:
        raise ValueError("route supervision mask must be [B]")

    phrase_mask = phrase_valid_mask.bool()
    target = teacher_target.float() * phrase_mask.to(torch.float32)
    target = target / target.sum(dim=1, keepdim=True).clamp_min(_EPS)
    probability = (
        student_probability.float()
        * phrase_mask.to(torch.float32)
    )
    probability = probability / probability.sum(
        dim=1, keepdim=True
    ).clamp_min(_EPS)

    valid_row = (
        route_supervision_mask.view(-1).bool()
        & phrase_mask.sum(dim=1).ge(2)
        & target.gt(0).sum(dim=1).ge(2)
    )
    if not bool(valid_row.any()):
        return student_probability.sum() * 0.0
    log_target = target.clamp_min(_EPS).log()
    log_probability = probability.clamp_min(_EPS).log()
    row_kl = (
        target * (log_target - log_probability)
    ).sum(dim=1)
    return row_kl[valid_row].mean()


def _rank_values(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values, dim=0)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(
        values.numel(),
        device=values.device,
        dtype=torch.float32,
    )
    return ranks


def masked_phrase_spearman(
    teacher_target: torch.Tensor,
    student_probability: torch.Tensor,
    phrase_valid_mask: torch.Tensor,
    route_supervision_mask: torch.Tensor,
) -> torch.Tensor:
    values = []
    for row in range(teacher_target.shape[0]):
        if not bool(route_supervision_mask.view(-1)[row]):
            continue
        mask = phrase_valid_mask[row].bool()
        if int(mask.sum()) < 2:
            continue
        target = teacher_target[row][mask].float()
        student = student_probability[row][mask].float()
        target_rank = _rank_values(target)
        student_rank = _rank_values(student)
        target_rank = target_rank - target_rank.mean()
        student_rank = student_rank - student_rank.mean()
        denominator = torch.sqrt(
            target_rank.pow(2).sum() * student_rank.pow(2).sum()
            + _EPS
        )
        values.append(
            (target_rank * student_rank).sum() / denominator
        )
    if not values:
        return student_probability.sum() * 0.0
    return torch.stack(values).mean()


def masked_distribution_entropy(
    probability: torch.Tensor,
    phrase_valid_mask: torch.Tensor,
) -> torch.Tensor:
    mask = phrase_valid_mask.bool()
    valid_row = mask.any(dim=1)
    if not bool(valid_row.any()):
        return probability.sum() * 0.0
    p = probability.float().clamp_min(_EPS)
    entropy = -(p * p.log() * mask.to(p.dtype)).sum(dim=1)
    return entropy[valid_row].mean()


def masked_top1_agreement(
    teacher_target: torch.Tensor,
    student_probability: torch.Tensor,
    route_supervision_mask: torch.Tensor,
) -> torch.Tensor:
    valid = route_supervision_mask.view(-1).bool()
    if not bool(valid.any()):
        return student_probability.sum() * 0.0
    match = teacher_target.argmax(dim=1).eq(
        student_probability.argmax(dim=1)
    )
    return match[valid].float().mean()


def aggregate_identity_phrase_route_objectives(
    global_sdm: torch.Tensor,
    global_itc: torch.Tensor,
    local_sdm: torch.Tensor,
    local_itc: torch.Tensor,
    observation_sdm: torch.Tensor,
    observation_itc: torch.Tensor,
    final_sdm: torch.Tensor,
    final_itc: torch.Tensor,
    group_nce: torch.Tensor,
    route_kl: torch.Tensor,
    auxiliary_weight: float,
) -> Dict[str, torch.Tensor]:
    """Exact v16.6.0/v16.7.0 objective.

    L = 0.5*(L_global + L_local)
      + 0.5*L_observation
      + 0.5*L_final
      + lambda*L_group
      + lambda*L_route
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
    phrase_route_loss = float(auxiliary_weight) * route_kl
    observation_objective = observation_sdm + observation_itc
    final_objective = final_sdm + final_itc
    return {
        "sdm_loss": sdm_loss,
        "itc_loss": itc_loss,
        "identity_group_loss": identity_group_loss,
        "phrase_route_loss": phrase_route_loss,
        "anchor_objective": 0.5 * (
            global_sdm + global_itc + local_sdm + local_itc
        ),
        "observation_objective": observation_objective,
        "final_objective": final_objective,
        "balanced_main_objective": (
            _OBSERVATION_MAIN_WEIGHT * observation_objective
            + _FINAL_MAIN_WEIGHT * final_objective
        ),
        "observation_main_weight": observation_sdm.new_tensor(
            _OBSERVATION_MAIN_WEIGHT
        ),
        "final_main_weight": final_sdm.new_tensor(
            _FINAL_MAIN_WEIGHT
        ),
    }


__all__ = [
    "RelativePhraseRouter",
    "ZeroInitializedPhraseIdentityResidual",
    "aggregate_identity_phrase_route_objectives",
    "build_identity_final_embedding",
    "identity_residual_score",
    "itc_from_similarity",
    "masked_distribution_entropy",
    "masked_phrase_softmax",
    "masked_phrase_spearman",
    "masked_top1_agreement",
    "paired_identity_group_nce",
    "phrase_attention_pool",
    "phrase_route_kl_divergence",
    "sdm_from_similarity",
]

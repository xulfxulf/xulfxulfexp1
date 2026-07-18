"""HIRE-v2 v16.4.0 group-conditioned token-routing components.

This first token-routing version deliberately avoids an external MLLM teacher.
It tests the core hypothesis with an online, detached relationship teacher:

- the complete observation remains the current-pair retrieval subject;
- same-PID, different-image supports estimate which selected text tokens are
  discriminative and consistently observable across the identity group;
- only the predicted propagable-token pool is allowed to add a zero-initialized
  residual to the text identity representation;
- non-propagable and unobservable tokens remain available to the complete
  current-pair observation and are only excluded from the new identity residual.

The online target is not presented as a final identity/state annotation.  It is
an immediately trainable precursor to a future MLLM-supervised three-way router.
"""

from __future__ import annotations

from typing import Dict, Tuple

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


def _masked_softmax(
    logits: torch.Tensor,
    mask: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    if logits.shape != mask.shape:
        raise ValueError("logits and mask must have the same shape")
    mask = mask.bool()
    minimum = torch.finfo(logits.dtype).min
    probability = F.softmax(logits.masked_fill(~mask, minimum), dim=dim)
    probability = probability * mask.to(dtype=probability.dtype)
    denominator = probability.sum(dim=dim, keepdim=True).clamp_min(_EPS)
    probability = probability / denominator
    empty = ~mask.any(dim=dim, keepdim=True)
    return torch.where(empty, torch.zeros_like(probability), probability)


def _masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    mask_f = mask.to(dtype=values.dtype)
    while mask_f.ndim < values.ndim:
        mask_f = mask_f.unsqueeze(-1)
    numerator = (values * mask_f).sum(dim=dim)
    denominator = mask_f.sum(dim=dim).clamp_min(1.0)
    return numerator / denominator


def _masked_std(
    values: torch.Tensor,
    mask: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    mean = _masked_mean(values, mask, dim=dim)
    centered = values - mean.unsqueeze(dim)
    mask_f = mask.to(dtype=values.dtype)
    while mask_f.ndim < values.ndim:
        mask_f = mask_f.unsqueeze(-1)
    # Invalid MaxSim slots contain finfo.min. Mask before squaring so padded
    # supports cannot overflow and turn an otherwise valid variance into NaN.
    centered = torch.where(
        mask_f.bool(),
        centered,
        torch.zeros_like(centered),
    )
    variance = centered.pow(2).sum(dim=dim)
    denominator = mask_f.sum(dim=dim).clamp_min(1.0)
    return torch.sqrt(variance / denominator + _EPS)


def masked_standardize(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Standardize within each query over valid selected text tokens."""
    if values.ndim != 2 or mask.shape != values.shape:
        raise ValueError("values and mask must be [B,M]")
    mean = _masked_mean(values, mask, dim=1)
    std = _masked_std(values, mask, dim=1)
    standardized = (values - mean.unsqueeze(1)) / std.unsqueeze(1).clamp_min(
        1e-4
    )
    return torch.where(mask, standardized, torch.zeros_like(standardized))


class AttentionRawTokenSelector(nn.Module):
    """Expose normalized raw CLIP tokens using the existing RDE attention ratio."""

    def __init__(self, ratio: float = 0.3):
        super().__init__()
        if not 0.0 < ratio <= 1.0:
            raise ValueError("token-selection ratio must be in (0,1]")
        self.ratio = float(ratio)

    def select_image(
        self,
        tokens: torch.Tensor,
        attention: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if tokens.ndim != 3 or attention.ndim != 3:
            raise ValueError("image tokens and attention must be rank-3")
        if tokens.shape[:2] != attention.shape[:2]:
            raise ValueError("image token and attention shapes differ")
        patch_count = tokens.shape[1] - 1
        if patch_count < 1:
            raise ValueError("image sequence has no patch tokens")
        count = max(1, min(patch_count, int(patch_count * self.ratio)))
        scores = attention[:, 0, :].detach().float().clone()
        scores[:, 0] = torch.finfo(scores.dtype).min
        indices = scores.topk(
            k=count,
            dim=-1,
            largest=True,
            sorted=True,
        ).indices
        gather = indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
        selected = torch.gather(tokens.detach(), dim=1, index=gather)
        selected = F.normalize(selected.float(), dim=-1)
        mask = torch.ones(
            selected.shape[:2],
            dtype=torch.bool,
            device=selected.device,
        )
        return {
            "tokens": selected,
            "mask": mask,
            "indices": indices,
            "attention": torch.gather(scores, dim=1, index=indices),
        }

    def select_text(
        self,
        tokens: torch.Tensor,
        token_ids: torch.Tensor,
        attention: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if tokens.ndim != 3 or token_ids.ndim != 2 or attention.ndim != 3:
            raise ValueError("text inputs must be [B,L,D], [B,L], [B,L,L]")
        if tokens.shape[:2] != token_ids.shape:
            raise ValueError("text token features and IDs differ")
        if attention.shape[:2] != token_ids.shape:
            raise ValueError("text attention and IDs differ")

        batch_size, sequence_length = token_ids.shape
        eot = token_ids.argmax(dim=-1)
        valid = token_ids.ne(0)
        valid[:, 0] = False
        valid[
            torch.arange(batch_size, device=token_ids.device),
            eot,
        ] = False

        scores = attention[
            torch.arange(batch_size, device=token_ids.device),
            eot,
            :,
        ].detach().float().clone()
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)

        count = max(1, int(max(1, sequence_length - 2) * self.ratio))
        count = min(count, max(1, sequence_length - 2))
        indices = scores.topk(
            k=count,
            dim=-1,
            largest=True,
            sorted=True,
        ).indices
        gather = indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
        selected = torch.gather(tokens.detach(), dim=1, index=gather)
        selected = F.normalize(selected.float(), dim=-1)
        selected_valid = torch.gather(valid, dim=1, index=indices)
        selected_scores = torch.gather(scores, dim=1, index=indices)
        selected = torch.where(
            selected_valid.unsqueeze(-1),
            selected,
            torch.zeros_like(selected),
        )
        weights = _masked_softmax(
            selected_scores,
            selected_valid,
            dim=1,
        )
        return {
            "tokens": selected,
            "mask": selected_valid,
            "indices": indices,
            "attention": selected_scores,
            "weights": weights,
        }


def choose_hard_negative_indices(
    observation_score: torch.Tensor,
    pids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Choose one detached highest-observation different-PID image per query."""
    if observation_score.ndim != 2 or (
        observation_score.shape[0] != observation_score.shape[1]
    ):
        raise ValueError("observation score must be square")
    pids = pids.view(-1)
    negative = pids[:, None].ne(pids[None, :])
    minimum = torch.finfo(observation_score.dtype).min
    masked = observation_score.detach().masked_fill(~negative, minimum)
    valid = negative.any(dim=1)
    indices = masked.argmax(dim=1)
    # A row without a negative is ignored by the route objective.
    indices = torch.where(valid, indices, torch.zeros_like(indices))
    return indices, valid


def token_patch_maxsim(
    text_tokens: torch.Tensor,
    image_tokens: torch.Tensor,
    image_mask: torch.Tensor,
) -> torch.Tensor:
    """MaxSim from each selected text token to each selected image patch."""
    if text_tokens.ndim != 3 or image_tokens.ndim != 3:
        raise ValueError("text and image tokens must be rank-3")
    if text_tokens.shape[0] != image_tokens.shape[0]:
        raise ValueError("text/image batch sizes differ")
    if text_tokens.shape[-1] != image_tokens.shape[-1]:
        raise ValueError("text/image token dimensions differ")
    similarity = torch.einsum(
        "bmd,bnd->bmn",
        text_tokens,
        image_tokens,
    )
    minimum = torch.finfo(similarity.dtype).min
    similarity = similarity.masked_fill(
        ~image_mask[:, None, :].bool(),
        minimum,
    )
    return similarity.max(dim=-1).values


def support_token_patch_maxsim(
    text_tokens: torch.Tensor,
    support_tokens: torch.Tensor,
    support_patch_mask: torch.Tensor,
) -> torch.Tensor:
    """Token-to-patch MaxSim for K same-PID support images: [B,M,K]."""
    if text_tokens.ndim != 3 or support_tokens.ndim != 4:
        raise ValueError("expected text [B,M,D] and support [B,K,N,D]")
    if text_tokens.shape[0] != support_tokens.shape[0]:
        raise ValueError("text/support batch sizes differ")
    if text_tokens.shape[-1] != support_tokens.shape[-1]:
        raise ValueError("text/support token dimensions differ")
    similarity = torch.einsum(
        "bmd,bknd->bmkn",
        text_tokens,
        support_tokens,
    )
    minimum = torch.finfo(similarity.dtype).min
    similarity = similarity.masked_fill(
        ~support_patch_mask[:, None, :, :].bool(),
        minimum,
    )
    return similarity.max(dim=-1).values


def build_group_propagability_targets(
    text_pack: Dict[str, torch.Tensor],
    anchor_image_pack: Dict[str, torch.Tensor],
    support_image_pack: Dict[str, torch.Tensor],
    support_mask: torch.Tensor,
    hard_negative_image_pack: Dict[str, torch.Tensor],
    hard_negative_valid: torch.Tensor,
    minimum_supports: int = 2,
) -> Dict[str, torch.Tensor]:
    """Build detached soft identity-propagability targets.

    A token receives a high target only when:
    1. it is relevant to the current anchor relative to the hardest different
       identity in the random batch; and
    2. it remains discriminative and consistent over same-PID support images.

    The target is query-relative and threshold-free:
        stable = mean(support MaxSim) - std(support MaxSim) - hard-negative MaxSim
        pair   = anchor MaxSim - hard-negative MaxSim
        q_id   = sigmoid(zscore(stable)) * sigmoid(zscore(pair))

    State and unknown are intentionally merged into the non-identity remainder
    in v16.4.0.  The complete pair branch continues to use every reliable token.
    """
    text_tokens = text_pack["tokens"].detach()
    text_mask = text_pack["mask"].bool()
    support_mask = support_mask.bool()
    if support_mask.ndim != 2:
        raise ValueError("support mask must be [B,K]")

    anchor_score = token_patch_maxsim(
        text_tokens,
        anchor_image_pack["tokens"].detach(),
        anchor_image_pack["mask"],
    )
    hard_negative_score = token_patch_maxsim(
        text_tokens,
        hard_negative_image_pack["tokens"].detach(),
        hard_negative_image_pack["mask"],
    )
    support_score = support_token_patch_maxsim(
        text_tokens,
        support_image_pack["tokens"].detach(),
        support_image_pack["mask"],
    )

    support_valid = support_mask[:, None, :].expand_as(support_score)
    support_count = support_mask.sum(dim=1)
    group_valid = support_count.ge(int(minimum_supports))
    support_mean = _masked_mean(
        support_score,
        support_valid,
        dim=2,
    )
    support_std = _masked_std(
        support_score,
        support_valid,
        dim=2,
    )

    stable_margin = support_mean - support_std - hard_negative_score
    pair_margin = anchor_score - hard_negative_score
    stable_z = masked_standardize(stable_margin, text_mask)
    pair_z = masked_standardize(pair_margin, text_mask)
    target = torch.sigmoid(stable_z) * torch.sigmoid(pair_z)

    valid = (
        text_mask
        & group_valid.unsqueeze(1)
        & hard_negative_valid.unsqueeze(1)
    )
    target = torch.where(valid, target, torch.zeros_like(target)).detach()
    return {
        "target": target,
        "valid": valid,
        "anchor_score": anchor_score.detach(),
        "hard_negative_score": hard_negative_score.detach(),
        "support_mean": support_mean.detach(),
        "support_std": support_std.detach(),
        "stable_margin": stable_margin.detach(),
        "pair_margin": pair_margin.detach(),
        "group_valid": group_valid,
        "support_count": support_count,
    }


class TokenPropagabilityRouter(nn.Module):
    """Predict one identity-propagability probability for each selected word."""

    def __init__(self, dim: int):
        super().__init__()
        hidden = max(1, int(dim) // 4)
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        # Initial prediction is exactly 0.5 for every valid token.
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(
        self,
        token_features: torch.Tensor,
        text_observation: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if token_features.ndim != 3:
            raise ValueError("token features must be [B,M,D]")
        if text_observation.shape != (
            token_features.shape[0],
            token_features.shape[2],
        ):
            raise ValueError("text observation is incompatible with tokens")
        if token_mask.shape != token_features.shape[:2]:
            raise ValueError("token mask is incompatible with tokens")
        contextual = self.norm(
            token_features.detach()
            + text_observation.detach().unsqueeze(1)
        )
        hidden = F.gelu(self.fc1(contextual))
        logits = self.fc2(hidden).squeeze(-1)
        probability = torch.sigmoid(logits)
        probability = torch.where(
            token_mask.bool(),
            probability,
            torch.zeros_like(probability),
        )
        return {
            "logits": logits,
            "probability": probability,
        }


class ZeroInitializedIdentityTokenResidual(nn.Module):
    """Inject a routed text-token pool into the text identity representation."""

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(
        self,
        base_identity_raw: torch.Tensor,
        token_features: torch.Tensor,
        token_attention: torch.Tensor,
        identity_probability: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if base_identity_raw.ndim != 2 or token_features.ndim != 3:
            raise ValueError("identity raw must be [B,D], tokens [B,M,D]")
        if token_attention.shape != token_mask.shape:
            raise ValueError("token attention and mask must match")
        if identity_probability.shape != token_mask.shape:
            raise ValueError("identity probability and mask must match")
        weight = (
            token_attention
            * identity_probability
            * token_mask.to(dtype=token_attention.dtype)
        )
        denominator = weight.sum(dim=1, keepdim=True).clamp_min(_EPS)
        pooled = (
            token_features.detach()
            * weight.unsqueeze(-1)
        ).sum(dim=1) / denominator
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


def token_route_binary_cross_entropy(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape or valid.shape != target.shape:
        raise ValueError("route prediction, target, and mask must match")
    valid = valid.bool()
    if not bool(valid.any()):
        return prediction.sum() * 0.0
    return F.binary_cross_entropy(
        prediction[valid],
        target[valid],
    )


def aggregate_identity_token_route_objectives(
    global_sdm: torch.Tensor,
    global_itc: torch.Tensor,
    local_sdm: torch.Tensor,
    local_itc: torch.Tensor,
    observation_sdm: torch.Tensor,
    observation_itc: torch.Tensor,
    final_sdm: torch.Tensor,
    final_itc: torch.Tensor,
    group_nce: torch.Tensor,
    route_bce: torch.Tensor,
    auxiliary_weight: float,
) -> Dict[str, torch.Tensor]:
    """Aggregate the exact v16.4.0 objective.

    L = 0.5*(L_global + L_local)
      + 0.5*L_observation
      + 0.5*L_final
      + lambda*L_group
      + lambda*L_route

    The retrieval path is exactly v16.2.1 at initialization because the routed
    token residual adapter is zero initialized.
    """
    if auxiliary_weight < 0.0:
        raise ValueError("auxiliary weight must be non-negative")
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
    token_route_loss = float(auxiliary_weight) * route_bce
    anchor_objective = 0.5 * (
        global_sdm + global_itc + local_sdm + local_itc
    )
    observation_objective = observation_sdm + observation_itc
    final_objective = final_sdm + final_itc
    return {
        "sdm_loss": sdm_loss,
        "itc_loss": itc_loss,
        "identity_group_loss": identity_group_loss,
        "token_route_loss": token_route_loss,
        "anchor_objective": anchor_objective,
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


def masked_correlation(
    left: torch.Tensor,
    right: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    valid = mask.bool()
    if int(valid.sum()) < 2:
        return left.sum() * 0.0
    x = left[valid].float()
    y = right[valid].float()
    x = x - x.mean()
    y = y - y.mean()
    denominator = (
        torch.sqrt(x.pow(2).sum() + _EPS)
        * torch.sqrt(y.pow(2).sum() + _EPS)
    )
    return (x * y).sum() / denominator


__all__ = [
    "AttentionRawTokenSelector",
    "TokenPropagabilityRouter",
    "ZeroInitializedIdentityTokenResidual",
    "aggregate_identity_token_route_objectives",
    "build_group_propagability_targets",
    "build_identity_final_embedding",
    "choose_hard_negative_indices",
    "identity_residual_score",
    "itc_from_similarity",
    "masked_correlation",
    "paired_identity_group_nce",
    "sdm_from_similarity",
    "token_route_binary_cross_entropy",
]

"""HIRE-v2 v16.3.0 state late-interaction components.

v16.3.0 keeps the complete v16.2.1 identity-balanced path unchanged and adds a
pair-conditioned state residual.  The state score is not a second global
embedding: selected text words interact with selected image patches only for
the top-K candidates produced by the identity-balanced base score.

The three new method settings are:
- number of selected text state tokens;
- number of selected image state patches;
- number of base candidates reranked by the state branch.

The state projection dimension is derived deterministically as one quarter of
the CLIP embedding dimension and is not exposed as a search parameter.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hire_v2_identity_balanced_components import (
    aggregate_identity_balanced_objectives,
)
from .hire_v2_identity_components import (
    itc_from_similarity,
    sdm_from_similarity,
)


_EPS = 1e-8
_OBSERVATION_MAIN_WEIGHT = 0.5
_IDENTITY_MAIN_WEIGHT = 0.25
_STATE_FINAL_MAIN_WEIGHT = 0.25


def _masked_softmax(
    values: torch.Tensor,
    mask: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    """Numerically stable softmax with exactly zero invalid positions."""
    if values.shape != mask.shape:
        raise ValueError("values and mask must have identical shapes")
    mask = mask.bool()
    negative = torch.finfo(values.dtype).min
    masked = values.masked_fill(~mask, negative)
    probability = F.softmax(masked, dim=dim)
    probability = probability * mask.to(dtype=values.dtype)
    denominator = probability.sum(dim=dim, keepdim=True).clamp_min(_EPS)
    probability = probability / denominator
    empty = ~mask.any(dim=dim, keepdim=True)
    return torch.where(empty, torch.zeros_like(probability), probability)


class SharedStateTokenProjection(nn.Module):
    """Shared image/text token projection for pair-conditioned state evidence.

    A shared map preserves the CLIP cross-modal axes.  Inputs are detached by
    the model so the state auxiliary cannot rewrite the strong observation or
    identity paths.  The rectangular projection is initialized with orthogonal
    rows for a meaningful, non-collapsed initial token geometry.
    """

    def __init__(self, input_dim: int, output_dim: Optional[int] = None):
        super().__init__()
        if input_dim < 4:
            raise ValueError("input_dim must be at least four")
        if output_dim is None:
            output_dim = max(1, input_dim // 4)
        if output_dim < 1 or output_dim > input_dim:
            raise ValueError("state output_dim must lie in [1, input_dim]")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.proj = nn.Linear(self.input_dim, self.output_dim, bias=False)
        nn.init.orthogonal_(self.proj.weight)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[-1] != self.input_dim:
            raise ValueError("state tokens must have shape [B,L,input_dim]")
        return F.normalize(self.proj(tokens.float()), dim=-1)


class AttentionStateTokenEncoder(nn.Module):
    """Select state words/patches from final CLIP attention and project them."""

    def __init__(
        self,
        input_dim: int,
        image_token_count: int = 16,
        text_token_count: int = 8,
        output_dim: Optional[int] = None,
    ):
        super().__init__()
        if image_token_count < 2:
            raise ValueError("state image token count must be at least two")
        if text_token_count < 1:
            raise ValueError("state text token count must be positive")
        self.image_token_count = int(image_token_count)
        self.text_token_count = int(text_token_count)
        self.projection = SharedStateTokenProjection(
            input_dim=input_dim,
            output_dim=output_dim,
        )

    def encode_image(
        self,
        tokens: torch.Tensor,
        attention: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Select top-attended image patches; CLS never enters state matching."""
        if tokens.ndim != 3 or attention.ndim != 3:
            raise ValueError("image tokens and attention must be rank-3")
        if tokens.shape[0] != attention.shape[0]:
            raise ValueError("image token/attention batch sizes differ")
        patch_count = tokens.shape[1] - 1
        if patch_count < 2:
            raise ValueError("state image branch requires at least two patches")
        count = min(self.image_token_count, patch_count)
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
        projected = self.projection(selected)
        mask = torch.ones(
            projected.shape[:2],
            dtype=torch.bool,
            device=projected.device,
        )
        attention_score = torch.gather(scores, dim=1, index=indices)
        return {
            "tokens": projected,
            "mask": mask,
            "attention": attention_score,
            "indices": indices,
        }

    def encode_text(
        self,
        tokens: torch.Tensor,
        token_ids: torch.Tensor,
        attention: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Select EOT-attended valid words and produce normalized word weights."""
        if tokens.ndim != 3 or token_ids.ndim != 2 or attention.ndim != 3:
            raise ValueError("text inputs must be [B,L,D], [B,L], [B,L,L]")
        if tokens.shape[:2] != token_ids.shape:
            raise ValueError("text token features and IDs have incompatible shapes")
        if attention.shape[:2] != token_ids.shape:
            raise ValueError("text attention and IDs have incompatible shapes")

        batch_size, sequence_length = token_ids.shape
        eot_indices = token_ids.argmax(dim=-1)
        valid = token_ids.ne(0)
        valid[:, 0] = False
        valid[
            torch.arange(batch_size, device=token_ids.device),
            eot_indices,
        ] = False

        scores = attention[
            torch.arange(batch_size, device=token_ids.device),
            eot_indices,
            :,
        ].detach().float().clone()
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        count = min(self.text_token_count, max(1, sequence_length - 2))
        indices = scores.topk(
            k=count,
            dim=-1,
            largest=True,
            sorted=True,
        ).indices
        gather = indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
        selected = torch.gather(tokens.detach(), dim=1, index=gather)
        selected_valid = torch.gather(valid, dim=1, index=indices)
        selected_scores = torch.gather(scores, dim=1, index=indices)
        projected = self.projection(selected)
        projected = torch.where(
            selected_valid.unsqueeze(-1),
            projected,
            torch.zeros_like(projected),
        )
        weights = _masked_softmax(
            selected_scores,
            selected_valid,
            dim=1,
        )
        return {
            "tokens": projected,
            "mask": selected_valid,
            "weights": weights,
            "attention": selected_scores,
            "indices": indices,
        }


class SignedBoundedStateGate(nn.Module):
    """Exactly-zero initialized, bounded state residual scale.

    ``tanh(0) == 0`` gives exact v16.2.1 behavior at initialization while
    preserving a non-zero derivative.  The state-pair auxiliary defines the
    positive score direction; a successful run is expected to learn a positive
    gate, but the signed parameter avoids a non-differentiable positivity clamp.
    """

    def __init__(self):
        super().__init__()
        self.raw = nn.Parameter(torch.zeros([], dtype=torch.float32))

    def forward(self) -> torch.Tensor:
        return torch.tanh(self.raw)


def build_state_candidate_indices(
    base_score: torch.Tensor,
    image_ids: torch.Tensor,
    topk: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select base top-K candidates and force all same-image positives in train."""
    if base_score.ndim != 2 or base_score.shape[0] != base_score.shape[1]:
        raise ValueError("training base score must be square [B,B]")
    image_ids = image_ids.view(-1)
    if image_ids.numel() != base_score.shape[0]:
        raise ValueError("image_ids do not match base score")
    if topk < 1:
        raise ValueError("state topk must be positive")

    batch_size = base_score.shape[0]
    k = min(int(topk), batch_size)
    positive = image_ids[:, None].eq(image_ids[None, :])
    # Cosine values lie in [-1, 1]; adding four safely prioritizes positives.
    priority = base_score.detach() + positive.to(base_score.dtype) * 4.0
    indices = priority.topk(k=k, dim=1, largest=True, sorted=True).indices
    mask = torch.zeros_like(base_score, dtype=torch.bool)
    mask.scatter_(1, indices, True)
    positive_covered = (mask & positive).sum(dim=1).eq(positive.sum(dim=1))
    if not bool(positive_covered.all()):
        raise RuntimeError("state candidate selection failed to include positives")
    return indices, mask, positive


def selected_state_late_interaction(
    text_pack: Dict[str, torch.Tensor],
    image_pack: Dict[str, torch.Tensor],
    candidate_indices: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Compute text-word to candidate-image-patch weighted MaxSim.

    Args:
        text_pack:
            tokens [Q,M,D], mask [Q,M], weights [Q,M].
        image_pack:
            tokens [G,N,D], mask [G,N].
        candidate_indices:
            [Q,K] gallery rows selected for each query.

    Returns:
        score [Q,K];
        peak_mean and peak_margin diagnostics.
    """
    text_tokens = text_pack["tokens"]
    text_mask = text_pack["mask"].bool()
    text_weights = text_pack["weights"]
    image_tokens = image_pack["tokens"]
    image_mask = image_pack["mask"].bool()

    if text_tokens.ndim != 3 or image_tokens.ndim != 3:
        raise ValueError("state token packs must be rank-3")
    if text_tokens.shape[0] != candidate_indices.shape[0]:
        raise ValueError("query and candidate-index batch sizes differ")
    if text_tokens.shape[-1] != image_tokens.shape[-1]:
        raise ValueError("text/image state token dimensions differ")

    query_count, candidate_count = candidate_indices.shape
    flat = candidate_indices.reshape(-1)
    selected_image_tokens = image_tokens.index_select(0, flat).reshape(
        query_count,
        candidate_count,
        image_tokens.shape[1],
        image_tokens.shape[2],
    )
    selected_image_mask = image_mask.index_select(0, flat).reshape(
        query_count,
        candidate_count,
        image_mask.shape[1],
    )

    similarity = torch.einsum(
        "qmd,qknd->qkmn",
        text_tokens,
        selected_image_tokens,
    )
    negative = torch.finfo(similarity.dtype).min
    similarity = similarity.masked_fill(
        ~selected_image_mask[:, :, None, :],
        negative,
    )
    top_values = similarity.topk(
        k=min(2, similarity.shape[-1]),
        dim=-1,
        largest=True,
        sorted=True,
    ).values
    peak = top_values[..., 0]
    if top_values.shape[-1] == 1:
        margin = torch.zeros_like(peak)
    else:
        margin = peak - top_values[..., 1]

    valid_word = text_mask[:, None, :]
    peak = torch.where(valid_word, peak, torch.zeros_like(peak))
    margin = torch.where(valid_word, margin, torch.zeros_like(margin))
    weight = text_weights[:, None, :] * valid_word.to(text_weights.dtype)
    denominator = weight.sum(dim=-1).clamp_min(_EPS)
    score = (peak * weight).sum(dim=-1) / denominator
    peak_mean = (peak * weight).sum(dim=-1) / denominator
    margin_mean = (margin * weight).sum(dim=-1) / denominator
    return {
        "score": score,
        "peak_mean": peak_mean,
        "margin_mean": margin_mean,
    }


def scatter_selected_state_scores(
    base_score: torch.Tensor,
    selected_score: torch.Tensor,
    candidate_indices: torch.Tensor,
) -> torch.Tensor:
    """Create a full score matrix whose non-candidate residual is exactly zero."""
    if selected_score.shape != candidate_indices.shape:
        raise ValueError("selected score and candidate indices must match")
    full = base_score.detach().clone()
    full.scatter_(1, candidate_indices, selected_score)
    return full


def masked_multi_positive_nce(
    scores: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    candidate_mask: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """Symmetric state NCE.

    Positives are same-image pairs.  Negatives are different-PID pairs.
    Same-PID/different-image pairs are ignored.  Only base top-K candidates
    participate.  The same definition is applied to text-to-image and to the
    transposed image-to-text direction.
    """
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("state scores must be square [B,B]")
    if positive.shape != scores.shape or negative.shape != scores.shape:
        raise ValueError("state relation masks must match scores")
    if candidate_mask.shape != scores.shape:
        raise ValueError("candidate mask must match scores")

    def one_direction(
        directional_scores: torch.Tensor,
        directional_positive: torch.Tensor,
        directional_negative: torch.Tensor,
        directional_candidates: torch.Tensor,
    ) -> torch.Tensor:
        positive_mask = directional_positive & directional_candidates
        negative_mask = directional_negative & directional_candidates
        allowed = positive_mask | negative_mask
        valid = positive_mask.any(dim=1) & negative_mask.any(dim=1)
        if not bool(valid.any()):
            return directional_scores.sum() * 0.0

        logits = directional_scores * logit_scale.to(
            device=directional_scores.device,
            dtype=directional_scores.dtype,
        )
        minimum = torch.finfo(logits.dtype).min
        allowed_logits = logits.masked_fill(~allowed, minimum)
        positive_logits = logits.masked_fill(~positive_mask, minimum)
        denominator = torch.logsumexp(allowed_logits, dim=1)
        numerator = torch.logsumexp(positive_logits, dim=1)
        return -(numerator[valid] - denominator[valid]).mean()

    t2i = one_direction(scores, positive, negative, candidate_mask)
    i2t = one_direction(
        scores.t(),
        positive.t(),
        negative.t(),
        candidate_mask.t(),
    )
    return 0.5 * (t2i + i2t)


def state_pair_nce(
    scores: torch.Tensor,
    pids: torch.Tensor,
    image_ids: torch.Tensor,
    candidate_mask: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    pids = pids.view(-1)
    image_ids = image_ids.view(-1)
    positive = image_ids[:, None].eq(image_ids[None, :])
    negative = pids[:, None].ne(pids[None, :])
    return masked_multi_positive_nce(
        scores=scores,
        positive=positive,
        negative=negative,
        candidate_mask=candidate_mask,
        logit_scale=logit_scale,
    )


def state_residual_score(
    identity_base_score: torch.Tensor,
    state_score: torch.Tensor,
    candidate_mask: torch.Tensor,
    state_gate: torch.Tensor,
) -> torch.Tensor:
    """Apply a top-K pair-conditioned residual to the v16.2.1 base score."""
    if identity_base_score.shape != state_score.shape:
        raise ValueError("base and state score matrices must match")
    if candidate_mask.shape != identity_base_score.shape:
        raise ValueError("candidate mask must match score matrices")
    residual = state_score - identity_base_score.detach()
    return identity_base_score + (
        candidate_mask.to(identity_base_score.dtype)
        * state_gate.to(
            device=identity_base_score.device,
            dtype=identity_base_score.dtype,
        )
        * residual
    )


def aggregate_identity_state_objectives(
    global_sdm: torch.Tensor,
    global_itc: torch.Tensor,
    local_sdm: torch.Tensor,
    local_itc: torch.Tensor,
    observation_sdm: torch.Tensor,
    observation_itc: torch.Tensor,
    identity_final_sdm: torch.Tensor,
    identity_final_itc: torch.Tensor,
    state_final_sdm: torch.Tensor,
    state_final_itc: torch.Tensor,
    identity_group_nce: torch.Tensor,
    state_nce: torch.Tensor,
    auxiliary_weight: float,
) -> Dict[str, torch.Tensor]:
    """Aggregate the exact v16.3.0 objective.

    L_anchor = 0.5 * (L_global + L_local)
    L_main   = 0.5 * L_observation
             + 0.25 * L_identity_final
             + 0.25 * L_state_final
    L_aux    = lambda * L_identity_group + lambda * L_state_pair

    At state-gate initialization, state_final == identity_final, so L_main
    exactly equals the v16.2.1 main retrieval objective.
    """
    if auxiliary_weight < 0.0:
        raise ValueError("auxiliary_weight must be non-negative")

    sdm_loss = (
        0.5 * (global_sdm + local_sdm)
        + _OBSERVATION_MAIN_WEIGHT * observation_sdm
        + _IDENTITY_MAIN_WEIGHT * identity_final_sdm
        + _STATE_FINAL_MAIN_WEIGHT * state_final_sdm
    )
    itc_loss = (
        0.5 * (global_itc + local_itc)
        + _OBSERVATION_MAIN_WEIGHT * observation_itc
        + _IDENTITY_MAIN_WEIGHT * identity_final_itc
        + _STATE_FINAL_MAIN_WEIGHT * state_final_itc
    )
    identity_group_loss = float(auxiliary_weight) * identity_group_nce
    state_pair_loss = float(auxiliary_weight) * state_nce

    anchor_objective = 0.5 * (
        global_sdm + global_itc + local_sdm + local_itc
    )
    observation_objective = observation_sdm + observation_itc
    identity_final_objective = identity_final_sdm + identity_final_itc
    state_final_objective = state_final_sdm + state_final_itc
    hierarchical_main_objective = (
        _OBSERVATION_MAIN_WEIGHT * observation_objective
        + _IDENTITY_MAIN_WEIGHT * identity_final_objective
        + _STATE_FINAL_MAIN_WEIGHT * state_final_objective
    )
    return {
        "sdm_loss": sdm_loss,
        "itc_loss": itc_loss,
        "identity_group_loss": identity_group_loss,
        "state_pair_loss": state_pair_loss,
        "anchor_objective": anchor_objective,
        "observation_objective": observation_objective,
        "identity_final_objective": identity_final_objective,
        "state_final_objective": state_final_objective,
        "hierarchical_main_objective": hierarchical_main_objective,
        "observation_main_weight": observation_sdm.new_tensor(
            _OBSERVATION_MAIN_WEIGHT
        ),
        "identity_main_weight": identity_final_sdm.new_tensor(
            _IDENTITY_MAIN_WEIGHT
        ),
        "state_final_main_weight": state_final_sdm.new_tensor(
            _STATE_FINAL_MAIN_WEIGHT
        ),
    }


__all__ = [
    "AttentionStateTokenEncoder",
    "SharedStateTokenProjection",
    "SignedBoundedStateGate",
    "aggregate_identity_state_objectives",
    "build_state_candidate_indices",
    "itc_from_similarity",
    "masked_multi_positive_nce",
    "scatter_selected_state_scores",
    "sdm_from_similarity",
    "selected_state_late_interaction",
    "state_pair_nce",
    "state_residual_score",
]

"""Core components for HIRE.

HIRE = Heterogeneity-aware Identity Random Effects.

This module intentionally depends only on PyTorch so it can be unit-tested
without loading CLIP checkpoints or datasets.  The token-selection layers are
adapted from the public RDE implementation, while the random-effects posterior,
probabilistic matching, state residual, and loss functions are specific to this
project.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


_EPS = 1e-8


def _identity_init(linear: nn.Linear) -> None:
    """Initialize a square linear layer as identity, otherwise Xavier."""
    if linear.weight.shape[0] == linear.weight.shape[1]:
        nn.init.eye_(linear.weight)
    else:
        nn.init.xavier_uniform_(linear.weight)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(dtype=values.dtype)
    while mask_f.ndim < values.ndim:
        mask_f = mask_f.unsqueeze(-1)
    denominator = mask_f.sum().clamp_min(1.0)
    return (values * mask_f).sum() / denominator


class TokenMLP(nn.Module):
    """Two-layer token MLP following the public RDE TSE implementation."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, dim = x.shape
        x = x.reshape(batch * length, dim)
        x = F.relu(self.bn1(self.fc1(x)), inplace=False)
        x = self.bn2(self.fc2(x))
        return x.reshape(batch, length, -1)


def _masked_token_max(features: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Max-pool tokens while ignoring invalid positions."""
    if features.ndim != 3 or valid_mask.ndim != 2:
        raise ValueError("features must be [B,L,D] and valid_mask [B,L]")
    if features.shape[:2] != valid_mask.shape:
        raise ValueError("features and valid_mask have incompatible shapes")
    neg = torch.finfo(features.dtype).min
    pooled = features.masked_fill(~valid_mask.unsqueeze(-1), neg).max(dim=1).values
    empty = ~valid_mask.any(dim=1)
    if empty.any():
        pooled = torch.where(empty.unsqueeze(-1), torch.zeros_like(pooled), pooled)
    return pooled


class RDEVisualTokenSelection(nn.Module):
    """RDE-style attention-guided visual token selection and pooling.

    Input tokens are the final projected CLIP visual tokens.  Attention is the
    final self-attention matrix averaged over heads, shaped [B,L,L].
    """

    def __init__(self, input_dim: int = 512, output_dim: int = 1024, ratio: float = 0.3):
        super().__init__()
        if not 0 < ratio <= 1:
            raise ValueError("ratio must be in (0, 1]")
        self.ratio = float(ratio)
        self.skip = nn.Linear(input_dim, output_dim)
        self.mlp = TokenMLP(input_dim, output_dim // 2, output_dim)

    def forward(self, tokens: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or attention.ndim != 3:
            raise ValueError("visual tokens and attention must be rank-3 tensors")
        if attention.shape[0] != tokens.shape[0] or attention.shape[1] != tokens.shape[1]:
            raise ValueError("visual tokens and attention have incompatible shapes")
        patch_count = tokens.shape[1] - 1
        if patch_count < 1:
            raise ValueError("visual sequence has no patch tokens")
        k = max(1, min(patch_count, int(patch_count * self.ratio)))

        scores = attention[:, 0, :].detach().clone()
        scores[:, 0] = torch.finfo(scores.dtype).min
        indices = scores.topk(k=k, dim=-1, largest=True, sorted=False).indices
        gather_index = indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
        selected = torch.gather(tokens, dim=1, index=gather_index)
        selected = F.normalize(selected.float(), dim=-1)
        transformed = self.mlp(selected) + self.skip(selected)
        valid = torch.ones(selected.shape[:2], dtype=torch.bool, device=selected.device)
        return _masked_token_max(transformed, valid)


class RDETextTokenSelection(nn.Module):
    """RDE-style attention-guided text token selection and pooling."""

    def __init__(self, input_dim: int = 512, output_dim: int = 1024, ratio: float = 0.3):
        super().__init__()
        if not 0 < ratio <= 1:
            raise ValueError("ratio must be in (0, 1]")
        self.ratio = float(ratio)
        self.skip = nn.Linear(input_dim, output_dim)
        self.mlp = TokenMLP(input_dim, output_dim // 2, output_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        token_ids: torch.Tensor,
        attention: torch.Tensor,
    ) -> torch.Tensor:
        if tokens.ndim != 3 or token_ids.ndim != 2 or attention.ndim != 3:
            raise ValueError("text inputs must be [B,L,D], [B,L], and [B,L,L]")
        if tokens.shape[:2] != token_ids.shape or attention.shape[:2] != token_ids.shape:
            raise ValueError("text tokens, ids, and attention have incompatible shapes")

        batch, length = token_ids.shape
        eot = token_ids.argmax(dim=-1)
        valid_tokens = token_ids.ne(0)
        valid_tokens[:, 0] = False
        valid_tokens[torch.arange(batch, device=token_ids.device), eot] = False

        # The EOT query summarizes the sentence in CLIP's causal transformer.
        scores = attention[torch.arange(batch, device=token_ids.device), eot, :].detach().clone()
        scores = scores.masked_fill(~valid_tokens, torch.finfo(scores.dtype).min)

        global_k = max(1, int(max(1, length - 2) * self.ratio))
        global_k = min(global_k, max(1, length - 2))
        indices = scores.topk(k=global_k, dim=-1, largest=True, sorted=False).indices
        gather_index = indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
        selected = torch.gather(tokens, dim=1, index=gather_index)
        selected = F.normalize(selected.float(), dim=-1)

        # A row can contain fewer valid words than global_k.  Gather the actual
        # validity of every selected index instead of assuming top-k ordering.
        selected_valid = torch.gather(valid_tokens, dim=1, index=indices)
        transformed = self.mlp(selected) + self.skip(selected)
        return _masked_token_max(transformed, selected_valid)


class ObservationFusion(nn.Module):
    """Fuse CLIP global and RDE fine-grained observations.

    The local projection is zero-initialized, so the initial observation is
    exactly the strong global CLIP representation.  Its learnable gate and
    projection then introduce fine-grained evidence end-to-end.
    """

    def __init__(self, global_dim: int, local_dim: int, output_dim: int):
        super().__init__()
        self.global_proj = nn.Linear(global_dim, output_dim, bias=False)
        self.local_proj = nn.Linear(local_dim, output_dim, bias=False)
        self.norm = nn.LayerNorm(output_dim)
        _identity_init(self.global_proj)
        nn.init.zeros_(self.local_proj.weight)
        self.local_gate_logit = nn.Parameter(torch.tensor(0.0))

    def forward(self, global_feature: torch.Tensor, local_feature: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.local_gate_logit)
        return self.norm(self.global_proj(global_feature.float()) + gate * self.local_proj(local_feature.float()))


class IdentityPosteriorHead(nn.Module):
    """Predict a diagonal-Gaussian identity observation posterior."""

    def __init__(self, input_dim: int, identity_dim: int, variance_floor: float = 1e-6):
        super().__init__()
        self.mean_proj = nn.Linear(input_dim, identity_dim, bias=False)
        self.mean_norm = nn.LayerNorm(identity_dim)
        self.variance_proj = nn.Linear(input_dim, identity_dim)
        self.variance_floor = float(variance_floor)
        _identity_init(self.mean_proj)
        nn.init.zeros_(self.variance_proj.weight)
        # softplus(0.5413) ~= 1.0, therefore all dimensions initially have
        # unit variance instead of arbitrary confidence.
        nn.init.constant_(self.variance_proj.bias, math.log(math.expm1(1.0)))

    def forward(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean_norm(self.mean_proj(observation.float()))
        # The uncertainty head must not distort the shared encoder merely to
        # make its variance prediction easy.
        variance = F.softplus(self.variance_proj(observation.detach().float())) + self.variance_floor
        return mean, variance


class StateResidualHead(nn.Module):
    """Predict the instance-level state residual in the shared latent space."""

    def __init__(self, input_dim: int, state_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, state_dim, bias=False)
        self.norm = nn.LayerNorm(state_dim)
        _identity_init(self.proj)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.norm(self.proj(observation.float())), dim=-1)


def heterogeneity_aware_posterior(
    means: torch.Tensor,
    variances: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
    detach_initial_center: bool = True,
) -> Dict[str, torch.Tensor]:
    """Fuse grouped observations using a random-effects trusted intersection.

    Args:
        means: [B,S,D] per-observation posterior means.
        variances: [B,S,D] positive diagonal variances.
        mask: [B,S] valid support observations.

    Returns a dictionary containing the fused mean, fused variance, within-group
    heterogeneity tau^2, effective precision, and row-validity mask.
    """
    if means.shape != variances.shape or means.ndim != 3:
        raise ValueError("means and variances must have shape [B,S,D]")
    if mask.shape != means.shape[:2]:
        raise ValueError("mask must have shape [B,S]")
    if (variances <= 0).any():
        raise ValueError("variances must be positive")

    mask_f = mask.to(dtype=means.dtype).unsqueeze(-1)
    valid = mask.any(dim=1)

    precision0 = mask_f / (variances + eps)
    precision0_sum = precision0.sum(dim=1).clamp_min(eps)
    center0 = (precision0 * means).sum(dim=1) / precision0_sum
    center_for_tau = center0.detach() if detach_initial_center else center0
    tau2 = (
        precision0 * (means - center_for_tau.unsqueeze(1)).pow(2)
    ).sum(dim=1) / precision0_sum

    effective_precision = mask_f / (variances + tau2.unsqueeze(1) + eps)
    precision_sum = effective_precision.sum(dim=1).clamp_min(eps)
    group_mean = (effective_precision * means).sum(dim=1) / precision_sum
    group_variance = 1.0 / precision_sum + tau2

    # Empty rows are possible for single-image identities.  Keep them finite;
    # callers must use valid to exclude them from posterior-supervised losses.
    group_mean = torch.where(valid.unsqueeze(-1), group_mean, torch.zeros_like(group_mean))
    group_variance = torch.where(valid.unsqueeze(-1), group_variance, torch.ones_like(group_variance))
    tau2 = torch.where(valid.unsqueeze(-1), tau2, torch.zeros_like(tau2))

    return {
        "mean": group_mean,
        "variance": group_variance.clamp_min(eps),
        "tau2": tau2.clamp_min(0.0),
        "precision": effective_precision,
        "valid": valid,
    }


def gaussian_pairwise_score(
    query_mean: torch.Tensor,
    query_variance: torch.Tensor,
    target_mean: torch.Tensor,
    target_variance: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Mutual-likelihood-style score between diagonal Gaussian posteriors."""
    if query_mean.shape != query_variance.shape or target_mean.shape != target_variance.shape:
        raise ValueError("mean and variance shapes must match within each side")
    if query_mean.ndim != 2 or target_mean.ndim != 2:
        raise ValueError("posterior tensors must be [N,D] and [M,D]")
    if query_mean.shape[-1] != target_mean.shape[-1]:
        raise ValueError("query and target dimensions differ")

    variance_sum = query_variance[:, None, :] + target_variance[None, :, :] + eps
    squared = (query_mean[:, None, :] - target_mean[None, :, :]).pow(2)
    distance = 0.5 * (squared / variance_sum + torch.log(variance_sum)).mean(dim=-1)
    return -distance


def symmetric_multi_positive_nce(
    scores: torch.Tensor,
    pids: torch.Tensor,
    temperature: float,
    row_valid: Optional[torch.Tensor] = None,
    column_valid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Symmetric multi-positive NCE with PID-defined positives."""
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("scores must be a square [B,B] matrix")
    pids = pids.view(-1)
    if pids.numel() != scores.shape[0]:
        raise ValueError("PID count does not match score matrix")
    if row_valid is None:
        row_valid = torch.ones(scores.shape[0], dtype=torch.bool, device=scores.device)
    if column_valid is None:
        column_valid = torch.ones(scores.shape[1], dtype=torch.bool, device=scores.device)
    positive = pids[:, None].eq(pids[None, :])
    valid_matrix = row_valid[:, None] & column_valid[None, :]
    positive = positive & valid_matrix
    logits = scores / float(temperature)
    neg_inf = torch.finfo(logits.dtype).min

    def direction(
        direction_logits: torch.Tensor,
        direction_pos: torch.Tensor,
        direction_valid_matrix: torch.Tensor,
        valid_rows: torch.Tensor,
    ) -> torch.Tensor:
        masked_logits = direction_logits.masked_fill(~direction_valid_matrix, neg_inf)
        # Avoid an all -inf softmax on rows that will be skipped anyway.
        masked_logits = torch.where(
            valid_rows.unsqueeze(1), masked_logits, torch.zeros_like(masked_logits)
        )
        log_prob = F.log_softmax(masked_logits, dim=1)
        counts = direction_pos.sum(dim=1)
        rows = valid_rows & counts.gt(0)
        if not rows.any():
            return scores.sum() * 0.0
        target = direction_pos.to(log_prob.dtype) / counts.clamp_min(1).unsqueeze(1)
        return -(target[rows] * log_prob[rows]).sum(dim=1).mean()

    forward_loss = direction(logits, positive, valid_matrix, row_valid)
    backward_loss = direction(
        logits.t(), positive.t(), valid_matrix.t(), column_valid
    )
    return 0.5 * (forward_loss + backward_loss)


def paired_state_nce(
    image_state: torch.Tensor,
    text_state: torch.Tensor,
    pids: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Original-pair state NCE; same-PID non-pairs are ignored, not negatives."""
    logits_t2i = text_state @ image_state.t() / float(temperature)
    pids = pids.view(-1)
    same_pid = pids[:, None].eq(pids[None, :])
    diagonal = torch.eye(pids.numel(), dtype=torch.bool, device=pids.device)
    allowed = (~same_pid) | diagonal
    neg_inf = torch.finfo(logits_t2i.dtype).min

    logits_t2i = logits_t2i.masked_fill(~allowed, neg_inf)
    labels = torch.arange(pids.numel(), device=pids.device)
    loss_t2i = F.cross_entropy(logits_t2i, labels)
    loss_i2t = F.cross_entropy(logits_t2i.t(), labels)
    return 0.5 * (loss_t2i + loss_i2t)


def all_negative_tal(
    scores: torch.Tensor,
    pids: torch.Tensor,
    tau: float = 0.015,
    margin: float = 0.1,
) -> torch.Tensor:
    """RDE-style Triplet Alignment Loss on an arbitrary score matrix."""
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("TAL expects a square score matrix")
    pids = pids.view(-1)
    positives = pids[:, None].eq(pids[None, :])
    negatives = ~positives
    neg_inf = torch.finfo(scores.dtype).min

    def one_direction(matrix: torch.Tensor, pos_mask: torch.Tensor, neg_mask: torch.Tensor) -> torch.Tensor:
        pos_logits = (matrix / tau).masked_fill(~pos_mask, neg_inf)
        pos_weights = F.softmax(pos_logits, dim=1).detach()
        positive_score = (pos_weights * matrix).sum(dim=1)
        negative_lse = tau * torch.logsumexp(
            (matrix / tau).masked_fill(~neg_mask, neg_inf), dim=1
        )
        valid = neg_mask.any(dim=1) & pos_mask.any(dim=1)
        losses = F.relu(-positive_score + negative_lse + margin)
        if not valid.any():
            return matrix.sum() * 0.0
        return losses[valid].mean()

    return 0.5 * (
        one_direction(scores, positives, negatives)
        + one_direction(scores.t(), positives.t(), negatives.t())
    )


def posterior_calibration_nll(
    support_mean: torch.Tensor,
    support_variance: torch.Tensor,
    group_mean: torch.Tensor,
    group_variance: torch.Tensor,
    support_mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Calibrate predicted variance to empirical within-identity dispersion.

    A raw Gaussian NLL can become arbitrarily negative when both residual and
    variance approach zero.  We therefore regress log variance to a detached,
    random-effects target variance.  The target combines each observation's
    squared deviation from the trusted group intersection with the fused group
    uncertainty, and the Smooth-L1 objective is non-negative and bounded below.
    """
    centered = support_mean - group_mean.detach().unsqueeze(1)
    target_variance = (
        centered.detach().pow(2) + group_variance.detach().unsqueeze(1)
    ).clamp_min(eps)
    calibration = F.smooth_l1_loss(
        torch.log(support_variance.clamp_min(eps)),
        torch.log(target_variance),
        reduction="none",
    ).mean(dim=-1)
    return _masked_mean(calibration, support_mask)


def residual_alignment_loss(
    predicted_image_state: torch.Tensor,
    predicted_text_state: torch.Tensor,
    image_observation: torch.Tensor,
    text_observation: torch.Tensor,
    image_group_mean: torch.Tensor,
    text_group_mean: torch.Tensor,
    image_group_valid: torch.Tensor,
    text_group_valid: torch.Tensor,
) -> torch.Tensor:
    """Align state heads with observation minus cross-modal identity posterior."""
    image_target = F.normalize(
        image_observation - text_group_mean.detach(), dim=-1
    )
    text_target = F.normalize(
        text_observation - image_group_mean.detach(), dim=-1
    )
    image_term = 1.0 - (predicted_image_state * image_target).sum(dim=-1)
    text_term = 1.0 - (predicted_text_state * text_target).sum(dim=-1)
    image_loss = _masked_mean(image_term, text_group_valid)
    text_loss = _masked_mean(text_term, image_group_valid)
    return 0.5 * (image_loss + text_loss)


def state_safety_loss(
    text_state: torch.Tensor,
    image_state: torch.Tensor,
    support_image_state: torch.Tensor,
    support_text_state: torch.Tensor,
    support_mask: torch.Tensor,
) -> torch.Tensor:
    """Prevent state residuals from strongly penalizing true same-ID supports."""
    text_to_support = torch.sum(text_state.unsqueeze(1) * support_image_state, dim=-1)
    image_to_support = torch.sum(image_state.unsqueeze(1) * support_text_state, dim=-1)
    penalty = 0.5 * (F.relu(-text_to_support) + F.relu(-image_to_support))
    return _masked_mean(penalty, support_mask)

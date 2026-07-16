"""Components for the HIRE-v2 anchor baseline.

The anchor baseline intentionally contains no support-bag, identity-posterior, or
state-residual path.  It only verifies that the retained CLIP global observation
and RDE-style fine-grained observation form a strong, explicitly supervised
retrieval space before hierarchical decomposition is reintroduced.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _masked_token_max(features: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Max-pool token features while excluding invalid positions."""
    if features.ndim != 3 or valid_mask.ndim != 2:
        raise ValueError("features must be [B,L,D] and valid_mask must be [B,L]")
    if features.shape[:2] != valid_mask.shape:
        raise ValueError("features and valid_mask have incompatible shapes")
    negative = torch.finfo(features.dtype).min
    pooled = features.masked_fill(~valid_mask.unsqueeze(-1), negative).max(dim=1).values
    empty_rows = ~valid_mask.any(dim=1)
    if empty_rows.any():
        pooled = torch.where(empty_rows.unsqueeze(-1), torch.zeros_like(pooled), pooled)
    return pooled


class TokenMLP(nn.Module):
    """RDE-style two-layer token MLP with optional token masking.

    Text sequences are padded to a fixed context length.  Invalid selected
    positions must not enter BatchNorm statistics; otherwise padding features can
    influence the fine-grained branch even though they are masked before pooling.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.output_dim = int(output_dim)

    @staticmethod
    def _apply_batch_norm(module: nn.BatchNorm1d, values: torch.Tensor) -> torch.Tensor:
        # BatchNorm raises for a single training value.  This only occurs in
        # synthetic edge cases or a degenerate one-token batch; use the stored
        # running statistics rather than silently including padded positions.
        if module.training and values.shape[0] == 1:
            return F.batch_norm(
                values,
                module.running_mean,
                module.running_var,
                module.weight,
                module.bias,
                training=False,
                momentum=0.0,
                eps=module.eps,
            )
        return module(values)

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("TokenMLP input must have shape [B,L,D]")
        batch_size, token_count, input_dim = x.shape
        flat = x.reshape(batch_size * token_count, input_dim)

        if valid_mask is None:
            hidden = F.relu(self._apply_batch_norm(self.bn1, self.fc1(flat)), inplace=False)
            output = self._apply_batch_norm(self.bn2, self.fc2(hidden))
            return output.reshape(batch_size, token_count, self.output_dim)

        if valid_mask.shape != x.shape[:2]:
            raise ValueError("valid_mask must match TokenMLP's first two dimensions")
        flat_mask = valid_mask.reshape(-1).bool()
        output = flat.new_zeros((flat.shape[0], self.output_dim))
        valid_indices = flat_mask.nonzero(as_tuple=False).view(-1)
        if valid_indices.numel() == 0:
            return output.reshape(batch_size, token_count, self.output_dim)

        valid_values = flat.index_select(0, valid_indices)
        hidden = F.relu(
            self._apply_batch_norm(self.bn1, self.fc1(valid_values)),
            inplace=False,
        )
        valid_output = self._apply_batch_norm(self.bn2, self.fc2(hidden))
        output = output.index_copy(0, valid_indices, valid_output)
        return output.reshape(batch_size, token_count, self.output_dim)


class RDEVisualTokenSelection(nn.Module):
    """Attention-guided visual token selection adapted from public RDE code."""

    def __init__(self, input_dim: int = 512, output_dim: int = 1024, ratio: float = 0.3):
        super().__init__()
        if not 0.0 < ratio <= 1.0:
            raise ValueError("ratio must be in (0, 1]")
        self.ratio = float(ratio)
        self.skip = nn.Linear(input_dim, output_dim)
        self.mlp = TokenMLP(input_dim, max(1, output_dim // 2), output_dim)

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
    """Attention-guided text token selection adapted from public RDE code."""

    def __init__(self, input_dim: int = 512, output_dim: int = 1024, ratio: float = 0.3):
        super().__init__()
        if not 0.0 < ratio <= 1.0:
            raise ValueError("ratio must be in (0, 1]")
        self.ratio = float(ratio)
        self.skip = nn.Linear(input_dim, output_dim)
        self.mlp = TokenMLP(input_dim, max(1, output_dim // 2), output_dim)

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

        batch_size, sequence_length = token_ids.shape
        eot_indices = token_ids.argmax(dim=-1)
        valid_tokens = token_ids.ne(0)
        valid_tokens[:, 0] = False
        valid_tokens[torch.arange(batch_size, device=token_ids.device), eot_indices] = False

        scores = attention[
            torch.arange(batch_size, device=token_ids.device), eot_indices, :
        ].detach().clone()
        scores = scores.masked_fill(~valid_tokens, torch.finfo(scores.dtype).min)

        global_k = max(1, int(max(1, sequence_length - 2) * self.ratio))
        global_k = min(global_k, max(1, sequence_length - 2))
        indices = scores.topk(k=global_k, dim=-1, largest=True, sorted=False).indices
        gather_index = indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
        selected = torch.gather(tokens, dim=1, index=gather_index)
        selected = F.normalize(selected.float(), dim=-1)
        selected_valid = torch.gather(valid_tokens, dim=1, index=indices)
        # Only valid selected words enter the token MLP and its BatchNorm
        # statistics.  Padded/special positions remain exactly masked.
        transformed = self.mlp(selected, selected_valid) + self.skip(selected)
        return _masked_token_max(transformed, selected_valid)


class ResidualObservationFusion(nn.Module):
    """Preserve CLIP geometry while learning a zero-initialized local residual.

    observation = normalize(normalize(global) + local_adapter(local)).

    Because local_adapter is initialized to zero, the initial observation is
    exactly the normalized CLIP global embedding, not a LayerNorm-transformed
    approximation of it.
    """

    def __init__(self, global_dim: int, local_dim: int):
        super().__init__()
        self.local_adapter = nn.Linear(local_dim, global_dim, bias=False)
        nn.init.zeros_(self.local_adapter.weight)

    def forward(
        self,
        global_feature: torch.Tensor,
        local_feature: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        global_feature = F.normalize(global_feature.float(), dim=-1)
        local_residual = self.local_adapter(local_feature.float())
        observation = F.normalize(global_feature + local_residual, dim=-1)
        return observation, local_residual


def aggregate_anchor_objectives(
    global_sdm: torch.Tensor,
    global_itc: torch.Tensor,
    local_sdm: torch.Tensor,
    local_itc: torch.Tensor,
    observation_sdm: torch.Tensor,
    observation_itc: torch.Tensor,
) -> dict:
    """Aggregate the documented HIRE-v2 anchor objective exactly.

    L_anchor = 0.5 * (L_global + L_local)
    L_total  = L_anchor + L_observation
    """
    sdm_loss = 0.5 * (global_sdm + local_sdm) + observation_sdm
    itc_loss = 0.5 * (global_itc + local_itc) + observation_itc
    return {
        "sdm_loss": sdm_loss,
        "itc_loss": itc_loss,
        "anchor_objective": 0.5 * (
            global_sdm + global_itc + local_sdm + local_itc
        ),
        "observation_objective": observation_sdm + observation_itc,
    }


class CLIPAttentionAdapter(object):
    """Expose final-layer CLIP attention without changing clip_model.py."""

    @staticmethod
    def _run_transformer_with_last_attention(
        transformer: nn.Module,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        last_attention = None
        blocks = list(transformer.resblocks)
        if not blocks:
            raise RuntimeError("CLIP transformer has no residual blocks")
        for index, block in enumerate(blocks):
            if index != len(blocks) - 1:
                x = block(x)
                continue
            normalized = block.ln_1(x)
            attention_mask = block.attn_mask
            if attention_mask is not None:
                attention_mask = attention_mask.to(
                    dtype=normalized.dtype,
                    device=normalized.device,
                )
            attention_output, last_attention = block.attn(
                normalized,
                normalized,
                normalized,
                need_weights=True,
                attn_mask=attention_mask,
            )
            x = x + attention_output
            x = x + block.mlp(block.ln_2(x))
        if last_attention is None:
            raise RuntimeError("failed to collect final CLIP attention")
        return x, last_attention

    @classmethod
    def encode_image(
        cls,
        clip_model: nn.Module,
        images: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        visual = clip_model.visual
        if not hasattr(visual, "transformer"):
            raise RuntimeError("HIRE-v2 anchor requires a CLIP Vision Transformer")
        x = visual.conv1(images.type(clip_model.dtype))
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        class_token = visual.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0],
            1,
            x.shape[-1],
            dtype=x.dtype,
            device=x.device,
        )
        x = torch.cat([class_token, x], dim=1)
        x = visual.ln_pre(x + visual.positional_embedding.to(x.dtype))
        x = x.permute(1, 0, 2)
        x, attention = cls._run_transformer_with_last_attention(visual.transformer, x)
        x = x.permute(1, 0, 2)
        x = visual.ln_post(x)
        if visual.proj is not None:
            x = x @ visual.proj
        return x, attention

    @classmethod
    def encode_text(
        cls,
        clip_model: nn.Module,
        token_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = clip_model.token_embedding(token_ids).type(clip_model.dtype)
        x = x + clip_model.positional_embedding.type(clip_model.dtype)
        x = x.permute(1, 0, 2)
        x, attention = cls._run_transformer_with_last_attention(clip_model.transformer, x)
        x = x.permute(1, 0, 2)
        x = clip_model.ln_final(x).type(clip_model.dtype)
        x = x @ clip_model.text_projection
        return x, attention

    @classmethod
    def forward(
        cls,
        clip_model: nn.Module,
        images: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        image_tokens, image_attention = cls.encode_image(clip_model, images)
        text_tokens, text_attention = cls.encode_text(clip_model, token_ids)
        return image_tokens, image_attention, text_tokens, text_attention

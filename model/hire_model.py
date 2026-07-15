"""Heterogeneity-aware hierarchical identity-state retrieval model.

The implementation is designed as an independent model path in the existing
IRRA-light repository.  It keeps the OpenAI CLIP ViT-B/16 backbone, ports RDE's
attention-guided token-selection idea, and replaces instance-only matching with
an explicit hierarchical random-effects decomposition:

    observation = latent identity effect + instance state effect + noise.

Same-ID supports estimate a dynamic identity posterior; they are never inserted
as ordinary strong positives.  The state head learns the remaining instance
residual and the final retrieval score is produced directly by identity and
state components without retaining a third baseline score.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .clip_model import build_CLIP_from_openai_pretrained, convert_weights
from .hire_components import (
    IdentityPosteriorHead,
    ObservationFusion,
    RDETextTokenSelection,
    RDEVisualTokenSelection,
    StateResidualHead,
    all_negative_tal,
    gaussian_pairwise_score,
    heterogeneity_aware_posterior,
    paired_state_nce,
    posterior_calibration_nll,
    residual_alignment_loss,
    state_safety_loss,
    symmetric_multi_positive_nce,
)


class CLIPAttentionAdapter(object):
    """Run the repository's CLIP while exposing final-layer attention.

    This adapter avoids changing model/clip_model.py and therefore preserves all
    existing modes byte-for-byte.  It reproduces the final transformer forward
    using the public modules already present in the repository.
    """

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
            mask = block.attn_mask
            if mask is not None:
                mask = mask.to(dtype=normalized.dtype, device=normalized.device)
            attention_output, last_attention = block.attn(
                normalized,
                normalized,
                normalized,
                need_weights=True,
                attn_mask=mask,
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
        image: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        visual = clip_model.visual
        if not hasattr(visual, "transformer"):
            raise RuntimeError("HIRE currently requires a CLIP Vision Transformer backbone")
        x = visual.conv1(image.type(clip_model.dtype))
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        class_token = visual.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
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


class HIRE(nn.Module):
    """Main HIRE model."""

    is_hire_model = True

    def __init__(self, args, num_classes: int = 0):
        super().__init__()
        self.args = args
        self.num_classes = int(num_classes)
        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(
            args.pretrain_choice,
            args.img_size,
            args.stride_size,
        )
        self.embed_dim = int(base_cfg["embed_dim"])
        self.observation_dim = int(getattr(args, "hire_observation_dim", self.embed_dim))
        self.identity_dim = int(getattr(args, "hire_identity_dim", self.observation_dim))
        self.state_dim = int(getattr(args, "hire_state_dim", self.observation_dim))
        self.tse_dim = int(getattr(args, "hire_tse_dim", 1024))
        if self.identity_dim != self.observation_dim or self.state_dim != self.observation_dim:
            raise ValueError(
                "The main HIRE formulation requires observation, identity, and state dimensions to match"
            )
        self.select_ratio = float(getattr(args, "hire_select_ratio", 0.3))
        self.posterior_temperature = float(getattr(args, "temperature", 0.02))
        self.tal_tau = float(getattr(args, "hire_tau", 0.015))
        self.tal_margin = float(getattr(args, "hire_margin", 0.1))
        self.support_encode_chunk = max(1, int(getattr(args, "batch_size", 64)))

        self.image_tse = RDEVisualTokenSelection(
            input_dim=self.embed_dim,
            output_dim=self.tse_dim,
            ratio=self.select_ratio,
        )
        self.text_tse = RDETextTokenSelection(
            input_dim=self.embed_dim,
            output_dim=self.tse_dim,
            ratio=self.select_ratio,
        )
        self.image_fusion = ObservationFusion(
            global_dim=self.embed_dim,
            local_dim=self.tse_dim,
            output_dim=self.observation_dim,
        )
        self.text_fusion = ObservationFusion(
            global_dim=self.embed_dim,
            local_dim=self.tse_dim,
            output_dim=self.observation_dim,
        )
        self.image_identity = IdentityPosteriorHead(
            self.observation_dim,
            self.identity_dim,
        )
        self.text_identity = IdentityPosteriorHead(
            self.observation_dim,
            self.identity_dim,
        )
        self.image_state = StateResidualHead(self.observation_dim, self.state_dim)
        self.text_state = StateResidualHead(self.observation_dim, self.state_dim)

        # The final score contains no independent baseline term.  Identity starts
        # dominant while state starts as a safe 0.1 residual; both scales are
        # learned end-to-end and require no manually searched fusion weight.
        self.log_identity_scale = nn.Parameter(torch.tensor(0.0))
        self.log_state_scale = nn.Parameter(torch.tensor(math.log(0.1)))

    @staticmethod
    def _eot_feature(tokens: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        indices = token_ids.argmax(dim=-1)
        return tokens[torch.arange(tokens.shape[0], device=tokens.device), indices]

    def _observations_from_tokens(
        self,
        image_tokens: torch.Tensor,
        image_attention: torch.Tensor,
        text_tokens: torch.Tensor,
        text_attention: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        image_global = image_tokens[:, 0, :].float()
        text_global = self._eot_feature(text_tokens, token_ids).float()
        image_local = self.image_tse(image_tokens.float(), image_attention.detach())
        text_local = self.text_tse(text_tokens.float(), token_ids, text_attention.detach())
        image_observation = self.image_fusion(image_global, image_local)
        text_observation = self.text_fusion(text_global, text_local)
        return image_observation, text_observation

    def _encode_main(
        self,
        images: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        image_tokens, image_attention, text_tokens, text_attention = CLIPAttentionAdapter.forward(
            self.base_model,
            images,
            token_ids,
        )
        image_observation, text_observation = self._observations_from_tokens(
            image_tokens,
            image_attention,
            text_tokens,
            text_attention,
            token_ids,
        )
        image_mean, image_variance = self.image_identity(image_observation)
        text_mean, text_variance = self.text_identity(text_observation)
        image_state = self.image_state(image_observation)
        text_state = self.text_state(text_observation)
        return {
            "image_observation": image_observation,
            "text_observation": text_observation,
            "image_mean": image_mean,
            "image_variance": image_variance,
            "text_mean": text_mean,
            "text_variance": text_variance,
            "image_state": image_state,
            "text_state": text_state,
        }

    def _encode_support_backbone(
        self,
        support_images: torch.Tensor,
        support_token_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        image_tokens_parts: List[torch.Tensor] = []
        image_attention_parts: List[torch.Tensor] = []
        text_tokens_parts: List[torch.Tensor] = []
        text_attention_parts: List[torch.Tensor] = []
        count = support_images.shape[0]
        with torch.no_grad():
            for start in range(0, count, self.support_encode_chunk):
                end = min(start + self.support_encode_chunk, count)
                i_tok, i_attn, t_tok, t_attn = CLIPAttentionAdapter.forward(
                    self.base_model,
                    support_images[start:end],
                    support_token_ids[start:end],
                )
                image_tokens_parts.append(i_tok.float())
                image_attention_parts.append(i_attn.float())
                text_tokens_parts.append(t_tok.float())
                text_attention_parts.append(t_attn.float())
        return (
            torch.cat(image_tokens_parts, dim=0),
            torch.cat(image_attention_parts, dim=0),
            torch.cat(text_tokens_parts, dim=0),
            torch.cat(text_attention_parts, dim=0),
        )

    def _encode_supports(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        required = {"support_images", "support_caption_ids", "support_mask"}
        missing = required - set(batch.keys())
        if missing:
            raise RuntimeError("HIRE requires support fields: {}".format(sorted(missing)))
        support_images = batch["support_images"]
        support_token_ids = batch["support_caption_ids"]
        support_mask = batch["support_mask"].bool()
        if support_images.ndim != 5 or support_token_ids.ndim != 3:
            raise RuntimeError("support images/tokens must be [B,S,C,H,W] and [B,S,L]")
        batch_size, support_size = support_mask.shape
        flat_images = support_images.reshape(batch_size * support_size, *support_images.shape[2:])
        flat_tokens = support_token_ids.reshape(batch_size * support_size, support_token_ids.shape[-1])
        image_tokens, image_attention, text_tokens, text_attention = self._encode_support_backbone(
            flat_images,
            flat_tokens,
        )
        image_observation, text_observation = self._observations_from_tokens(
            image_tokens,
            image_attention,
            text_tokens,
            text_attention,
            flat_tokens,
        )
        image_mean, image_variance = self.image_identity(image_observation)
        text_mean, text_variance = self.text_identity(text_observation)
        image_state = self.image_state(image_observation)
        text_state = self.text_state(text_observation)

        def reshape(x: torch.Tensor) -> torch.Tensor:
            return x.reshape(batch_size, support_size, -1)

        return {
            "mask": support_mask,
            "image_observation": reshape(image_observation),
            "text_observation": reshape(text_observation),
            "image_mean": reshape(image_mean),
            "image_variance": reshape(image_variance),
            "text_mean": reshape(text_mean),
            "text_variance": reshape(text_variance),
            "image_state": reshape(image_state),
            "text_state": reshape(text_state),
        }

    def _score_components(
        self,
        text_mean: torch.Tensor,
        text_variance: torch.Tensor,
        text_state: torch.Tensor,
        image_mean: torch.Tensor,
        image_variance: torch.Tensor,
        image_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        identity_score = gaussian_pairwise_score(
            text_mean,
            text_variance,
            image_mean,
            image_variance,
        )
        state_score = text_state @ image_state.t()
        identity_scale = torch.exp(self.log_identity_scale.clamp(-5.0, 5.0))
        state_scale = torch.exp(self.log_state_scale.clamp(-5.0, 5.0))
        final_score = identity_scale * identity_score + state_scale * state_score
        return identity_score, state_score, final_score

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pids = batch["pids"].view(-1)
        main = self._encode_main(batch["images"], batch["caption_ids"])
        support = self._encode_supports(batch)

        image_group = heterogeneity_aware_posterior(
            support["image_mean"],
            support["image_variance"],
            support["mask"],
        )
        text_group = heterogeneity_aware_posterior(
            support["text_mean"],
            support["text_variance"],
            support["mask"],
        )

        text_to_image_group = gaussian_pairwise_score(
            main["text_mean"],
            main["text_variance"],
            image_group["mean"],
            image_group["variance"],
        )
        image_to_text_group = gaussian_pairwise_score(
            main["image_mean"],
            main["image_variance"],
            text_group["mean"],
            text_group["variance"],
        )
        identity_set_nce = 0.5 * (
            symmetric_multi_positive_nce(
                text_to_image_group,
                pids,
                self.posterior_temperature,
                row_valid=torch.ones_like(image_group["valid"]),
                column_valid=image_group["valid"],
            )
            + symmetric_multi_positive_nce(
                image_to_text_group,
                pids,
                self.posterior_temperature,
                row_valid=torch.ones_like(text_group["valid"]),
                column_valid=text_group["valid"],
            )
        )
        uncertainty_calibration = 0.5 * (
            posterior_calibration_nll(
                support["image_mean"],
                support["image_variance"],
                image_group["mean"],
                image_group["variance"],
                support["mask"],
            )
            + posterior_calibration_nll(
                support["text_mean"],
                support["text_variance"],
                text_group["mean"],
                text_group["variance"],
                support["mask"],
            )
        )
        identity_posterior_loss = identity_set_nce + uncertainty_calibration

        state_pair_nce = paired_state_nce(
            main["image_state"],
            main["text_state"],
            pids,
            self.posterior_temperature,
        )
        residual_alignment = residual_alignment_loss(
            main["image_state"],
            main["text_state"],
            main["image_observation"],
            main["text_observation"],
            image_group["mean"],
            text_group["mean"],
            image_group["valid"],
            text_group["valid"],
        )
        safety = state_safety_loss(
            main["text_state"],
            main["image_state"],
            support["image_state"],
            support["text_state"],
            support["mask"],
        )
        state_hierarchical_loss = state_pair_nce + residual_alignment + safety

        identity_score, state_score, final_score = self._score_components(
            main["text_mean"],
            main["text_variance"],
            main["text_state"],
            main["image_mean"],
            main["image_variance"],
            main["image_state"],
        )
        joint_tal_loss = all_negative_tal(
            final_score,
            pids,
            tau=self.tal_tau,
            margin=self.tal_margin,
        )

        return {
            "joint_tal_loss": joint_tal_loss,
            "identity_posterior_loss": identity_posterior_loss,
            "state_hierarchical_loss": state_hierarchical_loss,
            "temperature": torch.tensor(self.posterior_temperature, device=pids.device),
            # Diagnostics intentionally avoid the substring 'loss' because the
            # existing training loop sums every return key containing it.
            "identity_set_nce": identity_set_nce.detach(),
            "uncertainty_calibration": uncertainty_calibration.detach(),
            "state_pair_nce": state_pair_nce.detach(),
            "residual_alignment": residual_alignment.detach(),
            "state_safety": safety.detach(),
            "mean_image_variance": main["image_variance"].mean().detach(),
            "mean_text_variance": main["text_variance"].mean().detach(),
            "mean_group_heterogeneity": 0.5 * (
                image_group["tau2"].mean() + text_group["tau2"].mean()
            ).detach(),
            "identity_scale": torch.exp(self.log_identity_scale.detach()),
            "state_scale": torch.exp(self.log_state_scale.detach()),
            "support_valid_ratio": support["mask"].any(dim=1).float().mean().detach(),
            "identity_score_mean": identity_score.mean().detach(),
            "state_score_mean": state_score.mean().detach(),
        }

    def _encode_image_retrieval(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Text inputs are not needed for image encoding, so call the image path
        # directly and apply exactly the same visual observation modules.
        image_tokens, image_attention = CLIPAttentionAdapter.encode_image(self.base_model, images)
        image_global = image_tokens[:, 0, :].float()
        image_local = self.image_tse(image_tokens.float(), image_attention.detach())
        observation = self.image_fusion(image_global, image_local)
        mean, variance = self.image_identity(observation)
        state = self.image_state(observation)
        return {"mean": mean, "variance": variance, "state": state}

    def _encode_text_retrieval(self, token_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        text_tokens, text_attention = CLIPAttentionAdapter.encode_text(self.base_model, token_ids)
        text_global = self._eot_feature(text_tokens, token_ids).float()
        text_local = self.text_tse(text_tokens.float(), token_ids, text_attention.detach())
        observation = self.text_fusion(text_global, text_local)
        mean, variance = self.text_identity(observation)
        state = self.text_state(observation)
        return {"mean": mean, "variance": variance, "state": state}

    def encode_image_retrieval(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self._encode_image_retrieval(images)

    def encode_text_retrieval(self, token_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self._encode_text_retrieval(token_ids)

    def compute_similarity_matrix(
        self,
        text_repr: Dict[str, torch.Tensor],
        image_repr: Dict[str, torch.Tensor],
        query_chunk: int = 128,
        gallery_chunk: int = 512,
    ) -> torch.Tensor:
        """Compute the final HIRE score matrix without materializing [Q,G,D]."""
        device = next(self.parameters()).device
        query_count = text_repr["mean"].shape[0]
        gallery_count = image_repr["mean"].shape[0]
        output = torch.empty(query_count, gallery_count, dtype=torch.float32)
        identity_scale = torch.exp(self.log_identity_scale.detach().clamp(-5.0, 5.0))
        state_scale = torch.exp(self.log_state_scale.detach().clamp(-5.0, 5.0))

        for q_start in range(0, query_count, query_chunk):
            q_end = min(q_start + query_chunk, query_count)
            q_mean = text_repr["mean"][q_start:q_end].to(device)
            q_var = text_repr["variance"][q_start:q_end].to(device)
            q_state = text_repr["state"][q_start:q_end].to(device)
            row_parts = []
            for g_start in range(0, gallery_count, gallery_chunk):
                g_end = min(g_start + gallery_chunk, gallery_count)
                g_mean = image_repr["mean"][g_start:g_end].to(device)
                g_var = image_repr["variance"][g_start:g_end].to(device)
                g_state = image_repr["state"][g_start:g_end].to(device)
                identity = gaussian_pairwise_score(q_mean, q_var, g_mean, g_var)
                state = q_state @ g_state.t()
                row_parts.append((identity_scale * identity + state_scale * state).float().cpu())
            output[q_start:q_end] = torch.cat(row_parts, dim=1)
        return output

    # Compatibility helpers.  Standard Evaluator dispatches to the HIRE-specific
    # methods, but these return identity means for third-party feature exporters.
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode_image_retrieval(images)["mean"]

    def encode_text(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.encode_text_retrieval(token_ids)["mean"]


def build_hire_model(args, num_classes: int = 0) -> HIRE:
    model = HIRE(args, num_classes=num_classes)
    # Keep the pretrained CLIP backbone in the repository's normal fp16 form;
    # all newly initialized HIRE modules remain fp32 for numerical stability of
    # Gaussian variance and random-effects calculations.
    convert_weights(model.base_model)
    return model

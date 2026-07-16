"""HIRE-v2 version-one baseline: anchored complete observations.

This model deliberately excludes identity posteriors, same-ID supports, and state
residuals.  It establishes a strong and interpretable observation space with:

1. CLIP ViT-B/16 global observations;
2. RDE-style fine-grained token-selection observations;
3. zero-initialized residual fusion that initially equals CLIP global geometry;
4. direct SDM + ITC supervision on global, local, and fused observations.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import objectives
from .clip_model import build_CLIP_from_openai_pretrained, convert_weights
from .hire_v2_anchor_components import (
    CLIPAttentionAdapter,
    RDETextTokenSelection,
    RDEVisualTokenSelection,
    ResidualObservationFusion,
    aggregate_anchor_objectives,
)


class HIREV2Anchor(nn.Module):
    """Anchored complete-observation baseline."""

    is_hire_v2_anchor_model = True

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
        self.tse_dim = int(getattr(args, "hire_v2_tse_dim", 1024))
        self.select_ratio = float(getattr(args, "hire_v2_select_ratio", 0.3))
        self.register_buffer(
            "logit_scale",
            torch.ones([]) * (1.0 / float(args.temperature)),
        )

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
        self.image_fusion = ResidualObservationFusion(
            global_dim=self.embed_dim,
            local_dim=self.tse_dim,
        )
        self.text_fusion = ResidualObservationFusion(
            global_dim=self.embed_dim,
            local_dim=self.tse_dim,
        )

    @staticmethod
    def _eot_feature(tokens: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        indices = token_ids.argmax(dim=-1)
        return tokens[
            torch.arange(tokens.shape[0], device=tokens.device),
            indices,
        ]

    def _representations_from_tokens(
        self,
        image_tokens: torch.Tensor,
        image_attention: torch.Tensor,
        text_tokens: torch.Tensor,
        text_attention: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        image_global = F.normalize(image_tokens[:, 0, :].float(), dim=-1)
        text_global = F.normalize(self._eot_feature(text_tokens, token_ids).float(), dim=-1)
        image_local = F.normalize(
            self.image_tse(image_tokens.float(), image_attention.detach()),
            dim=-1,
        )
        text_local = F.normalize(
            self.text_tse(text_tokens.float(), token_ids, text_attention.detach()),
            dim=-1,
        )
        image_observation, image_residual = self.image_fusion(image_global, image_local)
        text_observation, text_residual = self.text_fusion(text_global, text_local)
        return {
            "image_global": image_global,
            "text_global": text_global,
            "image_local": image_local,
            "text_local": text_local,
            "image_observation": image_observation,
            "text_observation": text_observation,
            "image_residual": image_residual,
            "text_residual": text_residual,
        }

    def _encode_joint(
        self,
        images: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        image_tokens, image_attention, text_tokens, text_attention = (
            CLIPAttentionAdapter.forward(self.base_model, images, token_ids)
        )
        return self._representations_from_tokens(
            image_tokens,
            image_attention,
            text_tokens,
            text_attention,
            token_ids,
        )

    def _retrieval_objectives(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        pids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scale = self.logit_scale.to(image_features.device)
        sdm = objectives.compute_sdm(
            image_features,
            text_features,
            pids,
            scale,
        )
        itc = objectives.compute_itc(
            image_features,
            text_features,
            scale,
        )
        return sdm, itc

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        encoded = self._encode_joint(batch["images"], batch["caption_ids"])
        pids = batch["pids"]

        global_sdm, global_itc = self._retrieval_objectives(
            encoded["image_global"],
            encoded["text_global"],
            pids,
        )
        local_sdm, local_itc = self._retrieval_objectives(
            encoded["image_local"],
            encoded["text_local"],
            pids,
        )
        observation_sdm, observation_itc = self._retrieval_objectives(
            encoded["image_observation"],
            encoded["text_observation"],
            pids,
        )

        # L_total = 0.5 * (L_global + L_local) + L_observation.
        # Centralize the aggregation in a tested helper so the implementation
        # cannot silently drift away from the design document.
        aggregated = aggregate_anchor_objectives(
            global_sdm,
            global_itc,
            local_sdm,
            local_itc,
            observation_sdm,
            observation_itc,
        )

        return {
            "sdm_loss": aggregated["sdm_loss"],
            "itc_loss": aggregated["itc_loss"],
            "temperature": 1.0 / self.logit_scale.to(pids.device),
            # Diagnostics intentionally avoid the substring "loss" because the
            # existing processor sums every returned key containing it.
            "global_sdm": global_sdm.detach(),
            "global_itc": global_itc.detach(),
            "local_sdm": local_sdm.detach(),
            "local_itc": local_itc.detach(),
            "observation_sdm": observation_sdm.detach(),
            "observation_itc": observation_itc.detach(),
            "anchor_objective": aggregated["anchor_objective"].detach(),
            "observation_objective": aggregated["observation_objective"].detach(),
            "image_local_residual_norm": encoded["image_residual"].norm(dim=-1).mean().detach(),
            "text_local_residual_norm": encoded["text_residual"].norm(dim=-1).mean().detach(),
        }

    def encode_image_retrieval(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        image_tokens, image_attention = CLIPAttentionAdapter.encode_image(
            self.base_model,
            images,
        )
        image_global = F.normalize(image_tokens[:, 0, :].float(), dim=-1)
        image_local = F.normalize(
            self.image_tse(image_tokens.float(), image_attention.detach()),
            dim=-1,
        )
        image_observation, _ = self.image_fusion(image_global, image_local)
        return {
            "global": image_global,
            "local": image_local,
            "observation": image_observation,
        }

    def encode_text_retrieval(self, token_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        text_tokens, text_attention = CLIPAttentionAdapter.encode_text(
            self.base_model,
            token_ids,
        )
        text_global = F.normalize(self._eot_feature(text_tokens, token_ids).float(), dim=-1)
        text_local = F.normalize(
            self.text_tse(text_tokens.float(), token_ids, text_attention.detach()),
            dim=-1,
        )
        text_observation, _ = self.text_fusion(text_global, text_local)
        return {
            "global": text_global,
            "local": text_local,
            "observation": text_observation,
        }

    # Standard Evaluator uses these methods and therefore selects checkpoints by
    # the fused observation score, which is the official version-one result.
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode_image_retrieval(images)["observation"]

    def encode_text(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.encode_text_retrieval(token_ids)["observation"]


def build_hire_v2_anchor_model(args, num_classes: int = 0) -> HIREV2Anchor:
    model = HIREV2Anchor(args, num_classes=num_classes)
    # Keep pretrained CLIP in the repository's normal fp16 format.  Newly
    # initialized token-selection and fusion modules remain fp32.
    convert_weights(model.base_model)
    return model

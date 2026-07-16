"""HIRE-v2 version two: anchored identity random effects only.

This path preserves version one's CLIP-global/RDE-local observation anchor and
adds one innovation: a same-ID support-image trusted intersection that trains a
shared identity mapping.  There is no state branch, no support text, no weak
positive pair loss, and no identity classifier.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .clip_model import convert_weights
from .hire_v2_anchor_components import CLIPAttentionAdapter
from .hire_v2_anchor_model import HIREV2Anchor
from .hire_v2_identity_components import (
    BoundedImageUncertainty,
    BoundedResidualGate,
    SharedIdentityMean,
    aggregate_identity_objectives,
    build_identity_final_embedding,
    heterogeneity_aware_identity_intersection,
    identity_residual_score,
    itc_from_similarity,
    paired_identity_group_nce,
    sdm_from_similarity,
)


class HIREV2Identity(HIREV2Anchor):
    """Anchored observation baseline plus identity random-effect residual."""

    is_hire_v2_anchor_model = False
    is_hire_v2_identity_model = True

    def __init__(self, args, num_classes: int = 0):
        super().__init__(args, num_classes=num_classes)
        self.support_size = int(getattr(args, "hire_v2_support_size", 3))
        self.auxiliary_weight = float(getattr(args, "hire_v2_aux_weight", 0.1))
        if self.support_size < 2:
            raise ValueError("HIRE-v2 identity requires at least two support images")
        if self.auxiliary_weight < 0.0:
            raise ValueError("HIRE-v2 identity auxiliary weight must be non-negative")

        # A single shared map prevents image/text identity axes from drifting
        # independently.  It starts exactly equal to the anchored observation.
        self.identity_mean = SharedIdentityMean(self.embed_dim)
        # Only support-image uncertainty is required in this text-to-image task.
        self.image_uncertainty = BoundedImageUncertainty(self.embed_dim)
        # The identity residual enters conservatively but is exactly zero at
        # initialization because identity_mean starts as an identity map.
        self.identity_gate = BoundedResidualGate(initial_value=0.1)
        self.support_encode_chunk = max(1, int(getattr(args, "batch_size", 64)))

    def _identity_from_observation(self, observation: torch.Tensor) -> torch.Tensor:
        # Identity objectives must not rewrite the strong anchor representation.
        # The anchor continues to learn only through global/local/final retrieval.
        return self.identity_mean(observation.detach())

    def _support_observations(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        required = {
            "support_images",
            "support_mask",
            "support_pids",
            "support_image_ids",
        }
        missing = required - set(batch.keys())
        if missing:
            raise RuntimeError(
                "HIRE-v2 identity requires support fields: {}".format(sorted(missing))
            )

        support_images = batch["support_images"]
        support_mask = batch["support_mask"].bool()
        if support_images.ndim != 5 or support_mask.ndim != 2:
            raise RuntimeError("support images/mask must be [B,S,C,H,W] and [B,S]")
        batch_size, support_size = support_mask.shape
        if support_size != self.support_size:
            raise RuntimeError(
                "support tensor size {} does not match configured {}".format(
                    support_size, self.support_size
                )
            )

        flat_images = support_images.reshape(
            batch_size * support_size, *support_images.shape[2:]
        )
        observation_parts: List[torch.Tensor] = []
        # Support observations are evidence for the identity head only.  Freezing
        # their CLIP/TSE/fusion path avoids a hidden support-bag gradient into the
        # anchor while main-batch anchor training remains fully end-to-end.
        # image_tse contains BatchNorm, so temporarily use evaluation statistics;
        # no-grad alone would still update running means/variances.
        image_tse_was_training = self.image_tse.training
        self.image_tse.eval()
        try:
            with torch.no_grad():
                for start in range(0, flat_images.shape[0], self.support_encode_chunk):
                    end = min(start + self.support_encode_chunk, flat_images.shape[0])
                    image_tokens, image_attention = CLIPAttentionAdapter.encode_image(
                        self.base_model, flat_images[start:end]
                    )
                    image_global = F.normalize(image_tokens[:, 0, :].float(), dim=-1)
                    image_local = F.normalize(
                        self.image_tse(image_tokens.float(), image_attention.detach()),
                        dim=-1,
                    )
                    image_observation, _ = self.image_fusion(image_global, image_local)
                    observation_parts.append(image_observation.float())
        finally:
            self.image_tse.train(image_tse_was_training)

        support_observation = torch.cat(observation_parts, dim=0).reshape(
            batch_size, support_size, -1
        )
        support_mean = self.identity_mean(support_observation.detach())
        support_variance = self.image_uncertainty(support_observation.detach())
        return {
            "observation": support_observation,
            "mean": support_mean,
            "variance": support_variance,
            "mask": support_mask,
        }

    @staticmethod
    def _validate_support_relations(batch: Dict[str, torch.Tensor]) -> None:
        mask = batch["support_mask"].bool()
        anchor_pids = batch["pids"].view(-1, 1)
        anchor_image_ids = batch["image_ids"].view(-1, 1)
        support_pids = batch["support_pids"]
        support_image_ids = batch["support_image_ids"]

        if support_pids.shape != mask.shape or support_image_ids.shape != mask.shape:
            raise RuntimeError("support PID/image-ID tensors must match support mask")
        if ((support_pids != anchor_pids) & mask).any():
            raise RuntimeError("identity supports cross a PID boundary")
        if ((support_image_ids == anchor_image_ids) & mask).any():
            raise RuntimeError("identity supports reuse the anchor image")

        # Every valid support position must contain a distinct image ID.
        for row in range(mask.shape[0]):
            valid_ids = support_image_ids[row][mask[row]]
            if valid_ids.numel() != torch.unique(valid_ids).numel():
                raise RuntimeError("identity support set contains duplicate image IDs")

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        encoded = self._encode_joint(batch["images"], batch["caption_ids"])
        pids = batch["pids"].view(-1)
        self._validate_support_relations(batch)

        global_sdm, global_itc = self._retrieval_objectives(
            encoded["image_global"], encoded["text_global"], pids
        )
        local_sdm, local_itc = self._retrieval_objectives(
            encoded["image_local"], encoded["text_local"], pids
        )

        image_identity = self._identity_from_observation(encoded["image_observation"])
        text_identity = self._identity_from_observation(encoded["text_observation"])
        support = self._support_observations(batch)
        group = heterogeneity_aware_identity_intersection(
            support["mean"], support["variance"], support["mask"], min_supports=2
        )

        observation_score = encoded["text_observation"] @ encoded["image_observation"].t()
        identity_score = text_identity @ image_identity.t()
        gate = self.identity_gate()
        final_score = identity_residual_score(
            observation_score, identity_score, gate
        )

        final_sdm = sdm_from_similarity(final_score, pids, self.logit_scale)
        final_itc = itc_from_similarity(final_score, self.logit_scale)

        group_score = text_identity @ group["mean"].t()
        group_nce = paired_identity_group_nce(
            group_score, pids, group["valid"], self.logit_scale
        )
        aggregated = aggregate_identity_objectives(
            global_sdm,
            global_itc,
            local_sdm,
            local_itc,
            final_sdm,
            final_itc,
            group_nce,
            self.auxiliary_weight,
        )

        valid_variance = support["variance"][support["mask"]]
        if valid_variance.numel() > 0:
            variance_mean = valid_variance.mean()
            variance_range = (
                self.image_uncertainty.variance_max
                - self.image_uncertainty.variance_min
            )
            low_threshold = self.image_uncertainty.variance_min + 0.05 * variance_range
            high_threshold = self.image_uncertainty.variance_max - 0.05 * variance_range
            variance_low_ratio = (valid_variance <= low_threshold).float().mean()
            variance_high_ratio = (valid_variance >= high_threshold).float().mean()
        else:
            zero = final_score.sum() * 0.0
            variance_mean = zero.detach()
            variance_low_ratio = zero.detach()
            variance_high_ratio = zero.detach()

        if group["valid"].any():
            heterogeneity_mean = group["tau2"][group["valid"]].mean()
        else:
            heterogeneity_mean = (final_score.sum() * 0.0).detach()

        identity_weight = self.identity_mean.proj.weight
        identity_matrix = torch.eye(
            identity_weight.shape[0],
            device=identity_weight.device,
            dtype=identity_weight.dtype,
        )
        projection_delta = (identity_weight - identity_matrix).norm()
        observation_identity_cosine = 0.5 * (
            (encoded["image_observation"].detach() * image_identity).sum(dim=-1).mean()
            + (encoded["text_observation"].detach() * text_identity).sum(dim=-1).mean()
        )

        return {
            "sdm_loss": aggregated["sdm_loss"],
            "itc_loss": aggregated["itc_loss"],
            "identity_group_loss": aggregated["identity_group_loss"],
            "temperature": 1.0 / self.logit_scale.to(pids.device),
            # Diagnostics intentionally avoid the substring "loss".
            "global_sdm": global_sdm.detach(),
            "global_itc": global_itc.detach(),
            "local_sdm": local_sdm.detach(),
            "local_itc": local_itc.detach(),
            "final_sdm": final_sdm.detach(),
            "final_itc": final_itc.detach(),
            "anchor_objective": aggregated["anchor_objective"].detach(),
            "final_objective": aggregated["final_objective"].detach(),
            "identity_group_nce": group_nce.detach(),
            "identity_gate": gate.detach(),
            "identity_score_delta_abs": (
                identity_score.detach() - observation_score.detach()
            ).abs().mean(),
            "observation_identity_cosine": observation_identity_cosine.detach(),
            "identity_projection_delta_norm": projection_delta.detach(),
            "support_valid_ratio": group["valid"].float().mean().detach(),
            "support_count_mean": group["count"].float().mean().detach(),
            "mean_image_variance": variance_mean.detach(),
            "variance_low_ratio": variance_low_ratio.detach(),
            "variance_high_ratio": variance_high_ratio.detach(),
            "mean_group_heterogeneity": heterogeneity_mean.detach(),
            "image_local_residual_norm": encoded["image_residual"].norm(dim=-1).mean().detach(),
            "text_local_residual_norm": encoded["text_residual"].norm(dim=-1).mean().detach(),
        }

    def encode_image_retrieval(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        encoded = super().encode_image_retrieval(images)
        identity = self._identity_from_observation(encoded["observation"])
        gate = self.identity_gate()
        final = build_identity_final_embedding(encoded["observation"], identity, gate)
        encoded.update({"identity": identity, "final": final})
        return encoded

    def encode_text_retrieval(self, token_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        encoded = super().encode_text_retrieval(token_ids)
        identity = self._identity_from_observation(encoded["observation"])
        gate = self.identity_gate()
        final = build_identity_final_embedding(encoded["observation"], identity, gate)
        encoded.update({"identity": identity, "final": final})
        return encoded

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode_image_retrieval(images)["final"]

    def encode_text(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.encode_text_retrieval(token_ids)["final"]


def build_hire_v2_identity_model(args, num_classes: int = 0) -> HIREV2Identity:
    model = HIREV2Identity(args, num_classes=num_classes)
    # Keep CLIP in the repository's normal fp16 representation.  All new
    # identity modules remain fp32 for stable variance/intersection computation.
    convert_weights(model.base_model)
    return model

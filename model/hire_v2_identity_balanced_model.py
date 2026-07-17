"""HIRE-v2 v16.2.1: anchor-balanced identity group consensus.

This version is an evidence-driven correction of v16.2.0.

Kept unchanged:
- CLIP global observation;
- RDE-style token selection;
- zero-initialized global/local residual fusion;
- dynamic same-PID, different-image support sampling;
- strict per-query leave-one group NCE;
- shared identity map and bounded residual gate;
- test-time support-free retrieval.

Changed:
- the observation score again receives its own direct SDM + ITC objective;
- observation and final retrieval objectives are fixed at 0.5 / 0.5;
- the empirically inactive variance/heterogeneity weighting is removed;
- support identity observations form a deterministic masked group consensus.

There is no state branch, support text, weak-positive pair loss, classifier,
MLLM teacher label, hard-negative pool, or view classifier.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F

from .clip_model import convert_weights
from .hire_v2_anchor_components import CLIPAttentionAdapter
from .hire_v2_anchor_model import HIREV2Anchor
from .hire_v2_identity_balanced_components import (
    BoundedResidualGate,
    SharedIdentityMean,
    aggregate_identity_balanced_objectives,
    build_identity_final_embedding,
    identity_residual_score,
    itc_from_similarity,
    masked_identity_group_consensus,
    paired_identity_group_nce,
    sdm_from_similarity,
)


class HIREV2IdentityBalanced(HIREV2Anchor):
    """v16.2.1 anchored observation plus identity-consensus residual."""

    is_hire_v2_anchor_model = False
    # Retain the generic identity flag so the existing component evaluator can
    # be used, while exposing a version-specific flag for audits.
    is_hire_v2_identity_model = True
    is_hire_v2_identity_balanced_model = True
    hire_v2_experiment_version = "v16.2.1"

    def __init__(self, args, num_classes: int = 0):
        super().__init__(args, num_classes=num_classes)
        self.support_size = int(getattr(args, "hire_v2_support_size", 3))
        self.auxiliary_weight = float(getattr(args, "hire_v2_aux_weight", 0.1))
        if self.support_size < 2:
            raise ValueError(
                "HIRE-v2 identity-balanced requires at least two support images"
            )
        if self.auxiliary_weight < 0.0:
            raise ValueError(
                "HIRE-v2 identity-balanced auxiliary weight must be non-negative"
            )

        # Image and text share one identity coordinate system.  Exact identity
        # initialization guarantees S_identity == S_observation at step zero.
        self.identity_mean = SharedIdentityMean(self.embed_dim)
        self.identity_gate = BoundedResidualGate(initial_value=0.1)
        self.support_encode_chunk = max(
            1, int(getattr(args, "batch_size", 64))
        )

    def _identity_from_observation(
        self, observation: torch.Tensor
    ) -> torch.Tensor:
        # Identity objectives operate on detached anchored observations.  The
        # strong global/local observation path is updated only by retrieval
        # objectives, never by the group auxiliary.
        return self.identity_mean(observation.detach())

    def _support_observations(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        required = {
            "support_images",
            "support_mask",
            "support_pids",
            "support_image_ids",
        }
        missing = required - set(batch.keys())
        if missing:
            raise RuntimeError(
                "HIRE-v2 identity-balanced requires support fields: {}".format(
                    sorted(missing)
                )
            )

        support_images = batch["support_images"]
        support_mask = batch["support_mask"].bool()
        if support_images.ndim != 5 or support_mask.ndim != 2:
            raise RuntimeError(
                "support images/mask must be [B,S,C,H,W] and [B,S]"
            )
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

        # Support images estimate the identity group only.  Their CLIP/TSE/fusion
        # path is frozen to avoid hidden support-bag gradients.  TSE is placed
        # in eval mode because it contains BatchNorm running statistics.
        image_tse_was_training = self.image_tse.training
        self.image_tse.eval()
        try:
            with torch.no_grad():
                for start in range(
                    0, flat_images.shape[0], self.support_encode_chunk
                ):
                    end = min(
                        start + self.support_encode_chunk,
                        flat_images.shape[0],
                    )
                    image_tokens, image_attention = (
                        CLIPAttentionAdapter.encode_image(
                            self.base_model, flat_images[start:end]
                        )
                    )
                    image_global = F.normalize(
                        image_tokens[:, 0, :].float(), dim=-1
                    )
                    image_local = F.normalize(
                        self.image_tse(
                            image_tokens.float(), image_attention.detach()
                        ),
                        dim=-1,
                    )
                    image_observation, _ = self.image_fusion(
                        image_global, image_local
                    )
                    observation_parts.append(image_observation.float())
        finally:
            self.image_tse.train(image_tse_was_training)

        support_observation = torch.cat(
            observation_parts, dim=0
        ).reshape(batch_size, support_size, -1)
        support_identity = self.identity_mean(
            support_observation.detach()
        )
        return {
            "observation": support_observation,
            "mean": support_identity,
            "mask": support_mask,
        }

    @staticmethod
    def _validate_support_relations(
        batch: Dict[str, torch.Tensor]
    ) -> None:
        mask = batch["support_mask"].bool()
        anchor_pids = batch["pids"].view(-1, 1)
        anchor_image_ids = batch["image_ids"].view(-1, 1)
        support_pids = batch["support_pids"]
        support_image_ids = batch["support_image_ids"]

        if (
            support_pids.shape != mask.shape
            or support_image_ids.shape != mask.shape
        ):
            raise RuntimeError(
                "support PID/image-ID tensors must match support mask"
            )
        if ((support_pids != anchor_pids) & mask).any():
            raise RuntimeError("identity supports cross a PID boundary")
        if ((support_image_ids == anchor_image_ids) & mask).any():
            raise RuntimeError("identity supports reuse the anchor image")

        for row in range(mask.shape[0]):
            valid_ids = support_image_ids[row][mask[row]]
            if valid_ids.numel() != torch.unique(valid_ids).numel():
                raise RuntimeError(
                    "identity support set contains duplicate image IDs"
                )

    def forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        encoded = self._encode_joint(
            batch["images"], batch["caption_ids"]
        )
        pids = batch["pids"].view(-1)
        self._validate_support_relations(batch)

        global_sdm, global_itc = self._retrieval_objectives(
            encoded["image_global"], encoded["text_global"], pids
        )
        local_sdm, local_itc = self._retrieval_objectives(
            encoded["image_local"], encoded["text_local"], pids
        )

        observation_score = (
            encoded["text_observation"]
            @ encoded["image_observation"].t()
        )
        observation_sdm = sdm_from_similarity(
            observation_score, pids, self.logit_scale
        )
        observation_itc = itc_from_similarity(
            observation_score, self.logit_scale
        )

        image_identity = self._identity_from_observation(
            encoded["image_observation"]
        )
        text_identity = self._identity_from_observation(
            encoded["text_observation"]
        )

        support = self._support_observations(batch)
        group = masked_identity_group_consensus(
            support["mean"], support["mask"], min_supports=2
        )

        identity_score = text_identity @ image_identity.t()
        gate = self.identity_gate()
        final_score = identity_residual_score(
            observation_score, identity_score, gate
        )
        final_sdm = sdm_from_similarity(
            final_score, pids, self.logit_scale
        )
        final_itc = itc_from_similarity(
            final_score, self.logit_scale
        )

        group_score = text_identity @ group["mean"].t()
        group_nce = paired_identity_group_nce(
            group_score, pids, group["valid"], self.logit_scale
        )

        aggregated = aggregate_identity_balanced_objectives(
            global_sdm=global_sdm,
            global_itc=global_itc,
            local_sdm=local_sdm,
            local_itc=local_itc,
            observation_sdm=observation_sdm,
            observation_itc=observation_itc,
            final_sdm=final_sdm,
            final_itc=final_itc,
            group_nce=group_nce,
            auxiliary_weight=self.auxiliary_weight,
        )

        identity_weight = self.identity_mean.proj.weight
        identity_matrix = torch.eye(
            identity_weight.shape[0],
            device=identity_weight.device,
            dtype=identity_weight.dtype,
        )
        projection_delta = (
            identity_weight - identity_matrix
        ).norm()
        observation_identity_cosine = 0.5 * (
            (
                encoded["image_observation"].detach()
                * image_identity
            ).sum(dim=-1).mean()
            + (
                encoded["text_observation"].detach()
                * text_identity
            ).sum(dim=-1).mean()
        )

        if group["valid"].any():
            group_dispersion = group["dispersion_scalar"][
                group["valid"]
            ].mean()
            group_support_cosine = group["mean_support_cosine"][
                group["valid"]
            ].mean()
        else:
            zero = final_score.sum() * 0.0
            group_dispersion = zero.detach()
            group_support_cosine = zero.detach()

        return {
            "sdm_loss": aggregated["sdm_loss"],
            "itc_loss": aggregated["itc_loss"],
            "identity_group_loss": aggregated[
                "identity_group_loss"
            ],
            "temperature": 1.0
            / self.logit_scale.to(pids.device),
            # Diagnostics intentionally avoid the substring "loss".
            "global_sdm": global_sdm.detach(),
            "global_itc": global_itc.detach(),
            "local_sdm": local_sdm.detach(),
            "local_itc": local_itc.detach(),
            "observation_sdm": observation_sdm.detach(),
            "observation_itc": observation_itc.detach(),
            "final_sdm": final_sdm.detach(),
            "final_itc": final_itc.detach(),
            "anchor_objective": aggregated[
                "anchor_objective"
            ].detach(),
            "observation_objective": aggregated[
                "observation_objective"
            ].detach(),
            "final_objective": aggregated[
                "final_objective"
            ].detach(),
            "balanced_main_objective": aggregated[
                "balanced_main_objective"
            ].detach(),
            "observation_main_weight": aggregated[
                "observation_main_weight"
            ].detach(),
            "final_main_weight": aggregated[
                "final_main_weight"
            ].detach(),
            "identity_group_nce": group_nce.detach(),
            "identity_gate": gate.detach(),
            "identity_score_delta_abs": (
                identity_score.detach()
                - observation_score.detach()
            ).abs().mean(),
            "observation_final_score_delta_abs": (
                final_score.detach()
                - observation_score.detach()
            ).abs().mean(),
            "observation_identity_cosine": (
                observation_identity_cosine.detach()
            ),
            "identity_projection_delta_norm": (
                projection_delta.detach()
            ),
            "support_valid_ratio": (
                group["valid"].float().mean().detach()
            ),
            "support_count_mean": (
                group["count"].float().mean().detach()
            ),
            "identity_group_dispersion": (
                group_dispersion.detach()
            ),
            "identity_group_support_cosine": (
                group_support_cosine.detach()
            ),
            "image_local_residual_norm": (
                encoded["image_residual"]
                .norm(dim=-1)
                .mean()
                .detach()
            ),
            "text_local_residual_norm": (
                encoded["text_residual"]
                .norm(dim=-1)
                .mean()
                .detach()
            ),
        }

    def encode_image_retrieval(
        self, images: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        encoded = super().encode_image_retrieval(images)
        identity = self._identity_from_observation(
            encoded["observation"]
        )
        gate = self.identity_gate()
        final = build_identity_final_embedding(
            encoded["observation"], identity, gate
        )
        encoded.update(
            {"identity": identity, "final": final}
        )
        return encoded

    def encode_text_retrieval(
        self, token_ids: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        encoded = super().encode_text_retrieval(token_ids)
        identity = self._identity_from_observation(
            encoded["observation"]
        )
        gate = self.identity_gate()
        final = build_identity_final_embedding(
            encoded["observation"], identity, gate
        )
        encoded.update(
            {"identity": identity, "final": final}
        )
        return encoded

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode_image_retrieval(images)["final"]

    def encode_text(
        self, token_ids: torch.Tensor
    ) -> torch.Tensor:
        return self.encode_text_retrieval(token_ids)["final"]


def build_hire_v2_identity_balanced_model(
    args, num_classes: int = 0
) -> HIREV2IdentityBalanced:
    model = HIREV2IdentityBalanced(
        args, num_classes=num_classes
    )
    # CLIP stays in the repository's normal fp16 representation.  The new
    # identity consensus modules remain fp32.
    convert_weights(model.base_model)
    return model

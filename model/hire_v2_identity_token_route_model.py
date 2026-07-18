"""HIRE-v2 v16.4.0: current-pair subject plus token-routed identity residual.

v16.4.0 is the first immediately trainable token-routing experiment.  It does
not use an external MLLM teacher.  Instead, a detached online relationship
teacher uses the current image, same-PID different-image supports, and one
highest-observation different-PID image to estimate a soft propagability target
for each selected text token.

The complete observation remains the retrieval subject and continues to use all
global/local evidence.  Only the new text identity residual is filtered by the
predicted token propagability.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F

from .clip_model import convert_weights
from .hire_v2_anchor_components import CLIPAttentionAdapter
from .hire_v2_identity_balanced_components import (
    build_identity_final_embedding,
    identity_residual_score,
    masked_identity_group_consensus,
    paired_identity_group_nce,
    sdm_from_similarity,
    itc_from_similarity,
)
from .hire_v2_identity_balanced_model import HIREV2IdentityBalanced
from .hire_v2_token_route_components import (
    AttentionRawTokenSelector,
    TokenPropagabilityRouter,
    ZeroInitializedIdentityTokenResidual,
    aggregate_identity_token_route_objectives,
    build_group_propagability_targets,
    choose_hard_negative_indices,
    masked_correlation,
    token_route_binary_cross_entropy,
)


class HIREV2IdentityTokenRoute(HIREV2IdentityBalanced):
    """v16.4.0 group-conditioned text-token identity routing."""

    is_hire_v2_anchor_model = False
    is_hire_v2_identity_model = True
    is_hire_v2_identity_balanced_model = False
    is_hire_v2_state_model = False
    is_hire_v2_identity_token_route_model = True
    hire_v2_experiment_version = "v16.4.0"

    def __init__(self, args, num_classes: int = 0):
        super().__init__(args, num_classes=num_classes)
        self.raw_token_selector = AttentionRawTokenSelector(
            ratio=self.select_ratio
        )
        self.token_router = TokenPropagabilityRouter(
            dim=self.embed_dim
        )
        self.identity_token_residual = (
            ZeroInitializedIdentityTokenResidual(
                dim=self.embed_dim
            )
        )

    def _support_bundle(
        self,
        batch: Dict[str, torch.Tensor],
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
                "v16.4.0 requires support fields: {}".format(
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
                    support_size,
                    self.support_size,
                )
            )

        flat_images = support_images.reshape(
            batch_size * support_size,
            *support_images.shape[2:]
        )
        observation_parts: List[torch.Tensor] = []
        patch_parts: List[torch.Tensor] = []
        patch_mask_parts: List[torch.Tensor] = []

        image_tse_was_training = self.image_tse.training
        self.image_tse.eval()
        try:
            with torch.no_grad():
                for start in range(
                    0,
                    flat_images.shape[0],
                    self.support_encode_chunk,
                ):
                    end = min(
                        start + self.support_encode_chunk,
                        flat_images.shape[0],
                    )
                    image_tokens, image_attention = (
                        CLIPAttentionAdapter.encode_image(
                            self.base_model,
                            flat_images[start:end],
                        )
                    )
                    image_global = F.normalize(
                        image_tokens[:, 0, :].float(),
                        dim=-1,
                    )
                    image_local = F.normalize(
                        self.image_tse(
                            image_tokens.float(),
                            image_attention.detach(),
                        ),
                        dim=-1,
                    )
                    image_observation, _ = self.image_fusion(
                        image_global,
                        image_local,
                    )
                    patch_pack = self.raw_token_selector.select_image(
                        image_tokens,
                        image_attention,
                    )
                    observation_parts.append(
                        image_observation.float()
                    )
                    patch_parts.append(
                        patch_pack["tokens"].float()
                    )
                    patch_mask_parts.append(
                        patch_pack["mask"].bool()
                    )
        finally:
            self.image_tse.train(image_tse_was_training)

        support_observation = torch.cat(
            observation_parts,
            dim=0,
        ).reshape(batch_size, support_size, -1)
        patch_tokens_flat = torch.cat(
            patch_parts,
            dim=0,
        )
        patch_mask_flat = torch.cat(
            patch_mask_parts,
            dim=0,
        )
        patch_count = patch_tokens_flat.shape[1]
        support_patch_tokens = patch_tokens_flat.reshape(
            batch_size,
            support_size,
            patch_count,
            -1,
        )
        support_patch_mask = patch_mask_flat.reshape(
            batch_size,
            support_size,
            patch_count,
        )
        support_patch_mask = (
            support_patch_mask
            & support_mask.unsqueeze(-1)
        )

        support_identity = self.identity_mean(
            support_observation.detach()
        )
        return {
            "observation": support_observation,
            "mean": support_identity,
            "mask": support_mask,
            "patch_tokens": support_patch_tokens,
            "patch_mask": support_patch_mask,
        }

    def _routed_text_identity(
        self,
        text_observation: torch.Tensor,
        text_pack: Dict[str, torch.Tensor],
        route_probability: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        base_raw = self.identity_mean.proj(
            text_observation.detach().float()
        )
        return self.identity_token_residual(
            base_identity_raw=base_raw,
            token_features=text_pack["tokens"],
            token_attention=text_pack["weights"],
            identity_probability=route_probability,
            token_mask=text_pack["mask"],
        )

    @staticmethod
    def _valid_mean(
        values: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        valid = valid.bool()
        if not bool(valid.any()):
            return values.sum() * 0.0
        return values[valid].float().mean()

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        (
            image_tokens,
            image_attention,
            text_tokens,
            text_attention,
        ) = CLIPAttentionAdapter.forward(
            self.base_model,
            batch["images"],
            batch["caption_ids"],
        )
        encoded = self._representations_from_tokens(
            image_tokens=image_tokens,
            image_attention=image_attention,
            text_tokens=text_tokens,
            text_attention=text_attention,
            token_ids=batch["caption_ids"],
        )
        image_pack = self.raw_token_selector.select_image(
            image_tokens,
            image_attention,
        )
        text_pack = self.raw_token_selector.select_text(
            text_tokens,
            batch["caption_ids"],
            text_attention,
        )

        pids = batch["pids"].view(-1)
        self._validate_support_relations(batch)

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

        observation_score = (
            encoded["text_observation"]
            @ encoded["image_observation"].t()
        )
        observation_sdm = sdm_from_similarity(
            observation_score,
            pids,
            self.logit_scale,
        )
        observation_itc = itc_from_similarity(
            observation_score,
            self.logit_scale,
        )

        support = self._support_bundle(batch)
        group = masked_identity_group_consensus(
            support["mean"],
            support["mask"],
            min_supports=2,
        )

        hard_negative_indices, hard_negative_valid = (
            choose_hard_negative_indices(
                observation_score,
                pids,
            )
        )
        hard_negative_pack = {
            "tokens": image_pack["tokens"].index_select(
                0,
                hard_negative_indices,
            ),
            "mask": image_pack["mask"].index_select(
                0,
                hard_negative_indices,
            ),
        }
        route_target = build_group_propagability_targets(
            text_pack=text_pack,
            anchor_image_pack=image_pack,
            support_image_pack={
                "tokens": support["patch_tokens"],
                "mask": support["patch_mask"],
            },
            support_mask=support["mask"],
            hard_negative_image_pack=hard_negative_pack,
            hard_negative_valid=hard_negative_valid,
            minimum_supports=2,
        )
        route_prediction = self.token_router(
            token_features=text_pack["tokens"],
            text_observation=encoded["text_observation"],
            token_mask=text_pack["mask"],
        )
        route_bce = token_route_binary_cross_entropy(
            prediction=route_prediction["probability"],
            target=route_target["target"],
            valid=route_target["valid"],
        )

        image_identity = self._identity_from_observation(
            encoded["image_observation"]
        )
        routed_text = self._routed_text_identity(
            text_observation=encoded["text_observation"],
            text_pack=text_pack,
            route_probability=route_prediction["probability"],
        )
        text_identity = routed_text["identity"]

        identity_score = text_identity @ image_identity.t()
        identity_gate = self.identity_gate()
        final_score = identity_residual_score(
            observation_score,
            identity_score,
            identity_gate,
        )
        final_sdm = sdm_from_similarity(
            final_score,
            pids,
            self.logit_scale,
        )
        final_itc = itc_from_similarity(
            final_score,
            self.logit_scale,
        )

        group_score = text_identity @ group["mean"].t()
        group_nce = paired_identity_group_nce(
            group_score,
            pids,
            group["valid"],
            self.logit_scale,
        )

        aggregated = aggregate_identity_token_route_objectives(
            global_sdm=global_sdm,
            global_itc=global_itc,
            local_sdm=local_sdm,
            local_itc=local_itc,
            observation_sdm=observation_sdm,
            observation_itc=observation_itc,
            final_sdm=final_sdm,
            final_itc=final_itc,
            group_nce=group_nce,
            route_bce=route_bce,
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

        if bool(group["valid"].any()):
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

        valid_route = route_target["valid"]
        route_probability = route_prediction["probability"]
        route_target_value = route_target["target"]
        route_entropy = -(
            route_probability.clamp(1e-6, 1.0 - 1e-6)
            * route_probability.clamp(1e-6, 1.0 - 1e-6).log()
            + (1.0 - route_probability).clamp(1e-6, 1.0)
            * (1.0 - route_probability).clamp(1e-6, 1.0).log()
        )
        route_correlation = masked_correlation(
            route_probability,
            route_target_value,
            valid_route,
        )

        return {
            "sdm_loss": aggregated["sdm_loss"],
            "itc_loss": aggregated["itc_loss"],
            "identity_group_loss": aggregated[
                "identity_group_loss"
            ],
            "token_route_loss": aggregated[
                "token_route_loss"
            ],
            "temperature": 1.0
            / self.logit_scale.to(pids.device),
            # Existing v16.2.1 diagnostics.
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
            "identity_gate": identity_gate.detach(),
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
            # v16.4.0 route diagnostics.
            "token_route_bce": route_bce.detach(),
            "token_route_valid_ratio": (
                valid_route.float().mean().detach()
            ),
            "token_route_probability_mean": self._valid_mean(
                route_probability,
                valid_route,
            ).detach(),
            "token_route_probability_std": (
                route_probability[valid_route].float().std(
                    unbiased=False
                ).detach()
                if int(valid_route.sum()) > 1
                else route_bce.detach() * 0.0
            ),
            "token_route_target_mean": self._valid_mean(
                route_target_value,
                valid_route,
            ).detach(),
            "token_route_target_std": (
                route_target_value[valid_route].float().std(
                    unbiased=False
                ).detach()
                if int(valid_route.sum()) > 1
                else route_bce.detach() * 0.0
            ),
            "token_route_high_ratio": (
                self._valid_mean(
                    route_probability.gt(0.5).float(),
                    valid_route,
                ).detach()
            ),
            "token_route_entropy": self._valid_mean(
                route_entropy,
                valid_route,
            ).detach(),
            "token_route_target_correlation": (
                route_correlation.detach()
            ),
            "token_route_stable_margin": self._valid_mean(
                route_target["stable_margin"],
                valid_route,
            ).detach(),
            "token_route_pair_margin": self._valid_mean(
                route_target["pair_margin"],
                valid_route,
            ).detach(),
            "token_route_support_std": self._valid_mean(
                route_target["support_std"],
                valid_route,
            ).detach(),
            "token_route_hard_negative_valid_ratio": (
                hard_negative_valid.float().mean().detach()
            ),
            "token_route_selected_count": (
                text_pack["mask"].float().sum(dim=1).mean().detach()
            ),
            "identity_token_residual_norm": (
                routed_text["residual"]
                .norm(dim=-1)
                .mean()
                .detach()
            ),
            "identity_token_weight_sum": (
                routed_text["weight"]
                .sum(dim=1)
                .mean()
                .detach()
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
        self,
        images: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # Image identity remains exactly the v16.2.1 identity path.
        return super().encode_image_retrieval(images)

    def encode_text_retrieval(
        self,
        token_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        text_tokens, text_attention = CLIPAttentionAdapter.encode_text(
            self.base_model,
            token_ids,
        )
        text_global = F.normalize(
            self._eot_feature(text_tokens, token_ids).float(),
            dim=-1,
        )
        text_local = F.normalize(
            self.text_tse(
                text_tokens.float(),
                token_ids,
                text_attention.detach(),
            ),
            dim=-1,
        )
        text_observation, _ = self.text_fusion(
            text_global,
            text_local,
        )
        text_pack = self.raw_token_selector.select_text(
            text_tokens,
            token_ids,
            text_attention,
        )
        route_prediction = self.token_router(
            token_features=text_pack["tokens"],
            text_observation=text_observation,
            token_mask=text_pack["mask"],
        )
        routed_text = self._routed_text_identity(
            text_observation=text_observation,
            text_pack=text_pack,
            route_probability=route_prediction["probability"],
        )
        identity = routed_text["identity"]
        final = build_identity_final_embedding(
            text_observation,
            identity,
            self.identity_gate(),
        )
        return {
            "global": text_global,
            "local": text_local,
            "observation": text_observation,
            "identity": identity,
            "final": final,
            "token_route_probability": (
                route_prediction["probability"]
            ),
            "token_route_mask": text_pack["mask"],
            "identity_token_residual": routed_text["residual"],
        }

    def encode_text(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.encode_text_retrieval(token_ids)["final"]


def build_hire_v2_identity_token_route_model(
    args,
    num_classes: int = 0,
) -> HIREV2IdentityTokenRoute:
    model = HIREV2IdentityTokenRoute(
        args,
        num_classes=num_classes,
    )
    # CLIP remains in the repository's established fp16 representation.
    # The router and token residual modules remain fp32.
    convert_weights(model.base_model)
    return model

"""HIRE-v2 v16.6.0/v16.7.0 phrase-relative identity routing.

The two versions share one architecture and differ only in the offline teacher
file supplied by the dataset:

- ``identity_phrase_route`` expects v16.6.0 propagation targets;
- ``identity_phrase_route_cmp`` expects v16.7.0 comparative targets.

No independent state score, state gate, candidate reranking, test-time support,
or test-time MLLM is used.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from .clip_model import convert_weights
from .hire_v2_anchor_components import CLIPAttentionAdapter
from .hire_v2_identity_balanced_components import (
    masked_identity_group_consensus,
)
from .hire_v2_identity_balanced_model import HIREV2IdentityBalanced
from .hire_v2_phrase_route_components import (
    RelativePhraseRouter,
    ZeroInitializedPhraseIdentityResidual,
    aggregate_identity_phrase_route_objectives,
    build_identity_final_embedding,
    identity_residual_score,
    itc_from_similarity,
    masked_distribution_entropy,
    masked_phrase_spearman,
    masked_top1_agreement,
    paired_identity_group_nce,
    phrase_attention_pool,
    phrase_route_kl_divergence,
    sdm_from_similarity,
)


class HIREV2IdentityPhraseRoute(HIREV2IdentityBalanced):
    """v16.2.1 current-pair subject plus phrase-routed identity residual."""

    is_hire_v2_anchor_model = False
    is_hire_v2_identity_model = True
    is_hire_v2_identity_balanced_model = False
    is_hire_v2_state_model = False
    is_hire_v2_identity_token_route_model = False
    is_hire_v2_phrase_route_model = True

    def __init__(self, args, num_classes: int = 0):
        super().__init__(args, num_classes=num_classes)
        mode = str(getattr(args, "hire_v2_mode", "identity_phrase_route"))
        if mode not in {"identity_phrase_route", "identity_phrase_route_cmp"}:
            raise ValueError("Unsupported phrase-route mode: {}".format(mode))
        self.phrase_route_mode = mode
        self.hire_v2_experiment_version = (
            "v16.7.0" if mode == "identity_phrase_route_cmp" else "v16.6.0"
        )
        self.phrase_router = RelativePhraseRouter(self.embed_dim)
        self.phrase_identity_residual = ZeroInitializedPhraseIdentityResidual(
            self.embed_dim
        )

    def _phrase_pack(
        self,
        text_tokens: torch.Tensor,
        text_attention: torch.Tensor,
        token_ids: torch.Tensor,
        phrase_token_mask: torch.Tensor,
        phrase_valid_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return phrase_attention_pool(
            text_tokens=text_tokens,
            token_ids=token_ids,
            text_attention=text_attention,
            phrase_token_mask=phrase_token_mask,
            phrase_valid_mask=phrase_valid_mask,
        )

    def _routed_text_identity(
        self,
        text_observation: torch.Tensor,
        phrase_pack: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        route = self.phrase_router(
            phrase_features=phrase_pack["features"],
            text_observation=text_observation,
            phrase_valid_mask=phrase_pack["valid"],
        )
        base_identity_raw = self.identity_mean.proj(
            text_observation.detach().float()
        )
        routed = self.phrase_identity_residual(
            base_identity_raw=base_identity_raw,
            phrase_features=phrase_pack["features"],
            phrase_probability=route["probability"],
            phrase_valid_mask=phrase_pack["valid"],
        )
        routed.update(
            {
                "route_logits": route["logits"],
                "route_probability": route["probability"],
            }
        )
        return routed

    @staticmethod
    def _valid_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.bool()
        if not bool(mask.any()):
            return values.sum() * 0.0
        return values[mask].float().mean()

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        required = {
            "phrase_token_mask",
            "phrase_valid_mask",
            "phrase_route_target",
            "phrase_route_supervision",
        }
        missing = required - set(batch)
        if missing:
            raise RuntimeError(
                "Phrase-route batch is missing fields: {}".format(sorted(missing))
            )

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
        phrase_pack = self._phrase_pack(
            text_tokens=text_tokens,
            text_attention=text_attention,
            token_ids=batch["caption_ids"],
            phrase_token_mask=batch["phrase_token_mask"],
            phrase_valid_mask=batch["phrase_valid_mask"],
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
        routed_text = self._routed_text_identity(
            text_observation=encoded["text_observation"],
            phrase_pack=phrase_pack,
        )
        text_identity = routed_text["identity"]

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

        route_supervision = batch["phrase_route_supervision"].view(-1).bool()
        route_kl = phrase_route_kl_divergence(
            teacher_target=batch["phrase_route_target"],
            student_probability=routed_text["route_probability"],
            phrase_valid_mask=phrase_pack["valid"],
            route_supervision_mask=route_supervision,
        )

        aggregated = aggregate_identity_phrase_route_objectives(
            global_sdm=global_sdm,
            global_itc=global_itc,
            local_sdm=local_sdm,
            local_itc=local_itc,
            observation_sdm=observation_sdm,
            observation_itc=observation_itc,
            final_sdm=final_sdm,
            final_itc=final_itc,
            group_nce=group_nce,
            route_kl=route_kl,
            auxiliary_weight=self.auxiliary_weight,
        )

        identity_weight = self.identity_mean.proj.weight
        identity_matrix = torch.eye(
            identity_weight.shape[0],
            device=identity_weight.device,
            dtype=identity_weight.dtype,
        )
        projection_delta = (identity_weight - identity_matrix).norm()
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

        route_valid_phrase = (
            phrase_pack["valid"]
            & route_supervision[:, None]
        )
        teacher_target = batch["phrase_route_target"].float()
        route_probability = routed_text["route_probability"].float()
        route_spearman = masked_phrase_spearman(
            teacher_target=teacher_target,
            student_probability=route_probability,
            phrase_valid_mask=phrase_pack["valid"],
            route_supervision_mask=route_supervision,
        )
        route_top1 = masked_top1_agreement(
            teacher_target=teacher_target,
            student_probability=route_probability,
            route_supervision_mask=route_supervision,
        )
        route_entropy = masked_distribution_entropy(
            route_probability, phrase_pack["valid"]
        )
        teacher_entropy = masked_distribution_entropy(
            teacher_target,
            phrase_pack["valid"] & route_supervision[:, None],
        )
        if int(phrase_pack["valid"].sum()) > 1:
            route_probability_std = route_probability[
                phrase_pack["valid"]
            ].std(unbiased=False)
        else:
            route_probability_std = route_kl.detach() * 0.0

        return {
            "sdm_loss": aggregated["sdm_loss"],
            "itc_loss": aggregated["itc_loss"],
            "identity_group_loss": aggregated["identity_group_loss"],
            "phrase_route_loss": aggregated["phrase_route_loss"],
            "temperature": 1.0 / self.logit_scale.to(pids.device),
            "global_sdm": global_sdm.detach(),
            "global_itc": global_itc.detach(),
            "local_sdm": local_sdm.detach(),
            "local_itc": local_itc.detach(),
            "observation_sdm": observation_sdm.detach(),
            "observation_itc": observation_itc.detach(),
            "final_sdm": final_sdm.detach(),
            "final_itc": final_itc.detach(),
            "anchor_objective": aggregated["anchor_objective"].detach(),
            "observation_objective": aggregated[
                "observation_objective"
            ].detach(),
            "final_objective": aggregated["final_objective"].detach(),
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
                identity_score.detach() - observation_score.detach()
            ).abs().mean(),
            "observation_final_score_delta_abs": (
                final_score.detach() - observation_score.detach()
            ).abs().mean(),
            "observation_identity_cosine": observation_identity_cosine.detach(),
            "identity_projection_delta_norm": projection_delta.detach(),
            "support_valid_ratio": group["valid"].float().mean().detach(),
            "support_count_mean": group["count"].float().mean().detach(),
            "identity_group_dispersion": group_dispersion.detach(),
            "identity_group_support_cosine": group_support_cosine.detach(),
            "phrase_route_kl": route_kl.detach(),
            "phrase_route_supervision_ratio": route_supervision.float().mean().detach(),
            "phrase_route_valid_phrase_ratio": route_valid_phrase.float().mean().detach(),
            "phrase_route_teacher_entropy": teacher_entropy.detach(),
            "phrase_route_student_entropy": route_entropy.detach(),
            "phrase_route_spearman": route_spearman.detach(),
            "phrase_route_top1_agreement": route_top1.detach(),
            "phrase_route_probability_max": route_probability.max(dim=1).values.mean().detach(),
            "phrase_route_probability_std": route_probability_std.detach(),
            "phrase_count_mean": phrase_pack["valid"].float().sum(dim=1).mean().detach(),
            "phrase_identity_residual_norm": routed_text["residual"].norm(dim=-1).mean().detach(),
            "image_local_residual_norm": encoded["image_residual"].norm(dim=-1).mean().detach(),
            "text_local_residual_norm": encoded["text_residual"].norm(dim=-1).mean().detach(),
        }

    def encode_text_retrieval(
        self,
        token_ids: torch.Tensor,
        phrase_token_mask: torch.Tensor = None,
        phrase_valid_mask: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        if phrase_token_mask is None or phrase_valid_mask is None:
            raise RuntimeError(
                "Phrase-route inference requires deterministic phrase spans"
            )
        text_tokens, text_attention = CLIPAttentionAdapter.encode_text(
            self.base_model, token_ids
        )
        text_global = F.normalize(
            self._eot_feature(text_tokens, token_ids).float(), dim=-1
        )
        text_local = F.normalize(
            self.text_tse(
                text_tokens.float(), token_ids, text_attention.detach()
            ),
            dim=-1,
        )
        text_observation, _ = self.text_fusion(text_global, text_local)
        phrase_pack = self._phrase_pack(
            text_tokens=text_tokens,
            text_attention=text_attention,
            token_ids=token_ids,
            phrase_token_mask=phrase_token_mask,
            phrase_valid_mask=phrase_valid_mask,
        )
        routed_text = self._routed_text_identity(
            text_observation=text_observation,
            phrase_pack=phrase_pack,
        )
        identity = routed_text["identity"]
        final = build_identity_final_embedding(
            text_observation, identity, self.identity_gate()
        )
        return {
            "global": text_global,
            "local": text_local,
            "observation": text_observation,
            "identity": identity,
            "final": final,
            "phrase_probability": routed_text["route_probability"],
            "phrase_valid": phrase_pack["valid"],
            "phrase_identity_residual": routed_text["residual"],
        }

    def encode_text(
        self,
        token_ids: torch.Tensor,
        phrase_token_mask: torch.Tensor = None,
        phrase_valid_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        return self.encode_text_retrieval(
            token_ids,
            phrase_token_mask=phrase_token_mask,
            phrase_valid_mask=phrase_valid_mask,
        )["final"]


def build_hire_v2_identity_phrase_route_model(
    args, num_classes: int = 0
) -> HIREV2IdentityPhraseRoute:
    model = HIREV2IdentityPhraseRoute(args, num_classes=num_classes)
    convert_weights(model.base_model)
    return model

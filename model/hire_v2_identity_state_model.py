"""HIRE-v2 v16.3.0: identity-balanced base plus state late interaction.

The complete v16.2.1 identity path is preserved.  The only new mechanism is a
pair-conditioned state branch that reranks the identity-balanced top-K
candidates through selected text-word / image-patch late interaction.

State supervision:
- same image_id: positive;
- different PID: negative;
- same PID but different image_id: ignored.

The state branch never consumes the same-PID support set.  Support images remain
exclusive to the identity group-consensus objective.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from .clip_model import convert_weights
from .hire_v2_anchor_components import CLIPAttentionAdapter
from .hire_v2_identity_balanced_components import (
    build_identity_final_embedding,
    identity_residual_score,
    masked_identity_group_consensus,
    paired_identity_group_nce,
)
from .hire_v2_identity_balanced_model import HIREV2IdentityBalanced
from .hire_v2_state_components import (
    AttentionStateTokenEncoder,
    SignedBoundedStateGate,
    aggregate_identity_state_objectives,
    build_state_candidate_indices,
    itc_from_similarity,
    scatter_selected_state_scores,
    sdm_from_similarity,
    selected_state_late_interaction,
    state_pair_nce,
    state_residual_score,
)


class HIREV2IdentityState(HIREV2IdentityBalanced):
    """v16.3.0 identity group consensus plus state compatibility residual."""

    is_hire_v2_anchor_model = False
    is_hire_v2_identity_model = True
    is_hire_v2_identity_balanced_model = False
    is_hire_v2_state_model = True
    hire_v2_experiment_version = "v16.3.0"

    def __init__(self, args, num_classes: int = 0):
        super().__init__(args, num_classes=num_classes)
        self.state_topk = int(getattr(args, "hire_v2_state_topk", 50))
        self.state_image_tokens = int(
            getattr(args, "hire_v2_state_image_tokens", 16)
        )
        self.state_text_tokens = int(
            getattr(args, "hire_v2_state_text_tokens", 8)
        )
        if self.state_topk < 1:
            raise ValueError("HIRE-v2 state top-K must be positive")
        if self.state_image_tokens < 2:
            raise ValueError("HIRE-v2 state image-token count must be >= 2")
        if self.state_text_tokens < 1:
            raise ValueError("HIRE-v2 state text-token count must be positive")

        # State dimension is derived from the backbone rather than introduced as
        # a fourth tunable setting.
        self.state_dim = max(1, self.embed_dim // 4)
        self.state_encoder = AttentionStateTokenEncoder(
            input_dim=self.embed_dim,
            image_token_count=self.state_image_tokens,
            text_token_count=self.state_text_tokens,
            output_dim=self.state_dim,
        )
        # Exactly zero at initialization: v16.3.0 begins with exact v16.2.1
        # retrieval values and gradients.
        self.state_gate = SignedBoundedStateGate()

    def _base_representations_from_raw(
        self,
        image_tokens: torch.Tensor,
        image_attention: torch.Tensor,
        text_tokens: torch.Tensor,
        text_attention: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return self._representations_from_tokens(
            image_tokens=image_tokens,
            image_attention=image_attention,
            text_tokens=text_tokens,
            text_attention=text_attention,
            token_ids=token_ids,
        )

    def _state_packs_from_raw(
        self,
        image_tokens: torch.Tensor,
        image_attention: torch.Tensor,
        text_tokens: torch.Tensor,
        text_attention: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        # Detaching raw CLIP tokens is a central protection rule.  State-pair
        # supervision updates only the state projection and the state gate.
        image_pack = self.state_encoder.encode_image(
            image_tokens.detach(),
            image_attention.detach(),
        )
        text_pack = self.state_encoder.encode_text(
            text_tokens.detach(),
            token_ids,
            text_attention.detach(),
        )
        return image_pack, text_pack

    @staticmethod
    def _relation_diagnostics(
        state_score: torch.Tensor,
        candidate_mask: torch.Tensor,
        pids: torch.Tensor,
        image_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        pids = pids.view(-1)
        image_ids = image_ids.view(-1)
        positive = image_ids[:, None].eq(image_ids[None, :]) & candidate_mask
        negative = pids[:, None].ne(pids[None, :]) & candidate_mask
        zero = state_score.sum() * 0.0
        positive_mean = (
            state_score[positive].mean() if bool(positive.any()) else zero
        )
        negative_mean = (
            state_score[negative].mean() if bool(negative.any()) else zero
        )
        return {
            "state_positive_score": positive_mean,
            "state_negative_score": negative_mean,
            "state_positive_negative_margin": positive_mean - negative_mean,
        }

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
        encoded = self._base_representations_from_raw(
            image_tokens,
            image_attention,
            text_tokens,
            text_attention,
            batch["caption_ids"],
        )
        image_state_pack, text_state_pack = self._state_packs_from_raw(
            image_tokens,
            image_attention,
            text_tokens,
            text_attention,
            batch["caption_ids"],
        )

        pids = batch["pids"].view(-1)
        image_ids = batch["image_ids"].view(-1)
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

        # v16.2.1 identity path: unchanged.
        image_identity = self._identity_from_observation(
            encoded["image_observation"]
        )
        text_identity = self._identity_from_observation(
            encoded["text_observation"]
        )
        support = self._support_observations(batch)
        group = masked_identity_group_consensus(
            support["mean"],
            support["mask"],
            min_supports=2,
        )
        identity_score = text_identity @ image_identity.t()
        identity_gate = self.identity_gate()
        identity_final_score = identity_residual_score(
            observation_score,
            identity_score,
            identity_gate,
        )
        identity_final_sdm = sdm_from_similarity(
            identity_final_score,
            pids,
            self.logit_scale,
        )
        identity_final_itc = itc_from_similarity(
            identity_final_score,
            self.logit_scale,
        )
        group_score = text_identity @ group["mean"].t()
        identity_group_nce = paired_identity_group_nce(
            group_score,
            pids,
            group["valid"],
            self.logit_scale,
        )

        # State candidates are determined only by a detached identity-balanced
        # score.  All same-image positives are guaranteed to enter in training.
        (
            candidate_indices,
            candidate_mask,
            positive_mask,
        ) = build_state_candidate_indices(
            identity_final_score,
            image_ids,
            self.state_topk,
        )
        selected_state = selected_state_late_interaction(
            text_state_pack,
            image_state_pack,
            candidate_indices,
        )
        state_score = scatter_selected_state_scores(
            identity_final_score,
            selected_state["score"],
            candidate_indices,
        )
        state_nce = state_pair_nce(
            scores=state_score,
            pids=pids,
            image_ids=image_ids,
            candidate_mask=candidate_mask,
            logit_scale=self.logit_scale,
        )

        state_gate = self.state_gate()
        state_final_score = state_residual_score(
            identity_base_score=identity_final_score,
            state_score=state_score,
            candidate_mask=candidate_mask,
            state_gate=state_gate,
        )
        state_final_sdm = sdm_from_similarity(
            state_final_score,
            pids,
            self.logit_scale,
        )
        state_final_itc = itc_from_similarity(
            state_final_score,
            self.logit_scale,
        )

        aggregated = aggregate_identity_state_objectives(
            global_sdm=global_sdm,
            global_itc=global_itc,
            local_sdm=local_sdm,
            local_itc=local_itc,
            observation_sdm=observation_sdm,
            observation_itc=observation_itc,
            identity_final_sdm=identity_final_sdm,
            identity_final_itc=identity_final_itc,
            state_final_sdm=state_final_sdm,
            state_final_itc=state_final_itc,
            identity_group_nce=identity_group_nce,
            state_nce=state_nce,
            auxiliary_weight=self.auxiliary_weight,
        )

        identity_weight = self.identity_mean.proj.weight
        identity_matrix = torch.eye(
            identity_weight.shape[0],
            device=identity_weight.device,
            dtype=identity_weight.dtype,
        )
        identity_projection_delta = (
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
            zero = state_final_score.sum() * 0.0
            group_dispersion = zero.detach()
            group_support_cosine = zero.detach()

        relation = self._relation_diagnostics(
            state_score=state_score,
            candidate_mask=candidate_mask,
            pids=pids,
            image_ids=image_ids,
        )
        selected_delta = (
            selected_state["score"]
            - identity_final_score.detach().gather(
                1,
                candidate_indices,
            )
        )
        positive_coverage = (
            (candidate_mask & positive_mask).sum(dim=1)
            .eq(positive_mask.sum(dim=1))
            .float()
            .mean()
        )

        return {
            "sdm_loss": aggregated["sdm_loss"],
            "itc_loss": aggregated["itc_loss"],
            "identity_group_loss": aggregated["identity_group_loss"],
            "state_pair_loss": aggregated["state_pair_loss"],
            "temperature": 1.0 / self.logit_scale.to(pids.device),
            # Base observation and identity diagnostics.
            "global_sdm": global_sdm.detach(),
            "global_itc": global_itc.detach(),
            "local_sdm": local_sdm.detach(),
            "local_itc": local_itc.detach(),
            "observation_sdm": observation_sdm.detach(),
            "observation_itc": observation_itc.detach(),
            "identity_final_sdm": identity_final_sdm.detach(),
            "identity_final_itc": identity_final_itc.detach(),
            "state_final_sdm": state_final_sdm.detach(),
            "state_final_itc": state_final_itc.detach(),
            "anchor_objective": aggregated["anchor_objective"].detach(),
            "observation_objective": aggregated[
                "observation_objective"
            ].detach(),
            "identity_final_objective": aggregated[
                "identity_final_objective"
            ].detach(),
            "state_final_objective": aggregated[
                "state_final_objective"
            ].detach(),
            "hierarchical_main_objective": aggregated[
                "hierarchical_main_objective"
            ].detach(),
            "observation_main_weight": aggregated[
                "observation_main_weight"
            ].detach(),
            "identity_main_weight": aggregated[
                "identity_main_weight"
            ].detach(),
            "state_final_main_weight": aggregated[
                "state_final_main_weight"
            ].detach(),
            "identity_group_nce": identity_group_nce.detach(),
            "identity_gate": identity_gate.detach(),
            "identity_score_delta_abs": (
                identity_score.detach()
                - observation_score.detach()
            ).abs().mean(),
            "observation_identity_cosine": (
                observation_identity_cosine.detach()
            ),
            "identity_projection_delta_norm": (
                identity_projection_delta.detach()
            ),
            "support_valid_ratio": (
                group["valid"].float().mean().detach()
            ),
            "support_count_mean": (
                group["count"].float().mean().detach()
            ),
            "identity_group_dispersion": group_dispersion.detach(),
            "identity_group_support_cosine": (
                group_support_cosine.detach()
            ),
            # State-specific diagnostics.
            "state_pair_nce": state_nce.detach(),
            "state_gate": state_gate.detach(),
            "state_candidate_ratio": (
                candidate_mask.float().mean().detach()
            ),
            "state_positive_coverage": positive_coverage.detach(),
            "state_score_delta_abs": (
                selected_delta.abs().mean().detach()
            ),
            "identity_state_final_delta_abs": (
                state_final_score.detach()
                - identity_final_score.detach()
            ).abs().mean(),
            "state_peak_mean": (
                selected_state["peak_mean"].mean().detach()
            ),
            "state_peak_margin": (
                selected_state["margin_mean"].mean().detach()
            ),
            "state_positive_score": relation[
                "state_positive_score"
            ].detach(),
            "state_negative_score": relation[
                "state_negative_score"
            ].detach(),
            "state_positive_negative_margin": relation[
                "state_positive_negative_margin"
            ].detach(),
            "state_text_token_count": (
                text_state_pack["mask"]
                .float()
                .sum(dim=1)
                .mean()
                .detach()
            ),
            "state_image_token_count": (
                image_state_pack["mask"]
                .float()
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

    def encode_image_state_retrieval(
        self,
        images: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        image_tokens, image_attention = CLIPAttentionAdapter.encode_image(
            self.base_model,
            images,
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
        image_identity = self._identity_from_observation(
            image_observation
        )
        identity_final = build_identity_final_embedding(
            image_observation,
            image_identity,
            self.identity_gate(),
        )
        state_pack = self.state_encoder.encode_image(
            image_tokens.detach(),
            image_attention.detach(),
        )
        return {
            "global": image_global,
            "local": image_local,
            "observation": image_observation,
            "identity": image_identity,
            "identity_final": identity_final,
            "state_tokens": state_pack["tokens"],
            "state_mask": state_pack["mask"],
        }

    def encode_text_state_retrieval(
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
        text_identity = self._identity_from_observation(
            text_observation
        )
        identity_final = build_identity_final_embedding(
            text_observation,
            text_identity,
            self.identity_gate(),
        )
        state_pack = self.state_encoder.encode_text(
            text_tokens.detach(),
            token_ids,
            text_attention.detach(),
        )
        return {
            "global": text_global,
            "local": text_local,
            "observation": text_observation,
            "identity": text_identity,
            "identity_final": identity_final,
            "state_tokens": state_pack["tokens"],
            "state_mask": state_pack["mask"],
            "state_weights": state_pack["weights"],
        }

    # Generic component tooling receives the v16.2.1 identity-balanced vector.
    # The pair-conditioned state result is computed by the dedicated evaluator.
    def encode_image_retrieval(
        self,
        images: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        encoded = self.encode_image_state_retrieval(images)
        encoded["final"] = encoded["identity_final"]
        return encoded

    def encode_text_retrieval(
        self,
        token_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        encoded = self.encode_text_state_retrieval(token_ids)
        encoded["final"] = encoded["identity_final"]
        return encoded

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode_image_state_retrieval(images)[
            "identity_final"
        ]

    def encode_text(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.encode_text_state_retrieval(token_ids)[
            "identity_final"
        ]

    @torch.no_grad()
    def compute_state_reranked_similarity(
        self,
        text_repr: Dict[str, torch.Tensor],
        image_repr: Dict[str, torch.Tensor],
        query_chunk: int = 128,
    ) -> Dict[str, torch.Tensor]:
        """Compute full identity-base scores and exact top-K state reranking."""
        required_text = {
            "identity_final",
            "state_tokens",
            "state_mask",
            "state_weights",
        }
        required_image = {
            "identity_final",
            "state_tokens",
            "state_mask",
        }
        if required_text - set(text_repr):
            raise ValueError("missing text state representations")
        if required_image - set(image_repr):
            raise ValueError("missing image state representations")

        device = next(self.parameters()).device
        gallery_base = F.normalize(
            image_repr["identity_final"].float(),
            dim=-1,
        ).to(device)
        gallery_state = {
            "tokens": image_repr["state_tokens"].float().to(device),
            "mask": image_repr["state_mask"].bool().to(device),
        }
        query_count = text_repr["identity_final"].shape[0]
        gallery_count = gallery_base.shape[0]
        k = min(self.state_topk, gallery_count)
        base_output = torch.empty(
            query_count,
            gallery_count,
            dtype=torch.float32,
        )
        final_output = torch.empty_like(base_output)
        gate = self.state_gate().to(device=device, dtype=torch.float32)

        for start in range(0, query_count, int(query_chunk)):
            end = min(start + int(query_chunk), query_count)
            query_base = F.normalize(
                text_repr["identity_final"][start:end].float(),
                dim=-1,
            ).to(device)
            base_score = query_base @ gallery_base.t()
            candidate_indices = base_score.topk(
                k=k,
                dim=1,
                largest=True,
                sorted=True,
            ).indices
            query_state = {
                "tokens": text_repr["state_tokens"][
                    start:end
                ].float().to(device),
                "mask": text_repr["state_mask"][
                    start:end
                ].bool().to(device),
                "weights": text_repr["state_weights"][
                    start:end
                ].float().to(device),
            }
            selected_state = selected_state_late_interaction(
                query_state,
                gallery_state,
                candidate_indices,
            )["score"]
            selected_base = base_score.gather(
                1,
                candidate_indices,
            )
            selected_final = selected_base + gate * (
                selected_state - selected_base
            )
            final_score = base_score.clone()
            final_score.scatter_(
                1,
                candidate_indices,
                selected_final,
            )
            base_output[start:end] = base_score.float().cpu()
            final_output[start:end] = final_score.float().cpu()

        return {
            "identity_final": base_output,
            "state_final": final_output,
        }


def build_hire_v2_identity_state_model(
    args,
    num_classes: int = 0,
) -> HIREV2IdentityState:
    model = HIREV2IdentityState(
        args,
        num_classes=num_classes,
    )
    # Pretrained CLIP remains fp16; all new state modules stay fp32.
    convert_weights(model.base_model)
    return model

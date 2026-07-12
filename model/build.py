from model import objectives
from .clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class IRRA(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()
        self.irra_light = bool(getattr(args, 'irra_light', False)) or 'irra_light' in self.current_task
        self.irra_light_mode = getattr(args, 'irra_light_mode', 'single_pure')
        self.irra_light_bag = self.irra_light and self.irra_light_mode in {
            'single_proj_bag',
            'split_bag',
            'single_proj_bag_consistency',
            'split_bag_consistency',
        }
        self.irra_light_bag_consistency = self.irra_light and self.irra_light_mode in {
            'single_proj_bag_consistency',
            'split_bag_consistency',
        }
        self.irra_light_fast_bag = self.irra_light and self.irra_light_mode in {
            'split_bag_safe',
            'split_bag_state',
            'split_bag_state_hn',
        }
        self.irra_light_state_route = self.irra_light and self.irra_light_mode in {
            'split_bag_state',
            'split_bag_state_hn',
        }
        self.irra_light_hard_negative = (
            self.irra_light and self.irra_light_mode == 'split_bag_state_hn'
        )
        self.irra_light_single_proj = self.irra_light and self.irra_light_mode in {
            'single_proj_pure',
            'single_proj_id',
            'single_proj_bag',
            'single_proj_bag_consistency',
        }
        self.irra_light_split = self.irra_light and self.irra_light_mode in {
            'split_pure',
            'split_id',
            'split_bag',
            'split_bag_consistency',
            'split_bag_safe',
            'split_bag_state',
            'split_bag_state_hn',
        }
        self.irra_light_with_id = self.irra_light and self.irra_light_mode in {'single_id', 'single_proj_id', 'split_id'}

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']

        self.register_buffer('logit_scale', torch.ones([]) * (1 / args.temperature))

        if self.irra_light_single_proj:
            self.single_head = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
            nn.init.eye_(self.single_head.weight)

        if self.irra_light_split:
            self.identity_head = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
            self.state_head = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
            nn.init.eye_(self.identity_head.weight)
            nn.init.eye_(self.state_head.weight)

        if self.irra_light_with_id or ((not self.irra_light) and 'id' in args.loss_names):
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

        if (not self.irra_light) and 'mlm' in args.loss_names:
            self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                       layers=args.cmt_depth,
                                                       heads=self.embed_dim //
                                                       64)
            scale = self.cross_modal_transformer.width**-0.5
            
            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers)**-0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width)**-0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                            ('gelu', QuickGELU()),
                            ('ln', LayerNorm(self.embed_dim)),
                            ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))
            # init mlm head
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')

    def float_projection_heads(self):
        if self.irra_light_single_proj:
            self.single_head.float()
        if self.irra_light_split:
            self.identity_head.float()
            self.state_head.float()
        if self.irra_light_with_id and hasattr(self, 'classifier'):
            self.classifier.float()

    def _project_light_head(self, head, feats):
        return F.normalize(head(feats), dim=-1)

    def _masked_ratio_loss(self, pos_logits, neg_logits, neg_mask,
                           support_logits=None, support_mask=None, support_weights=None):
        neg_inf = torch.finfo(pos_logits.dtype).min
        pos_logits = pos_logits.unsqueeze(1)

        numerator_terms = [pos_logits]
        denominator_terms = [pos_logits]

        if support_logits is not None:
            if support_mask is None:
                raise ValueError("support_mask is required when support_logits is provided")
            if support_weights is not None:
                support_weights = support_weights.to(dtype=support_logits.dtype, device=support_logits.device)
                support_mask = support_mask & (support_weights > 0)
                support_logits = support_logits + torch.log(support_weights.clamp_min(1e-12))
            support_logits = support_logits.masked_fill(~support_mask, neg_inf)
            numerator_terms.append(support_logits)
            denominator_terms.append(support_logits)

        neg_logits = neg_logits.masked_fill(~neg_mask, neg_inf)
        denominator_terms.append(neg_logits)

        numerator = torch.logsumexp(torch.cat(numerator_terms, dim=1), dim=1)
        denominator = torch.logsumexp(torch.cat(denominator_terms, dim=1), dim=1)
        return -(numerator - denominator).mean()

    def _source_pair_loss(self, image_feats, text_feats, pids, logit_scale):
        pids = pids.view(-1)
        logits_t2i = torch.matmul(text_feats, image_feats.t()) * logit_scale
        logits_i2t = logits_t2i.t()
        neg_mask = pids.view(-1, 1) != pids.view(1, -1)

        t2i_loss = self._masked_ratio_loss(
            logits_t2i.diag(),
            logits_t2i,
            neg_mask,
        )
        i2t_loss = self._masked_ratio_loss(
            logits_i2t.diag(),
            logits_i2t,
            neg_mask,
        )
        return (t2i_loss + i2t_loss) / 2

    def _support_set_loss(self, image_feats, text_feats, support_i_feats,
                          support_t_feats, support_mask, pids, logit_scale,
                          support_weights=None):
        pids = pids.view(-1)
        neg_mask = pids.view(-1, 1) != pids.view(1, -1)

        logits_t2i = torch.matmul(text_feats, image_feats.t()) * logit_scale
        support_t2i = torch.sum(text_feats.unsqueeze(1) * support_i_feats, dim=-1) * logit_scale
        t2i_loss = self._masked_ratio_loss(
            logits_t2i.diag(),
            logits_t2i,
            neg_mask,
            support_logits=support_t2i,
            support_mask=support_mask,
            support_weights=support_weights,
        )

        logits_i2t = logits_t2i.t()
        support_i2t = torch.sum(image_feats.unsqueeze(1) * support_t_feats, dim=-1) * logit_scale
        i2t_loss = self._masked_ratio_loss(
            logits_i2t.diag(),
            logits_i2t,
            neg_mask,
            support_logits=support_i2t,
            support_mask=support_mask,
            support_weights=support_weights,
        )
        return (t2i_loss + i2t_loss) / 2

    def _encode_support_bag(self, batch):
        if 'support_images' not in batch or 'support_caption_ids' not in batch or 'support_mask' not in batch:
            raise RuntimeError("support-bag modes require support_images, support_caption_ids, and support_mask")

        support_images = batch['support_images']
        support_caption_ids = batch['support_caption_ids']
        support_mask = batch['support_mask'].bool()
        batch_size, support_size = support_mask.shape

        flat_images = support_images.reshape(
            batch_size * support_size,
            *support_images.shape[2:],
        )
        flat_caption_ids = support_caption_ids.reshape(
            batch_size * support_size,
            support_caption_ids.shape[-1],
        )

        support_i_chunks = []
        support_t_chunks = []
        support_encode_chunk = max(1, int(getattr(self.args, 'batch_size', 64)))
        with torch.no_grad():
            for start in range(0, flat_images.shape[0], support_encode_chunk):
                end = min(start + support_encode_chunk, flat_images.shape[0])
                support_image_feats, support_text_feats = self.base_model(
                    flat_images[start:end],
                    flat_caption_ids[start:end],
                )
                support_i_chunks.append(support_image_feats[:, 0, :].float())
                support_t_chunks.append(
                    support_text_feats[
                        torch.arange(support_text_feats.shape[0], device=flat_caption_ids.device),
                        flat_caption_ids[start:end].argmax(dim=-1),
                    ].float()
                )

        support_i_feats = torch.cat(support_i_chunks, dim=0)
        support_t_feats = torch.cat(support_t_chunks, dim=0)

        if self.irra_light_split:
            support_i_feats = self._project_light_head(self.identity_head, support_i_feats)
            support_t_feats = self._project_light_head(self.identity_head, support_t_feats)
        elif self.irra_light_single_proj:
            support_i_feats = self._project_light_head(self.single_head, support_i_feats)
            support_t_feats = self._project_light_head(self.single_head, support_t_feats)
        else:
            support_i_feats = F.normalize(support_i_feats, dim=-1)
            support_t_feats = F.normalize(support_t_feats, dim=-1)

        support_i_feats = support_i_feats.view(batch_size, support_size, -1)
        support_t_feats = support_t_feats.view(batch_size, support_size, -1)
        return support_i_feats, support_t_feats, support_mask

    def _encode_support_images_raw(self, batch):
        """Encode fast3 support images with a frozen CLIP backbone only."""
        required = {'support_images', 'support_mask'}
        missing = required - set(batch.keys())
        if missing:
            raise RuntimeError(f"v16 fast3 requires support image fields: {sorted(missing)}")

        support_images = batch['support_images']
        support_mask = batch['support_mask'].bool()
        if support_images.ndim != 5 or support_mask.ndim != 2:
            raise RuntimeError(
                "support_images must be [batch, support, C, H, W] and support_mask [batch, support]"
            )
        batch_size, support_size = support_mask.shape
        if support_images.shape[:2] != (batch_size, support_size):
            raise RuntimeError("support_images and support_mask shapes are inconsistent")

        flat_images = support_images.reshape(
            batch_size * support_size,
            *support_images.shape[2:],
        )
        raw_chunks = []
        support_encode_chunk = max(1, int(getattr(self.args, 'batch_size', 64)))
        with torch.no_grad():
            for start in range(0, flat_images.shape[0], support_encode_chunk):
                end = min(start + support_encode_chunk, flat_images.shape[0])
                encoded = self.base_model.encode_image(flat_images[start:end])
                raw_chunks.append(encoded[:, 0, :].float())
        support_raw_feats = torch.cat(raw_chunks, dim=0).view(batch_size, support_size, -1)
        return support_raw_feats, support_mask

    def _encode_hard_negative_images_raw(self, batch):
        """Encode one optional v16 fast3 hard-negative image per anchor."""
        required = {'hard_negative_image', 'hard_negative_mask'}
        missing = required - set(batch.keys())
        if missing:
            raise RuntimeError(f"v16 fast3 hard-negative mode requires: {sorted(missing)}")
        hard_negative_images = batch['hard_negative_image']
        hard_negative_mask = batch['hard_negative_mask'].bool().view(-1)
        with torch.no_grad():
            encoded = self.base_model.encode_image(hard_negative_images)
            hard_negative_raw_feats = encoded[:, 0, :].float()
        return hard_negative_raw_feats, hard_negative_mask

    def _masked_logmeanexp(self, logits, mask, weights=None):
        """Masked log(mean(exp(logits))) with a differentiable zero for empty rows."""
        if logits.shape != mask.shape:
            raise ValueError(
                f"logits/mask shape mismatch: {tuple(logits.shape)} vs {tuple(mask.shape)}"
            )
        valid_mask = mask.bool()
        adjusted_logits = logits
        if weights is None:
            normalizer = valid_mask.to(dtype=logits.dtype).sum(dim=1)
        else:
            if weights.shape != logits.shape:
                raise ValueError(
                    "weights must have the same shape as logits in _masked_logmeanexp"
                )
            weights = weights.to(dtype=logits.dtype, device=logits.device)
            valid_mask = valid_mask & (weights > 0)
            adjusted_logits = logits + torch.log(weights.clamp_min(1e-12))
            normalizer = torch.where(
                valid_mask,
                weights,
                torch.zeros_like(weights),
            ).sum(dim=1)

        valid_rows = normalizer > 0
        neg_inf = torch.finfo(logits.dtype).min
        masked_logits = adjusted_logits.masked_fill(~valid_mask, neg_inf)
        raw_score = torch.logsumexp(masked_logits, dim=1) - torch.log(
            normalizer.clamp_min(1.0)
        )
        zero = logits.sum(dim=1) * 0.0
        score = torch.where(valid_rows, raw_score, zero)
        if not torch.isfinite(score).all():
            raise RuntimeError("_masked_logmeanexp produced a non-finite score")
        return score, valid_rows

    def _support_bag_rank_loss(
        self,
        image_identity_feats,
        text_identity_feats,
        support_identity_feats,
        support_mask,
        support_weights,
        pids,
        logit_scale,
        hard_negative_feats=None,
        hard_negative_mask=None,
        hard_negative_image_ids=None,
        batch_image_ids=None,
    ):
        """One-way text-to-support-image rank loss for the v16 fast3 modes."""
        pids = pids.view(-1)
        support_mask = support_mask.bool()
        if support_identity_feats.shape[:2] != support_mask.shape:
            raise ValueError("support features and support mask have incompatible shapes")
        if support_weights is None:
            support_weights = torch.ones_like(support_mask, dtype=text_identity_feats.dtype)
        else:
            support_weights = support_weights.to(
                dtype=text_identity_feats.dtype,
                device=text_identity_feats.device,
            )
        if support_weights.shape != support_mask.shape:
            raise ValueError("support weights and support mask have incompatible shapes")

        support_logits = torch.sum(
            text_identity_feats.unsqueeze(1) * support_identity_feats,
            dim=-1,
        ) * logit_scale
        positive_score, support_valid_rows = self._masked_logmeanexp(
            support_logits,
            support_mask,
            weights=support_weights,
        )

        main_negative_logits = torch.matmul(
            text_identity_feats,
            image_identity_feats.t(),
        ) * logit_scale
        negative_mask = pids.view(-1, 1) != pids.view(1, -1)
        hard_negative_valid = torch.zeros_like(support_valid_rows)
        if hard_negative_feats is not None:
            if hard_negative_mask is None:
                raise ValueError("hard_negative_mask is required with hard_negative_feats")
            hard_negative_mask = hard_negative_mask.bool().view(-1)
            if hard_negative_feats.shape[0] != text_identity_feats.shape[0]:
                raise ValueError("hard-negative feature count must match batch size")
            if hard_negative_mask.shape[0] != text_identity_feats.shape[0]:
                raise ValueError("hard-negative mask count must match batch size")
            if hard_negative_image_ids is not None and batch_image_ids is not None:
                hard_negative_image_ids = hard_negative_image_ids.view(-1)
                batch_image_ids = batch_image_ids.view(-1)
                if hard_negative_image_ids.shape[0] != hard_negative_mask.shape[0]:
                    raise ValueError("hard-negative image IDs must match batch size")
                duplicate_in_batch = hard_negative_image_ids.view(-1, 1).eq(
                    batch_image_ids.view(1, -1)
                ).any(dim=1)
                hard_negative_mask = hard_negative_mask & ~duplicate_in_batch
            hard_negative_logits = torch.sum(
                text_identity_feats * hard_negative_feats,
                dim=-1,
            ).unsqueeze(1) * logit_scale
            main_negative_logits = torch.cat(
                [main_negative_logits, hard_negative_logits], dim=1
            )
            negative_mask = torch.cat(
                [negative_mask, hard_negative_mask.unsqueeze(1)], dim=1
            )
            hard_negative_valid = hard_negative_mask

        negative_score, negative_valid_rows = self._masked_logmeanexp(
            main_negative_logits,
            negative_mask,
        )
        valid_rows = support_valid_rows & negative_valid_rows
        if valid_rows.any():
            loss = F.softplus(negative_score[valid_rows] - positive_score[valid_rows]).mean()
        else:
            loss = text_identity_feats.sum() * 0.0
        return loss, valid_rows, support_valid_rows, hard_negative_valid

    def _state_nontransitive_loss(
        self,
        image_state_feats,
        text_state_feats,
        support_state_feats,
        support_mask,
        support_conflict_mask,
        logit_scale,
    ):
        """Keep state alignment local when a support image has an explicit conflict."""
        support_mask = support_mask.bool()
        support_conflict_mask = support_conflict_mask.bool()
        if support_mask.shape != support_conflict_mask.shape:
            raise ValueError("support conflict mask must match support mask")
        if support_state_feats.shape[:2] != support_mask.shape:
            raise ValueError("support state features and support mask have incompatible shapes")

        paired_scores = torch.sum(text_state_feats * image_state_feats, dim=-1) * logit_scale
        conflict_scores = torch.sum(
            text_state_feats.unsqueeze(1) * support_state_feats,
            dim=-1,
        ) * logit_scale
        conflict_score, conflict_valid_rows = self._masked_logmeanexp(
            conflict_scores,
            support_mask & support_conflict_mask,
        )
        if conflict_valid_rows.any():
            loss = F.softplus(
                conflict_score[conflict_valid_rows]
                - paired_scores[conflict_valid_rows]
            ).mean()
        else:
            loss = text_state_feats.sum() * 0.0
        return loss, conflict_valid_rows
    
    
    def cross_former(self, q, k, v):
        x = self.cross_attn(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

    def encode_image_heads(self, image):
        """Return identity and state image embeddings for split-head offline evaluation."""
        x = self.base_model.encode_image(image)
        x = x[:, 0, :].float()
        if self.irra_light_split:
            return {
                'identity': self._project_light_head(self.identity_head, x),
                'state': self._project_light_head(self.state_head, x),
            }
        if self.irra_light_single_proj:
            projected = self._project_light_head(self.single_head, x)
            return {'identity': projected, 'state': projected}
        return {'identity': x, 'state': x}

    def encode_text_heads(self, text):
        """Return identity and state text embeddings for split-head offline evaluation."""
        x = self.base_model.encode_text(text)
        x = x[torch.arange(x.shape[0], device=text.device), text.argmax(dim=-1)].float()
        if self.irra_light_split:
            return {
                'identity': self._project_light_head(self.identity_head, x),
                'state': self._project_light_head(self.state_head, x),
            }
        if self.irra_light_single_proj:
            projected = self._project_light_head(self.single_head, x)
            return {'identity': projected, 'state': projected}
        return {'identity': x, 'state': x}

    def encode_image(self, image):
        x = self.base_model.encode_image(image)
        x = x[:, 0, :].float()
        if self.irra_light_split:
            return self._project_light_head(self.identity_head, x)
        if self.irra_light_single_proj:
            return self._project_light_head(self.single_head, x)
        return x
        # return x.float() # for CLIP ResNet visual model

    def encode_text(self, text):
        x = self.base_model.encode_text(text)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()
        if self.irra_light_split:
            return self._project_light_head(self.identity_head, x)
        if self.irra_light_single_proj:
            return self._project_light_head(self.single_head, x)
        return x

    def forward(self, batch):
        ret = dict()

        images = batch['images']
        caption_ids = batch['caption_ids']
        image_feats, text_feats = self.base_model(images, caption_ids)
        i_feats = image_feats[:, 0, :].float()
        # i_feats = image_feats.float() # for CLIP ResNet visual model
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

        logit_scale = self.logit_scale.to(i_feats.device)
        ret.update({'temperature': 1 / logit_scale})

        if self.irra_light:
            if self.irra_light_split:
                identity_i_feats = self._project_light_head(self.identity_head, i_feats)
                identity_t_feats = self._project_light_head(self.identity_head, t_feats)
                state_i_feats = self._project_light_head(self.state_head, i_feats)
                state_t_feats = self._project_light_head(self.state_head, t_feats)
            elif self.irra_light_single_proj:
                identity_i_feats = self._project_light_head(self.single_head, i_feats)
                identity_t_feats = self._project_light_head(self.single_head, t_feats)
                state_i_feats = identity_i_feats
                state_t_feats = identity_t_feats
            else:
                identity_i_feats = i_feats
                identity_t_feats = t_feats
                state_i_feats = i_feats
                state_t_feats = t_feats

            if self.irra_light_fast_bag:
                if self.args.irra_light_identity_loss != 'sdm':
                    raise ValueError(
                        "v16 fast3 requires --irra_light_identity_loss sdm"
                    )
                if 'support_reliability' not in batch:
                    raise RuntimeError("v16 fast3 requires support_reliability")

                ret.update({
                    'identity_sdm_loss': objectives.compute_sdm(
                        identity_i_feats,
                        identity_t_feats,
                        batch['pids'],
                        logit_scale,
                    ),
                    'state_itc_loss': objectives.compute_itc(
                        state_i_feats,
                        state_t_feats,
                        logit_scale,
                    ),
                })

                support_raw_feats, support_mask = self._encode_support_images_raw(batch)
                support_identity_feats = self._project_light_head(
                    self.identity_head,
                    support_raw_feats,
                )
                support_weights = batch['support_reliability']

                if self.irra_light_state_route:
                    if 'support_conflict_mask' not in batch:
                        raise RuntimeError(
                            "v16 fast3 state modes require support_conflict_mask"
                        )
                    support_state_feats = self._project_light_head(
                        self.state_head,
                        support_raw_feats,
                    )
                    state_nontransitive_loss, conflict_valid_rows = (
                        self._state_nontransitive_loss(
                            state_i_feats,
                            state_t_feats,
                            support_state_feats,
                            support_mask,
                            batch['support_conflict_mask'],
                            logit_scale,
                        )
                    )
                    ret.update({
                        'state_nontransitive_loss': state_nontransitive_loss,
                        'support_conflict_anchor_ratio': conflict_valid_rows.float().mean(),
                    })

                hard_negative_feats = None
                hard_negative_mask = None
                hard_negative_image_ids = None
                if self.irra_light_hard_negative:
                    if 'hard_negative_image_id' not in batch:
                        raise RuntimeError(
                            "v16 fast3 hard-negative mode requires hard_negative_image_id"
                        )
                    hard_negative_raw_feats, hard_negative_mask = (
                        self._encode_hard_negative_images_raw(batch)
                    )
                    hard_negative_feats = self._project_light_head(
                        self.identity_head,
                        hard_negative_raw_feats,
                    )
                    hard_negative_image_ids = batch['hard_negative_image_id']

                identity_bag_loss, _valid_rows, support_valid_rows, hard_negative_valid = (
                    self._support_bag_rank_loss(
                        identity_i_feats,
                        identity_t_feats,
                        support_identity_feats,
                        support_mask,
                        support_weights,
                        batch['pids'],
                        logit_scale,
                        hard_negative_feats=hard_negative_feats,
                        hard_negative_mask=hard_negative_mask,
                        hard_negative_image_ids=hard_negative_image_ids,
                        batch_image_ids=batch.get('image_ids'),
                    )
                )
                ret.update({
                    'identity_bag_loss': identity_bag_loss,
                    'support_valid_ratio': support_valid_rows.float().mean(),
                })
                valid_weights = support_weights[support_mask]
                if valid_weights.numel() > 0:
                    valid_weights = valid_weights.float()
                    ret.update({
                        'support_rho_mean': valid_weights.mean(),
                        'support_rho_zero_ratio': (valid_weights <= 0).float().mean(),
                        'support_rho_mid_ratio': (
                            (valid_weights > 0) & (valid_weights < 1)
                        ).float().mean(),
                        'support_rho_one_ratio': (valid_weights >= 1).float().mean(),
                    })
                if self.irra_light_hard_negative:
                    ret['hard_negative_valid_ratio'] = (
                        hard_negative_valid.float().mean()
                    )
                return ret

            if self.irra_light_bag:
                support_i_feats, support_t_feats, support_mask = self._encode_support_bag(batch)
                support_weights = batch.get('support_reliability') if self.irra_light_bag_consistency else None
                ret.update({
                    'identity_src_loss': self._source_pair_loss(
                        identity_i_feats, identity_t_feats, batch['pids'], logit_scale),
                    'identity_set_loss': self._support_set_loss(
                        identity_i_feats, identity_t_feats,
                        support_i_feats, support_t_feats, support_mask,
                        batch['pids'], logit_scale,
                        support_weights=support_weights),
                })
                if support_weights is not None:
                    valid_weights = support_weights[support_mask]
                    if valid_weights.numel() > 0:
                        ret.update({
                            'support_rho_mean': valid_weights.float().mean(),
                            'support_rho_zero_ratio': (valid_weights <= 0).float().mean(),
                            'support_rho_mid_ratio': ((valid_weights > 0) & (valid_weights < 1)).float().mean(),
                            'support_rho_one_ratio': (valid_weights >= 1).float().mean(),
                        })
                if self.irra_light_split:
                    ret.update({
                        'state_src_loss': self._source_pair_loss(
                            state_i_feats, state_t_feats, batch['pids'], logit_scale)
                    })
                return ret

            if self.args.irra_light_identity_loss == 'sdm':
                ret.update({
                    'identity_sdm_loss': objectives.compute_sdm(
                        identity_i_feats, identity_t_feats, batch['pids'], logit_scale)
                })
            elif self.args.irra_light_identity_loss == 'itc':
                ret.update({
                    'identity_itc_loss': objectives.compute_itc(
                        identity_i_feats, identity_t_feats, logit_scale)
                })
            else:
                raise ValueError(f'Unsupported IRRA-light identity loss: {self.args.irra_light_identity_loss}')

            ret.update({
                'state_itc_loss': objectives.compute_itc(
                    state_i_feats, state_t_feats, logit_scale)
            })

            if self.irra_light_with_id:
                image_logits = self.classifier(identity_i_feats.float())
                text_logits = self.classifier(identity_t_feats.float())
                ret.update({
                    'id_loss': objectives.compute_id(
                        image_logits, text_logits, batch['pids']) * self.args.id_loss_weight
                })
                image_pred = torch.argmax(image_logits, dim=1)
                text_pred = torch.argmax(text_logits, dim=1)
                ret.update({'img_acc': (image_pred == batch['pids']).float().mean()})
                ret.update({'txt_acc': (text_pred == batch['pids']).float().mean()})
            return ret

        if 'itc' in self.current_task:
            ret.update({'itc_loss':objectives.compute_itc(i_feats, t_feats, logit_scale)})
        
        if 'sdm' in self.current_task:
            ret.update({'sdm_loss':objectives.compute_sdm(i_feats, t_feats, batch['pids'], logit_scale)})

        if 'cmpm' in self.current_task:
            ret.update({'cmpm_loss':objectives.compute_cmpm(i_feats, t_feats, batch['pids'])})
        
        if 'id' in self.current_task:
            image_logits = self.classifier(i_feats.half()).float()
            text_logits = self.classifier(t_feats.half()).float()
            ret.update({'id_loss':objectives.compute_id(image_logits, text_logits, batch['pids'])*self.args.id_loss_weight})

            image_pred = torch.argmax(image_logits, dim=1)
            text_pred = torch.argmax(text_logits, dim=1)

            image_precision = (image_pred == batch['pids']).float().mean()
            text_precision = (text_pred == batch['pids']).float().mean()
            ret.update({'img_acc': image_precision})
            ret.update({'txt_acc': text_precision})
        
        if 'mlm' in self.current_task:
            mlm_ids = batch['mlm_ids']

            mlm_feats = self.base_model.encode_text(mlm_ids)

            x = self.cross_former(mlm_feats, image_feats, image_feats)

            x = self.mlm_head(x)  # [batch_size, text_len, num_colors]

            scores = x.float().reshape(-1, self.args.vocab_size)
            mlm_labels = batch['mlm_labels'].reshape(-1)
            ret.update({'mlm_loss': objectives.compute_mlm(scores, mlm_labels)*self.args.mlm_loss_weight})

            pred = scores.max(1)[1]
            mlm_label_idx = torch.nonzero(mlm_labels)
            acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
            ret.update({'mlm_acc': acc})

        return ret


def build_model(args, num_classes=11003):
    model = IRRA(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    model.float_projection_heads()
    return model

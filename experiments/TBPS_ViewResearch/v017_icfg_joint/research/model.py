import math
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from view4.common import data_rng, preserve_rng, seed32, world
from view4.model import ViewCLIP
from view4.upstream import enable_visual_checkpointing, install_checkpoint_compatibility
from model.visual_transformer import visual_transformer
from model.text_transformer import text_transformers
from misc.build import load_checkpoint


def confidence(prob):
    p = prob.float()
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("Routing probabilities must have shape [batch,3]")
    if not torch.isfinite(p).all() or (p < 0).any() or not torch.allclose(p.sum(1), torch.ones_like(p[:, 0]), atol=1e-5):
        raise ValueError("Invalid routing probability simplex")
    entropy = -(p * p.clamp_min(1e-12).log()).sum(1)
    return (1. - entropy / math.log(3)).clamp(0., 1.)


def weighted_global_mean(values, weights):
    numerator = (values * weights).sum()
    denominator = weights.detach().sum()
    report = numerator.detach().clone()
    if dist.is_initialized():
        dist.all_reduce(denominator)
        dist.all_reduce(report)
    denominator = denominator.clamp_min(1e-12)
    return world() * numerator / denominator, report / denominator


class SoftViewCLIP(ViewCLIP):
    def __init__(self, official, num_classes, seed, cfg):
        super().__init__(official, visual_transformer(official), text_transformers(official),
                         num_classes, "E1", seed)
        self.experiment = cfg["version"]
        self.residual_alpha = cfg["alpha"]
        self.delta = cfg["delta"]
        d = official.model.embed_dim
        with data_rng(seed32(seed, "soft3_expert_init")):
            self.experts = nn.ModuleList([nn.Sequential(nn.LayerNorm(d, eps=1e-5),
                    nn.Linear(d, d // 4), nn.GELU(approximate="none"), nn.Linear(d // 4, d))
                    for _ in range(3)])
            for head in self.experts:
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)
        self.expert_queries = nn.Parameter(torch.zeros(3, d))

    def pool_patches(self, patches):
        if patches.ndim != 3 or patches.shape[1] == 0 or patches.shape[2] != self.expert_queries.shape[1]:
            raise ValueError("Expected nonempty projected visual patches [batch,patches,dim]")
        with torch.cuda.amp.autocast(enabled=False):
            values = patches.float()
            keys = F.layer_norm(values, (values.shape[-1],), eps=1e-5)
            attention = torch.einsum("bnd,gd->bgn", keys, self.expert_queries).softmax(-1)
            return torch.einsum("bgn,bnd->bgd", attention, values), attention

    def expert_outputs(self, h, patches):
        if h.ndim != 2 or h.shape[0] != patches.shape[0] or h.shape[-1] != patches.shape[-1]:
            raise ValueError("Shared/patch feature shape mismatch")
        with torch.cuda.amp.autocast(enabled=False):
            local, _ = self.pool_patches(patches)
            return torch.stack([head(h.float() + local[:, g]) for g, head in enumerate(self.experts)], 1)

    def route(self, h, probabilities, patches):
        with torch.cuda.amp.autocast(enabled=False):
            p = probabilities.float()
            if p.shape[0] != h.shape[0]:
                raise ValueError("Routing batch mismatch")
            c = confidence(p)
            outputs = self.expert_outputs(h, patches)
            return (outputs * p[:, :, None]).sum(1) * c[:, None]

    def fuse(self, h, probabilities, patches):
        r = self.route(h, probabilities, patches)
        with torch.cuda.amp.autocast(enabled=False):
            fused = (h.float() + self.residual_alpha * r).to(h.dtype)
        return F.normalize(fused, dim=-1, eps=1e-12), r

    def image_embedding(self, images, probabilities):
        h, dense = self.encode_image(images, return_dense=True)
        return self.fuse(h, probabilities, dense[:, 1:])[0]

    def auxiliary(self, h, support_h, batch, selected, patches, support_patches):
        a, b = h[selected].float(), support_h.float()
        pa, pb = batch["view_prob"][selected], batch["support_prob"][selected]
        with torch.cuda.amp.autocast(enabled=False):
            weight = (confidence(pa) * confidence(pb) * (1. - (pa * pb).sum(1))).detach()
            values = 1. - F.cosine_similarity(a, b, dim=-1, eps=1e-8)
            shared, shared_mean = weighted_global_mean(values, weight)
            # Detach BOTH encoder outputs; heads and pooling queries still receive gradients.
            ra = self.route(a.detach(), pa, patches[selected].detach())
            rb = self.route(b.detach(), pb, support_patches.detach())
            dec_values = F.cosine_similarity(ra, rb, dim=-1, eps=1e-8).square()
            decor, decor_mean = weighted_global_mean(dec_values, weight)
            counts = torch.stack([selected.sum(), ((ra.norm(dim=1) <= 1e-8) |
                                                  (rb.norm(dim=1) <= 1e-8)).sum()])
            if dist.is_initialized():
                dist.all_reduce(counts)
        return shared, decor, {"shared_mean": shared_mean.detach(), "decor_mean": decor_mean.detach(),
                              "support_pairs": counts[0], "zero_residual_pairs": counts[1]}

    def forward(self, batch, soft_label_mix, shared_weight, decorrelation_weight):
        h, dense = self.encode_image(batch["image"], return_dense=True)
        patches = dense[:, 1:]
        u, _ = self.encode_text(batch["text_tokens"], return_dense=True)
        v, r = self.fuse(h, batch["view_prob"], patches)
        t = F.normalize(u, dim=-1, eps=1e-12)
        vg, tg = self.all_gather(v), self.all_gather(t)
        logit_scale = self.logit_scale.exp()
        logit_scale.data = torch.clamp(logit_scale.data, max=100)
        ids = batch["person_id"]
        gathered_ids = self.all_gather(ids)
        q = ids[:, None].eq(gathered_ids[None, :]).float()
        q = q / q.sum(1, keepdim=True)
        with torch.no_grad():
            target_h, target_dense = self.encode_image(batch["image"], return_dense=True)
            vs, _ = self.fuse(target_h, batch["view_prob"], target_dense[:, 1:])
            ts = F.normalize(self.encode_text(batch["text_tokens"]), dim=-1, eps=1e-12)
            vsg, tsg = self.all_gather(vs), self.all_gather(ts)
        nitc = self.calc_contrastive(v, t, vs, ts, vg, tg, vsg, tsg, q, soft_label_mix, logit_scale)
        img_log = F.log_softmax(logit_scale * v @ tg.t(), dim=1)
        txt_log = F.log_softmax(logit_scale * t @ vg.t(), dim=1)
        target_log = (q + self.eps).log()
        ritc = .5 * (F.kl_div(target_log, img_log, log_target=True, reduction="batchmean") +
                     F.kl_div(target_log, txt_log, log_target=True, reduction="batchmean"))
        retrieval = nitc + ritc
        selected = batch["support_valid"].bool()
        if selected.any():
            if (batch["person_id"][selected] != batch["support_person_id"][selected]).any():
                raise ValueError("Support identity mismatch")
            if (batch["image_id"][selected] == batch["support_image_id"][selected]).any():
                raise ValueError("Same image is not cross-view evidence")
            for suffix in ("raw", "aug"):
                angle = (batch["theta_" + suffix][selected] - batch["support_theta_" + suffix][selected]).abs()
                if ((batch["view3_" + suffix][selected] == batch["support_view3_" + suffix][selected]) |
                        (torch.minimum(angle, 360. - angle) < self.delta)).any():
                    raise ValueError("Invalid cross-view support after augmentation")
            # Extra support dropout must not advance the next main-path RNG state.
            with preserve_rng():
                support_h, support_dense = self.encode_image(batch["support_image"][selected], return_dense=True)
                support_patches = support_dense[:, 1:]
        else:
            support_h = h[:0]
            support_patches = patches[:0]
        shared, decor, stats = self.auxiliary(h, support_h, batch, selected, patches, support_patches)
        total = retrieval + shared_weight * shared + decorrelation_weight * decor
        with torch.no_grad():
            rho = self.residual_alpha * r.norm(dim=-1) / h.float().norm(dim=-1).clamp_min(1e-12)
        return {"loss_total": total, "nitc": nitc.detach(), "ritc": ritc.detach(),
                "retrieval": retrieval.detach(), "rho": rho, **stats}


def build_model(official, cfg, num_classes, initializer=None):
    if cfg['ritc_eps'] != .001:
        raise ValueError('V017 requires R-ITC target floor 0.001')
    official.experiment.ritc_eps = cfg['ritc_eps']
    install_checkpoint_compatibility()
    model = SoftViewCLIP(official, num_classes, cfg["seed"], cfg)
    if model.eps != cfg['ritc_eps']:
        raise RuntimeError('R-ITC target floor was not passed to the model')
    enable_visual_checkpointing(model)
    if initializer:
        if (cfg['initialization'] != 'openai_clip' or
                Path(initializer).resolve() != Path(cfg['clip_checkpoint']).resolve()):
            raise ValueError('V017 initializes from the original OpenAI checkpoint only')
        model, result = load_checkpoint(model, official)
        expected = {k for k in model.state_dict() if k.startswith("experts.") or k == "expert_queries"}
        if set(result.missing_keys) != expected or result.unexpected_keys:
            raise RuntimeError("Initializer shape/key mismatch: " + str(result))
    model.visual.freeze_conv1 = False
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    model.train()
    if not all(p.requires_grad for p in model.parameters()):
        raise RuntimeError("Unexpected frozen model parameter")
    return model

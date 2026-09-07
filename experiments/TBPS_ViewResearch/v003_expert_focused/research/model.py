import math

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from view4.common import data_rng, load_training_checkpoint, preserve_rng, seed32, world
from view4.model import ViewCLIP
from view4.upstream import enable_visual_checkpointing, install_checkpoint_compatibility
from model.visual_transformer import visual_transformer
from model.text_transformer import text_transformers


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

    def route(self, h, probabilities):
        with torch.cuda.amp.autocast(enabled=False):
            p = probabilities.float()
            c = confidence(p)
            outputs = torch.stack([head(h.float()) for head in self.experts], dim=1)
            return (outputs * p[:, :, None]).sum(1) * c[:, None]

    def auxiliary(self, h, support_h, batch, selected):
        a, b = h[selected].float(), support_h.float()
        pa, pb = batch["view_prob"][selected], batch["support_prob"][selected]
        with torch.cuda.amp.autocast(enabled=False):
            weight = (confidence(pa) * confidence(pb) * (1. - (pa * pb).sum(1))).detach()
            values = 1. - F.cosine_similarity(a, b, dim=-1, eps=1e-8)
            shared, shared_mean = weighted_global_mean(values, weight)
            # Only the experts receive gradients from cross-view residual decorrelation.
            ra, rb = self.route(a.detach(), pa), self.route(b.detach(), pb)
            dec_values = F.cosine_similarity(ra, rb, dim=-1, eps=1e-8).square()
            decor, decor_mean = weighted_global_mean(dec_values, weight)
            counts = torch.stack([selected.sum(), ((ra.norm(dim=1) <= 1e-8) |
                                                  (rb.norm(dim=1) <= 1e-8)).sum()])
            if dist.is_initialized():
                dist.all_reduce(counts)
        return shared, decor, {"shared_mean": shared_mean.detach(), "decor_mean": decor_mean.detach(),
                              "support_pairs": counts[0], "zero_residual_pairs": counts[1]}

    def forward(self, batch, soft_label_mix, shared_weight, decorrelation_weight):
        h, _ = self.encode_image(batch["image"], return_dense=True)
        u, _ = self.encode_text(batch["text_tokens"], return_dense=True)
        v, r = self.fuse(h, batch["view_prob"])
        t = F.normalize(u, dim=-1, eps=1e-12)
        vg, tg = self.all_gather(v), self.all_gather(t)
        logit_scale = self.logit_scale.exp()
        logit_scale.data = torch.clamp(logit_scale.data, max=100)
        ids = batch["person_id"]
        gathered_ids = self.all_gather(ids)
        q = ids[:, None].eq(gathered_ids[None, :]).float()
        q = q / q.sum(1, keepdim=True)
        with torch.no_grad():
            vs, _ = self.fuse(self.encode_image(batch["image"]), batch["view_prob"])
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
                support_h = self.encode_image(batch["support_image"][selected])
        else:
            support_h = h[:0]
        shared, decor, stats = self.auxiliary(h, support_h, batch, selected)
        total = retrieval + shared_weight * shared + decorrelation_weight * decor
        with torch.no_grad():
            rho = self.residual_alpha * r.norm(dim=-1) / h.float().norm(dim=-1).clamp_min(1e-12)
        return {"loss_total": total, "nitc": nitc.detach(), "ritc": ritc.detach(),
                "retrieval": retrieval.detach(), "rho": rho, **stats}


def build_model(official, cfg, num_classes, initializer=None):
    install_checkpoint_compatibility()
    model = SoftViewCLIP(official, num_classes, cfg["seed"], cfg)
    enable_visual_checkpointing(model)
    if initializer:
        saved = load_training_checkpoint(initializer)
        if saved.get("experiment") != "E0" or saved.get("run_kind") != "formal":
            raise ValueError("Expected a validated E0 checkpoint")
        result = model.load_state_dict(saved["model"], strict=False)
        expected = {k for k in model.state_dict() if k.startswith("experts.")}
        if set(result.missing_keys) != expected or result.unexpected_keys:
            raise RuntimeError("Initializer shape/key mismatch: " + str(result))
        del saved
    model.visual.freeze_conv1 = False
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    model.train()
    if not all(p.requires_grad for p in model.parameters()):
        raise RuntimeError("Unexpected frozen model parameter")
    return model

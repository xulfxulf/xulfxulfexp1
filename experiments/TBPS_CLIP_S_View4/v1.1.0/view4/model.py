from view4.common import load_training_checkpoint
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from .common import data_rng, seed32, world
from .upstream import bootstrap, install_checkpoint_compatibility, enable_visual_checkpointing

bootstrap()
from model.tbps_model import CLIP
from model.visual_transformer import visual_transformer
from model.text_transformer import text_transformers
from misc.build import load_checkpoint


def pair_mask(batch, delta=60):
    ids, images = batch["person_id"], batch["image_id"]
    valid = (ids[:, None] == ids[None, :]) & (images[:, None] != images[None, :])
    for suffix in ("raw", "aug"):
        view, theta = batch["view_" + suffix], batch["theta_" + suffix].float()
        diff = (theta[:, None] - theta[None, :]).abs()
        circular = torch.minimum(diff, 360.0 - diff)
        valid &= (view[:, None] != view[None, :]) & (circular >= delta)
    return valid.triu(diagonal=1)


def decorrelation(r_dec, batch, delta=60, distributed=True):
    mask = pair_mask(batch, delta)
    a, b = mask.nonzero(as_tuple=True)
    local_count = torch.tensor(a.numel(), device=r_dec.device, dtype=torch.long)
    if a.numel():
        values = F.cosine_similarity(r_dec[a], r_dec[b], dim=-1, eps=1e-8).square()
        local_sum = values.sum()
        norms_a, norms_b = r_dec[a].norm(dim=-1), r_dec[b].norm(dim=-1)
        zeros = ((norms_a == 0) | (norms_b == 0)).sum()
        near = ((norms_a <= 1e-8) | (norms_b <= 1e-8)).sum()
    else:
        local_sum = r_dec.sum() * 0.0
        zeros, near = local_count.clone(), local_count.clone()
    counts = torch.stack([local_count, zeros, near])
    report_sum = local_sum.detach().clone()
    size = world() if distributed else 1
    if distributed and dist.is_initialized():
        dist.all_reduce(counts)
        dist.all_reduce(report_sum)
    denominator = counts[0].clamp_min(1).float()
    loss = size * local_sum / denominator
    return loss, {"ortho_sum": report_sum, "ortho_pairs": counts[0],
                  "ortho_zero_pairs": counts[1], "ortho_near_zero_pairs": counts[2],
                  "ortho_mean": report_sum / denominator}


class ViewCLIP(CLIP):
    def __init__(self, config, visual, text, num_classes, experiment, seed):
        super().__init__(config, visual, text, num_classes, eps=config.experiment.ritc_eps)
        if experiment not in ("E0", "E1", "E2", "E3"):
            raise ValueError(experiment)
        self.experiment, self.residual_alpha, self.delta = experiment, 0.2, 60
        if experiment in ("E2", "E3"):
            d = config.model.embed_dim
            with data_rng(seed32(seed, "expert_init")):
                self.experts = nn.ModuleList([nn.Sequential(
                    nn.LayerNorm(d, eps=1e-5), nn.Linear(d, d // 4),
                    nn.GELU(approximate="none"), nn.Linear(d // 4, d)) for _ in range(4)])
                for head in self.experts:
                    nn.init.zeros_(head[-1].weight)
                    nn.init.zeros_(head[-1].bias)

    @property
    def has_experts(self):
        return hasattr(self, "experts")

    def all_gather(self, tensor):
        if not dist.is_initialized():
            return tensor
        return super().all_gather(tensor)

    def route(self, h, view):
        if not self.has_experts:
            return torch.zeros_like(h, dtype=torch.float32)
        if view.ndim != 1 or view.shape[0] != h.shape[0] or ((view < 0) | (view > 3)).any():
            raise ValueError("Invalid image view IDs")
        with torch.cuda.amp.autocast(enabled=False):
            r = torch.zeros_like(h, dtype=torch.float32)
            for group, head in enumerate(self.experts):
                indices = torch.where(view == group)[0]
                if indices.numel():
                    r = r.index_copy(0, indices, head(h[indices].float()))
        return r

    def fuse(self, h, view):
        if not self.has_experts:
            return F.normalize(h, dim=-1, eps=1e-12), torch.zeros_like(h, dtype=torch.float32)
        r = self.route(h, view)
        with torch.cuda.amp.autocast(enabled=False):
            fused = (h.float() + self.residual_alpha * r).to(h.dtype)
        return F.normalize(fused, dim=-1, eps=1e-12), r

    def image_embedding(self, images, views):
        return self.fuse(self.encode_image(images), views)[0]

    def forward(self, batch, soft_label_mix, decorrelation_weight):
        if not 0 <= soft_label_mix <= .5 or not 0 <= decorrelation_weight <= .01:
            raise ValueError("Invalid fixed-group loss schedule")
        if self.experiment != "E3" and decorrelation_weight != 0:
            raise ValueError("Only E3 can enable residual decorrelation")
        h, _ = self.encode_image(batch["image"], return_dense=True)
        u, _ = self.encode_text(batch["text_tokens"], return_dense=True)
        v, r = self.fuse(h, batch["view_aug"])
        t = F.normalize(u, dim=-1, eps=1e-12)
        vg, tg = self.all_gather(v), self.all_gather(t)
        logit_scale = self.logit_scale.exp()
        logit_scale.data = torch.clamp(logit_scale.data, max=100)
        ids = batch["person_id"]
        gathered_ids = self.all_gather(ids)
        q = ids[:, None].eq(gathered_ids[None, :]).float()
        q = q / q.sum(1, keepdim=True)
        # A fresh no-grad train-mode pass, including the same active experts and text dropout.
        with torch.no_grad():
            vs, _ = self.fuse(self.encode_image(batch["image"]), batch["view_aug"])
            ts = F.normalize(self.encode_text(batch["text_tokens"]), dim=-1, eps=1e-12)
            vsg, tsg = self.all_gather(vs), self.all_gather(ts)
        nitc = self.calc_contrastive(v, t, vs, ts, vg, tg, vsg, tsg, q, soft_label_mix, logit_scale)
        img_log = F.log_softmax(logit_scale * v @ tg.t(), dim=1)
        txt_log = F.log_softmax(logit_scale * t @ vg.t(), dim=1)
        target_log = (q + self.eps).log()
        ritc = .5 * (F.kl_div(target_log, img_log, log_target=True, reduction="batchmean") +
                     F.kl_div(target_log, txt_log, log_target=True, reduction="batchmean"))
        retrieval = nitc + ritc
        ortho = retrieval.detach() * 0
        stats = {}
        if self.has_experts:
            if self.experiment == "E3" and decorrelation_weight > 0:
                r_dec = self.route(h.detach(), batch["view_aug"])
                ortho, stats = decorrelation(r_dec, batch, self.delta)
            else:
                with torch.no_grad():
                    ortho, stats = decorrelation(self.route(h.detach(), batch["view_aug"]), batch, self.delta)
        # E2 and E3 have exactly the same backward graph throughout epoch 1.
        total = retrieval + decorrelation_weight * ortho if decorrelation_weight > 0 else retrieval
        with torch.no_grad():
            rho = self.residual_alpha * r.float().norm(dim=-1) / (h.float().norm(dim=-1) + 1e-12)
        return {"loss_total": total, "nitc": nitc.detach(), "ritc": ritc.detach(),
                "retrieval": retrieval.detach(), "rho": rho.detach(), **stats}


def build_model(official, experiment, seed, num_train_ids, e0_checkpoint=None, load_openai=True):
    install_checkpoint_compatibility()
    model = ViewCLIP(official, visual_transformer(official), text_transformers(official),
                     num_train_ids, experiment, seed)
    enable_visual_checkpointing(model)
    if e0_checkpoint:
        saved = load_training_checkpoint(e0_checkpoint, map_location="cpu")
        if saved.get("experiment") != "E0" or saved.get("run_kind") != "formal":
            raise ValueError("E1-E3 require this runner's formal S E0, not the earlier full TBPS checkpoint")
        result = model.load_state_dict(saved["model"], strict=False)
        del saved
    elif load_openai:
        model, result = load_checkpoint(model, official)
    else:
        result = None
    if result is not None:
        expected_missing = {k for k in model.state_dict() if k.startswith("experts.")} if model.has_experts else set()
        if set(result.missing_keys) != expected_missing or result.unexpected_keys:
            raise RuntimeError("Unexpected checkpoint mismatch: " + str(result))
    if experiment != "E0":
        model.visual.freeze_conv1 = False
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    model.train()
    if experiment != "E0" and not all(p.requires_grad for p in model.parameters()):
        raise RuntimeError("Unexpected frozen original parameter after train()")
    return model


def build_optimizer(model, lr):
    groups, audit = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        decay = 0. if parameter.ndim < 2 or any(x in name for x in ("bias", "ln", "bn")) else .02
        ratio = 10. if name.startswith("experts.") else 1.
        groups.append({"params": [parameter], "lr": lr * ratio, "ratio": ratio, "weight_decay": decay})
        audit.append({"name": name, "shape": list(parameter.shape), "ratio": ratio, "weight_decay": decay})
    return torch.optim.AdamW(groups, lr=lr, betas=(.9, .98), eps=1e-8), audit

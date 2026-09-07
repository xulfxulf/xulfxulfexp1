import itertools

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from view4.common import preserve_rng
from research.data import SoftGalleryDataset
from misc.eval import metric_eval
from text_utils.tokenizer import tokenize


@torch.no_grad()
def evaluate(model, records, dataset_root, device):
    data = SoftGalleryDataset(records, dataset_root)
    training = model.training
    with preserve_rng():
        try:
            model.eval()
            images = torch.empty((len(data), model.embed_dim), device=device, dtype=torch.float32)
            seen = torch.zeros(len(data), dtype=torch.int64)
            for batch in DataLoader(data, batch_size=32, shuffle=False, num_workers=0):
                idx = batch["gallery_index"]
                if seen[idx].any() or [records[i]["image_id"] for i in idx.tolist()] != batch["image_id"].tolist():
                    raise RuntimeError("Gallery alignment/duplication error")
                images[idx.to(device)] = model.image_embedding(batch["image"].to(device),
                                                     batch["view_prob"].to(device)).float()
                seen[idx] += 1
            if not (seen == 1).all():
                raise RuntimeError("Incomplete gallery coverage")
            texts = []
            # No query view, source image, PID or candidate metadata enters text encoding.
            for start in range(0, len(data.query_texts), 256):
                tokens = tokenize(data.query_texts[start:start + 256], context_length=77).to(device)
                texts.append(F.normalize(model.encode_text(tokens), dim=-1, eps=1e-12).float())
            texts = torch.cat(texts)
            if not torch.isfinite(images).all() or not torch.isfinite(texts).all():
                raise FloatingPointError("Nonfinite retrieval embeddings")
            return metric_eval(texts @ images.t(), data.gallery_person_ids, data.query_person_ids)
        finally:
            model.train(training)


@torch.no_grad()
def expert_diagnostics(model, records, indices, root, device):
    data, outputs = SoftGalleryDataset(records, root), []
    training = model.training
    with preserve_rng():
        try:
            model.eval()
            for batch in DataLoader(Subset(data, indices), batch_size=32, shuffle=False):
                h = model.encode_image(batch["image"].to(device)).float()
                outputs.append(torch.stack([head(h) for head in model.experts], 1).cpu())
        finally:
            model.train(training)
    output = torch.cat(outputs)
    norms = output.norm(dim=-1)
    result = {"count": len(output), "head_norm_mean": norms.mean(0).tolist(), "same_input_heads": {}}
    for a, b in itertools.combinations(range(3), 2):
        valid = (norms[:, a] > 1e-8) & (norms[:, b] > 1e-8)
        result["same_input_heads"]["%d_%d" % (a, b)] = {
            "valid": int(valid.sum()), "cos2": F.cosine_similarity(output[valid, a], output[valid, b],
                             dim=-1, eps=1e-8).square().mean().item() if valid.any() else None}
    return result

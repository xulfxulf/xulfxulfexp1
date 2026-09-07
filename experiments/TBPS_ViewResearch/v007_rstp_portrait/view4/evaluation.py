import itertools
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from .common import preserve_rng, write_json
from .data import GalleryDataset
from .upstream import bootstrap

bootstrap()
from misc.eval import metric_eval
from text_utils.tokenizer import tokenize


@torch.no_grad()
def feature_export(model, records, dataset_root, device):
    dataset = GalleryDataset(records, dataset_root)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    training = model.training
    with preserve_rng():
        try:
            model.eval()
            images = torch.empty((len(dataset), model.embed_dim), device=device, dtype=torch.float32)
            seen = torch.zeros(len(dataset), dtype=torch.int64)
            for batch in loader:
                index = batch["gallery_index"]
                if seen[index].any():
                    raise RuntimeError("Duplicated gallery index")
                if [records[i]["image_id"] for i in index.tolist()] != batch["image_id"].tolist():
                    raise RuntimeError("Gallery image_id alignment failure")
                features = model.image_embedding(batch["image"].to(device), batch["view_raw"].to(device))
                images[index.to(device)] = features.float()
                seen[index] += 1
            if not torch.all(seen == 1):
                raise RuntimeError("Gallery coverage is not exactly once")
            texts = []
            for start in range(0, len(dataset.query_texts), 256):
                tokens = tokenize(dataset.query_texts[start:start + 256], context_length=77).to(device)
                texts.append(F.normalize(model.encode_text(tokens), dim=-1, eps=1e-12).float())
            texts = torch.cat(texts)
            for features in (images, texts):
                if not torch.isfinite(features).all():
                    raise FloatingPointError("Nonfinite evaluation features")
            return texts, images, dataset
        finally:
            model.train(training)


def evaluate(model, records, dataset_root, device):
    texts, images, dataset = feature_export(model, records, dataset_root, device)
    return metric_eval(texts @ images.t(), dataset.gallery_person_ids, dataset.query_person_ids)


@torch.no_grad()
def expert_diagnostics(model, catalog, subset_indices, dataset_root, device):
    if not model.has_experts:
        return {"applicable": False}
    dataset = GalleryDataset(catalog["train"], dataset_root)
    training = model.training
    outputs = []
    with preserve_rng():
        try:
            model.eval()
            loader = DataLoader(Subset(dataset, subset_indices), batch_size=32, shuffle=False, num_workers=0)
            for batch in loader:
                h = model.encode_image(batch["image"].to(device)).float()
                outputs.append(torch.stack([head(h) for head in model.experts], dim=1).cpu())
        finally:
            model.train(training)
    outputs = torch.cat(outputs)
    norms = outputs.norm(dim=-1)
    result = {"applicable": True, "count": len(outputs), "same_input_all_heads": {},
              "head_norm_mean": norms.mean(0).tolist()}
    for a, b in itertools.combinations(range(4), 2):
        valid = (norms[:, a] > 1e-8) & (norms[:, b] > 1e-8)
        values = F.cosine_similarity(outputs[valid, a], outputs[valid, b], dim=-1, eps=1e-8).square()
        result["same_input_all_heads"]["%d_%d" % (a, b)] = {
            "mean_cos2": values.mean().item() if valid.any() else None,
            "valid": int(valid.sum()), "degenerate_fraction": 1 - valid.float().mean().item()}
    return result

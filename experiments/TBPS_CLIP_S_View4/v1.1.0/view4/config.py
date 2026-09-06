from pathlib import Path

import yaml

from .common import ROOT


def load_config(path):
    with open(path, encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    required = {"dataset_root", "annotation_root", "orientation", "clip_checkpoint", "prepared_dir",
                "world_size", "local_batch", "epochs", "steps_per_epoch", "P", "K", "workers",
                "alpha", "delta", "lambda_max", "seeds", "nltk_data"}
    if required - cfg.keys():
        raise ValueError("Missing config: " + str(sorted(required - cfg.keys())))
    if (cfg["world_size"], cfg["local_batch"], cfg["epochs"], cfg["steps_per_epoch"],
            cfg["P"], cfg["K"]) != (4, 80, 5, 212, 160, 2):
        raise ValueError("Group G0 is fixed to 4x80, 5 epochs, 212 steps, P160 K2")
    if (cfg["alpha"], cfg["delta"], cfg["lambda_max"], cfg["seeds"]) != (0.2, 60, 0.01, [1, 2, 3]):
        raise ValueError("Group G0 method constants cannot be silently overridden")
    for key in ("dataset_root", "annotation_root", "orientation", "clip_checkpoint", "prepared_dir", "nltk_data"):
        if not Path(cfg[key]).is_absolute():
            raise ValueError(key + " must be absolute")
    return cfg


def official_config(cfg, device="cpu"):
    from easydict import EasyDict
    with open(ROOT / "vendor/TBPS-CLIP/config/s.config.yaml", encoding="utf-8") as stream:
        official = EasyDict(yaml.safe_load(stream))
    official.device = str(device)
    official.model.checkpoint = cfg["clip_checkpoint"]
    official.model.ckpt_type = "original_clip"
    official.misc.seed = 1
    official.anno_dir = cfg["annotation_root"]
    official.image_dir = str(Path(cfg["dataset_root"]) / "imgs")
    return official


def lr_value(step, experiment):
    start, peak, end = (1e-6, 1e-4, 5e-6) if experiment == "E0" else (1e-7, 1e-5, 5e-7)
    if step < 212:
        return start + (peak - start) * step / 211
    import math
    return end + 0.5 * (peak - end) * (1 + math.cos(math.pi * (step - 212) / 848))


def loss_weights(epoch, step, experiment):
    beta = 0.5 * min(1.0, ((epoch - 1) * 212 + step) / 212)
    lam = 0.0
    if experiment == "E3" and epoch >= 2:
        lam = 0.01 * step / 211 if epoch == 2 else 0.01
    return beta, lam


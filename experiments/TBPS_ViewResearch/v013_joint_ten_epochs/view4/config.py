from pathlib import Path

import yaml

from .common import ROOT
from .geometry import INPUT_SIZE, check_geometry


def load_config(path):
    with open(path, encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    required = {"dataset_root", "annotation_root", "orientation", "clip_checkpoint", "prepared_dir",
                "world_size", "local_batch", "epochs", "steps_per_epoch", "P", "K", "workers",
                "alpha", "delta", "lambda_max", "seeds", "nltk_data"}
    if required - cfg.keys():
        raise ValueError("Missing config: " + str(sorted(required - cfg.keys())))
    check_geometry(cfg)
    if (cfg["world_size"], cfg["local_batch"], cfg["steps_per_epoch"],
            cfg["P"], cfg["K"]) != (4, 80, 212, 160, 2):
        raise ValueError("CUHK seed1 setup is fixed to 4x80, 212 steps, P160 K2")
    if cfg.get("dataset") != "CUHK-PEDES" or cfg["epochs"] != 10:
        raise ValueError("Only ten-epoch CUHK portrait E0 is enabled in this snapshot")
    if (cfg["alpha"], cfg["delta"], cfg["lambda_max"], cfg["seeds"]) != (0.2, 60, 0.01, [1]):
        raise ValueError("CUHK seed1 constants cannot be silently overridden")
    for key in ("dataset_root", "annotation_root", "orientation", "clip_checkpoint", "prepared_dir", "nltk_data"):
        if not Path(cfg[key]).is_absolute():
            raise ValueError(key + " must be absolute")
    return cfg


def official_config(cfg, device="cpu"):
    check_geometry(cfg)
    from easydict import EasyDict
    with open(ROOT / "vendor/TBPS-CLIP/config/s.config.yaml", encoding="utf-8") as stream:
        official = EasyDict(yaml.safe_load(stream))
    official.device = str(device)
    official.model.checkpoint = cfg["clip_checkpoint"]
    official.model.ckpt_type = "original_clip"
    official.misc.seed = 1
    official.experiment.input_resolution = list(INPUT_SIZE)
    official.schedule.epoch = cfg["epochs"]
    official.anno_dir = cfg["annotation_root"]
    official.image_dir = str(Path(cfg["dataset_root"]) / "imgs")
    return official


def lr_value(step, experiment, epochs=5, steps_per_epoch=212):
    start, peak, end = (1e-6, 1e-4, 5e-6) if experiment == "E0" else (1e-7, 1e-5, 5e-7)
    if epochs < 2 or steps_per_epoch < 2 or not 0 <= step < epochs * steps_per_epoch:
        raise ValueError("Learning-rate step is outside the configured training horizon")
    if step < steps_per_epoch:
        return start + (peak - start) * step / (steps_per_epoch - 1)
    import math
    decay_steps = (epochs - 1) * steps_per_epoch
    return end + 0.5 * (peak - end) * (1 + math.cos(math.pi * (step - steps_per_epoch) / decay_steps))


def loss_weights(epoch, step, experiment, steps_per_epoch=212):
    beta = 0.5 * min(1.0, ((epoch - 1) * steps_per_epoch + step) / steps_per_epoch)
    lam = 0.0
    if experiment == "E3" and epoch >= 2:
        lam = 0.01 * step / (steps_per_epoch - 1) if epoch == 2 else 0.01
    return beta, lam

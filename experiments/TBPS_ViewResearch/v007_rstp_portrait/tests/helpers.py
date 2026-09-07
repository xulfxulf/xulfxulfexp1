import copy

import torch
from easydict import EasyDict
from torch import nn

from view4.model import ViewCLIP


class TinyVisual(nn.Module):
    def __init__(self, width=16):
        super().__init__()
        self.freeze_conv1 = False
        self.conv1 = nn.Linear(6, width)
        self.proj = nn.Parameter(torch.eye(width))

    def forward(self, x, return_dense=False):
        h = self.conv1(x) @ self.proj
        return (h, h[:, None]) if return_dense else h


class TinyText(nn.Module):
    def __init__(self, width=16):
        super().__init__()
        self.projection = nn.Linear(5, width)
        self.dropout = nn.Dropout(.05)

    def forward(self, tokens, return_dense=False):
        x = tokens[:, :5].float() / 50000
        h = self.dropout(self.projection(x))
        return (h, h[:, None]) if return_dense else h


def tiny_model(experiment="E3", seed=1):
    config = EasyDict({"device": "cpu", "model": {"embed_dim": 16, "use_gather": True},
                      "experiment": {"ss": False, "id": False, "mlm": False, "citc": False,
                       "mvs_image": False, "ritc": True, "ritc_eps": .01, "ritc_ratio": 1.,
                       "nitc_ratio": 1., "back_trans": True, "backtrans_p": .1,
                       "eda_alpha": .05, "text_length": 77}})
    model = ViewCLIP(config, TinyVisual(), TinyText(), 4, experiment, seed)
    return model


def tiny_batch(n=8):
    ids = torch.arange(n) // 2
    raw = torch.tensor(([0, 2] * ((n + 1) // 2))[:n])
    angles = torch.where(raw == 0, torch.tensor(180.), torch.tensor(0.))
    return {"image": torch.randn(n, 6), "text_tokens": torch.randint(0, 49000, (n, 77)),
            "person_id": ids, "image_id": torch.arange(n), "view_raw": raw,
            "view_aug": raw.clone(), "theta_raw": angles, "theta_aug": angles.clone()}

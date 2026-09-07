import os
import sys

import torch
from torch.utils.checkpoint import checkpoint

from .common import ROOT


def bootstrap(nltk_data=None):
    vendor = str(ROOT / "vendor/TBPS-CLIP")
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    if nltk_data:
        os.environ["NLTK_DATA"] = nltk_data
        import nltk
        if nltk_data not in nltk.data.path:
            nltk.data.path.insert(0, nltk_data)


def install_checkpoint_compatibility():
    from model.base_transformer import Transformer

    def forward(self, x):
        x = self.dropout(x)
        if self.checkpoint and self.training and torch.is_grad_enabled():
            for block in self.resblocks[:-1]:
                x = checkpoint(block, x, use_reentrant=False, preserve_rng_state=True)
            return self.resblocks[-1](x)
        return self.resblocks(x)

    # Do not edit the pinned reference. Both reference and new model use this adapter.
    Transformer.forward = forward


def enable_visual_checkpointing(model):
    """Enable activation recomputation on the visual tower, not the text tower.

    The pinned upstream factory constructs both towers with checkpoint=False.
    Installing a compatible forward alone does not enable recomputation.
    """
    transformer = model.visual.transformer
    if len(transformer.resblocks) != 12:
        raise ValueError("Expected the 12-block ViT-B/16 visual encoder")
    transformer.checkpoint = True
    return model


bootstrap()

"""Model factory with backward-compatible HIRE and HIRE-v2 dispatch."""

from .build import build_model as build_irra_model
from .hire_model import build_hire_model
from .hire_v2_anchor_model import build_hire_v2_anchor_model


def build_model(args, num_classes=11003):
    if getattr(args, "hire_v2", False) or getattr(args, "loss_names", "") == "hire_v2_anchor":
        return build_hire_v2_anchor_model(args, num_classes=num_classes)
    if getattr(args, "hire", False) or getattr(args, "loss_names", "") == "hire":
        return build_hire_model(args, num_classes=num_classes)
    return build_irra_model(args, num_classes=num_classes)

"""Model factory with backward-compatible HIRE dispatch."""

from .build import build_model as build_irra_model
from .hire_model import build_hire_model


def build_model(args, num_classes=11003):
    if getattr(args, "hire", False) or getattr(args, "loss_names", "") == "hire":
        return build_hire_model(args, num_classes=num_classes)
    return build_irra_model(args, num_classes=num_classes)

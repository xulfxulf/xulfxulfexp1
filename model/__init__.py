"""Model factory with backward-compatible HIRE and HIRE-v2 dispatch."""

from .build import build_model as build_irra_model
from .hire_model import build_hire_model
from .hire_v2_anchor_model import build_hire_v2_anchor_model
from .hire_v2_identity_model import build_hire_v2_identity_model
from .hire_v2_identity_balanced_model import (
    build_hire_v2_identity_balanced_model,
)
from .hire_v2_identity_state_model import (
    build_hire_v2_identity_state_model,
)
from .hire_v2_identity_token_route_model import (
    build_hire_v2_identity_token_route_model,
)


def build_model(args, num_classes=11003):
    if getattr(args, "hire_v2", False):
        mode = getattr(args, "hire_v2_mode", "anchor")
        if mode == "anchor":
            return build_hire_v2_anchor_model(
                args, num_classes=num_classes
            )
        if mode == "identity":
            return build_hire_v2_identity_model(
                args, num_classes=num_classes
            )
        if mode == "identity_balanced":
            return build_hire_v2_identity_balanced_model(
                args, num_classes=num_classes
            )
        if mode == "identity_state":
            return build_hire_v2_identity_state_model(
                args, num_classes=num_classes
            )
        if mode == "identity_token_route":
            return build_hire_v2_identity_token_route_model(
                args, num_classes=num_classes
            )
        raise ValueError(
            "unsupported HIRE-v2 mode: {}".format(mode)
        )
    loss_names = getattr(args, "loss_names", "")
    if loss_names == "hire_v2_anchor":
        return build_hire_v2_anchor_model(
            args, num_classes=num_classes
        )
    if loss_names == "hire_v2_identity":
        return build_hire_v2_identity_model(
            args, num_classes=num_classes
        )
    if loss_names == "hire_v2_identity_balanced":
        return build_hire_v2_identity_balanced_model(
            args, num_classes=num_classes
        )
    if loss_names == "hire_v2_identity_state":
        return build_hire_v2_identity_state_model(
            args, num_classes=num_classes
        )
    if loss_names == "hire_v2_identity_token_route":
        return build_hire_v2_identity_token_route_model(
            args, num_classes=num_classes
        )
    if getattr(args, "hire", False) or loss_names == "hire":
        return build_hire_model(args, num_classes=num_classes)
    return build_irra_model(args, num_classes=num_classes)

#!/usr/bin/env python
"""Static and mathematical audit for HIRE-v2 identity-only delivery."""

from __future__ import annotations

import importlib.util
import os.path as op
import sys

import torch
import torch.nn.functional as F

PROJECT_ROOT = op.abspath(op.join(op.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

_COMPONENT_PATH = op.join(PROJECT_ROOT, "model", "hire_v2_identity_components.py")
_SPEC = importlib.util.spec_from_file_location("hire_v2_identity_components_audit", _COMPONENT_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
BoundedImageUncertainty = _MODULE.BoundedImageUncertainty
BoundedResidualGate = _MODULE.BoundedResidualGate
SharedIdentityMean = _MODULE.SharedIdentityMean
build_identity_final_embedding = _MODULE.build_identity_final_embedding
heterogeneity_aware_identity_intersection = _MODULE.heterogeneity_aware_identity_intersection
identity_residual_score = _MODULE.identity_residual_score
paired_identity_group_nce = _MODULE.paired_identity_group_nce


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    torch.manual_seed(7)
    dim = 8
    mean_head = SharedIdentityMean(dim)
    observation = F.normalize(torch.randn(4, dim), dim=-1)
    identity = mean_head(observation)
    require(torch.allclose(observation, identity, atol=1e-6), "identity map init mismatch")

    uncertainty = BoundedImageUncertainty(dim)
    variance = uncertainty(observation)
    midpoint = 0.5 * (uncertainty.variance_min + uncertainty.variance_max)
    require(torch.allclose(variance, torch.full_like(variance, midpoint), atol=1e-6), "variance init mismatch")
    require((variance > 0).all(), "variance must be positive")

    supports = identity[:3].unsqueeze(0).repeat(2, 1, 1)
    support_variance = variance[:3].unsqueeze(0).repeat(2, 1, 1)
    mask = torch.tensor([[1, 1, 1], [1, 0, 0]], dtype=torch.bool)
    posterior = heterogeneity_aware_identity_intersection(
        supports, support_variance, mask, min_supports=2
    )
    require(bool(posterior["valid"][0]), "three-support row must be valid")
    require(not bool(posterior["valid"][1]), "one-support row must be invalid")
    require(torch.isfinite(posterior["mean"]).all(), "posterior must be finite")

    pids = torch.tensor([0, 1, 2, 3])
    scores = torch.randn(4, 4, requires_grad=True)
    group_loss = paired_identity_group_nce(
        scores, pids, torch.ones(4, dtype=torch.bool), torch.tensor(50.0)
    )
    require(torch.isfinite(group_loss), "group NCE must be finite")
    group_loss.backward()
    require(scores.grad is not None and scores.grad.abs().sum() > 0, "group NCE gradient missing")

    gate_module = BoundedResidualGate(0.1)
    gate = gate_module()
    observation_score = observation @ observation.t()
    identity_score = identity @ identity.t()
    final_score = identity_residual_score(observation_score, identity_score, gate)
    require(torch.allclose(final_score, observation_score, atol=1e-6), "initial residual must equal anchor")

    final_embedding = build_identity_final_embedding(observation, identity, gate)
    inferred = final_embedding @ final_embedding.t()
    expected = (1.0 - gate) * observation_score + gate * identity_score
    require(torch.allclose(inferred, expected, atol=1e-6), "final embedding score mismatch")

    required_paths = [
        "model/hire_v2_identity_components.py",
        "model/hire_v2_identity_model.py",
        "datasets/hire_v2_identity_dataset.py",
        "tools/hire_v2/eval_identity_components.py",
        "run_hire_v2_identity_4090_tag.sh",
    ]
    for relative in required_paths:
        require(op.isfile(op.join(PROJECT_ROOT, relative)), "missing {}".format(relative))

    print("HIRE-v2 identity audit passed")
    print("- shared identity map starts exactly at anchored observation")
    print("- bounded uncertainty starts at a finite uniform midpoint")
    print("- trusted intersection handles valid and invalid support rows")
    print("- strict leave-one group NCE is finite and differentiable")
    print("- final retrieval embedding exactly matches the residual score")


if __name__ == "__main__":
    main()

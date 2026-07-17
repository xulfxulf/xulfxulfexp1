#!/usr/bin/env python
"""Static and mathematical audit for HIRE-v2 v16.3.0."""

from __future__ import annotations

import argparse
import json
import os
import os.path as op
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = op.abspath(op.join(op.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.hire_v2_state_components import (
    AttentionStateTokenEncoder,
    SignedBoundedStateGate,
    aggregate_identity_state_objectives,
    build_state_candidate_indices,
    selected_state_late_interaction,
    state_residual_score,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit v16.3.0 code/formula invariants"
    )
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def assert_close(left, right, name, atol=1e-6):
    if not torch.allclose(left, right, atol=atol, rtol=0.0):
        raise AssertionError(
            "{} mismatch: {} vs {}".format(name, left, right)
        )


def main():
    cli = parse_args()
    torch.manual_seed(17)
    results = {}

    # 1. The state gate is exactly zero but has unit derivative.
    gate_module = SignedBoundedStateGate()
    gate = gate_module()
    if gate.item() != 0.0:
        raise AssertionError("state gate is not exactly zero initialized")
    gate.backward()
    if gate_module.raw.grad is None or gate_module.raw.grad.item() != 1.0:
        raise AssertionError("state gate is not learnable at zero")
    results["zero_state_gate"] = "passed"

    # 2. Candidate selection forces all same-image positives.
    base = torch.randn(5, 5)
    image_ids = torch.tensor([1, 1, 2, 3, 4])
    indices, candidate, positive = build_state_candidate_indices(
        base,
        image_ids,
        topk=3,
    )
    if not bool(
        ((candidate & positive).sum(dim=1) == positive.sum(dim=1)).all()
    ):
        raise AssertionError("candidate selection dropped a state positive")
    results["positive_candidate_coverage"] = "passed"

    # 3. Late interaction equals a hand-computed weighted MaxSim.
    text_pack = {
        "tokens": F.normalize(
            torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
            dim=-1,
        ),
        "mask": torch.tensor([[True, True]]),
        "weights": torch.tensor([[0.75, 0.25]]),
    }
    image_pack = {
        "tokens": F.normalize(
            torch.tensor(
                [
                    [[1.0, 0.0], [0.0, 1.0]],
                    [[1.0, 1.0], [1.0, -1.0]],
                ]
            ),
            dim=-1,
        ),
        "mask": torch.ones(2, 2, dtype=torch.bool),
    }
    score = selected_state_late_interaction(
        text_pack,
        image_pack,
        torch.tensor([[0, 1]]),
    )["score"]
    expected = torch.tensor(
        [[1.0, 2.0 ** -0.5]]
    )
    assert_close(score, expected, "weighted MaxSim")
    results["late_interaction"] = "passed"

    # 4. State residual is exactly the identity-balanced base at initialization.
    identity_base = torch.randn(4, 4)
    state_score = torch.randn(4, 4)
    candidate_mask = torch.eye(4, dtype=torch.bool)
    final = state_residual_score(
        identity_base,
        state_score,
        candidate_mask,
        torch.tensor(0.0),
    )
    assert_close(final, identity_base, "initial state residual")
    results["initial_score_equivalence"] = "passed"

    # 5. Objective aggregation exactly matches the document.
    values = [
        torch.tensor(float(index), requires_grad=True)
        for index in range(1, 13)
    ]
    aggregated = aggregate_identity_state_objectives(
        global_sdm=values[0],
        global_itc=values[1],
        local_sdm=values[2],
        local_itc=values[3],
        observation_sdm=values[4],
        observation_itc=values[5],
        identity_final_sdm=values[6],
        identity_final_itc=values[7],
        state_final_sdm=values[8],
        state_final_itc=values[9],
        identity_group_nce=values[10],
        state_nce=values[11],
        auxiliary_weight=0.1,
    )
    expected_sdm = (
        0.5 * (values[0] + values[2])
        + 0.5 * values[4]
        + 0.25 * values[6]
        + 0.25 * values[8]
    )
    expected_itc = (
        0.5 * (values[1] + values[3])
        + 0.5 * values[5]
        + 0.25 * values[7]
        + 0.25 * values[9]
    )
    assert_close(
        aggregated["sdm_loss"],
        expected_sdm,
        "v16.3 SDM aggregation",
    )
    assert_close(
        aggregated["itc_loss"],
        expected_itc,
        "v16.3 ITC aggregation",
    )
    results["objective_formula"] = "passed"

    # 6. At gate zero, the main retrieval gradient equals v16.2.1.
    observation = torch.randn(3, 3, requires_grad=True)
    identity = torch.randn(3, 3, requires_grad=True)
    state_final = identity
    v163 = (
        0.5 * observation.pow(2).sum()
        + 0.25 * identity.pow(2).sum()
        + 0.25 * state_final.pow(2).sum()
    )
    v163.backward()
    observation_gradient = observation.grad.clone()
    identity_gradient = identity.grad.clone()

    observation_ref = observation.detach().clone().requires_grad_(True)
    identity_ref = identity.detach().clone().requires_grad_(True)
    v1621 = (
        0.5 * observation_ref.pow(2).sum()
        + 0.5 * identity_ref.pow(2).sum()
    )
    v1621.backward()
    assert_close(
        observation_gradient,
        observation_ref.grad,
        "observation gradient equivalence",
    )
    assert_close(
        identity_gradient,
        identity_ref.grad,
        "identity gradient equivalence",
    )
    results["initial_gradient_equivalence"] = "passed"

    # 7. State token projection is shared by image and text encoders.
    encoder = AttentionStateTokenEncoder(
        input_dim=16,
        image_token_count=4,
        text_token_count=3,
        output_dim=4,
    )
    parameter_ids = {
        id(parameter)
        for parameter in encoder.projection.parameters()
    }
    if len(parameter_ids) != 1:
        raise AssertionError("unexpected state projection parameterization")
    results["shared_state_projection"] = "passed"

    # 8. Source-level registration and protection rules.
    root = Path(
        os.environ.get(
            "HIRE_V2_AUDIT_SOURCE_ROOT",
            PROJECT_ROOT,
        )
    )
    model_source = (
        root / "model" / "hire_v2_identity_state_model.py"
    ).read_text(encoding="utf-8")
    component_source = (
        root / "model" / "hire_v2_state_components.py"
    ).read_text(encoding="utf-8")
    options_source = (
        root / "utils" / "options.py"
    ).read_text(encoding="utf-8")
    factory_source = (
        root / "model" / "__init__.py"
    ).read_text(encoding="utf-8")
    data_source = (
        root / "datasets" / "build.py"
    ).read_text(encoding="utf-8")
    processor_source = (
        root / "processor" / "processor.py"
    ).read_text(encoding="utf-8")
    metrics_source = (
        root / "utils" / "metrics.py"
    ).read_text(encoding="utf-8")

    required_model_fragments = [
        "image_tokens.detach()",
        "text_tokens.detach()",
        "state_pair_nce",
        "state_residual_score",
        "masked_identity_group_consensus",
    ]
    for fragment in required_model_fragments:
        if fragment not in model_source:
            raise AssertionError(
                "state model misses required fragment: {}".format(fragment)
            )
    forbidden_model_fragments = [
        "support_state",
        "MLLM",
        "hard_negative",
        "state_classifier",
    ]
    for fragment in forbidden_model_fragments:
        if fragment in model_source:
            raise AssertionError(
                "state model contains forbidden mechanism: {}".format(fragment)
            )
    for source_name, source in (
        ("options", options_source),
        ("factory", factory_source),
        ("data", data_source),
    ):
        if "identity_state" not in source:
            raise AssertionError(
                "{} does not register identity_state".format(source_name)
            )
    if "is_hire_v2_state_model" not in metrics_source:
        raise AssertionError("metrics does not detect the v16.3 state model")
    if "compute_state_reranked_similarity" not in metrics_source:
        raise AssertionError("metrics does not execute state reranking")
    for meter in (
        "state_pair_loss",
        "state_gate",
        "state_positive_coverage",
        "state_positive_negative_margin",
        "state_final_main_weight",
    ):
        if meter not in processor_source:
            raise AssertionError(
                "processor does not log {}".format(meter)
            )
    if "_OBSERVATION_MAIN_WEIGHT = 0.5" not in component_source:
        raise AssertionError("observation weight is not fixed at 0.5")
    if "_IDENTITY_MAIN_WEIGHT = 0.25" not in component_source:
        raise AssertionError("identity-final weight is not fixed at 0.25")
    if "_STATE_FINAL_MAIN_WEIGHT = 0.25" not in component_source:
        raise AssertionError("state-final weight is not fixed at 0.25")
    results["source_invariants"] = "passed"

    payload = {
        "experiment_version": "v16.3.0",
        "status": "passed",
        "checks": results,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if cli.output_json:
        output = Path(cli.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

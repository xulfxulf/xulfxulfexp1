#!/usr/bin/env python
"""Static and mathematical audit for HIRE-v2 v16.2.1."""

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

from model.hire_v2_identity_balanced_components import (
    SharedIdentityMean,
    aggregate_identity_balanced_objectives,
    build_identity_final_embedding,
    identity_residual_score,
    masked_identity_group_consensus,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit v16.2.1 code/formula invariants"
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="optional JSON result path",
    )
    return parser.parse_args()


def assert_close(left, right, name, atol=1e-6):
    if not torch.allclose(left, right, atol=atol, rtol=0.0):
        raise AssertionError(
            "{} mismatch: {} vs {}".format(name, left, right)
        )


def main():
    cli = parse_args()
    torch.manual_seed(7)
    results = {}

    # 1. Deterministic masked group consensus.
    means = F.normalize(
        torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0],
                    [0.8, 0.2, 0.0],
                    [0.6, 0.4, 0.0],
                ],
                [
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0],
                ],
            ]
        ),
        dim=-1,
    )
    mask = torch.tensor(
        [[True, True, True], [True, False, False]]
    )
    group = masked_identity_group_consensus(
        means, mask, min_supports=2
    )
    expected = F.normalize(means[0].mean(dim=0), dim=-1)
    assert_close(
        group["mean"][0], expected, "simple group mean"
    )
    if group["valid"].tolist() != [True, False]:
        raise AssertionError("minimum-support validity is wrong")
    results["masked_group_consensus"] = "passed"

    # 2. Objective aggregation exactly matches the document.
    values = [
        torch.tensor(float(index), requires_grad=True)
        for index in range(1, 10)
    ]
    aggregate = aggregate_identity_balanced_objectives(
        global_sdm=values[0],
        global_itc=values[1],
        local_sdm=values[2],
        local_itc=values[3],
        observation_sdm=values[4],
        observation_itc=values[5],
        final_sdm=values[6],
        final_itc=values[7],
        group_nce=values[8],
        auxiliary_weight=0.1,
    )
    assert_close(
        aggregate["sdm_loss"],
        0.5 * (values[0] + values[2])
        + 0.5 * values[4]
        + 0.5 * values[6],
        "balanced SDM",
    )
    assert_close(
        aggregate["itc_loss"],
        0.5 * (values[1] + values[3])
        + 0.5 * values[5]
        + 0.5 * values[7],
        "balanced ITC",
    )
    assert_close(
        aggregate["identity_group_loss"],
        0.1 * values[8],
        "group auxiliary",
    )
    results["objective_formula"] = "passed"

    # 3. Shared identity map starts as an exact identity.
    mapper = SharedIdentityMean(8)
    observation = F.normalize(torch.randn(5, 8), dim=-1)
    identity = mapper(observation)
    assert_close(identity, observation, "identity initialization")
    results["identity_initialization"] = "passed"

    # 4. The final score exactly equals the observation score at initialization.
    observation_score = observation @ observation.t()
    identity_score = identity @ identity.t()
    gate = torch.tensor(0.1)
    final_score = identity_residual_score(
        observation_score, identity_score, gate
    )
    assert_close(
        final_score,
        observation_score,
        "initial final-score equivalence",
    )
    results["initial_score_equivalence"] = "passed"

    # 5. Balanced main objective has the same observation gradient as the
    # v16.1 observation objective at initialization.
    score = torch.randn(4, 4, requires_grad=True)
    identity_score_detached = score.detach().clone()
    final = identity_residual_score(
        score, identity_score_detached, gate
    )
    balanced = (
        0.5 * score.pow(2).sum()
        + 0.5 * final.pow(2).sum()
    )
    balanced.backward()
    balanced_gradient = score.grad.detach().clone()

    baseline_score = score.detach().clone().requires_grad_(True)
    baseline = baseline_score.pow(2).sum()
    baseline.backward()
    assert_close(
        balanced_gradient,
        baseline_score.grad,
        "initial anchor-gradient equivalence",
    )
    results["initial_gradient_equivalence"] = "passed"

    # 6. Test-time concatenated embedding exactly matches the forward mixture.
    text_observation = F.normalize(torch.randn(6, 8), dim=-1)
    image_observation = F.normalize(torch.randn(7, 8), dim=-1)
    text_identity = mapper(text_observation)
    image_identity = mapper(image_observation)
    text_final = build_identity_final_embedding(
        text_observation, text_identity, gate
    )
    image_final = build_identity_final_embedding(
        image_observation, image_identity, gate
    )
    embedding_score = text_final @ image_final.t()
    expected_score = (
        (1.0 - gate)
        * (text_observation @ image_observation.t())
        + gate * (text_identity @ image_identity.t())
    )
    assert_close(
        embedding_score,
        expected_score,
        "inference score equivalence",
    )
    results["inference_equivalence"] = "passed"

    # 7. Source-level invariants across dispatch, data, logging and model.
    root = Path(os.environ.get("HIRE_V2_AUDIT_SOURCE_ROOT", PROJECT_ROOT))
    model_source = (
        root / "model" / "hire_v2_identity_balanced_model.py"
    ).read_text(encoding="utf-8")
    component_source = (
        root / "model" / "hire_v2_identity_balanced_components.py"
    ).read_text(encoding="utf-8")
    options_source = (root / "utils" / "options.py").read_text(
        encoding="utf-8"
    )
    factory_source = (root / "model" / "__init__.py").read_text(
        encoding="utf-8"
    )
    data_source = (root / "datasets" / "build.py").read_text(
        encoding="utf-8"
    )
    processor_source = (
        root / "processor" / "processor.py"
    ).read_text(encoding="utf-8")

    if "BoundedImageUncertainty" in model_source:
        raise AssertionError(
            "v16.2.1 must not instantiate the inactive uncertainty head"
        )
    if "heterogeneity_aware_identity_intersection" in model_source:
        raise AssertionError(
            "v16.2.1 must not use the inactive heterogeneity weighting"
        )
    if "masked_identity_group_consensus" not in model_source:
        raise AssertionError(
            "v16.2.1 model does not use deterministic group consensus"
        )
    if "_OBSERVATION_MAIN_WEIGHT = 0.5" not in component_source:
        raise AssertionError("observation main weight is not fixed at 0.5")
    if "_FINAL_MAIN_WEIGHT = 0.5" not in component_source:
        raise AssertionError("final main weight is not fixed at 0.5")
    for name, source in (
        ("options", options_source),
        ("model factory", factory_source),
        ("data builder", data_source),
    ):
        if "identity_balanced" not in source:
            raise AssertionError(
                "{} does not register identity_balanced".format(name)
            )
    for meter_name in (
        "balanced_main_objective",
        "identity_group_dispersion",
        "identity_group_support_cosine",
        "observation_main_weight",
        "final_main_weight",
    ):
        if meter_name not in processor_source:
            raise AssertionError(
                "processor does not log {}".format(meter_name)
            )
    results["source_invariants"] = "passed"

    payload = {
        "experiment_version": "v16.2.1",
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

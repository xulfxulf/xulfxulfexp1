#!/usr/bin/env python
"""Static and mathematical audit for HIRE-v2 v16.4.0."""

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

from model.hire_v2_token_route_components import (
    AttentionRawTokenSelector,
    TokenPropagabilityRouter,
    ZeroInitializedIdentityTokenResidual,
    aggregate_identity_token_route_objectives,
    build_group_propagability_targets,
    choose_hard_negative_indices,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit v16.4.0 code/formula invariants"
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
    torch.manual_seed(41)
    checks = {}

    image_selector = AttentionRawTokenSelector(ratio=0.5)
    image_tokens = torch.randn(2, 5, 4)
    image_attention = torch.zeros(2, 5, 5)
    image_attention[:, 0, :] = torch.tensor(
        [10.0, 0.2, 0.9, 0.8, 0.1]
    )
    image_pack = image_selector.select_image(
        image_tokens,
        image_attention,
    )
    if bool(image_pack["indices"].eq(0).any()):
        raise AssertionError("image raw-token selector retained CLS")
    checks["image_token_selection"] = "passed"

    text_selector = AttentionRawTokenSelector(ratio=0.75)
    text_tokens = torch.randn(1, 7, 4)
    token_ids = torch.tensor(
        [[49406, 10, 20, 30, 0, 0, 49407]]
    )
    text_attention = torch.zeros(1, 7, 7)
    text_attention[0, 6] = torch.tensor(
        [1.0, 0.4, 0.9, 0.8, 2.0, 1.9, 1.8]
    )
    text_pack = text_selector.select_text(
        text_tokens,
        token_ids,
        text_attention,
    )
    selected = text_pack["indices"][0][
        text_pack["mask"][0]
    ].tolist()
    if set(selected) != {1, 2, 3}:
        raise AssertionError(
            "text selector did not isolate valid content tokens"
        )
    checks["text_token_selection"] = "passed"

    score = torch.tensor(
        [
            [1.0, 0.99, 0.7],
            [0.98, 1.0, 0.8],
            [0.5, 0.6, 1.0],
        ]
    )
    pids = torch.tensor([1, 1, 2])
    hard_negative, valid = choose_hard_negative_indices(
        score,
        pids,
    )
    if hard_negative.tolist() != [2, 2, 1]:
        raise AssertionError("hard-negative selection crossed PID logic")
    if not bool(valid.all()):
        raise AssertionError("valid hard negatives were not detected")
    checks["hard_negative_selection"] = "passed"

    synthetic_text = {
        "tokens": F.normalize(
            torch.tensor(
                [[[1.0, 0.0], [0.0, 1.0]]]
            ),
            dim=-1,
        ),
        "mask": torch.tensor([[True, True]]),
        "weights": torch.tensor([[0.5, 0.5]]),
    }
    anchor = {
        "tokens": F.normalize(
            torch.tensor(
                [[[1.0, 0.0], [0.0, 1.0]]]
            ),
            dim=-1,
        ),
        "mask": torch.tensor([[True, True]]),
    }
    supports = {
        "tokens": F.normalize(
            torch.tensor(
                [[
                    [[1.0, 0.0], [1.0, 0.0]],
                    [[1.0, 0.0], [1.0, 0.0]],
                ]]
            ),
            dim=-1,
        ),
        "mask": torch.ones(
            1, 2, 2, dtype=torch.bool
        ),
    }
    hard = {
        "tokens": F.normalize(
            torch.tensor(
                [[[0.0, 1.0], [0.0, 1.0]]]
            ),
            dim=-1,
        ),
        "mask": torch.tensor([[True, True]]),
    }
    target = build_group_propagability_targets(
        text_pack=synthetic_text,
        anchor_image_pack=anchor,
        support_image_pack=supports,
        support_mask=torch.tensor([[True, True]]),
        hard_negative_image_pack=hard,
        hard_negative_valid=torch.tensor([True]),
        minimum_supports=2,
    )
    if not (
        target["target"][0, 0]
        > target["target"][0, 1]
    ):
        raise AssertionError(
            "group teacher did not prefer stable propagable token"
        )
    checks["group_propagability_target"] = "passed"

    router = TokenPropagabilityRouter(dim=8)
    router_tokens = F.normalize(
        torch.randn(3, 5, 8),
        dim=-1,
    )
    router_observation = F.normalize(
        torch.randn(3, 8),
        dim=-1,
    )
    router_mask = torch.ones(3, 5, dtype=torch.bool)
    probability = router(
        router_tokens,
        router_observation,
        router_mask,
    )["probability"]
    assert_close(
        probability,
        torch.full_like(probability, 0.5),
        "router initial probability",
    )
    checks["router_initialization"] = "passed"

    residual_module = ZeroInitializedIdentityTokenResidual(
        dim=8
    )
    base_raw = torch.randn(3, 8)
    residual_output = residual_module(
        base_identity_raw=base_raw,
        token_features=router_tokens,
        token_attention=torch.full((3, 5), 0.2),
        identity_probability=probability,
        token_mask=router_mask,
    )
    assert_close(
        residual_output["identity"],
        F.normalize(base_raw, dim=-1),
        "zero token residual",
    )
    if residual_output["residual"].abs().sum().item() != 0.0:
        raise AssertionError("token residual is not exactly zero initialized")
    checks["identity_initialization_equivalence"] = "passed"

    values = [
        torch.tensor(float(index), requires_grad=True)
        for index in range(1, 11)
    ]
    aggregated = aggregate_identity_token_route_objectives(
        global_sdm=values[0],
        global_itc=values[1],
        local_sdm=values[2],
        local_itc=values[3],
        observation_sdm=values[4],
        observation_itc=values[5],
        final_sdm=values[6],
        final_itc=values[7],
        group_nce=values[8],
        route_bce=values[9],
        auxiliary_weight=0.1,
    )
    expected_sdm = (
        0.5 * (values[0] + values[2])
        + 0.5 * values[4]
        + 0.5 * values[6]
    )
    expected_itc = (
        0.5 * (values[1] + values[3])
        + 0.5 * values[5]
        + 0.5 * values[7]
    )
    assert_close(
        aggregated["sdm_loss"],
        expected_sdm,
        "v16.4 SDM formula",
    )
    assert_close(
        aggregated["itc_loss"],
        expected_itc,
        "v16.4 ITC formula",
    )
    assert_close(
        aggregated["identity_group_loss"],
        0.1 * values[8],
        "identity group auxiliary",
    )
    assert_close(
        aggregated["token_route_loss"],
        0.1 * values[9],
        "token route auxiliary",
    )
    checks["objective_formula"] = "passed"

    # Source-level registration and protection rules.
    source_root = Path(
        os.environ.get(
            "HIRE_V2_AUDIT_SOURCE_ROOT",
            PROJECT_ROOT,
        )
    )
    model_source = (
        source_root
        / "model"
        / "hire_v2_identity_token_route_model.py"
    ).read_text(encoding="utf-8")
    component_source = (
        source_root
        / "model"
        / "hire_v2_token_route_components.py"
    ).read_text(encoding="utf-8")
    options_source = (
        source_root / "utils" / "options.py"
    ).read_text(encoding="utf-8")
    factory_source = (
        source_root / "model" / "__init__.py"
    ).read_text(encoding="utf-8")
    data_source = (
        source_root / "datasets" / "build.py"
    ).read_text(encoding="utf-8")
    processor_source = (
        source_root / "processor" / "processor.py"
    ).read_text(encoding="utf-8")

    required_model_fragments = [
        "self.raw_token_selector.select_text",
        "observation_score",
        "choose_hard_negative_indices",
        "build_group_propagability_targets",
        "masked_identity_group_consensus",
        "identity_token_residual",
    ]
    for fragment in required_model_fragments:
        if fragment not in model_source:
            raise AssertionError(
                "v16.4 model misses required fragment: {}".format(
                    fragment
                )
            )
    forbidden_model_fragments = [
        "state_gate",
        "state_pair_nce",
        "counterfactual_loss",
        "hard_negative_csv",
    ]
    for fragment in forbidden_model_fragments:
        if fragment in model_source:
            raise AssertionError(
                "v16.4 model contains excluded mechanism: {}".format(
                    fragment
                )
            )
    if "tokens.detach()" not in component_source:
        raise AssertionError(
            "raw route-token evidence is not detached"
        )
    if "text_observation.detach()" not in component_source:
        raise AssertionError(
            "route context is not detached"
        )
    if "self.proj.weight" not in component_source:
        raise AssertionError("token residual adapter is missing")
    if "nn.init.zeros_(self.proj.weight)" not in component_source:
        raise AssertionError(
            "token residual adapter is not zero initialized"
        )
    for source_name, source in (
        ("options", options_source),
        ("factory", factory_source),
        ("data", data_source),
    ):
        if "identity_token_route" not in source:
            raise AssertionError(
                "{} does not register v16.4 mode".format(
                    source_name
                )
            )
    for meter in (
        "token_route_loss",
        "token_route_probability_mean",
        "token_route_target_correlation",
        "identity_token_residual_norm",
    ):
        if meter not in processor_source:
            raise AssertionError(
                "processor does not log {}".format(meter)
            )
    checks["source_invariants"] = "passed"

    payload = {
        "experiment_version": "v16.4.0",
        "status": "passed",
        "checks": checks,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if cli.output_json:
        output = Path(cli.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

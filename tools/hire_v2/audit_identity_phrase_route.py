#!/usr/bin/env python
"""Static and mathematical audit for HIRE-v2 v16.6.0 and v16.7.0."""

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

from model.hire_v2_phrase_route_components import (
    RelativePhraseRouter,
    ZeroInitializedPhraseIdentityResidual,
    aggregate_identity_phrase_route_objectives,
    phrase_route_kl_divergence,
)
from tools.mllm.phrase_teacher_common import (
    comparative_raw_score,
    propagation_raw_score,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Audit v16.6/v16.7 phrase route")
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def assert_close(left, right, name, atol=1e-6):
    if not torch.allclose(left, right, atol=atol, rtol=0.0):
        raise AssertionError("{} mismatch: {} vs {}".format(name, left, right))


def main():
    cli = parse_args()
    torch.manual_seed(66)
    checks = {}

    router = RelativePhraseRouter(dim=8)
    phrase_features = F.normalize(torch.randn(2, 4, 8), dim=-1)
    observation = F.normalize(torch.randn(2, 8), dim=-1)
    phrase_mask = torch.tensor(
        [[True, True, False, False], [True, True, True, False]]
    )
    probability = router(
        phrase_features, observation, phrase_mask
    )["probability"]
    assert_close(
        probability[0, :2], torch.tensor([0.5, 0.5]), "uniform route row one"
    )
    assert_close(
        probability[1, :3],
        torch.tensor([1.0 / 3.0] * 3),
        "uniform route row two",
    )
    checks["uniform_phrase_router"] = "passed"

    residual = ZeroInitializedPhraseIdentityResidual(dim=8)
    base = torch.randn(2, 8)
    output = residual(base, phrase_features, probability, phrase_mask)
    assert_close(
        output["identity"], F.normalize(base, dim=-1), "zero phrase residual"
    )
    if output["residual"].abs().sum().item() != 0.0:
        raise AssertionError("phrase residual is not exactly zero initialized")
    checks["v1621_initial_equivalence"] = "passed"

    teacher = torch.tensor([[0.75, 0.25, 0.0], [0.5, 0.5, 0.0]])
    student = torch.tensor([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]])
    mask = torch.tensor([[True, True, False], [True, True, False]])
    supervision = torch.tensor([True, False])
    route_kl = phrase_route_kl_divergence(
        teacher, student, mask, supervision
    )
    expected_kl = 0.75 * torch.log(torch.tensor(1.5)) + 0.25 * torch.log(
        torch.tensor(0.5)
    )
    assert_close(route_kl, expected_kl, "relative route KL")
    checks["relative_route_kl"] = "passed"

    propagation = propagation_raw_score(
        "support", "unknown", ["support", "support", "contradiction"]
    )
    if abs(propagation["propagation_raw_score"] - 4.0 / 9.0) > 1e-8:
        raise AssertionError("v16.6 propagation formula mismatch")
    checks["v1660_teacher_formula"] = "passed"

    comparison = comparative_raw_score(
        propagation["propagation_raw_score"], "unknown"
    )
    if abs(comparison["comparative_raw_score"] - 2.0 / 9.0) > 1e-8:
        raise AssertionError("v16.7 comparative formula mismatch")
    checks["v1670_teacher_formula"] = "passed"

    values = [torch.tensor(float(index)) for index in range(1, 11)]
    aggregated = aggregate_identity_phrase_route_objectives(
        global_sdm=values[0],
        global_itc=values[1],
        local_sdm=values[2],
        local_itc=values[3],
        observation_sdm=values[4],
        observation_itc=values[5],
        final_sdm=values[6],
        final_itc=values[7],
        group_nce=values[8],
        route_kl=values[9],
        auxiliary_weight=0.1,
    )
    assert_close(
        aggregated["sdm_loss"],
        0.5 * (values[0] + values[2])
        + 0.5 * values[4]
        + 0.5 * values[6],
        "SDM objective",
    )
    assert_close(
        aggregated["itc_loss"],
        0.5 * (values[1] + values[3])
        + 0.5 * values[5]
        + 0.5 * values[7],
        "ITC objective",
    )
    assert_close(
        aggregated["identity_group_loss"],
        0.1 * values[8],
        "identity group objective",
    )
    assert_close(
        aggregated["phrase_route_loss"],
        0.1 * values[9],
        "phrase route objective",
    )
    checks["objective_formula"] = "passed"

    root = Path(os.environ.get("HIRE_V2_AUDIT_SOURCE_ROOT", PROJECT_ROOT))
    model_source = (
        root / "model" / "hire_v2_identity_phrase_route_model.py"
    ).read_text(encoding="utf-8")
    component_source = (
        root / "model" / "hire_v2_phrase_route_components.py"
    ).read_text(encoding="utf-8")
    options_source = (root / "utils" / "options.py").read_text(encoding="utf-8")
    factory_source = (root / "model" / "__init__.py").read_text(encoding="utf-8")
    data_source = (root / "datasets" / "build.py").read_text(encoding="utf-8")
    metrics_source = (root / "utils" / "metrics.py").read_text(encoding="utf-8")
    processor_source = (root / "processor" / "processor.py").read_text(
        encoding="utf-8"
    )

    for mode in ("identity_phrase_route", "identity_phrase_route_cmp"):
        for name, source in (
            ("options", options_source),
            ("factory", factory_source),
            ("data", data_source),
        ):
            if mode not in source:
                raise AssertionError("{} does not register {}".format(name, mode))
    if "build_hire_v2_identity_phrase_route_model" not in factory_source:
        raise AssertionError("phrase modes do not share the same model builder")
    if "v16.7.0" not in model_source or "v16.6.0" not in model_source:
        raise AssertionError("model does not expose both experiment versions")
    for fragment in (
        "phrase_route_kl_divergence",
        "masked_identity_group_consensus",
        "identity_residual_score",
        "phrase_identity_residual",
    ):
        if fragment not in model_source:
            raise AssertionError("model misses {}".format(fragment))
    for forbidden in (
        "state_gate",
        "state_pair_nce",
        "compute_state_reranked_similarity",
        "hard_negative_image",
    ):
        if forbidden in model_source:
            raise AssertionError("phrase model contains excluded {}".format(forbidden))
    if "nn.init.zeros_(self.proj.weight)" not in component_source:
        raise AssertionError("phrase residual is not zero initialized")
    if "phrase_token_mask" not in metrics_source:
        raise AssertionError("standard evaluator does not pass phrase spans")
    for meter in (
        "phrase_route_loss",
        "phrase_route_spearman",
        "phrase_identity_residual_norm",
    ):
        if meter not in processor_source:
            raise AssertionError("processor does not log {}".format(meter))
    checks["shared_architecture_and_registration"] = "passed"

    teacher_common = (
        root / "tools" / "mllm" / "phrase_teacher_common.py"
    ).read_text(encoding="utf-8")
    merge166 = (
        root / "tools" / "mllm" / "merge_phrase_teacher.py"
    ).read_text(encoding="utf-8")
    merge167 = (
        root / "tools" / "mllm" / "merge_phrase_comparative_teacher.py"
    ).read_text(encoding="utf-8")
    for fragment in (
        "n_support / count",
        "1.0 - n_contradiction / count",
        '"support": 0.0',
        '"unknown": 0.5',
        '"contradiction": 1.0',
    ):
        if fragment not in teacher_common:
            raise AssertionError("teacher code misses design fragment {}".format(fragment))
    if '"experiment_version": "v16.6.0"' not in merge166:
        raise AssertionError("v16.6 merge version mismatch")
    if '"experiment_version": "v16.7.0"' not in merge167:
        raise AssertionError("v16.7 merge version mismatch")
    checks["teacher_target_source_invariants"] = "passed"

    payload = {
        "versions": ["v16.6.0", "v16.7.0"],
        "status": "passed",
        "checks": checks,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if cli.output_json:
        target = Path(cli.output_json).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()

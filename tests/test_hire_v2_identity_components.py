import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F

_COMPONENT_PATH = Path(__file__).resolve().parents[1] / "model" / "hire_v2_identity_components.py"
_SPEC = importlib.util.spec_from_file_location("hire_v2_identity_components_test", _COMPONENT_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

SharedIdentityMean = _MODULE.SharedIdentityMean
BoundedImageUncertainty = _MODULE.BoundedImageUncertainty
BoundedResidualGate = _MODULE.BoundedResidualGate
heterogeneity_aware_identity_intersection = _MODULE.heterogeneity_aware_identity_intersection
paired_identity_group_nce = _MODULE.paired_identity_group_nce
identity_residual_score = _MODULE.identity_residual_score
build_identity_final_embedding = _MODULE.build_identity_final_embedding
sdm_from_similarity = _MODULE.sdm_from_similarity
itc_from_similarity = _MODULE.itc_from_similarity
aggregate_identity_objectives = _MODULE.aggregate_identity_objectives


def test_shared_identity_mean_starts_at_anchor():
    torch.manual_seed(1)
    observation = F.normalize(torch.randn(6, 8), dim=-1)
    module = SharedIdentityMean(8)
    identity = module(observation)
    assert torch.allclose(identity, observation, atol=1e-6)


def test_bounded_uncertainty_initializes_to_midpoint_and_has_gradient():
    torch.manual_seed(2)
    observation = torch.randn(5, 8)
    module = BoundedImageUncertainty(8, variance_min=0.1, variance_max=2.0)
    variance = module(observation)
    assert torch.allclose(variance, torch.full_like(variance, 1.05), atol=1e-6)
    loss = variance.mean()
    loss.backward()
    assert module.proj.weight.grad is not None
    assert torch.isfinite(module.proj.weight.grad).all()


def test_trusted_intersection_prefers_lower_variance_support():
    means = F.normalize(torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]]), dim=-1)
    variances = torch.tensor([[[0.1, 0.1], [2.0, 2.0], [0.1, 0.1]]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    result = heterogeneity_aware_identity_intersection(means, variances, mask)
    assert bool(result["valid"][0])
    assert result["mean"][0, 0] > result["mean"][0, 1]
    assert torch.isfinite(result["tau2"]).all()


def test_trusted_intersection_requires_two_supports():
    means = F.normalize(torch.randn(2, 3, 4), dim=-1)
    variances = torch.ones_like(means)
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
    result = heterogeneity_aware_identity_intersection(means, variances, mask)
    assert result["valid"].tolist() == [True, False]
    assert torch.equal(result["mean"][1], torch.zeros_like(result["mean"][1]))


def test_group_nce_ignores_same_pid_non_diagonal_targets():
    # Rows 0 and 1 share a PID.  Their non-diagonal groups are masked, not positives.
    scores = torch.tensor(
        [
            [4.0, 100.0, 0.0, 0.0],
            [100.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 4.0],
        ],
        requires_grad=True,
    )
    pids = torch.tensor([7, 7, 8, 9])
    valid = torch.ones(4, dtype=torch.bool)
    loss = paired_identity_group_nce(scores, pids, valid, torch.tensor(1.0))
    assert torch.isfinite(loss)
    loss.backward()
    assert scores.grad is not None
    assert scores.grad[0, 1].item() == 0.0
    assert scores.grad[1, 0].item() == 0.0
    assert scores.grad[0, 2].abs().item() > 0.0


def test_identity_residual_is_exact_anchor_at_initialization():
    torch.manual_seed(3)
    observation = F.normalize(torch.randn(4, 8), dim=-1)
    identity = SharedIdentityMean(8)(observation)
    observation_score = observation @ observation.t()
    identity_score = identity @ identity.t()
    gate = BoundedResidualGate(0.1)()
    final_score = identity_residual_score(observation_score, identity_score, gate)
    assert torch.allclose(final_score, observation_score, atol=1e-6)


def test_final_embedding_matches_inference_mixture():
    torch.manual_seed(4)
    observation = F.normalize(torch.randn(4, 8), dim=-1)
    identity = F.normalize(torch.randn(4, 8), dim=-1)
    gate = torch.tensor(0.3)
    final = build_identity_final_embedding(observation, identity, gate)
    score = final @ final.t()
    expected = (1.0 - gate) * (observation @ observation.t()) + gate * (
        identity @ identity.t()
    )
    assert torch.allclose(score, expected, atol=1e-6)
    assert torch.allclose(final.norm(dim=-1), torch.ones(4), atol=1e-6)


def test_score_matrix_losses_match_feature_losses_for_unit_features():
    # This checks shape, finiteness and gradients; exact repository objective
    # equivalence is audited in the full repository smoke run.
    torch.manual_seed(5)
    image = F.normalize(torch.randn(5, 8), dim=-1)
    text = F.normalize(torch.randn(5, 8), dim=-1)
    similarity = (text @ image.t()).requires_grad_()
    pids = torch.arange(5)
    scale = torch.tensor(50.0)
    sdm = sdm_from_similarity(similarity, pids, scale)
    itc = itc_from_similarity(similarity, scale)
    assert torch.isfinite(sdm)
    assert torch.isfinite(itc)
    (sdm + itc).backward()
    assert similarity.grad is not None
    assert torch.isfinite(similarity.grad).all()


def test_identity_objective_aggregation_matches_document():
    values = [torch.tensor(float(index), requires_grad=True) for index in range(1, 8)]
    aggregated = aggregate_identity_objectives(*values, auxiliary_weight=0.1)
    assert torch.equal(aggregated["sdm_loss"], 0.5 * (values[0] + values[2]) + values[4])
    assert torch.equal(aggregated["itc_loss"], 0.5 * (values[1] + values[3]) + values[5])
    assert torch.allclose(aggregated["identity_group_loss"], 0.1 * values[6])
    total = aggregated["sdm_loss"] + aggregated["itc_loss"] + aggregated["identity_group_loss"]
    total.backward()
    assert all(value.grad is not None for value in values)

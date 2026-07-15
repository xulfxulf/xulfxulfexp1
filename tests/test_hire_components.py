import math
import importlib.util
import pathlib

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "hire_components", ROOT / "model" / "hire_components.py"
)
hire_components = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hire_components)

all_negative_tal = hire_components.all_negative_tal
gaussian_pairwise_score = hire_components.gaussian_pairwise_score
heterogeneity_aware_posterior = hire_components.heterogeneity_aware_posterior
paired_state_nce = hire_components.paired_state_nce
posterior_calibration_nll = hire_components.posterior_calibration_nll
residual_alignment_loss = hire_components.residual_alignment_loss
state_safety_loss = hire_components.state_safety_loss
symmetric_multi_positive_nce = hire_components.symmetric_multi_positive_nce


def test_random_effects_intersection_prefers_low_variance_consensus():
    means = torch.tensor([[[0.0], [0.1], [3.0]]], dtype=torch.float32)
    variances = torch.tensor([[[0.1], [0.1], [10.0]]], dtype=torch.float32)
    mask = torch.tensor([[1, 1, 1]], dtype=torch.bool)
    result = heterogeneity_aware_posterior(means, variances, mask)
    assert result["valid"].item()
    assert 0.0 <= result["mean"].item() < 0.5
    assert result["variance"].item() > 0
    assert result["tau2"].item() > 0


def test_random_effects_handles_empty_rows_without_nan():
    means = torch.zeros(2, 3, 4)
    variances = torch.ones_like(means)
    mask = torch.tensor([[1, 0, 0], [0, 0, 0]], dtype=torch.bool)
    result = heterogeneity_aware_posterior(means, variances, mask)
    assert result["valid"].tolist() == [True, False]
    for key in ("mean", "variance", "tau2"):
        assert torch.isfinite(result[key]).all()


def test_gaussian_pairwise_score_prefers_matching_mean():
    q_mean = torch.tensor([[0.0, 0.0]])
    q_var = torch.ones_like(q_mean)
    g_mean = torch.tensor([[0.0, 0.0], [2.0, 2.0]])
    g_var = torch.ones_like(g_mean)
    scores = gaussian_pairwise_score(q_mean, q_var, g_mean, g_var)
    assert scores.shape == (1, 2)
    assert scores[0, 0] > scores[0, 1]


def test_multi_positive_nce_is_finite_and_differentiable():
    scores = torch.tensor(
        [[3.0, 2.0, -1.0], [2.5, 3.0, -1.0], [-1.0, -1.0, 3.0]],
        requires_grad=True,
    )
    pids = torch.tensor([1, 1, 2])
    loss = symmetric_multi_positive_nce(scores, pids, temperature=0.1)
    assert torch.isfinite(loss)
    loss.backward()
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()


def test_state_nce_ignores_same_pid_nonpair():
    image = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    text = image.clone().detach().requires_grad_(True)
    pids = torch.tensor([1, 1, 2, 3])
    loss = paired_state_nce(image, text, pids, temperature=0.1)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(text.grad).all()


def test_tal_is_nonnegative_and_differentiable():
    scores = torch.randn(5, 5, requires_grad=True)
    pids = torch.arange(5)
    loss = all_negative_tal(scores, pids, tau=0.015, margin=0.1)
    assert loss.item() >= 0
    loss.backward()
    assert torch.isfinite(scores.grad).all()


def test_calibration_and_residual_losses_are_finite():
    support_mean = torch.randn(2, 3, 4)
    support_var = torch.rand(2, 3, 4) + 0.2
    mask = torch.tensor([[1, 1, 1], [1, 0, 0]], dtype=torch.bool)
    group = heterogeneity_aware_posterior(support_mean, support_var, mask)
    calibration = posterior_calibration_nll(
        support_mean, support_var, group["mean"], group["variance"], mask
    )
    assert calibration.item() >= 0
    assert torch.isfinite(calibration)

    image_obs = torch.randn(2, 4)
    text_obs = torch.randn(2, 4)
    image_state = torch.nn.functional.normalize(torch.randn(2, 4), dim=-1)
    text_state = torch.nn.functional.normalize(torch.randn(2, 4), dim=-1)
    residual = residual_alignment_loss(
        image_state,
        text_state,
        image_obs,
        text_obs,
        group["mean"],
        group["mean"],
        group["valid"],
        group["valid"],
    )
    assert torch.isfinite(residual)


def test_state_safety_only_penalizes_negative_same_id_scores():
    text = torch.tensor([[1.0, 0.0]])
    image = torch.tensor([[1.0, 0.0]])
    support_image = torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]])
    support_text = support_image.clone()
    mask = torch.tensor([[1, 1]], dtype=torch.bool)
    loss = state_safety_loss(text, image, support_image, support_text, mask)
    assert math.isclose(loss.item(), 0.5, rel_tol=1e-5, abs_tol=1e-5)


def test_observation_fusion_starts_from_global_representation():
    fusion = hire_components.ObservationFusion(global_dim=4, local_dim=6, output_dim=4)
    global_feature = torch.randn(3, 4)
    local_feature = torch.randn(3, 6)
    output = fusion(global_feature, local_feature)
    expected = fusion.norm(global_feature)
    assert torch.allclose(output, expected, atol=1e-6, rtol=1e-6)

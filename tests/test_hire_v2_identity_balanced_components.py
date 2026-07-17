import torch
import torch.nn.functional as F

from model.hire_v2_identity_balanced_components import (
    SharedIdentityMean,
    aggregate_identity_balanced_objectives,
    build_identity_final_embedding,
    identity_residual_score,
    masked_identity_group_consensus,
    paired_identity_group_nce,
)


def test_masked_identity_group_consensus_is_simple_mean():
    means = F.normalize(
        torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0],
                    [0.8, 0.2, 0.0],
                    [0.6, 0.4, 0.0],
                ]
            ]
        ),
        dim=-1,
    )
    mask = torch.tensor([[True, True, True]])
    result = masked_identity_group_consensus(means, mask)
    expected = F.normalize(means.mean(dim=1), dim=-1)
    assert torch.allclose(result["mean"], expected, atol=1e-6)
    assert result["valid"].tolist() == [True]
    assert result["count"].tolist() == [3]


def test_invalid_group_is_finite_and_zero():
    means = F.normalize(torch.randn(2, 3, 5), dim=-1)
    mask = torch.tensor(
        [[True, False, False], [False, False, False]]
    )
    result = masked_identity_group_consensus(means, mask)
    assert result["valid"].tolist() == [False, False]
    assert torch.isfinite(result["mean"]).all()
    assert torch.equal(
        result["mean"], torch.zeros_like(result["mean"])
    )


def test_group_dispersion_is_nonnegative():
    means = F.normalize(torch.randn(4, 3, 8), dim=-1)
    mask = torch.ones(4, 3, dtype=torch.bool)
    result = masked_identity_group_consensus(means, mask)
    assert (result["dispersion"] >= 0).all()
    assert (result["dispersion_scalar"] >= 0).all()


def test_balanced_objective_formula():
    tensors = [
        torch.tensor(float(index))
        for index in range(1, 10)
    ]
    result = aggregate_identity_balanced_objectives(
        global_sdm=tensors[0],
        global_itc=tensors[1],
        local_sdm=tensors[2],
        local_itc=tensors[3],
        observation_sdm=tensors[4],
        observation_itc=tensors[5],
        final_sdm=tensors[6],
        final_itc=tensors[7],
        group_nce=tensors[8],
        auxiliary_weight=0.1,
    )
    assert torch.allclose(
        result["sdm_loss"],
        0.5 * (tensors[0] + tensors[2])
        + 0.5 * tensors[4]
        + 0.5 * tensors[6],
    )
    assert torch.allclose(
        result["itc_loss"],
        0.5 * (tensors[1] + tensors[3])
        + 0.5 * tensors[5]
        + 0.5 * tensors[7],
    )
    assert torch.allclose(
        result["identity_group_loss"], 0.1 * tensors[8]
    )


def test_identity_map_initializes_to_observation():
    torch.manual_seed(1)
    mapper = SharedIdentityMean(16)
    observation = F.normalize(torch.randn(7, 16), dim=-1)
    identity = mapper(observation)
    assert torch.allclose(identity, observation, atol=1e-6)


def test_final_score_initially_equals_observation():
    torch.manual_seed(2)
    mapper = SharedIdentityMean(8)
    text = F.normalize(torch.randn(5, 8), dim=-1)
    image = F.normalize(torch.randn(6, 8), dim=-1)
    text_identity = mapper(text)
    image_identity = mapper(image)
    observation_score = text @ image.t()
    identity_score = text_identity @ image_identity.t()
    final = identity_residual_score(
        observation_score, identity_score, torch.tensor(0.1)
    )
    assert torch.allclose(
        final, observation_score, atol=1e-6
    )


def test_balanced_gradient_matches_anchor_at_initialization():
    torch.manual_seed(3)
    gate = torch.tensor(0.1)
    observation = torch.randn(4, 4, requires_grad=True)
    identity = observation.detach().clone()
    final = identity_residual_score(
        observation, identity, gate
    )
    balanced = (
        0.5 * observation.pow(2).sum()
        + 0.5 * final.pow(2).sum()
    )
    balanced.backward()
    balanced_gradient = observation.grad.clone()

    baseline = observation.detach().clone().requires_grad_(True)
    baseline.pow(2).sum().backward()
    assert torch.allclose(
        balanced_gradient, baseline.grad, atol=1e-6
    )


def test_final_embedding_matches_convex_score():
    torch.manual_seed(4)
    gate = torch.tensor(0.17)
    text_observation = F.normalize(torch.randn(5, 8), dim=-1)
    image_observation = F.normalize(torch.randn(7, 8), dim=-1)
    mapper = SharedIdentityMean(8)
    text_identity = mapper(text_observation)
    image_identity = mapper(image_observation)

    text_final = build_identity_final_embedding(
        text_observation, text_identity, gate
    )
    image_final = build_identity_final_embedding(
        image_observation, image_identity, gate
    )
    actual = text_final @ image_final.t()
    expected = (
        (1.0 - gate)
        * (text_observation @ image_observation.t())
        + gate * (text_identity @ image_identity.t())
    )
    assert torch.allclose(actual, expected, atol=1e-6)


def test_paired_group_nce_ignores_same_pid_non_diagonal():
    scores = torch.tensor(
        [
            [1.0, 100.0, 0.0],
            [100.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        requires_grad=True,
    )
    pids = torch.tensor([1, 1, 2])
    valid = torch.tensor([True, True, True])
    loss = paired_identity_group_nce(
        scores, pids, valid, torch.tensor(1.0)
    )
    assert torch.isfinite(loss)
    loss.backward()
    # Same-PID non-diagonal positions are masked and receive no gradient.
    assert scores.grad[0, 1].item() == 0.0
    assert scores.grad[1, 0].item() == 0.0

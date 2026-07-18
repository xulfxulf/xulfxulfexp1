import torch
import torch.nn.functional as F

from model.hire_v2_token_route_components import (
    AttentionRawTokenSelector,
    TokenPropagabilityRouter,
    ZeroInitializedIdentityTokenResidual,
    aggregate_identity_token_route_objectives,
    build_group_propagability_targets,
    choose_hard_negative_indices,
    masked_correlation,
    token_route_binary_cross_entropy,
)


def test_image_selector_excludes_cls():
    selector = AttentionRawTokenSelector(ratio=0.5)
    tokens = torch.randn(2, 5, 4)
    attention = torch.zeros(2, 5, 5)
    attention[:, 0, :] = torch.tensor(
        [10.0, 0.1, 0.9, 0.8, 0.2]
    )
    result = selector.select_image(tokens, attention)
    assert result["tokens"].shape == (2, 2, 4)
    assert not result["indices"].eq(0).any()
    assert set(result["indices"][0].tolist()) == {2, 3}


def test_text_selector_excludes_special_and_padding():
    selector = AttentionRawTokenSelector(ratio=0.75)
    tokens = torch.randn(1, 7, 4)
    token_ids = torch.tensor(
        [[49406, 11, 22, 33, 0, 0, 49407]]
    )
    attention = torch.zeros(1, 7, 7)
    attention[0, 6] = torch.tensor(
        [1.0, 0.4, 0.9, 0.8, 2.0, 1.9, 1.8]
    )
    result = selector.select_text(
        tokens,
        token_ids,
        attention,
    )
    selected = result["indices"][0][
        result["mask"][0]
    ].tolist()
    assert set(selected) == {1, 2, 3}
    assert torch.allclose(
        result["weights"].sum(dim=1),
        torch.ones(1),
        atol=1e-6,
    )
    assert (
        result["tokens"][0, ~result["mask"][0]]
        .abs()
        .sum()
        .item()
        == 0.0
    )


def test_hard_negative_selection_respects_pid():
    score = torch.tensor(
        [
            [1.0, 0.99, 0.7],
            [0.98, 1.0, 0.8],
            [0.5, 0.6, 1.0],
        ]
    )
    pids = torch.tensor([1, 1, 2])
    indices, valid = choose_hard_negative_indices(
        score,
        pids,
    )
    assert valid.tolist() == [True, True, True]
    assert indices.tolist() == [2, 2, 1]


def test_group_target_prefers_cross_image_stable_token():
    # Token 0 ([1,0]) matches anchor and both supports and rejects the hard
    # negative. Token 1 ([0,1]) matches only anchor/hard-negative and is not
    # supported by the identity group.
    text_pack = {
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
    hard_negative = {
        "tokens": F.normalize(
            torch.tensor(
                [[[0.0, 1.0], [0.0, 1.0]]]
            ),
            dim=-1,
        ),
        "mask": torch.tensor([[True, True]]),
    }
    result = build_group_propagability_targets(
        text_pack=text_pack,
        anchor_image_pack=anchor,
        support_image_pack=supports,
        support_mask=torch.tensor([[True, True]]),
        hard_negative_image_pack=hard_negative,
        hard_negative_valid=torch.tensor([True]),
        minimum_supports=2,
    )
    assert result["valid"].all()
    assert (
        result["target"][0, 0].item()
        > result["target"][0, 1].item()
    )
    assert result["target"][0, 0].item() > 0.5
    assert result["target"][0, 1].item() < 0.2


def test_group_target_masks_padded_support_before_variance():
    text_pack = {
        "tokens": F.normalize(
            torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
            dim=-1,
        ),
        "mask": torch.tensor([[True, True]]),
    }
    anchor = {
        "tokens": F.normalize(
            torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
            dim=-1,
        ),
        "mask": torch.tensor([[True, True]]),
    }
    supports = {
        "tokens": F.normalize(
            torch.tensor(
                [[
                    [[1.0, 0.0], [0.9, 0.1]],
                    [[1.0, 0.0], [0.8, 0.2]],
                    [[0.0, 0.0], [0.0, 0.0]],
                ]]
            ),
            dim=-1,
        ),
        "mask": torch.tensor(
            [[[True, True], [True, True], [False, False]]]
        ),
    }
    hard_negative = {
        "tokens": F.normalize(
            torch.tensor([[[0.0, 1.0], [0.1, 0.9]]]),
            dim=-1,
        ),
        "mask": torch.tensor([[True, True]]),
    }

    result = build_group_propagability_targets(
        text_pack=text_pack,
        anchor_image_pack=anchor,
        support_image_pack=supports,
        support_mask=torch.tensor([[True, True, False]]),
        hard_negative_image_pack=hard_negative,
        hard_negative_valid=torch.tensor([True]),
        minimum_supports=2,
    )

    assert result["valid"].all()
    for key in ("support_std", "stable_margin", "target"):
        assert torch.isfinite(result[key]).all(), key


def test_router_initial_probability_is_half():
    router = TokenPropagabilityRouter(dim=8)
    tokens = F.normalize(torch.randn(3, 5, 8), dim=-1)
    observation = F.normalize(torch.randn(3, 8), dim=-1)
    mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, True, True, False],
            [True, True, False, False, False],
        ]
    )
    output = router(tokens, observation, mask)
    assert torch.allclose(
        output["probability"][mask],
        torch.full_like(
            output["probability"][mask],
            0.5,
        ),
        atol=1e-6,
    )
    assert (
        output["probability"][~mask]
        .abs()
        .sum()
        .item()
        == 0.0
    )


def test_zero_token_residual_preserves_identity_at_initialization():
    module = ZeroInitializedIdentityTokenResidual(dim=8)
    base = torch.randn(4, 8)
    tokens = F.normalize(torch.randn(4, 5, 8), dim=-1)
    attention = torch.full((4, 5), 0.2)
    probability = torch.rand(4, 5)
    mask = torch.ones(4, 5, dtype=torch.bool)
    result = module(
        base_identity_raw=base,
        token_features=tokens,
        token_attention=attention,
        identity_probability=probability,
        token_mask=mask,
    )
    expected = F.normalize(base, dim=-1)
    assert torch.allclose(
        result["identity"],
        expected,
        atol=1e-6,
    )
    assert result["residual"].abs().sum().item() == 0.0


def test_route_bce_masks_invalid_tokens_and_backpropagates():
    prediction = torch.tensor(
        [[0.5, 0.6], [0.2, 0.8]],
        requires_grad=True,
    )
    target = torch.tensor(
        [[0.4, 0.9], [0.1, 0.7]]
    )
    valid = torch.tensor(
        [[True, False], [True, True]]
    )
    loss = token_route_binary_cross_entropy(
        prediction,
        target,
        valid,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert prediction.grad[0, 1].item() == 0.0
    assert prediction.grad[0, 0].abs().item() > 0.0


def test_objective_formula_matches_document():
    values = [
        torch.tensor(float(index))
        for index in range(1, 11)
    ]
    result = aggregate_identity_token_route_objectives(
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
    assert torch.allclose(
        result["sdm_loss"],
        expected_sdm,
    )
    assert torch.allclose(
        result["itc_loss"],
        expected_itc,
    )
    assert torch.allclose(
        result["identity_group_loss"],
        0.1 * values[8],
    )
    assert torch.allclose(
        result["token_route_loss"],
        0.1 * values[9],
    )


def test_masked_correlation():
    left = torch.tensor(
        [[0.0, 0.5, 1.0, 0.0]]
    )
    right = torch.tensor(
        [[0.0, 0.5, 1.0, 100.0]]
    )
    mask = torch.tensor(
        [[True, True, True, False]]
    )
    correlation = masked_correlation(
        left,
        right,
        mask,
    )
    assert torch.allclose(
        correlation,
        torch.tensor(1.0),
        atol=1e-6,
    )

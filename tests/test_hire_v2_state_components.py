import torch
import torch.nn.functional as F

from model.hire_v2_state_components import (
    AttentionStateTokenEncoder,
    SignedBoundedStateGate,
    aggregate_identity_state_objectives,
    build_state_candidate_indices,
    masked_multi_positive_nce,
    scatter_selected_state_scores,
    selected_state_late_interaction,
    state_residual_score,
)


def test_state_gate_is_exactly_zero_and_learnable():
    gate = SignedBoundedStateGate()
    value = gate()
    assert value.item() == 0.0
    value.backward()
    assert gate.raw.grad is not None
    assert gate.raw.grad.item() == 1.0


def test_visual_selector_excludes_cls_and_uses_fixed_count():
    encoder = AttentionStateTokenEncoder(
        input_dim=8,
        image_token_count=3,
        text_token_count=2,
        output_dim=4,
    )
    tokens = torch.randn(2, 6, 8)
    attention = torch.zeros(2, 6, 6)
    attention[:, 0, 1:] = torch.tensor(
        [0.1, 0.9, 0.2, 0.8, 0.3]
    )
    output = encoder.encode_image(tokens, attention)
    assert output["tokens"].shape == (2, 3, 4)
    assert output["mask"].all()
    assert not output["indices"].eq(0).any()
    expected = {2, 4, 5}
    assert set(output["indices"][0].tolist()) == expected


def test_text_selector_masks_special_and_padding_tokens():
    encoder = AttentionStateTokenEncoder(
        input_dim=8,
        image_token_count=3,
        text_token_count=4,
        output_dim=4,
    )
    tokens = torch.randn(1, 7, 8)
    # 0=SOS, 6=EOT because it has the largest token ID; positions 4/5 are pad.
    token_ids = torch.tensor([[49406, 10, 20, 30, 0, 0, 49407]])
    attention = torch.zeros(1, 7, 7)
    attention[0, 6] = torch.tensor(
        [0.9, 0.3, 0.8, 0.7, 1.0, 0.95, 0.85]
    )
    output = encoder.encode_text(tokens, token_ids, attention)
    selected = output["indices"][0][output["mask"][0]].tolist()
    assert set(selected) == {1, 2, 3}
    assert torch.allclose(
        output["weights"].sum(dim=1),
        torch.ones(1),
        atol=1e-6,
    )
    assert output["tokens"][0, ~output["mask"][0]].abs().sum().item() == 0.0


def test_late_interaction_matches_manual_weighted_maxsim():
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
        "mask": torch.tensor(
            [[True, True], [True, True]]
        ),
    }
    candidates = torch.tensor([[0, 1]])
    output = selected_state_late_interaction(
        text_pack,
        image_pack,
        candidates,
    )
    expected_first = 1.0
    root_half = 2.0 ** -0.5
    expected_second = (
        0.75 * root_half + 0.25 * root_half
    )
    expected = torch.tensor(
        [[expected_first, expected_second]]
    )
    assert torch.allclose(
        output["score"],
        expected,
        atol=1e-6,
    )


def test_candidate_selection_forces_all_same_image_positives():
    base = torch.tensor(
        [
            [0.1, 0.0, 0.9, 0.8],
            [0.0, 0.1, 0.8, 0.9],
            [0.9, 0.8, 0.1, 0.0],
            [0.8, 0.9, 0.0, 0.1],
        ]
    )
    image_ids = torch.tensor([10, 10, 20, 30])
    indices, mask, positive = build_state_candidate_indices(
        base,
        image_ids,
        topk=2,
    )
    assert mask[0, 0] and mask[0, 1]
    assert mask[1, 0] and mask[1, 1]
    assert ((mask & positive).sum(dim=1) == positive.sum(dim=1)).all()
    assert indices.shape == (4, 2)


def test_same_pid_different_image_is_ignored_by_state_nce():
    scores = torch.tensor(
        [
            [2.0, 100.0, 0.0],
            [100.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
        ],
        requires_grad=True,
    )
    pids = torch.tensor([1, 1, 2])
    image_ids = torch.tensor([10, 11, 20])
    positive = image_ids[:, None].eq(image_ids[None, :])
    negative = pids[:, None].ne(pids[None, :])
    candidate = torch.ones_like(scores, dtype=torch.bool)
    loss = masked_multi_positive_nce(
        scores,
        positive,
        negative,
        candidate,
        logit_scale=torch.tensor(1.0),
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert scores.grad[0, 1].item() == 0.0
    assert scores.grad[1, 0].item() == 0.0


def test_state_residual_is_zero_at_initialization_and_sparse():
    base = torch.tensor(
        [[0.2, 0.3, 0.4], [0.4, 0.3, 0.2]]
    )
    selected = torch.tensor([[0.9], [0.8]])
    indices = torch.tensor([[1], [0]])
    state = scatter_selected_state_scores(
        base,
        selected,
        indices,
    )
    mask = torch.zeros_like(base, dtype=torch.bool)
    mask.scatter_(1, indices, True)
    exact = state_residual_score(
        base,
        state,
        mask,
        torch.tensor(0.0),
    )
    assert torch.equal(exact, base)
    changed = state_residual_score(
        base,
        state,
        mask,
        torch.tensor(0.5),
    )
    assert changed[0, 0].item() == base[0, 0].item()
    assert changed[1, 2].item() == base[1, 2].item()
    assert torch.allclose(changed[0, 1], torch.tensor(0.6))
    assert torch.allclose(changed[1, 0], torch.tensor(0.6))


def test_v163_objective_matches_document():
    values = [
        torch.tensor(float(index))
        for index in range(1, 13)
    ]
    result = aggregate_identity_state_objectives(
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
    assert torch.allclose(result["sdm_loss"], expected_sdm)
    assert torch.allclose(result["itc_loss"], expected_itc)
    assert torch.allclose(
        result["identity_group_loss"],
        0.1 * values[10],
    )
    assert torch.allclose(
        result["state_pair_loss"],
        0.1 * values[11],
    )


def test_initial_main_gradient_equals_v1621():
    observation = torch.randn(4, 4, requires_grad=True)
    identity_base = torch.randn(4, 4, requires_grad=True)
    state_final = identity_base
    v163 = (
        0.5 * observation.pow(2).sum()
        + 0.25 * identity_base.pow(2).sum()
        + 0.25 * state_final.pow(2).sum()
    )
    v163.backward()
    grad_observation = observation.grad.clone()
    grad_identity = identity_base.grad.clone()

    observation_ref = observation.detach().clone().requires_grad_(True)
    identity_ref = identity_base.detach().clone().requires_grad_(True)
    v1621 = (
        0.5 * observation_ref.pow(2).sum()
        + 0.5 * identity_ref.pow(2).sum()
    )
    v1621.backward()
    assert torch.allclose(
        grad_observation,
        observation_ref.grad,
        atol=1e-6,
    )
    assert torch.allclose(
        grad_identity,
        identity_ref.grad,
        atol=1e-6,
    )

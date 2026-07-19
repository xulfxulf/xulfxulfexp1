import torch
import torch.nn.functional as F

from model.hire_v2_phrase_route_components import (
    RelativePhraseRouter,
    ZeroInitializedPhraseIdentityResidual,
    aggregate_identity_phrase_route_objectives,
    masked_phrase_softmax,
    phrase_attention_pool,
    phrase_route_kl_divergence,
)


def test_phrase_attention_pool_uses_only_phrase_tokens():
    tokens = F.normalize(
        torch.tensor(
            [
                [
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                    [0.0, 0.0],
                ]
            ]
        ),
        dim=-1,
    )
    token_ids = torch.tensor([[49406, 10, 20, 49407, 0]])
    attention = torch.zeros(1, 5, 5)
    attention[0, 3] = torch.tensor([0.1, 0.9, 0.8, 0.2, 0.0])
    phrase_mask = torch.zeros(1, 5, 5, dtype=torch.bool)
    phrase_mask[0, 0, 1] = True
    phrase_mask[0, 1, 2] = True
    phrase_valid = torch.tensor([[True, True, False, False, False]])
    result = phrase_attention_pool(
        tokens, token_ids, attention, phrase_mask, phrase_valid
    )
    assert result["valid"].tolist() == [[True, True, False, False, False]]
    assert torch.allclose(result["features"][0, 0], torch.tensor([1.0, 0.0]))
    assert torch.allclose(result["features"][0, 1], torch.tensor([0.0, 1.0]))


def test_masked_phrase_softmax_is_uniform_at_zero_logits():
    logits = torch.zeros(2, 4)
    mask = torch.tensor(
        [[True, True, False, False], [True, True, True, False]]
    )
    probability = masked_phrase_softmax(logits, mask)
    assert torch.allclose(probability[0, :2], torch.tensor([0.5, 0.5]))
    assert torch.allclose(
        probability[1, :3], torch.tensor([1 / 3, 1 / 3, 1 / 3])
    )
    assert probability[~mask].abs().sum().item() == 0.0


def test_router_initializes_to_uniform_distribution():
    router = RelativePhraseRouter(dim=8)
    features = F.normalize(torch.randn(3, 5, 8), dim=-1)
    observation = F.normalize(torch.randn(3, 8), dim=-1)
    mask = torch.tensor(
        [
            [True, True, False, False, False],
            [True, True, True, False, False],
            [True, False, False, False, False],
        ]
    )
    probability = router(features, observation, mask)["probability"]
    assert torch.allclose(probability[0, :2], torch.tensor([0.5, 0.5]))
    assert torch.allclose(
        probability[1, :3], torch.tensor([1 / 3, 1 / 3, 1 / 3])
    )
    assert probability[2, 0].item() == 1.0


def test_zero_phrase_residual_preserves_base_identity():
    module = ZeroInitializedPhraseIdentityResidual(dim=8)
    base = torch.randn(4, 8)
    phrases = F.normalize(torch.randn(4, 5, 8), dim=-1)
    probability = torch.full((4, 5), 0.2)
    mask = torch.ones(4, 5, dtype=torch.bool)
    output = module(base, phrases, probability, mask)
    assert output["residual"].abs().sum().item() == 0.0
    assert torch.allclose(output["identity"], F.normalize(base, dim=-1))


def test_route_kl_matches_manual_distribution():
    teacher = torch.tensor([[0.75, 0.25, 0.0], [0.5, 0.5, 0.0]])
    student = torch.tensor([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]])
    mask = torch.tensor([[True, True, False], [True, True, False]])
    supervision = torch.tensor([True, False])
    loss = phrase_route_kl_divergence(
        teacher, student, mask, supervision
    )
    expected = 0.75 * torch.log(torch.tensor(1.5)) + 0.25 * torch.log(
        torch.tensor(0.5)
    )
    assert torch.allclose(loss, expected, atol=1e-6)


def test_route_kl_ignores_invalid_rows():
    student = torch.tensor([[0.5, 0.5], [0.5, 0.5]], requires_grad=True)
    teacher = torch.tensor([[0.8, 0.2], [1.0, 0.0]])
    mask = torch.ones(2, 2, dtype=torch.bool)
    supervision = torch.tensor([True, False])
    loss = phrase_route_kl_divergence(
        teacher, student, mask, supervision
    )
    loss.backward()
    assert student.grad[1].abs().sum().item() == 0.0
    assert student.grad[0].abs().sum().item() > 0.0


def test_objective_matches_document():
    values = [torch.tensor(float(index)) for index in range(1, 11)]
    result = aggregate_identity_phrase_route_objectives(
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
    assert torch.allclose(
        result["sdm_loss"],
        0.5 * (values[0] + values[2])
        + 0.5 * values[4]
        + 0.5 * values[6],
    )
    assert torch.allclose(
        result["itc_loss"],
        0.5 * (values[1] + values[3])
        + 0.5 * values[5]
        + 0.5 * values[7],
    )
    assert torch.allclose(result["identity_group_loss"], 0.1 * values[8])
    assert torch.allclose(result["phrase_route_loss"], 0.1 * values[9])

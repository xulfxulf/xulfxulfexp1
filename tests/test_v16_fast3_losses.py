import inspect
from types import SimpleNamespace

import torch
import torch.nn as nn

import model.build as build_module
from model.build import IRRA
from tools.v16.prepare_fast3_inputs import derive_hard_negatives_from_relation_pairs


def _loss_model():
    model = IRRA.__new__(IRRA)
    nn.Module.__init__(model)
    return model


def test_masked_logmeanexp_matches_manual_calculation():
    model = _loss_model()
    logits = torch.log(torch.tensor([[2.0, 3.0, 5.0]]))
    score, valid = model._masked_logmeanexp(
        logits,
        torch.tensor([[True, False, True]]),
    )
    expected = torch.log(torch.tensor([(2.0 + 5.0) / 2.0]))
    assert valid.tolist() == [True]
    assert torch.allclose(score, expected, atol=1e-6)


def test_zero_weight_support_is_masked():
    model = _loss_model()
    logits = torch.log(torch.tensor([[2.0, 5.0]]))
    score, valid = model._masked_logmeanexp(
        logits,
        torch.tensor([[True, True]]),
        weights=torch.tensor([[0.0, 1.0]]),
    )
    assert valid.tolist() == [True]
    assert torch.allclose(score, torch.log(torch.tensor([5.0])), atol=1e-6)


def test_no_valid_support_returns_differentiable_zero():
    model = _loss_model()
    image_feats = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    text_feats = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    support_feats = torch.zeros(2, 1, 2, requires_grad=True)
    loss, valid_rows, support_valid_rows, _ = model._support_bag_rank_loss(
        image_feats,
        text_feats,
        support_feats,
        torch.tensor([[False], [False]]),
        torch.zeros(2, 1),
        torch.tensor([0, 1]),
        torch.tensor(1.0),
    )
    assert not valid_rows.any()
    assert not support_valid_rows.any()
    assert loss.item() == 0.0
    loss.backward()
    assert text_feats.grad is not None
    assert torch.allclose(text_feats.grad, torch.zeros_like(text_feats.grad))


def test_hard_negative_is_added_and_duplicate_is_masked():
    model = _loss_model()
    image_feats = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    text_feats = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    support_feats = image_feats.unsqueeze(1).clone()
    support_mask = torch.ones(2, 1, dtype=torch.bool)
    support_weights = torch.ones(2, 1)
    pids = torch.tensor([0, 1])
    no_hn, _, _, _ = model._support_bag_rank_loss(
        image_feats,
        text_feats,
        support_feats,
        support_mask,
        support_weights,
        pids,
        torch.tensor(1.0),
    )
    with_hn, _, _, hn_valid = model._support_bag_rank_loss(
        image_feats,
        text_feats,
        support_feats,
        support_mask,
        support_weights,
        pids,
        torch.tensor(1.0),
        hard_negative_feats=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        hard_negative_mask=torch.tensor([True, False]),
        hard_negative_image_ids=torch.tensor([999, -1]),
        batch_image_ids=torch.tensor([10, 11]),
    )
    assert hn_valid.tolist() == [True, False]
    assert with_hn > no_hn
    duplicate_hn, _, _, duplicate_valid = model._support_bag_rank_loss(
        image_feats,
        text_feats,
        support_feats,
        support_mask,
        support_weights,
        pids,
        torch.tensor(1.0),
        hard_negative_feats=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        hard_negative_mask=torch.tensor([True, False]),
        hard_negative_image_ids=torch.tensor([10, -1]),
        batch_image_ids=torch.tensor([10, 11]),
    )
    assert duplicate_valid.tolist() == [False, False]
    assert torch.allclose(duplicate_hn, no_hn, atol=1e-6)


def test_state_nontransitive_loss_uses_only_explicit_conflicts():
    model = _loss_model()
    image_state = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    text_state = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    support_state = torch.tensor(
        [
            [[0.5, 0.0], [100.0, 0.0]],
            [[0.0, 0.5], [0.0, 100.0]],
        ],
        requires_grad=True,
    )
    support_mask = torch.ones(2, 2, dtype=torch.bool)
    conflict_mask = torch.tensor([[True, False], [False, False]])
    loss, valid_rows = model._state_nontransitive_loss(
        image_state,
        text_state,
        support_state,
        support_mask,
        conflict_mask,
        torch.tensor(1.0),
    )
    expected = torch.nn.functional.softplus(torch.tensor(0.5 - 1.0))
    assert valid_rows.tolist() == [True, False]
    assert torch.allclose(loss, expected, atol=1e-6)
    loss.backward()
    assert support_state.grad is not None
    assert torch.allclose(support_state.grad[0, 1], torch.zeros(2))


def test_hard_negative_cannot_enter_state_loss_signature():
    parameters = inspect.signature(IRRA._state_nontransitive_loss).parameters
    assert "hard_negative_feats" not in parameters
    assert "hard_negative_mask" not in parameters


class _TinyClip(nn.Module):
    def __init__(self, dim=4):
        super().__init__()
        self.image_projection = nn.Linear(3, dim)
        self.text_projection = nn.Linear(dim, dim)

    def encode_image(self, images):
        pooled = images.mean(dim=(-1, -2))
        return self.image_projection(pooled).unsqueeze(1)

    def encode_text(self, caption_ids):
        one_hot = torch.nn.functional.one_hot(caption_ids % 4, num_classes=4).float()
        return self.text_projection(one_hot)

    def forward(self, images, caption_ids):
        return self.encode_image(images), self.encode_text(caption_ids)


def _tiny_fast3_model(mode):
    model = _loss_model()
    model.args = SimpleNamespace(batch_size=4, irra_light_identity_loss="sdm")
    model.irra_light = True
    model.irra_light_mode = mode
    model.irra_light_bag = False
    model.irra_light_bag_consistency = False
    model.irra_light_fast_bag = True
    model.irra_light_state_route = mode in {"split_bag_state", "split_bag_state_hn"}
    model.irra_light_hard_negative = mode == "split_bag_state_hn"
    model.irra_light_single_proj = False
    model.irra_light_split = True
    model.irra_light_with_id = False
    model.base_model = _TinyClip()
    model.identity_head = nn.Linear(4, 4, bias=False)
    model.state_head = nn.Linear(4, 4, bias=False)
    nn.init.eye_(model.identity_head.weight)
    nn.init.eye_(model.state_head.weight)
    model.register_buffer("logit_scale", torch.tensor(1.0))
    return model


def _tiny_fast3_batch(include_hard_negative):
    batch_size = 4
    support_size = 2
    batch = {
        "images": torch.randn(batch_size, 3, 2, 2),
        "caption_ids": torch.tensor([[1, 2, 3]] * batch_size),
        "pids": torch.tensor([0, 0, 1, 1]),
        "image_ids": torch.tensor([10, 11, 12, 13]),
        "support_images": torch.randn(batch_size, support_size, 3, 2, 2),
        "support_mask": torch.ones(batch_size, support_size, dtype=torch.bool),
        "support_reliability": torch.tensor(
            [[1.0, 0.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
        ),
        "support_conflict_mask": torch.tensor(
            [[True, False], [False, False], [True, False], [False, False]]
        ),
    }
    if include_hard_negative:
        batch.update({
            "hard_negative_image": torch.randn(batch_size, 3, 2, 2),
            "hard_negative_mask": torch.tensor([True, True, False, True]),
            "hard_negative_image_id": torch.tensor([100, 101, -1, 103]),
            "hard_negative_pid": torch.tensor([1, 1, -1, 0]),
        })
    return batch


def test_all_fast3_modes_run_one_batch_forward_backward():
    expected_loss_keys = {
        "split_bag_safe": {
            "identity_sdm_loss",
            "state_itc_loss",
            "identity_bag_loss",
        },
        "split_bag_state": {
            "identity_sdm_loss",
            "state_itc_loss",
            "identity_bag_loss",
            "state_nontransitive_loss",
        },
        "split_bag_state_hn": {
            "identity_sdm_loss",
            "state_itc_loss",
            "identity_bag_loss",
            "state_nontransitive_loss",
        },
    }
    for mode, expected_keys in expected_loss_keys.items():
        model = _tiny_fast3_model(mode)
        ret = model(_tiny_fast3_batch(mode == "split_bag_state_hn"))
        assert expected_keys.issubset(ret)
        total_loss = sum(value for key, value in ret.items() if "loss" in key)
        assert torch.isfinite(total_loss)
        total_loss.backward()
        assert model.identity_head.weight.grad is not None
        assert model.state_head.weight.grad is not None
        if mode == "split_bag_safe":
            assert "state_nontransitive_loss" not in ret
        if mode == "split_bag_state_hn":
            assert "hard_negative_valid_ratio" in ret


def test_fast3_modes_construct_split_heads_without_legacy_bag(monkeypatch):
    def fake_clip_builder(_choice, _img_size, _stride_size):
        return _TinyClip(), {"embed_dim": 4}

    monkeypatch.setattr(
        build_module,
        "build_CLIP_from_openai_pretrained",
        fake_clip_builder,
    )
    for mode in ("split_bag_safe", "split_bag_state", "split_bag_state_hn"):
        args = SimpleNamespace(
            loss_names="irra_light",
            irra_light=True,
            irra_light_mode=mode,
            pretrain_choice="unit-test",
            img_size=(4, 4),
            stride_size=1,
            temperature=1.0,
            irra_light_identity_loss="sdm",
        )
        model = build_module.IRRA(args, num_classes=2)
        assert model.irra_light_split
        assert not model.irra_light_bag
        assert hasattr(model, "identity_head")
        assert hasattr(model, "state_head")


def test_native_relation_pairs_are_aggregated_without_model_inference():
    train_index = {
        "image_by_id": {
            0: {"pid": 0, "original_pid": 10},
            1: {"pid": 0, "original_pid": 10},
            2: {"pid": 1, "original_pid": 11},
        },
    }
    relation_rows = [
        {
            "pair_id": f"p{index}",
            "split": "train",
            "relation_type": "E_different_identity_high_similarity_candidate",
            "anchor_text_index": str(index),
            "anchor_image_index": str(0 if index < 2 else 1),
            "anchor_pid": "0",
            "candidate_image_index": "2",
            "candidate_pid": "1",
        }
        for index in range(3)
    ]
    relation_fields = list(relation_rows[0])
    score_rows = [
        {"mode": "split_pure", "head": "id", "pair_id": f"p{index}", "score": str(0.9 - index / 100)}
        for index in range(3)
    ]
    score_fields = list(score_rows[0])
    candidates, pair_count = derive_hard_negatives_from_relation_pairs(
        relation_rows,
        relation_fields,
        score_rows,
        score_fields,
        train_index,
        "split_pure",
        "id",
    )
    assert pair_count == 3
    assert candidates == [{
        "anchor_pid": 0,
        "negative_pid": 1,
        "negative_image_id": 2,
        "trigger_caption_count": 3,
        "trigger_image_count": 2,
        "rank": 1,
    }]

import json

import torch

from datasets.phrase_route_io import (
    PhraseRouteTable,
    normalized_caption_sha1,
    phrase_record_to_tensors,
    write_jsonl,
)


def sample_row(version="v16.6.0", route_kind="propagation"):
    caption = "a person wears a black shirt and white shoes"
    return {
        "experiment_version": version,
        "route_kind": route_kind,
        "split": "train",
        "record_index": 0,
        "image_id": 2,
        "caption": caption,
        "caption_sha1": normalized_caption_sha1(caption),
        "route_supervision_valid": True,
        "phrases": [
            {
                "phrase_id": "p1",
                "text": "black shirt",
                "token_positions": [4, 5],
                "category": "upper",
                "propagation_raw_score": 1.0,
                "target_weight": 0.75,
            },
            {
                "phrase_id": "p2",
                "text": "white shoes",
                "token_positions": [7, 8],
                "category": "shoes",
                "propagation_raw_score": 0.5,
                "target_weight": 0.25,
            },
        ],
    }


def test_phrase_record_to_tensors_has_fixed_text_length_axis():
    tensors = phrase_record_to_tensors(sample_row(), text_length=12)
    assert tensors["phrase_token_mask"].shape == (12, 12)
    assert tensors["phrase_valid_mask"].shape == (12,)
    assert tensors["phrase_route_target"].shape == (12,)
    assert tensors["phrase_valid_mask"].sum().item() == 2
    assert torch.allclose(
        tensors["phrase_route_target"][:2], torch.tensor([0.75, 0.25])
    )


def test_phrase_route_table_validates_version_and_caption(tmp_path):
    path = tmp_path / "labels.jsonl"
    write_jsonl(str(path), [sample_row()])
    table = PhraseRouteTable(
        str(path),
        split="train",
        expected_version="v16.6.0",
        expected_route_kind="propagation",
    )
    row = table.validate_caption(
        0, "a person wears a black shirt and white shoes", image_id=2
    )
    assert row["record_index"] == 0


def test_phrase_route_table_rejects_mismatched_caption(tmp_path):
    path = tmp_path / "labels.jsonl"
    write_jsonl(str(path), [sample_row()])
    table = PhraseRouteTable(str(path), split="train")
    try:
        table.validate_caption(0, "different caption", image_id=2)
    except RuntimeError:
        pass
    else:
        raise AssertionError("caption mismatch was not rejected")


def test_comparative_raw_score_is_loaded_first():
    row = sample_row("v16.7.0", "comparative")
    row["phrases"][0]["comparative_raw_score"] = 0.4
    tensors = phrase_record_to_tensors(row, text_length=12)
    assert torch.allclose(tensors["phrase_teacher_raw_score"][0], torch.tensor(0.4))

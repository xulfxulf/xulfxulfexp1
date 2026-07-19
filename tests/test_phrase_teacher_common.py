from tools.mllm.phrase_teacher_common import (
    comparative_raw_score,
    normalize_record_phrase_targets,
    propagation_raw_score,
    strict_label,
)


def test_strict_label_requires_four_identical_judgments():
    label, agreement = strict_label(["support"] * 4)
    assert label == "support" and agreement
    label, agreement = strict_label(
        ["support", "support", "unknown", "support"]
    )
    assert label == "unknown" and not agreement


def test_propagation_formula_matches_design():
    score = propagation_raw_score(
        "support", "unknown", ["support", "support", "contradiction"]
    )
    assert abs(score["propagation_raw_score"] - 4.0 / 9.0) < 1e-8


def test_anchor_contradiction_blocks_propagation():
    score = propagation_raw_score(
        "contradiction", "unknown", ["support", "support", "support"]
    )
    assert score["propagation_raw_score"] == 0.0


def test_comparative_factor_matches_design():
    assert comparative_raw_score(0.8, "support")["comparative_raw_score"] == 0.0
    assert comparative_raw_score(0.8, "unknown")["comparative_raw_score"] == 0.4
    assert comparative_raw_score(0.8, "contradiction")["comparative_raw_score"] == 0.8


def test_relative_normalization_requires_two_positive_phrases():
    phrases = [
        {"phrase_id": "a", "propagation_raw_score": 1.0},
        {"phrase_id": "b", "propagation_raw_score": 0.5},
    ]
    output, valid = normalize_record_phrase_targets(
        phrases, "propagation_raw_score"
    )
    assert valid
    assert abs(sum(item["target_weight"] for item in output) - 1.0) < 1e-8
    output, valid = normalize_record_phrase_targets(
        [{"phrase_id": "a", "propagation_raw_score": 1.0}],
        "propagation_raw_score",
    )
    assert not valid


def test_teacher_payload_maps_supports_by_image_id():
    from tools.mllm.phrase_teacher_common import validate_teacher_payload

    case = {
        "case_id": "c1",
        "phrases": [{"phrase_id": "p1"}],
        "supports": [{"image_id": 10}, {"image_id": 20}],
    }
    payload = {
        "case_id": "c1",
        "phrases": [
            {
                "phrase_id": "p1",
                "anchor": "support",
                "sibling": "unknown",
                "support_by_image_id": {
                    "20": "contradiction",
                    "10": "support",
                },
            }
        ],
    }
    parsed = validate_teacher_payload(case, payload)
    assert parsed["p1"]["support_by_image_id"]["10"] == "support"
    assert parsed["p1"]["support_by_image_id"]["20"] == "contradiction"


def test_comparative_teacher_requires_hard_negative_label():
    from tools.mllm.phrase_teacher_common import validate_teacher_payload

    case = {
        "case_id": "c2",
        "phrases": [{"phrase_id": "p1"}],
        "supports": [{"image_id": 10}],
        "hard_negative": {"image_id": 30},
    }
    payload = {
        "case_id": "c2",
        "phrases": [
            {
                "phrase_id": "p1",
                "anchor": "support",
                "sibling": "unknown",
                "support_by_image_id": {"10": "support"},
                "hard_negative": "contradiction",
            }
        ],
    }
    parsed = validate_teacher_payload(case, payload)
    assert parsed["p1"]["hard_negative"] == "contradiction"


def test_teacher_prompt_uses_exact_support_image_ids_not_positions():
    from tools.mllm.phrase_teacher_common import build_teacher_prompt

    case = {
        "case_id": "c3",
        "caption": "a person in a red shirt",
        "sibling_caption": "the person wears red",
        "image_id": 3,
        "anchor_image_path": "/tmp/anchor.jpg",
        "phrases": [{"phrase_id": "p1", "text": "red shirt"}],
        "supports": [
            {"image_id": 6, "path": "/tmp/6.jpg"},
            {"image_id": 5, "path": "/tmp/5.jpg"},
            {"image_id": 8, "path": "/tmp/8.jpg"},
        ],
    }

    prompt = build_teacher_prompt(case, "forward")

    assert '"6": "support|contradiction|unknown"' in prompt
    assert '"5": "support|contradiction|unknown"' in prompt
    assert '"8": "support|contradiction|unknown"' in prompt
    assert "not image_1/image_2 positions" in prompt
    assert '"123"' not in prompt

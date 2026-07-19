import json

import pytest

from tools.mllm.build_phrase_teacher_cases import quality_subset
from tools.mllm.run_phrase_teacher import attempt_counts, existing_case_ids


CATEGORIES = ("upper", "lower", "shoes", "bag", "hat", "hair", "pose")


def _case(case_id, *categories):
    return {
        "case_id": case_id,
        "phrases": [
            {"phrase_id": "{}_{}".format(case_id, category), "category": category}
            for category in categories
        ],
    }


def test_quality_subset_contains_exact_unique_quota_per_category():
    cases = [_case("overlap", *CATEGORIES)]
    for category in CATEGORIES:
        cases.extend(
            [_case("{}_{}".format(category, index), category) for index in range(3)]
        )

    selected = quality_subset(cases, per_category=2)

    assert len(selected) == 2 * len(CATEGORIES)
    assert len({row["case_id"] for row in selected}) == len(selected)
    assert {
        category: sum(
            row["quality_sampling_category"] == category for row in selected
        )
        for category in CATEGORIES
    } == {category: 2 for category in CATEGORIES}


def test_quality_subset_raises_instead_of_silently_underfilling():
    cases = [_case("{}_only".format(category), category) for category in CATEGORIES]
    with pytest.raises(RuntimeError, match="cannot provide"):
        quality_subset(cases, per_category=2)


def test_failed_teacher_cases_remain_retryable(tmp_path):
    output = tmp_path / "teacher.jsonl"
    rows = [
        {"case_id": "failed", "parsed_ok": False},
        {"case_id": "passed", "parsed_ok": True},
        {"case_id": "failed", "parsed_ok": False},
    ]
    output.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    assert existing_case_ids(output) == {"passed"}
    assert attempt_counts(output) == {"failed": 2, "passed": 1}

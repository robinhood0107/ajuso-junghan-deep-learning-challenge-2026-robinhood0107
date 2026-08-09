from __future__ import annotations

import math

import pytest

from deep_challenge.evaluation import (
    compare_predictions_paired,
    evaluate_predictions,
    holm_bonferroni,
    is_canonical_integer,
)


@pytest.mark.parametrize("value", ["0", "42", "-42", "3431577212128939"])
def test_canonical_integer_accepts_valid(value: str) -> None:
    assert is_canonical_integer(value)


@pytest.mark.parametrize("value", ["+1", "01", "-0", "1.0", "1e3", " 1", ""])
def test_canonical_integer_rejects_invalid(value: str) -> None:
    assert not is_canonical_integer(value)


def test_evaluate_reports_missing_extra_invalid_and_incorrect() -> None:
    result = evaluate_predictions(
        {"a": "1", "b": "2", "c": "-3", "d": "0"},
        {"a": "1", "b": "02", "c": "4", "extra": "9"},
    )
    assert result.total == 4
    assert result.correct == 1
    assert result.accuracy == 0.25
    assert result.missing_ids == ("d",)
    assert result.extra_ids == ("extra",)
    assert result.invalid_ids == ("b",)
    assert result.incorrect_ids == ("b", "c", "d")


def test_evaluate_rejects_empty_or_invalid_gold() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        evaluate_predictions({}, {})
    with pytest.raises(ValueError, match="non-canonical"):
        evaluate_predictions({"a": "01"}, {"a": "1"})


def test_paired_comparison_counts_and_is_deterministic() -> None:
    gold = {"a": "1", "b": "2", "c": "3", "d": "4"}
    prediction_a = {"a": "1", "b": "2", "c": "0", "d": "0"}
    prediction_b = {"a": "1", "b": "0", "c": "3", "d": "4"}
    first = compare_predictions_paired(
        gold, prediction_a, prediction_b, bootstrap_samples=500, seed=7
    )
    second = compare_predictions_paired(
        gold, prediction_a, prediction_b, bootstrap_samples=500, seed=7
    )
    assert first == second
    assert first.both_correct == 1
    assert first.only_a_correct == 1
    assert first.only_b_correct == 2
    assert first.both_wrong == 0
    assert first.accuracy_a == 0.5
    assert first.accuracy_b == 0.75
    assert first.delta_b_minus_a == 0.25
    assert 0.0 <= first.mcnemar_exact_p_value <= 1.0
    assert first.bootstrap_delta_ci_low <= 0.25 <= first.bootstrap_delta_ci_high
    assert first.bootstrap_unit == "row_iid"
    assert first.bootstrap_group_count == 4


def test_paired_comparison_identical_predictions() -> None:
    result = compare_predictions_paired(
        {"a": "1", "b": "2"},
        {"a": "1", "b": "0"},
        {"a": "1", "b": "0"},
        bootstrap_samples=20,
    )
    assert result.delta_b_minus_a == 0.0
    assert result.mcnemar_exact_p_value == 1.0
    assert math.isclose(result.bootstrap_delta_ci_low, 0.0)
    assert math.isclose(result.bootstrap_delta_ci_high, 0.0)


def test_paired_comparison_resamples_complete_duplicate_groups() -> None:
    gold = {"a": "1", "b": "2", "c": "3", "d": "4"}
    prediction_a = {"a": "1", "b": "2", "c": "0", "d": "0"}
    prediction_b = {"a": "0", "b": "0", "c": "3", "d": "4"}
    result = compare_predictions_paired(
        gold,
        prediction_a,
        prediction_b,
        bootstrap_samples=200,
        seed=9,
        group_by_id={"a": "g1", "b": "g1", "c": "g2", "d": "g2"},
    )
    assert result.bootstrap_unit == "duplicate_cluster"
    assert result.bootstrap_group_count == 2
    assert result.bootstrap_delta_ci_low == -1.0
    assert result.bootstrap_delta_ci_high == 1.0


def test_paired_comparison_validates_group_mapping() -> None:
    with pytest.raises(ValueError, match="missing gold IDs"):
        compare_predictions_paired(
            {"a": "1", "b": "2"},
            {"a": "1", "b": "0"},
            {"a": "0", "b": "2"},
            bootstrap_samples=10,
            group_by_id={"a": "g"},
        )


def test_holm_bonferroni_controls_familywise_error() -> None:
    adjusted = holm_bonferroni({"first": 0.01, "second": 0.03, "third": 0.04})
    by_name = {item.hypothesis: item for item in adjusted}
    assert by_name["first"].reject is True
    assert by_name["first"].adjusted_p_value == pytest.approx(0.03)
    assert by_name["second"].reject is False
    assert by_name["second"].adjusted_p_value == pytest.approx(0.06)
    assert by_name["third"].reject is False
    assert by_name["third"].adjusted_p_value == pytest.approx(0.06)


def test_holm_bonferroni_validates_inputs() -> None:
    with pytest.raises(ValueError):
        holm_bonferroni({"x": 0.1}, alpha=1.0)
    with pytest.raises(ValueError):
        holm_bonferroni({"x": float("nan")})


@pytest.mark.parametrize(
    ("samples", "confidence"),
    [(0, 0.95), (10, 0.0), (10, 1.0)],
)
def test_paired_comparison_validates_parameters(samples: int, confidence: float) -> None:
    with pytest.raises(ValueError):
        compare_predictions_paired(
            {"a": "1"},
            {"a": "1"},
            {"a": "1"},
            bootstrap_samples=samples,
            confidence=confidence,
        )

"""Exact-integer evaluation and paired model comparison."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

_CANONICAL_INTEGER = re.compile(r"^(?:0|-?[1-9]\d*)$")


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Exact-match evaluation over a fixed gold ID set."""

    total: int
    correct: int
    accuracy: float
    missing_ids: tuple[str, ...]
    extra_ids: tuple[str, ...]
    invalid_ids: tuple[str, ...]
    incorrect_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PairedComparison:
    """Paired correctness table with exact McNemar and bootstrap interval."""

    total: int
    both_correct: int
    only_a_correct: int
    only_b_correct: int
    both_wrong: int
    accuracy_a: float
    accuracy_b: float
    delta_b_minus_a: float
    mcnemar_exact_p_value: float
    bootstrap_delta_ci_low: float
    bootstrap_delta_ci_high: float
    bootstrap_unit: str
    bootstrap_group_count: int
    bootstrap_samples: int


@dataclass(frozen=True, slots=True)
class HolmAdjustment:
    """One family-wise-error-controlled p-value decision."""

    hypothesis: str
    raw_p_value: float
    adjusted_p_value: float
    reject: bool


def is_canonical_integer(value: str) -> bool:
    """Return whether *value* is the canonical submission integer form."""

    return _CANONICAL_INTEGER.fullmatch(value) is not None


def evaluate_predictions(
    gold: Mapping[str, str], predictions: Mapping[str, str]
) -> EvaluationResult:
    """Evaluate predictions without coercing malformed values.

    Missing and invalid predictions count as incorrect.  Extra prediction IDs are
    reported but do not change the gold-set denominator.
    """

    if not gold:
        raise ValueError("gold mapping must not be empty")
    invalid_gold = sorted(key for key, value in gold.items() if not is_canonical_integer(value))
    if invalid_gold:
        raise ValueError(f"gold contains non-canonical integer values: {invalid_gold[:5]}")

    gold_ids = set(gold)
    prediction_ids = set(predictions)
    missing = tuple(sorted(gold_ids - prediction_ids))
    extra = tuple(sorted(prediction_ids - gold_ids))
    invalid = tuple(
        sorted(
            key
            for key in gold_ids & prediction_ids
            if not isinstance(predictions[key], str)
            or not is_canonical_integer(predictions[key])
        )
    )
    invalid_set = set(invalid)
    correct_ids = {
        key
        for key in gold_ids & prediction_ids
        if key not in invalid_set and predictions[key] == gold[key]
    }
    incorrect = tuple(sorted(gold_ids - correct_ids))
    total = len(gold)
    correct = len(correct_ids)
    return EvaluationResult(
        total=total,
        correct=correct,
        accuracy=correct / total,
        missing_ids=missing,
        extra_ids=extra,
        invalid_ids=invalid,
        incorrect_ids=incorrect,
    )


def _correctness_vector(
    ids: Iterable[str], gold: Mapping[str, str], predictions: Mapping[str, str]
) -> list[int]:
    return [
        int(
            isinstance(predictions.get(identifier), str)
            and is_canonical_integer(predictions[identifier])
            and predictions[identifier] == gold[identifier]
        )
        for identifier in ids
    ]


def _mcnemar_exact_p_value(only_a: int, only_b: int) -> float:
    discordant = only_a + only_b
    if discordant == 0:
        return 1.0
    tail_end = min(only_a, only_b)
    tail_probability = sum(math.comb(discordant, k) for k in range(tail_end + 1)) / (
        2**discordant
    )
    return min(1.0, 2.0 * tail_probability)


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between zero and one")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def compare_predictions_paired(
    gold: Mapping[str, str],
    predictions_a: Mapping[str, str],
    predictions_b: Mapping[str, str],
    *,
    bootstrap_samples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
    group_by_id: Mapping[str, str] | None = None,
) -> PairedComparison:
    """Compare two systems with paired statistics and a cluster bootstrap.

    Pass the immutable duplicate/template ``group_by_id`` mapping used to build
    validation splits.  Complete groups are then resampled, preserving within-
    template dependence.  Omitting it explicitly falls back to row-IID units.
    """

    if not gold:
        raise ValueError("gold mapping must not be empty")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between zero and one")
    invalid_gold = [key for key, value in gold.items() if not is_canonical_integer(value)]
    if invalid_gold:
        raise ValueError("gold contains non-canonical integer values")

    identifiers = sorted(gold)
    correct_a = _correctness_vector(identifiers, gold, predictions_a)
    correct_b = _correctness_vector(identifiers, gold, predictions_b)
    both_correct = sum(a == 1 and b == 1 for a, b in zip(correct_a, correct_b, strict=True))
    only_a = sum(a == 1 and b == 0 for a, b in zip(correct_a, correct_b, strict=True))
    only_b = sum(a == 0 and b == 1 for a, b in zip(correct_a, correct_b, strict=True))
    both_wrong = len(identifiers) - both_correct - only_a - only_b

    differences = [b - a for a, b in zip(correct_a, correct_b, strict=True)]
    sample_size = len(differences)
    group_sums, group_sizes, bootstrap_unit = _bootstrap_groups(
        identifiers, differences, group_by_id
    )
    bootstrap_deltas = _cluster_bootstrap_deltas(
        group_sums,
        group_sizes,
        samples=bootstrap_samples,
        seed=seed,
    )
    alpha = 1.0 - confidence
    accuracy_a = sum(correct_a) / sample_size
    accuracy_b = sum(correct_b) / sample_size
    return PairedComparison(
        total=sample_size,
        both_correct=both_correct,
        only_a_correct=only_a,
        only_b_correct=only_b,
        both_wrong=both_wrong,
        accuracy_a=accuracy_a,
        accuracy_b=accuracy_b,
        delta_b_minus_a=accuracy_b - accuracy_a,
        mcnemar_exact_p_value=_mcnemar_exact_p_value(only_a, only_b),
        bootstrap_delta_ci_low=_percentile(bootstrap_deltas, alpha / 2.0),
        bootstrap_delta_ci_high=_percentile(bootstrap_deltas, 1.0 - alpha / 2.0),
        bootstrap_unit=bootstrap_unit,
        bootstrap_group_count=len(group_sums),
        bootstrap_samples=bootstrap_samples,
    )


def _bootstrap_groups(
    identifiers: Sequence[str],
    differences: Sequence[int],
    group_by_id: Mapping[str, str] | None,
) -> tuple[np.ndarray, np.ndarray, str]:
    if group_by_id is None:
        return (
            np.asarray(differences, dtype=np.int64),
            np.ones(len(differences), dtype=np.int64),
            "row_iid",
        )

    missing = [identifier for identifier in identifiers if identifier not in group_by_id]
    if missing:
        raise ValueError(f"group_by_id is missing gold IDs: {missing[:5]}")
    invalid = [
        identifier
        for identifier in identifiers
        if not isinstance(group_by_id[identifier], str) or not group_by_id[identifier]
    ]
    if invalid:
        raise ValueError(f"group_by_id contains invalid labels: {invalid[:5]}")

    differences_by_group: defaultdict[str, list[int]] = defaultdict(list)
    for identifier, difference in zip(identifiers, differences, strict=True):
        differences_by_group[group_by_id[identifier]].append(difference)
    groups = sorted(differences_by_group)
    return (
        np.asarray([sum(differences_by_group[group]) for group in groups], dtype=np.int64),
        np.asarray([len(differences_by_group[group]) for group in groups], dtype=np.int64),
        "duplicate_cluster",
    )


def _cluster_bootstrap_deltas(
    group_sums: np.ndarray,
    group_sizes: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> list[float]:
    """Vectorized, bounded-memory bootstrap over whole groups."""

    rng = np.random.default_rng(seed)
    group_count = int(group_sums.size)
    if np.all(group_sizes == 1):
        counts = np.bincount(group_sums + 1, minlength=3)
        draws = rng.multinomial(
            group_count,
            counts / group_count,
            size=samples,
        )
        deltas = (draws[:, 2] - draws[:, 0]) / group_count
        return sorted(float(value) for value in deltas)

    deltas = np.empty(samples, dtype=np.float64)
    max_index_cells = 1_000_000
    batch_size = max(1, min(samples, max_index_cells // group_count))
    for start in range(0, samples, batch_size):
        stop = min(samples, start + batch_size)
        indices = rng.integers(0, group_count, size=(stop - start, group_count))
        numerators = group_sums[indices].sum(axis=1)
        denominators = group_sizes[indices].sum(axis=1)
        deltas[start:stop] = numerators / denominators
    deltas.sort()
    return [float(value) for value in deltas]


def holm_bonferroni(
    p_values: Mapping[str, float], *, alpha: float = 0.05
) -> tuple[HolmAdjustment, ...]:
    """Apply Holm's step-down correction and preserve caller hypothesis order."""

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between zero and one")
    for hypothesis, value in p_values.items():
        if not isinstance(hypothesis, str) or not hypothesis:
            raise ValueError("hypothesis names must be non-empty strings")
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"invalid p-value for {hypothesis!r}: {value!r}")

    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    count = len(ordered)
    by_hypothesis: dict[str, HolmAdjustment] = {}
    previous_adjusted = 0.0
    still_rejecting = True
    for index, (hypothesis, raw_p_value) in enumerate(ordered):
        remaining = count - index
        adjusted = min(1.0, max(previous_adjusted, remaining * raw_p_value))
        previous_adjusted = adjusted
        reject = still_rejecting and raw_p_value <= alpha / remaining
        if not reject:
            still_rejecting = False
        by_hypothesis[hypothesis] = HolmAdjustment(
            hypothesis=hypothesis,
            raw_p_value=raw_p_value,
            adjusted_p_value=adjusted,
            reject=reject,
        )
    return tuple(by_hypothesis[hypothesis] for hypothesis in p_values)


__all__ = [
    "EvaluationResult",
    "HolmAdjustment",
    "PairedComparison",
    "compare_predictions_paired",
    "evaluate_predictions",
    "holm_bonferroni",
    "is_canonical_integer",
]

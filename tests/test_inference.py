from __future__ import annotations

import pytest

from deep_challenge.inference import (
    AdaptiveBudgetPolicy,
    Candidate,
    aggregate_candidates,
    deterministic_seed,
)


def test_seed_is_stable_unique_by_index_and_validates_inputs() -> None:
    first = deterministic_seed("val-1", 0, salt="final-v1")
    assert first == deterministic_seed("val-1", 0, salt="final-v1")
    assert first != deterministic_seed("val-1", 1, salt="final-v1")
    assert 0 <= first < 2**63
    with pytest.raises(ValueError):
        deterministic_seed("", 0, salt="x")
    with pytest.raises(ValueError):
        deterministic_seed("a", -1, salt="x")


def test_majority_and_verifier_weight_select_answer() -> None:
    result = aggregate_candidates(
        [
            Candidate("Final answer: 4", 1, route="cot"),
            Candidate("Final answer: 4", 2, route="tir", verifier_passed=True),
            Candidate("Final answer: 5", 3, route="cot"),
        ],
        deduplicate_exact_traces=False,
    )
    assert result.status == "selected"
    assert result.selected_answer == 4
    assert result.vote_table[0].vote_count == 2
    assert result.vote_table[0].verified_vote_count == 1


def test_exact_duplicate_trace_is_not_double_counted() -> None:
    result = aggregate_candidates(
        [
            Candidate("Reasoning A. Final answer: 4", 1),
            Candidate("Reasoning A. Final answer: 4", 2),
            Candidate("Reasoning B. Final answer: 5", 3),
        ]
    )
    assert result.status == "tie"
    assert sum(item.included_in_vote for item in result.candidates) == 2
    assert any(item.exclusion_reason == "duplicate_trace" for item in result.candidates)


def test_invalid_candidates_do_not_vote() -> None:
    result = aggregate_candidates(
        [Candidate("Final answer: 4.5", 1), Candidate("No answer", 2)]
    )
    assert result.status == "no_valid_answer"
    assert result.selected_answer is None


def test_tie_does_not_use_numeric_order_as_hidden_answer() -> None:
    result = aggregate_candidates(
        [Candidate("Final answer: 100", 1), Candidate("Final answer: -2", 2)]
    )
    assert result.status == "tie"
    assert result.selected_answer is None


def test_adaptive_policy_stops_only_with_confidence_and_verification() -> None:
    policy = AdaptiveBudgetPolicy()
    verified = aggregate_candidates(
        [
            Candidate("Final answer: 4", 1, verifier_passed=True),
            Candidate("Answer: 4", 2),
            Candidate("#### 4", 3),
            Candidate("Final answer: 5", 4),
        ]
    )
    assert verified.confidence_share >= 0.75
    assert policy.should_stop(4, verified)
    assert policy.next_budget(4, verified) is None

    unverified = aggregate_candidates(
        [
            Candidate("Final answer: 4", 1),
            Candidate("Answer: 4", 2),
            Candidate("#### 4", 3),
            Candidate("Final answer: 5", 4),
        ]
    )
    assert not policy.should_stop(4, unverified)
    assert policy.next_budget(4, unverified) == 8


def test_policy_validates_budgets() -> None:
    with pytest.raises(ValueError):
        AdaptiveBudgetPolicy(budgets=(4, 1))
    with pytest.raises(ValueError):
        AdaptiveBudgetPolicy(early_stop_share=0)


def test_mathematically_equal_float_weights_remain_a_tie() -> None:
    result = aggregate_candidates(
        [
            Candidate("Final answer: 4", 1, route="a"),
            Candidate("Final answer: 5", 2, route="b"),
        ],
        route_weights={"a": 0.1 + 0.2, "b": 0.15 + 0.15},
    )
    assert result.status == "tie"
    assert result.selected_answer is None


def test_generation_identity_conflict_is_rejected_independent_of_order() -> None:
    first = Candidate("Reasoning one. Final answer: 4", 7, sample_index=0)
    repeated = Candidate("Reasoning two. Final answer: 5", 7, sample_index=0)
    with pytest.raises(ValueError, match="conflicting completion"):
        aggregate_candidates([first, repeated])
    with pytest.raises(ValueError, match="conflicting completion"):
        aggregate_candidates([repeated, first])


def test_generation_identity_normalizes_digest_case_before_deduplication() -> None:
    lower = Candidate(
        "Final answer: 4",
        7,
        sample_index=0,
        prompt_sha256="a" * 64,
        checkpoint_sha256="b" * 64,
        generation_config_sha256="c" * 64,
    )
    upper = Candidate(
        "Final answer: 4",
        7,
        sample_index=0,
        prompt_sha256="A" * 64,
        checkpoint_sha256="B" * 64,
        generation_config_sha256="C" * 64,
    )
    result = aggregate_candidates([lower, upper], require_complete_provenance=True)
    assert result.vote_table[0].vote_count == 1
    assert result.candidates[1].exclusion_reason == "duplicate_generation_identity"


def test_complete_candidate_provenance_can_be_required() -> None:
    digest = "a" * 64
    complete = Candidate(
        "Final answer: 4",
        7,
        sample_index=0,
        prompt_sha256=digest,
        checkpoint_sha256=digest,
        generation_config_sha256=digest,
    )
    assert complete.has_complete_provenance
    assert aggregate_candidates([complete], require_complete_provenance=True).selected_answer == 4
    with pytest.raises(ValueError, match="complete generation provenance"):
        aggregate_candidates(
            [Candidate("Final answer: 4", 8)], require_complete_provenance=True
        )


def test_candidate_rejects_malformed_provenance() -> None:
    with pytest.raises(ValueError, match="sample_index"):
        Candidate("Final answer: 4", 1, sample_index=-1)
    with pytest.raises(ValueError, match="SHA-256"):
        Candidate("Final answer: 4", 1, prompt_sha256="bad")


def test_vote_weights_have_a_safe_upper_bound() -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        aggregate_candidates(
            [Candidate("Final answer: 4", 1)], route_weights={"cot": 1e308}
        )
    with pytest.raises(ValueError, match="below the supported precision"):
        aggregate_candidates(
            [Candidate("Final answer: 4", 1, route="tiny")],
            route_weights={"tiny": 1e-12},
        )


@pytest.mark.parametrize("field", ["greedy", "verifier_passed"])
def test_duplicate_generation_identity_rejects_vote_metadata_conflicts(field: str) -> None:
    common = {"completion": "Final answer: 4", "seed": 9, "sample_index": 0}
    first = Candidate(**common, **{field: False})
    second = Candidate(**common, **{field: True})
    with pytest.raises(ValueError, match="vote metadata"):
        aggregate_candidates([first, second])
    with pytest.raises(ValueError, match="vote metadata"):
        aggregate_candidates([second, first])

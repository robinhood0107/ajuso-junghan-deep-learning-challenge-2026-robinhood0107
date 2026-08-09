"""Model-agnostic candidate parsing, voting, and deterministic seed policy."""

from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction

from .answers import AnswerParseResult, parse_answer


@dataclass(frozen=True, slots=True)
class Candidate:
    """One raw model completion and its generation provenance."""

    completion: str
    seed: int
    route: str = "cot"
    greedy: bool = False
    verifier_passed: bool | None = None
    sample_index: int | None = None
    prompt_sha256: str | None = None
    checkpoint_sha256: str | None = None
    generation_config_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.completion, str):
            raise TypeError("completion must be a string")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not self.route:
            raise ValueError("route must not be empty")
        if self.sample_index is not None and self.sample_index < 0:
            raise ValueError("sample_index must be non-negative when supplied")
        for field_name in (
            "prompt_sha256",
            "checkpoint_sha256",
            "generation_config_sha256",
        ):
            value = getattr(self, field_name)
            if value is not None and re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
                raise ValueError(f"{field_name} must be a 64-character SHA-256 hex digest")
            if value is not None:
                object.__setattr__(self, field_name, value.lower())

    @property
    def has_complete_provenance(self) -> bool:
        """Return whether every run-defining digest and sample index is present."""

        return self.sample_index is not None and all(
            value is not None
            for value in (
                self.prompt_sha256,
                self.checkpoint_sha256,
                self.generation_config_sha256,
            )
        )

    @property
    def generation_identity(self) -> tuple[object, ...]:
        """Stable identity used to reject accidentally repeated generations."""

        return (
            self.route,
            self.seed,
            self.sample_index,
            self.prompt_sha256,
            self.checkpoint_sha256,
            self.generation_config_sha256,
        )


@dataclass(frozen=True, slots=True)
class EvaluatedCandidate:
    """Candidate paired with the deterministic parser result."""

    candidate: Candidate
    parse: AnswerParseResult
    included_in_vote: bool
    exclusion_reason: str | None


@dataclass(frozen=True, slots=True)
class VoteEntry:
    """Aggregated support for one integer answer."""

    answer: int
    weighted_score: float
    vote_count: int
    verified_vote_count: int
    route_count: int
    greedy_support: bool


@dataclass(frozen=True, slots=True)
class AggregationResult:
    """Voting result that preserves ties instead of inventing an answer."""

    status: str
    selected_answer: int | None
    confidence_share: float
    vote_table: tuple[VoteEntry, ...]
    candidates: tuple[EvaluatedCandidate, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class AdaptiveBudgetPolicy:
    """Predeclared candidate-count escalation policy."""

    budgets: tuple[int, ...] = (1, 4, 8, 16, 32)
    early_stop_share: float = 0.75
    require_verified_vote: bool = True

    def __post_init__(self) -> None:
        if not self.budgets or any(value <= 0 for value in self.budgets):
            raise ValueError("budgets must contain positive integers")
        if tuple(sorted(set(self.budgets))) != self.budgets:
            raise ValueError("budgets must be strictly increasing and unique")
        if not 0.0 < self.early_stop_share <= 1.0:
            raise ValueError("early_stop_share must be in (0, 1]")

    def should_stop(self, generated_count: int, result: AggregationResult) -> bool:
        """Return whether evidence meets the predeclared early-stop rule."""

        if generated_count < self.budgets[0] or result.status != "selected":
            return False
        selected = next(
            entry for entry in result.vote_table if entry.answer == result.selected_answer
        )
        verified = selected.verified_vote_count > 0
        return result.confidence_share >= self.early_stop_share and (
            verified or not self.require_verified_vote
        )

    def next_budget(self, generated_count: int, result: AggregationResult) -> int | None:
        """Return the next total candidate target, or ``None`` when done/exhausted."""

        if self.should_stop(generated_count, result):
            return None
        return next((budget for budget in self.budgets if budget > generated_count), None)


def deterministic_seed(problem_id: str, sample_index: int, *, salt: str) -> int:
    """Derive a stable 63-bit seed independent of batch order and Python hash state."""

    if not problem_id:
        raise ValueError("problem_id must not be empty")
    if sample_index < 0:
        raise ValueError("sample_index must be non-negative")
    if not salt:
        raise ValueError("salt must not be empty")
    material = f"seed-v1\0{salt}\0{problem_id}\0{sample_index}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & ((1 << 63) - 1)


def aggregate_candidates(
    candidates: Sequence[Candidate],
    *,
    route_weights: Mapping[str, float] | None = None,
    verifier_bonus: float = 0.25,
    deduplicate_exact_traces: bool = True,
    deduplicate_generation_identity: bool = True,
    require_complete_provenance: bool = False,
) -> AggregationResult:
    """Parse and vote over candidates using transparent, deterministic features.

    Ties on every declared tie-break feature remain unresolved.  The caller must
    generate more candidates, use a predeclared selector, or invoke an explicit
    emergency fallback rather than using numeric ordering as hidden policy.
    """

    if not candidates:
        raise ValueError("candidates must not be empty")
    if not math.isfinite(verifier_bonus) or verifier_bonus < 0:
        raise ValueError("verifier_bonus must be non-negative")
    weights = dict(route_weights or {})
    if any(not math.isfinite(value) or value <= 0 for value in weights.values()):
        raise ValueError("route weights must be finite and positive")
    if require_complete_provenance and any(
        not candidate.has_complete_provenance for candidate in candidates
    ):
        raise ValueError("every candidate must include complete generation provenance")

    exact_weights = {route: _exact_weight(value) for route, value in weights.items()}
    exact_verifier_bonus = _exact_weight(verifier_bonus)

    evidence_by_generation_identity: dict[tuple[object, ...], tuple[str, bool, bool | None]] = {}
    for candidate in candidates:
        trace_hash = hashlib.sha256(candidate.completion.strip().encode()).hexdigest()
        evidence = (trace_hash, candidate.greedy, candidate.verifier_passed)
        previous = evidence_by_generation_identity.setdefault(
            candidate.generation_identity, evidence
        )
        if previous != evidence:
            raise ValueError(
                "conflicting completion or vote metadata share one generation identity; "
                "seed/config provenance is not unique"
            )

    evaluated: list[EvaluatedCandidate] = []
    seen_trace_hashes: set[str] = set()
    seen_generation_identities: set[tuple[object, ...]] = set()
    by_answer: defaultdict[int, list[tuple[Candidate, Fraction]]] = defaultdict(list)
    for candidate in candidates:
        parsed = parse_answer(candidate.completion)
        trace_hash = hashlib.sha256(candidate.completion.strip().encode()).hexdigest()
        duplicate_trace = deduplicate_exact_traces and trace_hash in seen_trace_hashes
        duplicate_generation = (
            deduplicate_generation_identity
            and candidate.generation_identity in seen_generation_identities
        )
        if not duplicate_trace:
            seen_trace_hashes.add(trace_hash)
        if not duplicate_generation:
            seen_generation_identities.add(candidate.generation_identity)
        if not parsed.ok:
            evaluated.append(
                EvaluatedCandidate(candidate, parsed, False, f"parser_{parsed.status}")
            )
            continue
        if duplicate_generation:
            evaluated.append(
                EvaluatedCandidate(
                    candidate, parsed, False, "duplicate_generation_identity"
                )
            )
            continue
        if duplicate_trace:
            evaluated.append(EvaluatedCandidate(candidate, parsed, False, "duplicate_trace"))
            continue
        assert parsed.value is not None
        weight = exact_weights.get(candidate.route, Fraction(1, 1))
        if candidate.verifier_passed is True:
            weight += exact_verifier_bonus
        by_answer[parsed.value].append((candidate, weight))
        evaluated.append(EvaluatedCandidate(candidate, parsed, True, None))

    if not by_answer:
        return AggregationResult(
            status="no_valid_answer",
            selected_answer=None,
            confidence_share=0.0,
            vote_table=(),
            candidates=tuple(evaluated),
            reason="no_candidate_produced_an_unambiguous_integer",
        )

    table: list[VoteEntry] = []
    exact_scores: dict[int, Fraction] = {}
    for answer, supports in by_answer.items():
        source_candidates = [candidate for candidate, _ in supports]
        exact_score = sum((weight for _, weight in supports), start=Fraction(0, 1))
        exact_scores[answer] = exact_score
        table.append(
            VoteEntry(
                answer=answer,
                weighted_score=float(exact_score),
                vote_count=len(supports),
                verified_vote_count=sum(
                    candidate.verifier_passed is True for candidate in source_candidates
                ),
                route_count=len({candidate.route for candidate in source_candidates}),
                greedy_support=any(candidate.greedy for candidate in source_candidates),
            )
        )
    table.sort(
        key=lambda entry: (
            -exact_scores[entry.answer],
            -entry.verified_vote_count,
            -entry.route_count,
            -entry.vote_count,
            -int(entry.greedy_support),
            entry.answer,
        )
    )
    top = table[0]
    top_rank = (
        exact_scores[top.answer],
        top.verified_vote_count,
        top.route_count,
        top.vote_count,
        top.greedy_support,
    )
    tied = [
        entry
        for entry in table
        if (
            exact_scores[entry.answer],
            entry.verified_vote_count,
            entry.route_count,
            entry.vote_count,
            entry.greedy_support,
        )
        == top_rank
    ]
    total_score = sum(exact_scores.values(), start=Fraction(0, 1))
    confidence_share = float(exact_scores[top.answer] / total_score)
    if len(tied) > 1:
        return AggregationResult(
            status="tie",
            selected_answer=None,
            confidence_share=confidence_share,
            vote_table=tuple(table),
            candidates=tuple(evaluated),
            reason="top_answers_tied_on_all_predeclared_features",
        )
    return AggregationResult(
        status="selected",
        selected_answer=top.answer,
        confidence_share=confidence_share,
        vote_table=tuple(table),
        candidates=tuple(evaluated),
        reason="unique_top_answer",
    )


def _exact_weight(value: float) -> Fraction:
    """Map a finite binary float to a stable rational with decimal-scale tolerance."""

    if value > 1_000_000:
        raise ValueError("route weights and verifier bonus must not exceed 1,000,000")
    exact = Fraction(value).limit_denominator(1_000_000_000)
    if value > 0 and exact == 0:
        raise ValueError("positive vote weight is below the supported precision")
    return exact


__all__ = [
    "AdaptiveBudgetPolicy",
    "AggregationResult",
    "Candidate",
    "EvaluatedCandidate",
    "VoteEntry",
    "aggregate_candidates",
    "deterministic_seed",
]

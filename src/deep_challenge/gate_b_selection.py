"""Fail-closed development comparison and locked-holdout selection gates.

Everything in this module is CPU-only.  It validates raw development JSONL
against the immutable split and organizer-train answers before computing paired
statistics.  It never accepts leaderboard/test IDs and never emits locked
holdout IDs, questions, or answers into an artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .answers import parse_answer
from .data import MathRecord
from .evaluation import compare_predictions_paired, holm_bonferroni
from .gate_b import DEFAULT_GATE_B_CONFIG, PINNED_MODEL_REVISION, GateBValidationError
from .gate_b_runtime import (
    BASE_MODEL_CHECKPOINT_SHA256,
    AdapterArtifactEvidence,
    build_fold_sft_plan,
    validate_adapter_artifact,
)
from .inference import deterministic_seed
from .model_preflight import OFFICIAL_MODEL_ID
from .provenance import canonical_json_bytes, sha256_file
from .splits import (
    SplitManifest,
    SplitPartition,
    eligible_training_ids,
    eligible_validation_ids,
    expand_hard_group_exclusions,
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_TRAIN_ID_RE = re.compile(r"train-\d{6}\Z")
_LABEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
_MAX_JSONL_LINE_BYTES = 16 * 1024 * 1024
_COMPARISON_SCHEMA = "gate-b-development-comparison-v2"
_PROBE_DECISION_SCHEMA = "gate-b-candidate-probe-decision-v1"
_OOF_COMPARISON_SCHEMA = "gate-b-development-oof-comparison-v1"
_BASE_OOF_SCHEMA = "gate-b-base-development-oof-v1"
_BASE_OOF_FOLD_COUNT = 5
_FREEZE_SCHEMA = "gate-b-selection-freeze-v3"
_BASE_FREEZE_SCHEMA = "gate-b-base-selection-freeze-v1"
_HOLDOUT_SCHEMA = "gate-b-locked-holdout-access-v1"
HOLDOUT_ACCESS_ACKNOWLEDGEMENT = "OPEN_LOCKED_HOLDOUT_ONCE_AFTER_FREEZE"
PRIMARY_ONLY_ROUTING_POLICY = "primary_only"
INVALID_PRIMARY_FALLBACK_ROUTING_POLICY = (
    "primary_then_fallback_only_on_primary_parse_failure"
)
FIXED_BASE_METHOD_KIND = "fixed_base"
ADAPTER_METHOD_KIND = "adapter"

_BASELINE_RECORD_KEYS = frozenset(
    {
        "schema_version",
        "problem_id",
        "question_sha256",
        "route",
        "sample_index",
        "seed",
        "model_id",
        "revision",
        "prompt_sha256",
        "checkpoint_sha256",
        "config_sha256",
        "decoding_policy",
        "decoding_policy_sha256",
        "raw_completion",
        "raw_completion_sha256",
        "finish_reason",
        "input_token_count",
        "output_token_count",
        "parse",
        "reference_answer",
        "exact_match",
        "latency_ms",
        "peak_vram_allocated_bytes",
        "split_version",
        "split_sha256",
        "source_groups_sha256",
        "fold",
        "partition",
        "split_partition",
        "group_id",
        "eligibility_ids_sha256",
    }
)
_RUN_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "records_file",
        "records_bytes",
        "records_sha256",
        "record_count",
        "problem_count",
        "samples_per_problem",
        "model_id",
        "revision",
        "route",
        "checkpoint_sha256",
        "config_sha256",
        "decoding_policy",
        "decoding_policy_sha256",
        "split_version",
        "split_sha256",
        "source_groups_sha256",
        "fold",
        "partition",
        "split_partition",
        "eligibility_ids_sha256",
        "parser_status_counts",
        "finish_reason_counts",
        "exact_match_count",
        "exact_match_accuracy",
        "input_token_count_total",
        "output_token_count_total",
        "peak_vram_allocated_bytes_max",
        "execution_evidence",
        "generation_evidence",
    }
)
_EXECUTION_EVIDENCE_KEYS = frozenset(
    {"schema_version", "source_manifest", "config_file", "runtime_gate"}
)
_EXECUTION_SOURCE_KEYS = frozenset(
    {"path", "file", "sha256", "tree_sha256", "file_count"}
)
_EXECUTION_CONFIG_KEYS = frozenset({"path", "sha256", "config_sha256"})
_EXECUTION_RUNTIME_KEYS = frozenset(
    {"preflight_report", "gpu_smoke_report", "gpu_device_name"}
)
_EXECUTION_RUNTIME_FILE_KEYS = frozenset({"path", "sha256"})
_GENERATION_EVIDENCE_KEYS = frozenset(
    {
        "schema_version",
        "seed_sequence_sha256",
        "prompt_sha256_sequence_sha256",
        "latency_ms",
    }
)
_LATENCY_EVIDENCE_KEYS = frozenset({"count", "total", "min", "max", "mean"})


@dataclass(frozen=True, slots=True)
class NamedDevelopmentRun:
    """One named development JSONL input."""

    label: str
    records_path: str | Path
    manifest_path: str | Path


@dataclass(frozen=True, slots=True)
class FoldDevelopmentRun:
    """One named development run bound to its validation fold."""

    fold: int
    label: str
    records_path: str | Path
    manifest_path: str | Path
    method_kind: str
    adapter_path: str | Path | None = None


@dataclass(frozen=True, slots=True)
class GateBSelectionWriteResult:
    """Identity of one no-overwrite selection artifact."""

    path: str
    size_bytes: int
    sha256: str
    payload_sha256: str


@dataclass(frozen=True, slots=True)
class LockedHoldoutAccess:
    """Validated in-memory holdout authorization; IDs are never serialized."""

    receipt_path: str
    receipt_sha256: str
    freeze_sha256: str
    split_sha256: str
    eligible_ids: tuple[str, ...]
    primary_label: str
    primary_checkpoint_sha256: str
    fallback_label: str | None
    fallback_checkpoint_sha256: str | None
    routing_policy: str


@dataclass(frozen=True, slots=True)
class FrozenSelectionMethods:
    """Development-frozen method identities safe to inspect before holdout access."""

    freeze_path: str
    freeze_sha256: str
    split_sha256: str
    fold: int
    train_file_sha256: str
    exclusions_file_sha256: str
    excluded_ids_sha256: str
    split_artifact_sha256: str
    development_shard_sha256: str
    primary_label: str
    primary_checkpoint_sha256: str
    fallback_label: str | None
    fallback_checkpoint_sha256: str | None
    routing_policy: str


@dataclass(frozen=True, slots=True)
class _ValidatedRun:
    label: str
    path: Path
    sha256: str
    size_bytes: int
    manifest_path: Path
    manifest_sha256: str
    manifest_size_bytes: int
    checkpoint_sha256: str
    config_sha256: str
    eligibility_ids_sha256: str
    predictions: Mapping[str, str]
    exact_match_count: int


@dataclass(frozen=True, slots=True)
class _ValidatedMethodProvenance:
    method_kind: str
    training_method_fingerprint: str
    adapter_artifact: Mapping[str, str] | None


def compare_development_runs(
    reference: NamedDevelopmentRun,
    candidates: Iterable[NamedDevelopmentRun],
    records: Iterable[MathRecord],
    *,
    split_manifest: SplitManifest,
    fold: int,
    excluded_ids: Iterable[str],
    train_file_sha256: str,
    exclusions_file_sha256: str,
    split_artifact_sha256: str,
    development_shard_sha256: str,
    output_path: str | Path,
    bootstrap_samples: int = 10_000,
    confidence: float = 0.95,
    alpha: float = 0.05,
    seed: int = 20_260_804,
) -> GateBSelectionWriteResult:
    """Validate and compare one reference with one or more candidate runs.

    Exactly one deterministic generation (``sample_index=0``) must exist for
    every group-safe validation ID.  Missing, invalid, or conflicting parser
    results remain wrong predictions; no fallback value is introduced.
    """

    if not isinstance(split_manifest, SplitManifest):
        raise TypeError("split_manifest must be a SplitManifest")
    split_manifest.validate()
    train_digest = _required_sha256(train_file_sha256, "train_file_sha256")
    exclusions_digest = _required_sha256(
        exclusions_file_sha256, "exclusions_file_sha256"
    )
    split_artifact_digest = _required_sha256(
        split_artifact_sha256, "split_artifact_sha256"
    )
    development_shard_digest = _required_sha256(
        development_shard_sha256, "development_shard_sha256"
    )
    if isinstance(bootstrap_samples, bool) or not isinstance(bootstrap_samples, int):
        raise GateBValidationError("bootstrap_samples must be an integer")
    if bootstrap_samples <= 0:
        raise GateBValidationError("bootstrap_samples must be positive")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise GateBValidationError("seed must be a non-negative integer")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise GateBValidationError("confidence must be numeric")
    if not math.isfinite(float(confidence)) or not 0.0 < float(confidence) < 1.0:
        raise GateBValidationError("confidence must be strictly between zero and one")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise GateBValidationError("alpha must be numeric")
    if not math.isfinite(float(alpha)) or not 0.0 < float(alpha) < 1.0:
        raise GateBValidationError("alpha must be strictly between zero and one")

    candidate_runs = tuple(candidates)
    if not candidate_runs:
        raise GateBValidationError("at least one candidate development run is required")
    named_runs = (reference, *candidate_runs)
    labels = tuple(_validated_label(item.label) for item in named_runs)
    if len(set(labels)) != len(labels):
        raise GateBValidationError("development run labels must be unique")

    exclusions = tuple(excluded_ids)
    expected_ids = eligible_validation_ids(split_manifest, fold, exclusions)
    record_by_id = _validated_exact_records(records, expected_ids)
    assignment_by_id = split_manifest.assignment_by_id()
    validated = tuple(
        _load_development_run(
            item,
            expected_ids=expected_ids,
            record_by_id=record_by_id,
            split_manifest=split_manifest,
            assignment_by_id=assignment_by_id,
            fold=fold,
        )
        for item in named_runs
    )

    gold = {problem_id: str(record_by_id[problem_id].answer) for problem_id in expected_ids}
    group_by_id = {
        problem_id: assignment_by_id[problem_id].group_id for problem_id in expected_ids
    }
    reference_run = validated[0]
    comparisons: list[dict[str, Any]] = []
    fallback_routing_comparisons: list[dict[str, Any]] = []
    raw_p_values: dict[str, float] = {}
    for candidate in validated[1:]:
        comparison = compare_predictions_paired(
            gold,
            reference_run.predictions,
            candidate.predictions,
            bootstrap_samples=bootstrap_samples,
            confidence=float(confidence),
            seed=seed,
            group_by_id=group_by_id,
        )
        raw_p_values[candidate.label] = comparison.mcnemar_exact_p_value
        comparisons.append(
            {
                "reference_label": reference_run.label,
                "candidate_label": candidate.label,
                **asdict(comparison),
            }
        )
    for primary_run in validated:
        for fallback_run in validated:
            if primary_run.label == fallback_run.label:
                continue
            primary_invalid_ids = tuple(
                problem_id
                for problem_id in expected_ids
                if problem_id not in primary_run.predictions
            )
            routed_predictions = dict(primary_run.predictions)
            resolved_count = 0
            for problem_id in primary_invalid_ids:
                fallback_value = fallback_run.predictions.get(problem_id)
                if fallback_value is not None:
                    routed_predictions[problem_id] = fallback_value
                    resolved_count += 1
            comparison = compare_predictions_paired(
                gold,
                primary_run.predictions,
                routed_predictions,
                bootstrap_samples=bootstrap_samples,
                confidence=float(confidence),
                seed=seed,
                group_by_id=group_by_id,
            )
            hypothesis = (
                f"fallback_on_invalid:{primary_run.label}->{fallback_run.label}"
            )
            raw_p_values[hypothesis] = comparison.mcnemar_exact_p_value
            fallback_routing_comparisons.append(
                {
                    "primary_label": primary_run.label,
                    "fallback_label": fallback_run.label,
                    "routing_policy": INVALID_PRIMARY_FALLBACK_ROUTING_POLICY,
                    "fallback_invocation_count": len(primary_invalid_ids),
                    "fallback_resolved_count": resolved_count,
                    "unresolved_count": len(primary_invalid_ids) - resolved_count,
                    **asdict(comparison),
                    "hypothesis": hypothesis,
                }
            )
    adjustments = {
        item.hypothesis: asdict(item)
        for item in holm_bonferroni(raw_p_values, alpha=float(alpha))
    }
    for comparison in comparisons:
        comparison["holm"] = adjustments[str(comparison["candidate_label"])]
    for comparison in fallback_routing_comparisons:
        comparison["holm"] = adjustments[str(comparison["hypothesis"])]

    eligibility_digest = _ids_sha256(expected_ids)
    run_evidence = {
        item.label: {
            "path": str(item.path),
            "size_bytes": item.size_bytes,
            "sha256": item.sha256,
            "manifest_path": str(item.manifest_path),
            "manifest_size_bytes": item.manifest_size_bytes,
            "manifest_sha256": item.manifest_sha256,
            "checkpoint_sha256": item.checkpoint_sha256,
            "config_sha256": item.config_sha256,
            "eligibility_ids_sha256": item.eligibility_ids_sha256,
            "problem_count": len(expected_ids),
            "exact_match_count": item.exact_match_count,
            "exact_match_accuracy": item.exact_match_count / len(expected_ids),
        }
        for item in validated
    }
    payload_without_hash: dict[str, Any] = {
        "schema_version": _COMPARISON_SCHEMA,
        "model_id": OFFICIAL_MODEL_ID,
        "revision": PINNED_MODEL_REVISION,
        "route": "direct_answer",
        "split_version": split_manifest.version,
        "split_sha256": split_manifest.sha256,
        "source_groups_sha256": split_manifest.source_groups_sha256,
        "fold": fold,
        "partition": "fold_validation",
        "locked_holdout_accessed": False,
        "leaderboard_or_test_used": False,
        "data_provenance": {
            "train_file_sha256": train_digest,
            "exclusions_file_sha256": exclusions_digest,
            "excluded_ids_sha256": _ids_sha256(tuple(sorted(exclusions))),
            "split_artifact_sha256": split_artifact_digest,
            "development_shard_sha256": development_shard_digest,
        },
        "eligibility_ids_sha256": eligibility_digest,
        "problem_count": len(expected_ids),
        "reference_label": reference_run.label,
        "run_order": list(labels),
        "runs": run_evidence,
        "statistics": {
            "paired": True,
            "bootstrap_unit": "duplicate_cluster",
            "bootstrap_samples": bootstrap_samples,
            "confidence": float(confidence),
            "seed": seed,
            "mcnemar": "exact_two_sided",
            "multiple_comparisons": "holm_bonferroni",
            "family_size": len(raw_p_values),
            "alpha": float(alpha),
        },
        "comparisons": comparisons,
        "fallback_routing_comparisons": fallback_routing_comparisons,
    }
    return _write_hashed_json_noreplace(output_path, payload_without_hash)


def decide_candidate_probe_promotion(
    comparison_artifact: str | Path,
    *,
    candidate_label: str,
    output_path: str | Path,
) -> GateBSelectionWriteResult:
    """Apply the fixed single-fold harm screen before spending more GPU time.

    This decision is intentionally cost-control evidence only.  A candidate is
    stopped when its paired delta is negative, the entire cluster-bootstrap
    interval is below zero, and its Holm-adjusted exact McNemar test rejects at
    the comparison artifact's alpha.  Every other outcome may continue to the
    complete OOF experiment, but can never be frozen from this probe alone.
    """

    source, comparison, comparison_sha = _load_hashed_json_artifact(
        comparison_artifact, expected_schema=_COMPARISON_SCHEMA
    )
    candidate = _validated_label(candidate_label)
    if comparison.get("partition") != "fold_validation":
        raise GateBValidationError("candidate probe must use one fold-validation run")
    if comparison.get("locked_holdout_accessed") is not False:
        raise GateBValidationError("candidate probe must not access the locked holdout")
    if comparison.get("leaderboard_or_test_used") is not False:
        raise GateBValidationError("candidate probe must not use leaderboard/test data")
    if comparison.get("model_id") != OFFICIAL_MODEL_ID:
        raise GateBValidationError("candidate probe does not bind the official model")
    if comparison.get("revision") != PINNED_MODEL_REVISION:
        raise GateBValidationError("candidate probe does not bind the pinned revision")

    reference = _validated_label(
        _required_string(comparison.get("reference_label"), "reference_label")
    )
    comparisons = comparison.get("comparisons")
    if not isinstance(comparisons, list):
        raise GateBValidationError("candidate probe comparisons must be a list")
    matches = [
        item
        for item in comparisons
        if isinstance(item, Mapping) and item.get("candidate_label") == candidate
    ]
    if len(matches) != 1:
        raise GateBValidationError(
            "candidate probe must contain exactly one comparison for the label"
        )
    evidence = matches[0]
    if evidence.get("reference_label") != reference:
        raise GateBValidationError("candidate probe reference label mismatch")

    statistics = comparison.get("statistics")
    if not isinstance(statistics, Mapping):
        raise GateBValidationError("candidate probe lacks statistical provenance")
    if statistics.get("paired") is not True:
        raise GateBValidationError("candidate probe statistics must be paired")
    if statistics.get("bootstrap_unit") != "duplicate_cluster":
        raise GateBValidationError("candidate probe must bootstrap duplicate clusters")
    if statistics.get("mcnemar") != "exact_two_sided":
        raise GateBValidationError("candidate probe must use exact two-sided McNemar")
    if statistics.get("multiple_comparisons") != "holm_bonferroni":
        raise GateBValidationError("candidate probe must use Holm correction")
    alpha = _required_probability(
        statistics.get("alpha"), "statistics alpha", open_interval=True
    )
    confidence = _required_probability(
        statistics.get("confidence"), "statistics confidence", open_interval=True
    )
    if not math.isclose(confidence, 1.0 - alpha, rel_tol=0.0, abs_tol=1e-15):
        raise GateBValidationError(
            "candidate probe confidence must equal one minus alpha"
        )
    bootstrap_samples = statistics.get("bootstrap_samples")
    if type(bootstrap_samples) is not int or bootstrap_samples <= 0:
        raise GateBValidationError(
            "candidate probe bootstrap_samples must be a positive integer"
        )
    family_size = statistics.get("family_size")
    if type(family_size) is not int or family_size <= 0:
        raise GateBValidationError(
            "candidate probe family_size must be a positive integer"
        )
    accuracy_reference = _required_probability(
        evidence.get("accuracy_a"),
        "reference accuracy",
        open_interval=False,
    )
    accuracy_candidate = _required_probability(
        evidence.get("accuracy_b"),
        "candidate accuracy",
        open_interval=False,
    )
    delta = _required_finite_number(
        evidence.get("delta_b_minus_a"), "candidate delta_b_minus_a"
    )
    if not math.isclose(
        delta,
        accuracy_candidate - accuracy_reference,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise GateBValidationError("candidate delta disagrees with run accuracies")
    ci_low = _required_finite_number(
        evidence.get("bootstrap_delta_ci_low"), "candidate bootstrap CI low"
    )
    ci_high = _required_finite_number(
        evidence.get("bootstrap_delta_ci_high"), "candidate bootstrap CI high"
    )
    if ci_low > ci_high:
        raise GateBValidationError("candidate bootstrap CI bounds are reversed")
    if ci_low < -1.0 or ci_high > 1.0:
        raise GateBValidationError("candidate bootstrap CI must remain within [-1, 1]")
    if evidence.get("bootstrap_unit") != "duplicate_cluster":
        raise GateBValidationError(
            "candidate evidence must use duplicate-cluster bootstrap"
        )
    if evidence.get("bootstrap_samples") != bootstrap_samples:
        raise GateBValidationError(
            "candidate bootstrap_samples disagree with statistical provenance"
        )
    bootstrap_group_count = evidence.get("bootstrap_group_count")
    if type(bootstrap_group_count) is not int or bootstrap_group_count <= 0:
        raise GateBValidationError(
            "candidate bootstrap_group_count must be a positive integer"
        )
    problem_count = comparison.get("problem_count")
    if type(problem_count) is not int or problem_count <= 0:
        raise GateBValidationError("candidate probe problem_count must be positive")
    contingency_fields = ("both_correct", "both_wrong", "only_a_correct", "only_b_correct")
    contingency: dict[str, int] = {}
    for field_name in contingency_fields:
        value = evidence.get(field_name)
        if type(value) is not int or value < 0:
            raise GateBValidationError(
                f"candidate contingency {field_name} must be non-negative"
            )
        contingency[field_name] = value
    evidence_total = evidence.get("total")
    if (
        sum(contingency.values()) != problem_count
        or type(evidence_total) is not int
        or evidence_total != problem_count
    ):
        raise GateBValidationError("candidate contingency does not match problem_count")
    contingency_accuracy_reference = (
        contingency["both_correct"] + contingency["only_a_correct"]
    ) / problem_count
    contingency_accuracy_candidate = (
        contingency["both_correct"] + contingency["only_b_correct"]
    ) / problem_count
    contingency_delta = (
        contingency["only_b_correct"] - contingency["only_a_correct"]
    ) / problem_count
    if not math.isclose(
        accuracy_reference,
        contingency_accuracy_reference,
        rel_tol=0.0,
        abs_tol=1e-15,
    ) or not math.isclose(
        accuracy_candidate,
        contingency_accuracy_candidate,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise GateBValidationError(
            "candidate accuracies disagree with the paired contingency"
        )
    if not math.isclose(
        delta,
        contingency_delta,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise GateBValidationError(
            "candidate delta disagrees with the paired contingency"
        )
    mcnemar_p = _required_probability(
        evidence.get("mcnemar_exact_p_value"),
        "candidate exact McNemar p-value",
        open_interval=False,
    )
    holm_family = _validated_probe_holm_family(
        comparison,
        comparisons=comparisons,
        alpha=alpha,
        family_size=family_size,
    )
    holm = holm_family[candidate]
    raw_p = holm.raw_p_value
    adjusted_p = holm.adjusted_p_value
    reject = holm.reject
    if raw_p != mcnemar_p:
        raise GateBValidationError("candidate Holm raw p-value disagrees with McNemar")

    significant_regression = delta < 0.0 and ci_high < 0.0 and reject
    action = (
        "stop_before_remaining_folds"
        if significant_regression
        else "continue_to_complete_oof"
    )
    payload_without_hash: dict[str, Any] = {
        "schema_version": _PROBE_DECISION_SCHEMA,
        "decision_scope": "single_fold_gpu_cost_control_only",
        "policy": {
            "name": "single_fold_significant_harm_screen_v1",
            "stop_if": (
                "delta_b_minus_a<0 and bootstrap_delta_ci_high<0 and "
                "holm_reject_at_comparison_alpha"
            ),
            "alpha": alpha,
        },
        "comparison_artifact": {
            "path": str(source),
            "size_bytes": source.stat().st_size,
            "sha256": comparison_sha,
            "payload_sha256": comparison["payload_sha256"],
        },
        "model_id": OFFICIAL_MODEL_ID,
        "revision": PINNED_MODEL_REVISION,
        "split_version": comparison.get("split_version"),
        "split_sha256": comparison.get("split_sha256"),
        "source_groups_sha256": comparison.get("source_groups_sha256"),
        "fold": comparison.get("fold"),
        "reference_label": reference,
        "candidate_label": candidate,
        "evidence": {
            "problem_count": problem_count,
            "accuracy_reference": accuracy_reference,
            "accuracy_candidate": accuracy_candidate,
            "delta_candidate_minus_reference": delta,
            "bootstrap_delta_ci_low": ci_low,
            "bootstrap_delta_ci_high": ci_high,
            "mcnemar_exact_p_value": mcnemar_p,
            "holm_adjusted_p_value": adjusted_p,
            "holm_reject": reject,
        },
        "significant_regression": significant_regression,
        "candidate_action": action,
        "candidate_full_oof_authorized": not significant_regression,
        "final_selection_eligible": False,
        "complete_oof_required_before_freeze": True,
        "selection_frozen": False,
        "locked_holdout_accessed": False,
        "leaderboard_or_test_used": False,
    }
    return _write_hashed_json_noreplace(output_path, payload_without_hash)


def compare_cross_fold_development_runs(
    reference_label: str,
    candidate_labels: Iterable[str],
    fold_runs: Iterable[FoldDevelopmentRun],
    records: Iterable[MathRecord],
    *,
    split_manifest: SplitManifest,
    deployment_fold: int,
    excluded_ids: Iterable[str],
    train_file_sha256: str,
    exclusions_file_sha256: str,
    split_artifact_sha256: str,
    development_shard_sha256: str,
    output_path: str | Path,
    bootstrap_samples: int = 10_000,
    confidence: float = 0.95,
    alpha: float = 0.05,
    seed: int = 20_260_804,
) -> GateBSelectionWriteResult:
    """Pool exact out-of-fold predictions before confirmatory selection.

    Every method label must provide exactly one complete validation run for
    every fold.  Statistics are recomputed over the union of out-of-fold rows,
    with the immutable duplicate-cluster mapping supplied to the paired
    bootstrap.  The deployment checkpoint is taken only from the predeclared
    ``deployment_fold``; fold-specific adapter checkpoints are never ensembled.
    """

    if not isinstance(split_manifest, SplitManifest):
        raise TypeError("split_manifest must be a SplitManifest")
    split_manifest.validate()
    if (
        isinstance(deployment_fold, bool)
        or not isinstance(deployment_fold, int)
        or deployment_fold < 0
        or deployment_fold >= split_manifest.n_folds
    ):
        raise GateBValidationError("deployment_fold is outside the split fold range")
    train_digest = _required_sha256(train_file_sha256, "train_file_sha256")
    exclusions_digest = _required_sha256(
        exclusions_file_sha256, "exclusions_file_sha256"
    )
    split_artifact_digest = _required_sha256(
        split_artifact_sha256, "split_artifact_sha256"
    )
    development_shard_digest = _required_sha256(
        development_shard_sha256, "development_shard_sha256"
    )
    if isinstance(bootstrap_samples, bool) or not isinstance(bootstrap_samples, int):
        raise GateBValidationError("bootstrap_samples must be an integer")
    if bootstrap_samples <= 0:
        raise GateBValidationError("bootstrap_samples must be positive")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise GateBValidationError("seed must be a non-negative integer")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise GateBValidationError("confidence must be numeric")
    if not math.isfinite(float(confidence)) or not 0.0 < float(confidence) < 1.0:
        raise GateBValidationError("confidence must be strictly between zero and one")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise GateBValidationError("alpha must be numeric")
    if not math.isfinite(float(alpha)) or not 0.0 < float(alpha) < 1.0:
        raise GateBValidationError("alpha must be strictly between zero and one")

    reference = _validated_label(reference_label)
    candidates = tuple(_validated_label(label) for label in candidate_labels)
    if not candidates:
        raise GateBValidationError("at least one candidate label is required")
    labels = (reference, *candidates)
    if len(set(labels)) != len(labels):
        raise GateBValidationError("cross-fold development labels must be unique")

    supplied_runs = tuple(fold_runs)
    indexed: dict[tuple[int, str], FoldDevelopmentRun] = {}
    for item in supplied_runs:
        if not isinstance(item, FoldDevelopmentRun):
            raise TypeError("fold_runs must contain FoldDevelopmentRun values")
        if (
            isinstance(item.fold, bool)
            or not isinstance(item.fold, int)
            or item.fold < 0
            or item.fold >= split_manifest.n_folds
        ):
            raise GateBValidationError("cross-fold run has an invalid fold")
        label = _validated_label(item.label)
        if label not in labels:
            raise GateBValidationError(
                f"cross-fold run label is not declared: {label!r}"
            )
        expected_kind = FIXED_BASE_METHOD_KIND if label == reference else ADAPTER_METHOD_KIND
        if item.method_kind != expected_kind:
            raise GateBValidationError(
                f"cross-fold run {label!r} must use method_kind={expected_kind!r}"
            )
        if expected_kind == FIXED_BASE_METHOD_KIND and item.adapter_path is not None:
            raise GateBValidationError("fixed-base OOF runs must not reference an adapter")
        if expected_kind == ADAPTER_METHOD_KIND and item.adapter_path is None:
            raise GateBValidationError("adapter OOF runs require an adapter artifact path")
        key = (item.fold, label)
        if key in indexed:
            raise GateBValidationError(f"duplicate cross-fold run: {key!r}")
        indexed[key] = item
    expected_keys = {
        (fold, label)
        for fold in range(split_manifest.n_folds)
        for label in labels
    }
    missing_keys = sorted(expected_keys - set(indexed))
    extra_keys = sorted(set(indexed) - expected_keys)
    if missing_keys or extra_keys:
        raise GateBValidationError(
            "cross-fold runs must cover every label on every fold; "
            f"missing={missing_keys[:5]!r}, extra={extra_keys[:5]!r}"
        )

    exclusions = tuple(excluded_ids)
    if len(exclusions) != len(set(exclusions)):
        raise GateBValidationError("excluded_ids must not contain duplicates")
    expected_by_fold = {
        fold: eligible_validation_ids(split_manifest, fold, exclusions)
        for fold in range(split_manifest.n_folds)
    }
    expected_oof_ids = tuple(
        sorted(problem_id for ids in expected_by_fold.values() for problem_id in ids)
    )
    if len(expected_oof_ids) != len(set(expected_oof_ids)):
        raise GateBValidationError("cross-fold validation IDs overlap")
    record_by_id = _validated_exact_records(records, expected_oof_ids)
    assignment_by_id = split_manifest.assignment_by_id()
    validated: dict[tuple[int, str], _ValidatedRun] = {}
    validated_methods: dict[tuple[int, str], _ValidatedMethodProvenance] = {}
    run_identity_owner: dict[tuple[str, str], tuple[int, str]] = {}
    adapter_identity_owner: dict[str, tuple[int, str]] = {}
    fingerprints_by_label: dict[str, set[str]] = {label: set() for label in labels}
    for fold in range(split_manifest.n_folds):
        fold_records = {
            problem_id: record_by_id[problem_id]
            for problem_id in expected_by_fold[fold]
        }
        for label in labels:
            item = indexed[(fold, label)]
            key = (fold, label)
            run = _load_development_run(
                NamedDevelopmentRun(label, item.records_path, item.manifest_path),
                expected_ids=expected_by_fold[fold],
                record_by_id=fold_records,
                split_manifest=split_manifest,
                assignment_by_id=assignment_by_id,
                fold=fold,
            )
            identity = (run.sha256, run.manifest_sha256)
            previous_run = run_identity_owner.setdefault(identity, key)
            if previous_run != key:
                raise GateBValidationError(
                    "OOF run evidence is reused across fold/label entries: "
                    f"{previous_run!r} and {key!r}"
                )
            method = _validate_oof_method_provenance(
                item,
                run=run,
                record_by_id=record_by_id,
                split_manifest=split_manifest,
                fold=fold,
                excluded_ids=exclusions,
                train_file_sha256=train_digest,
                exclusions_file_sha256=exclusions_digest,
                split_artifact_sha256=split_artifact_digest,
                development_shard_sha256=development_shard_digest,
            )
            if method.adapter_artifact is not None:
                artifact_sha = method.adapter_artifact["artifact_sha256"]
                previous_adapter = adapter_identity_owner.setdefault(artifact_sha, key)
                if previous_adapter != key:
                    raise GateBValidationError(
                        "adapter artifact is reused across OOF fold/label entries: "
                        f"{previous_adapter!r} and {key!r}"
                    )
            validated[key] = run
            validated_methods[key] = method
            fingerprints_by_label[label].add(method.training_method_fingerprint)

    inconsistent_fingerprints = sorted(
        label for label, values in fingerprints_by_label.items() if len(values) != 1
    )
    if inconsistent_fingerprints:
        raise GateBValidationError(
            "OOF folds mix different training-method fingerprints for labels: "
            f"{inconsistent_fingerprints!r}"
        )

    pooled_predictions: dict[str, dict[str, str]] = {label: {} for label in labels}
    for label in labels:
        for fold in range(split_manifest.n_folds):
            fold_predictions = validated[(fold, label)].predictions
            overlap = set(pooled_predictions[label]) & set(fold_predictions)
            if overlap:  # pragma: no cover - split validation already prevents this
                raise GateBValidationError("cross-fold prediction IDs overlap")
            pooled_predictions[label].update(fold_predictions)

    gold = {
        problem_id: str(record_by_id[problem_id].answer)
        for problem_id in expected_oof_ids
    }
    group_by_id = {
        problem_id: assignment_by_id[problem_id].group_id
        for problem_id in expected_oof_ids
    }
    comparisons: list[dict[str, Any]] = []
    fallback_routing_comparisons: list[dict[str, Any]] = []
    raw_p_values: dict[str, float] = {}
    for candidate in candidates:
        comparison = compare_predictions_paired(
            gold,
            pooled_predictions[reference],
            pooled_predictions[candidate],
            bootstrap_samples=bootstrap_samples,
            confidence=float(confidence),
            seed=seed,
            group_by_id=group_by_id,
        )
        raw_p_values[candidate] = comparison.mcnemar_exact_p_value
        comparisons.append(
            {
                "reference_label": reference,
                "candidate_label": candidate,
                **asdict(comparison),
            }
        )
    for primary in labels:
        for fallback in labels:
            if primary == fallback:
                continue
            primary_predictions = pooled_predictions[primary]
            fallback_predictions = pooled_predictions[fallback]
            primary_invalid_ids = tuple(
                problem_id
                for problem_id in expected_oof_ids
                if problem_id not in primary_predictions
            )
            routed_predictions = dict(primary_predictions)
            resolved_count = 0
            for problem_id in primary_invalid_ids:
                fallback_value = fallback_predictions.get(problem_id)
                if fallback_value is not None:
                    routed_predictions[problem_id] = fallback_value
                    resolved_count += 1
            comparison = compare_predictions_paired(
                gold,
                primary_predictions,
                routed_predictions,
                bootstrap_samples=bootstrap_samples,
                confidence=float(confidence),
                seed=seed,
                group_by_id=group_by_id,
            )
            hypothesis = f"fallback_on_invalid:{primary}->{fallback}"
            raw_p_values[hypothesis] = comparison.mcnemar_exact_p_value
            fallback_routing_comparisons.append(
                {
                    "primary_label": primary,
                    "fallback_label": fallback,
                    "routing_policy": INVALID_PRIMARY_FALLBACK_ROUTING_POLICY,
                    "fallback_invocation_count": len(primary_invalid_ids),
                    "fallback_resolved_count": resolved_count,
                    "unresolved_count": len(primary_invalid_ids) - resolved_count,
                    **asdict(comparison),
                    "hypothesis": hypothesis,
                }
            )
    adjustments = {
        item.hypothesis: asdict(item)
        for item in holm_bonferroni(raw_p_values, alpha=float(alpha))
    }
    for comparison in comparisons:
        comparison["holm"] = adjustments[str(comparison["candidate_label"])]
    for comparison in fallback_routing_comparisons:
        comparison["holm"] = adjustments[str(comparison["hypothesis"])]

    run_evidence: dict[str, dict[str, Any]] = {}
    for label in labels:
        fold_evidence: dict[str, dict[str, Any]] = {}
        for fold in range(split_manifest.n_folds):
            item = validated[(fold, label)]
            method = validated_methods[(fold, label)]
            fold_evidence[str(fold)] = {
                **_run_evidence(item, len(expected_by_fold[fold])),
                "method_kind": method.method_kind,
                "training_method_fingerprint": method.training_method_fingerprint,
                "adapter_artifact": method.adapter_artifact,
            }
        deployment = dict(fold_evidence[str(deployment_fold)])
        pooled_exact_count = sum(
            validated[(fold, label)].exact_match_count
            for fold in range(split_manifest.n_folds)
        )
        run_evidence[label] = {
            **deployment,
            "deployment_fold": deployment_fold,
            "fold_runs": fold_evidence,
            "oof_problem_count": len(expected_oof_ids),
            "oof_exact_match_count": pooled_exact_count,
            "oof_exact_match_accuracy": pooled_exact_count / len(expected_oof_ids),
        }

    payload_without_hash: dict[str, Any] = {
        "schema_version": _OOF_COMPARISON_SCHEMA,
        "model_id": OFFICIAL_MODEL_ID,
        "revision": PINNED_MODEL_REVISION,
        "route": "direct_answer",
        "split_version": split_manifest.version,
        "split_sha256": split_manifest.sha256,
        "source_groups_sha256": split_manifest.source_groups_sha256,
        "n_folds": split_manifest.n_folds,
        "folds": list(range(split_manifest.n_folds)),
        "fold": deployment_fold,
        "deployment_fold": deployment_fold,
        "partition": "out_of_fold_cross_validation",
        "deployment_partition": "fold_validation",
        "locked_holdout_accessed": False,
        "leaderboard_or_test_used": False,
        "data_provenance": {
            "train_file_sha256": train_digest,
            "exclusions_file_sha256": exclusions_digest,
            "excluded_ids_sha256": _ids_sha256(tuple(sorted(exclusions))),
            "split_artifact_sha256": split_artifact_digest,
            "development_shard_sha256": development_shard_digest,
        },
        "eligibility_ids_sha256": _ids_sha256(expected_oof_ids),
        "deployment_eligibility_ids_sha256": _ids_sha256(
            expected_by_fold[deployment_fold]
        ),
        "problem_count": len(expected_oof_ids),
        "reference_label": reference,
        "run_order": list(labels),
        "runs": run_evidence,
        "statistics": {
            "paired": True,
            "evidence_scope": "complete_out_of_fold_union",
            "bootstrap_unit": "duplicate_cluster",
            "bootstrap_samples": bootstrap_samples,
            "confidence": float(confidence),
            "seed": seed,
            "mcnemar": "exact_two_sided",
            "multiple_comparisons": "holm_bonferroni",
            "family_size": len(raw_p_values),
            "alpha": float(alpha),
        },
        "comparisons": comparisons,
        "fallback_routing_comparisons": fallback_routing_comparisons,
    }
    return _write_hashed_json_noreplace(output_path, payload_without_hash)


def verify_base_development_oof(
    base_label: str,
    fold_runs: Iterable[FoldDevelopmentRun],
    records: Iterable[MathRecord],
    *,
    split_manifest: SplitManifest,
    deployment_fold: int,
    excluded_ids: Iterable[str],
    train_file_sha256: str,
    exclusions_file_sha256: str,
    split_artifact_sha256: str,
    development_shard_sha256: str,
    output_path: str | Path,
) -> GateBSelectionWriteResult:
    """Qualify one pinned fixed-base run on every development fold.

    This is deliberately separate from the candidate comparison schema.  It
    proves that the fallback path has complete, non-reused OOF evidence without
    inventing a synthetic candidate, paired statistic, or adapter claim.
    """

    if not isinstance(split_manifest, SplitManifest):
        raise TypeError("split_manifest must be a SplitManifest")
    split_manifest.validate()
    if split_manifest.n_folds != _BASE_OOF_FOLD_COUNT:
        raise GateBValidationError("base OOF requires the locked five-fold split")
    if (
        isinstance(deployment_fold, bool)
        or not isinstance(deployment_fold, int)
        or deployment_fold < 0
        or deployment_fold >= split_manifest.n_folds
    ):
        raise GateBValidationError("deployment_fold is outside the split fold range")
    train_digest = _required_sha256(train_file_sha256, "train_file_sha256")
    exclusions_digest = _required_sha256(
        exclusions_file_sha256, "exclusions_file_sha256"
    )
    split_artifact_digest = _required_sha256(
        split_artifact_sha256, "split_artifact_sha256"
    )
    development_shard_digest = _required_sha256(
        development_shard_sha256, "development_shard_sha256"
    )
    label = _validated_label(base_label)
    supplied_runs = tuple(fold_runs)
    if len(supplied_runs) != split_manifest.n_folds:
        raise GateBValidationError(
            "base OOF requires exactly one fixed-base run for every fold"
        )
    indexed: dict[int, FoldDevelopmentRun] = {}
    for item in supplied_runs:
        if not isinstance(item, FoldDevelopmentRun):
            raise TypeError("fold_runs must contain FoldDevelopmentRun values")
        if (
            isinstance(item.fold, bool)
            or not isinstance(item.fold, int)
            or item.fold < 0
            or item.fold >= split_manifest.n_folds
        ):
            raise GateBValidationError("base OOF run has an invalid fold")
        if item.label != label:
            raise GateBValidationError("base OOF run label does not match base_label")
        if item.method_kind != FIXED_BASE_METHOD_KIND or item.adapter_path is not None:
            raise GateBValidationError("base OOF accepts fixed-base runs only")
        if item.fold in indexed:
            raise GateBValidationError(f"duplicate base OOF fold: {item.fold}")
        indexed[item.fold] = item
    if set(indexed) != set(range(split_manifest.n_folds)):
        raise GateBValidationError("base OOF folds are incomplete")

    exclusions = tuple(excluded_ids)
    if len(exclusions) != len(set(exclusions)):
        raise GateBValidationError("excluded_ids must not contain duplicates")
    expected_by_fold = {
        fold: eligible_validation_ids(split_manifest, fold, exclusions)
        for fold in range(split_manifest.n_folds)
    }
    expected_oof_ids = tuple(
        sorted(problem_id for ids in expected_by_fold.values() for problem_id in ids)
    )
    if len(expected_oof_ids) != len(set(expected_oof_ids)):
        raise GateBValidationError("base OOF validation IDs overlap")
    record_by_id = _validated_exact_records(records, expected_oof_ids)
    assignment_by_id = split_manifest.assignment_by_id()
    validated: dict[int, _ValidatedRun] = {}
    run_identities: set[tuple[str, str]] = set()
    for fold in range(split_manifest.n_folds):
        item = indexed[fold]
        fold_records = {
            problem_id: record_by_id[problem_id]
            for problem_id in expected_by_fold[fold]
        }
        run = _load_development_run(
            NamedDevelopmentRun(label, item.records_path, item.manifest_path),
            expected_ids=expected_by_fold[fold],
            record_by_id=fold_records,
            split_manifest=split_manifest,
            assignment_by_id=assignment_by_id,
            fold=fold,
        )
        identity = (run.sha256, run.manifest_sha256)
        if identity in run_identities:
            raise GateBValidationError("base OOF reuses run evidence across folds")
        run_identities.add(identity)
        method = _validate_oof_method_provenance(
            item,
            run=run,
            record_by_id=record_by_id,
            split_manifest=split_manifest,
            fold=fold,
            excluded_ids=exclusions,
            train_file_sha256=train_digest,
            exclusions_file_sha256=exclusions_digest,
            split_artifact_sha256=split_artifact_digest,
            development_shard_sha256=development_shard_digest,
        )
        if (
            method.method_kind != FIXED_BASE_METHOD_KIND
            or method.adapter_artifact is not None
            or method.training_method_fingerprint != _base_method_fingerprint()
        ):
            raise GateBValidationError("base OOF method provenance is invalid")
        validated[fold] = run

    fold_evidence = {
        str(fold): {
            **_run_evidence(validated[fold], len(expected_by_fold[fold])),
            "method_kind": FIXED_BASE_METHOD_KIND,
            "training_method_fingerprint": _base_method_fingerprint(),
            "adapter_artifact": None,
        }
        for fold in range(split_manifest.n_folds)
    }
    deployment = dict(fold_evidence[str(deployment_fold)])
    pooled_exact_count = sum(item.exact_match_count for item in validated.values())
    run_evidence = {
        **deployment,
        "deployment_fold": deployment_fold,
        "fold_runs": fold_evidence,
        "oof_problem_count": len(expected_oof_ids),
        "oof_exact_match_count": pooled_exact_count,
        "oof_exact_match_accuracy": pooled_exact_count / len(expected_oof_ids),
    }
    payload_without_hash: dict[str, Any] = {
        "schema_version": _BASE_OOF_SCHEMA,
        "model_id": OFFICIAL_MODEL_ID,
        "revision": PINNED_MODEL_REVISION,
        "route": "direct_answer",
        "split_version": split_manifest.version,
        "split_sha256": split_manifest.sha256,
        "source_groups_sha256": split_manifest.source_groups_sha256,
        "n_folds": split_manifest.n_folds,
        "folds": list(range(split_manifest.n_folds)),
        "fold": deployment_fold,
        "deployment_fold": deployment_fold,
        "partition": "out_of_fold_cross_validation",
        "deployment_partition": "fold_validation",
        "locked_holdout_accessed": False,
        "leaderboard_or_test_used": False,
        "data_provenance": {
            "train_file_sha256": train_digest,
            "exclusions_file_sha256": exclusions_digest,
            "excluded_ids_sha256": _ids_sha256(tuple(sorted(exclusions))),
            "split_artifact_sha256": split_artifact_digest,
            "development_shard_sha256": development_shard_digest,
        },
        "eligibility_ids_sha256": _ids_sha256(expected_oof_ids),
        "deployment_eligibility_ids_sha256": _ids_sha256(
            expected_by_fold[deployment_fold]
        ),
        "problem_count": len(expected_oof_ids),
        "base_label": label,
        "method_kind": FIXED_BASE_METHOD_KIND,
        "run_order": [label],
        "runs": {label: run_evidence},
        "qualification": {
            "evidence_scope": "complete_out_of_fold_union",
            "fixed_base_only": True,
            "exact_fold_coverage": True,
            "validation_ids_unique": True,
            "run_evidence_unique": True,
            "checkpoint_sha256": BASE_MODEL_CHECKPOINT_SHA256,
            "training_method_fingerprint": _base_method_fingerprint(),
        },
    }
    return _write_hashed_json_noreplace(output_path, payload_without_hash)


def _run_evidence(item: _ValidatedRun, problem_count: int) -> dict[str, Any]:
    return {
        "path": str(item.path),
        "size_bytes": item.size_bytes,
        "sha256": item.sha256,
        "manifest_path": str(item.manifest_path),
        "manifest_size_bytes": item.manifest_size_bytes,
        "manifest_sha256": item.manifest_sha256,
        "checkpoint_sha256": item.checkpoint_sha256,
        "config_sha256": item.config_sha256,
        "eligibility_ids_sha256": item.eligibility_ids_sha256,
        "problem_count": problem_count,
        "exact_match_count": item.exact_match_count,
        "exact_match_accuracy": item.exact_match_count / problem_count,
    }


def _validate_oof_method_provenance(
    item: FoldDevelopmentRun,
    *,
    run: _ValidatedRun,
    record_by_id: Mapping[str, MathRecord],
    split_manifest: SplitManifest,
    fold: int,
    excluded_ids: Sequence[str],
    train_file_sha256: str,
    exclusions_file_sha256: str,
    split_artifact_sha256: str,
    development_shard_sha256: str,
) -> _ValidatedMethodProvenance:
    if item.method_kind == FIXED_BASE_METHOD_KIND:
        if item.adapter_path is not None:
            raise GateBValidationError("fixed-base OOF run unexpectedly references an adapter")
        if run.checkpoint_sha256 != BASE_MODEL_CHECKPOINT_SHA256:
            raise GateBValidationError(
                "fixed-base OOF run does not use the pinned base checkpoint"
            )
        return _ValidatedMethodProvenance(
            method_kind=FIXED_BASE_METHOD_KIND,
            training_method_fingerprint=_base_method_fingerprint(),
            adapter_artifact=None,
        )

    if item.method_kind != ADAPTER_METHOD_KIND or item.adapter_path is None:
        raise GateBValidationError("candidate OOF run must reference one adapter artifact")
    adapter = validate_adapter_artifact(item.adapter_path, config=DEFAULT_GATE_B_CONFIG)
    training_ids = eligible_training_ids(split_manifest, fold, excluded_ids)
    validation_ids = eligible_validation_ids(split_manifest, fold, excluded_ids)
    training_records = tuple(record_by_id[problem_id] for problem_id in training_ids)
    validation_records = tuple(record_by_id[problem_id] for problem_id in validation_ids)
    plan = build_fold_sft_plan(
        training_records,
        validation_records,
        split_manifest=split_manifest,
        fold=fold,
        excluded_ids=excluded_ids,
        config=DEFAULT_GATE_B_CONFIG,
    )
    expected = {
        "config_sha256": DEFAULT_GATE_B_CONFIG.sha256,
        "split_version": split_manifest.version,
        "split_sha256": split_manifest.sha256,
        "source_groups_sha256": split_manifest.source_groups_sha256,
        "fold": fold,
        "excluded_ids_sha256": _ids_sha256(excluded_ids),
        "training_count": len(plan.training_ids),
        "training_ids_sha256": plan.training_ids_sha256,
        "validation_count": len(plan.validation_ids),
        "validation_ids_sha256": plan.validation_ids_sha256,
        "validation_examples_sha256": plan.validation_examples_sha256,
        "train_file_sha256": train_file_sha256,
        "exclusions_file_sha256": exclusions_file_sha256,
        "split_artifact_sha256": split_artifact_sha256,
        "development_shard_sha256": development_shard_sha256,
    }
    if adapter.training_target_kind == "direct_answer":
        expected["training_examples_sha256"] = plan.training_examples_sha256
    elif adapter.training_target_kind == "verified_concise_rationale":
        rationale_fields = (
            adapter.rationale_candidate_config_sha256,
            adapter.rationale_candidate_config_file_sha256,
            adapter.rationale_corpus_records_sha256,
            adapter.rationale_corpus_manifest_sha256,
            adapter.rationale_corpus_audit_sha256,
        )
        if any(value is None for value in rationale_fields):
            raise GateBValidationError(
                "OOF rationale adapter is missing corpus/config provenance"
            )
    else:
        raise GateBValidationError(
            f"OOF adapter has unsupported training target: {adapter.training_target_kind!r}"
        )
    mismatched = [
        field_name
        for field_name, expected_value in expected.items()
        if getattr(adapter, field_name) != expected_value
    ]
    if mismatched:
        raise GateBValidationError(
            "OOF adapter does not match its exact fold/data/example provenance: "
            f"{mismatched!r}"
        )
    if run.checkpoint_sha256 != adapter.artifact_sha256:
        raise GateBValidationError(
            "OOF adapter run checkpoint does not match the adapter artifact digest"
        )
    artifact = {
        "path": adapter.path,
        "artifact_sha256": adapter.artifact_sha256,
        "manifest_sha256": adapter.manifest_sha256,
        "checksums_sha256": adapter.checksums_sha256,
    }
    if adapter.training_target_kind == "verified_concise_rationale":
        assert adapter.rationale_candidate_config_sha256 is not None
        assert adapter.rationale_candidate_config_file_sha256 is not None
        assert adapter.rationale_corpus_records_sha256 is not None
        assert adapter.rationale_corpus_manifest_sha256 is not None
        assert adapter.rationale_corpus_audit_sha256 is not None
        artifact.update(
            {
                "training_target_kind": adapter.training_target_kind,
                "rationale_candidate_config_sha256": (
                    adapter.rationale_candidate_config_sha256
                ),
                "rationale_candidate_config_file_sha256": (
                    adapter.rationale_candidate_config_file_sha256
                ),
                "rationale_corpus_records_sha256": (
                    adapter.rationale_corpus_records_sha256
                ),
                "rationale_corpus_manifest_sha256": (
                    adapter.rationale_corpus_manifest_sha256
                ),
                "rationale_corpus_audit_sha256": (
                    adapter.rationale_corpus_audit_sha256
                ),
            }
        )
    return _ValidatedMethodProvenance(
        method_kind=ADAPTER_METHOD_KIND,
        training_method_fingerprint=_adapter_method_fingerprint(adapter),
        adapter_artifact=artifact,
    )


def _base_method_fingerprint() -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "schema_version": "gate-b-oof-method-fingerprint-v1",
                "method_kind": FIXED_BASE_METHOD_KIND,
                "model_id": OFFICIAL_MODEL_ID,
                "revision": PINNED_MODEL_REVISION,
                "checkpoint_sha256": BASE_MODEL_CHECKPOINT_SHA256,
                "config_sha256": DEFAULT_GATE_B_CONFIG.sha256,
            }
        )
    ).hexdigest()


def _adapter_method_fingerprint(adapter: AdapterArtifactEvidence) -> str:
    payload: dict[str, object] = {
        "schema_version": "gate-b-oof-method-fingerprint-v1",
        "method_kind": ADAPTER_METHOD_KIND,
        "model_id": OFFICIAL_MODEL_ID,
        "revision": PINNED_MODEL_REVISION,
        "config_sha256": adapter.config_sha256,
        "split_version": adapter.split_version,
        "split_sha256": adapter.split_sha256,
        "source_groups_sha256": adapter.source_groups_sha256,
        "excluded_ids_sha256": adapter.excluded_ids_sha256,
        "train_file_sha256": adapter.train_file_sha256,
        "exclusions_file_sha256": adapter.exclusions_file_sha256,
        "split_artifact_sha256": adapter.split_artifact_sha256,
        "development_shard_sha256": adapter.development_shard_sha256,
        "preflight_sha256": adapter.preflight_sha256,
        "gpu_smoke_sha256": adapter.gpu_smoke_sha256,
        "source_manifest_sha256": adapter.source_manifest_sha256,
        "source_tree_sha256": adapter.source_tree_sha256,
        "source_file_count": adapter.source_file_count,
        "response_only_labels": True,
        "truncation": False,
        "tensor_contract": "qwen2.5-3b-all-linear-lora-504",
    }
    if adapter.training_target_kind == "verified_concise_rationale":
        payload.update(
            {
                "schema_version": "gate-b-oof-method-fingerprint-v2",
                "training_target_kind": adapter.training_target_kind,
                "rationale_candidate_config_sha256": (
                    adapter.rationale_candidate_config_sha256
                ),
                "rationale_candidate_config_file_sha256": (
                    adapter.rationale_candidate_config_file_sha256
                ),
            }
        )
    return hashlib.sha256(
        canonical_json_bytes(payload)
    ).hexdigest()


def require_base_development_artifact(
    manifest_path: str | Path,
    records: Iterable[MathRecord],
    *,
    split_manifest: SplitManifest,
    fold: int,
    excluded_ids: Iterable[str],
    expected_checkpoint_sha256: str,
) -> None:
    """Require a complete fixed-base development run before adapter training.

    The JSONL path is taken only from the adjacent strict run manifest.  Every
    row, parser result, reference answer, seed, prompt, split binding, and
    aggregate is recomputed before the checkpoint identity is accepted.
    """

    source, payload = _load_run_manifest(manifest_path)
    filename = _required_string(payload.get("records_file"), "records_file")
    if Path(filename).name != filename or filename in {".", ".."}:
        raise GateBValidationError("base run records_file must be one safe basename")
    expected_checkpoint = _required_sha256(
        expected_checkpoint_sha256, "expected_checkpoint_sha256"
    )
    expected_ids = eligible_validation_ids(split_manifest, fold, tuple(excluded_ids))
    record_by_id = _validated_exact_records(records, expected_ids)
    validated = _load_development_run(
        NamedDevelopmentRun("required-base", source.parent / filename, source),
        expected_ids=expected_ids,
        record_by_id=record_by_id,
        split_manifest=split_manifest,
        assignment_by_id=split_manifest.assignment_by_id(),
        fold=fold,
    )
    if validated.checkpoint_sha256 != expected_checkpoint:
        raise GateBValidationError(
            "required base development run does not use the pinned base checkpoint"
        )
    if validated.config_sha256 != DEFAULT_GATE_B_CONFIG.sha256:
        raise GateBValidationError(
            "required base development run does not use the locked Gate B config"
        )


def freeze_development_selection(
    comparison_artifact: str | Path,
    *,
    primary_label: str,
    fallback_label: str | None,
    decision_note: str,
    source_manifest_path: str | Path,
    lockfile_path: str | Path,
    output_path: str | Path,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> GateBSelectionWriteResult:
    """Freeze primary/fallback identities from development evidence only."""

    comparison_path, comparison, comparison_file_sha = _load_hashed_json_artifact(
        comparison_artifact,
        expected_schema=_OOF_COMPARISON_SCHEMA,
    )
    comparison_schema = comparison.get("schema_version")
    expected_comparison = {
        "model_id": OFFICIAL_MODEL_ID,
        "revision": PINNED_MODEL_REVISION,
        "route": "direct_answer",
        "locked_holdout_accessed": False,
        "leaderboard_or_test_used": False,
        "partition": "out_of_fold_cross_validation",
    }
    mismatched_comparison = [
        key for key, value in expected_comparison.items() if comparison.get(key) != value
    ]
    if mismatched_comparison:
        raise GateBValidationError(
            "comparison artifact violates the development-only contract: "
            f"{mismatched_comparison!r}"
        )
    _required_sha256(comparison.get("split_sha256"), "comparison split_sha256")
    _required_sha256(
        comparison.get("source_groups_sha256"), "comparison source_groups_sha256"
    )
    fold = comparison.get("fold")
    if isinstance(fold, bool) or not isinstance(fold, int) or fold < 0:
        raise GateBValidationError("comparison fold must be a non-negative integer")
    n_folds = comparison.get("n_folds")
    statistics = comparison.get("statistics")
    if (
        comparison_schema != _OOF_COMPARISON_SCHEMA
        or isinstance(n_folds, bool)
        or not isinstance(n_folds, int)
        or n_folds < 2
        or comparison.get("deployment_fold") != fold
        or comparison.get("deployment_partition") != "fold_validation"
        or comparison.get("folds") != list(range(n_folds))
        or not isinstance(statistics, Mapping)
        or statistics.get("evidence_scope") != "complete_out_of_fold_union"
    ):
        raise GateBValidationError(
            "final freeze requires complete OOF evidence and a bound deployment fold"
        )
    _required_sha256(
        comparison.get("deployment_eligibility_ids_sha256"),
        "comparison deployment_eligibility_ids_sha256",
    )
    primary = _validated_label(primary_label)
    fallback = None if fallback_label is None else _validated_label(fallback_label)
    if fallback == primary:
        raise GateBValidationError("fallback_label must differ from primary_label")
    if not isinstance(decision_note, str) or not decision_note.strip():
        raise GateBValidationError("decision_note must be a non-empty string")
    if decision_note != decision_note.strip():
        raise GateBValidationError("decision_note must be trimmed")
    source_manifest = _regular_file_evidence(source_manifest_path, "source manifest")
    lockfile = _regular_file_evidence(lockfile_path, "environment lockfile")
    runs = comparison.get("runs")
    if not isinstance(runs, Mapping):
        raise GateBValidationError("comparison artifact runs must be an object")
    selected_labels = (primary,) if fallback is None else (primary, fallback)
    missing = [label for label in selected_labels if label not in runs]
    if missing:
        raise GateBValidationError(f"frozen labels are absent from comparison: {missing!r}")
    selected: dict[str, Any] = {}
    for role, label in zip(("primary", "fallback"), selected_labels, strict=False):
        evidence = runs[label]
        if not isinstance(evidence, Mapping):
            raise GateBValidationError(f"comparison run {label!r} is invalid")
        _revalidate_deployment_method_evidence(evidence, comparison, label)
        run_path = Path(_required_string(evidence.get("path"), f"run {label} path"))
        if not run_path.is_file() or run_path.is_symlink():
            raise GateBValidationError(f"frozen run file is missing or unsafe: {run_path}")
        expected_sha = _required_sha256(evidence.get("sha256"), f"run {label} sha256")
        if sha256_file(run_path) != expected_sha:
            raise GateBValidationError(f"frozen run bytes changed after comparison: {label}")
        run_manifest_path = Path(
            _required_string(evidence.get("manifest_path"), f"run {label} manifest path")
        )
        if not run_manifest_path.is_file() or run_manifest_path.is_symlink():
            raise GateBValidationError(
                f"frozen run manifest is missing or unsafe: {run_manifest_path}"
            )
        expected_manifest_sha = _required_sha256(
            evidence.get("manifest_sha256"), f"run {label} manifest sha256"
        )
        if sha256_file(run_manifest_path) != expected_manifest_sha:
            raise GateBValidationError(
                f"frozen run manifest changed after comparison: {label}"
            )
        _, run_manifest = _load_run_manifest(run_manifest_path)
        run_binding = {
            "checkpoint_sha256": evidence.get("checkpoint_sha256"),
            "config_sha256": evidence.get("config_sha256"),
            "eligibility_ids_sha256": evidence.get("eligibility_ids_sha256"),
            "model_id": comparison.get("model_id"),
            "revision": comparison.get("revision"),
            "route": comparison.get("route"),
            "split_version": comparison.get("split_version"),
            "split_sha256": comparison.get("split_sha256"),
            "source_groups_sha256": comparison.get("source_groups_sha256"),
            "fold": comparison.get("fold"),
            "partition": comparison.get("deployment_partition"),
        }
        run_binding["eligibility_ids_sha256"] = comparison.get(
            "deployment_eligibility_ids_sha256"
        )
        mismatched_run_binding = [
            key for key, value in run_binding.items() if run_manifest.get(key) != value
        ]
        if mismatched_run_binding:
            raise GateBValidationError(
                f"comparison evidence changed run manifest binding for {label}: "
                f"{mismatched_run_binding!r}"
            )
        selected[role] = {"label": label, **dict(evidence)}
    if fallback is None:
        routing_policy = PRIMARY_ONLY_ROUTING_POLICY
        routing_evidence = None
    else:
        routing_entries = comparison.get("fallback_routing_comparisons")
        if not isinstance(routing_entries, list):
            raise GateBValidationError(
                "comparison artifact lacks frozen fallback-routing evidence"
            )
        matching_routes = [
            item
            for item in routing_entries
            if isinstance(item, Mapping)
            and item.get("primary_label") == primary
            and item.get("fallback_label") == fallback
            and item.get("routing_policy")
            == INVALID_PRIMARY_FALLBACK_ROUTING_POLICY
        ]
        if len(matching_routes) != 1:
            raise GateBValidationError(
                "comparison artifact must contain exactly one selected fallback route"
            )
        routing_policy = INVALID_PRIMARY_FALLBACK_ROUTING_POLICY
        routing_evidence = dict(matching_routes[0])
    completed = now()
    if not isinstance(completed, datetime) or completed.tzinfo is None:
        raise GateBValidationError("freeze timestamp must be timezone-aware")
    payload_without_hash = {
        "schema_version": _FREEZE_SCHEMA,
        "frozen_at_utc": completed.astimezone(UTC).isoformat(),
        "selection_frozen": True,
        "selection_basis": "development_evidence_only",
        "locked_holdout_accessed": False,
        "leaderboard_used_for_selection": False,
        "decision_note": decision_note,
        "source_manifest": source_manifest,
        "environment_lockfile": lockfile,
        "comparison_artifact": {
            "path": str(comparison_path),
            "size_bytes": comparison_path.stat().st_size,
            "sha256": comparison_file_sha,
            "payload_sha256": comparison["payload_sha256"],
        },
        "comparison_scope": "complete_out_of_fold_union",
        "model_id": comparison.get("model_id"),
        "revision": comparison.get("revision"),
        "split_version": comparison.get("split_version"),
        "split_sha256": comparison.get("split_sha256"),
        "source_groups_sha256": comparison.get("source_groups_sha256"),
        "fold": comparison.get("fold"),
        "development_eligibility_ids_sha256": comparison.get(
            "eligibility_ids_sha256"
        ),
        "deployment_eligibility_ids_sha256": comparison.get(
            "deployment_eligibility_ids_sha256"
        ),
        "data_provenance": comparison.get("data_provenance"),
        "routing_policy": routing_policy,
        "fallback_routing_evidence": routing_evidence,
        "primary": selected["primary"],
        "fallback": selected.get("fallback"),
    }
    return _write_hashed_json_noreplace(output_path, payload_without_hash)


def freeze_base_development_selection(
    base_oof_artifact: str | Path,
    *,
    primary_label: str,
    decision_note: str,
    source_manifest_path: str | Path,
    lockfile_path: str | Path,
    output_path: str | Path,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> GateBSelectionWriteResult:
    """Freeze the single fixed-base method from complete OOF evidence."""

    oof_path, oof, oof_file_sha = _load_hashed_json_artifact(
        base_oof_artifact,
        expected_schema=_BASE_OOF_SCHEMA,
    )
    _require_base_oof_qualification(oof)
    primary = _validated_label(primary_label)
    if primary != oof.get("base_label"):
        raise GateBValidationError("primary_label does not match base OOF evidence")
    if not isinstance(decision_note, str) or not decision_note.strip():
        raise GateBValidationError("decision_note must be a non-empty string")
    if decision_note != decision_note.strip():
        raise GateBValidationError("decision_note must be trimmed")
    source_manifest = _regular_file_evidence(source_manifest_path, "source manifest")
    lockfile = _regular_file_evidence(lockfile_path, "environment lockfile")
    runs = oof.get("runs")
    if not isinstance(runs, Mapping) or set(runs) != {primary}:
        raise GateBValidationError("base OOF artifact must contain exactly one run label")
    evidence = runs.get(primary)
    if not isinstance(evidence, Mapping):
        raise GateBValidationError("base OOF deployment evidence is invalid")
    _revalidate_deployment_method_evidence(evidence, oof, primary)
    run_path = Path(_required_string(evidence.get("path"), "base run path"))
    if run_path.is_symlink() or not run_path.is_file():
        raise GateBValidationError("frozen base run file is missing or unsafe")
    if sha256_file(run_path) != _required_sha256(
        evidence.get("sha256"), "base run sha256"
    ):
        raise GateBValidationError("frozen base run bytes changed after qualification")
    run_manifest_path = Path(
        _required_string(evidence.get("manifest_path"), "base run manifest path")
    )
    if run_manifest_path.is_symlink() or not run_manifest_path.is_file():
        raise GateBValidationError("frozen base run manifest is missing or unsafe")
    if sha256_file(run_manifest_path) != _required_sha256(
        evidence.get("manifest_sha256"), "base run manifest sha256"
    ):
        raise GateBValidationError("frozen base run manifest changed after qualification")
    _, run_manifest = _load_run_manifest(run_manifest_path)
    run_binding = {
        "checkpoint_sha256": evidence.get("checkpoint_sha256"),
        "config_sha256": evidence.get("config_sha256"),
        "eligibility_ids_sha256": oof.get("deployment_eligibility_ids_sha256"),
        "model_id": oof.get("model_id"),
        "revision": oof.get("revision"),
        "route": oof.get("route"),
        "split_version": oof.get("split_version"),
        "split_sha256": oof.get("split_sha256"),
        "source_groups_sha256": oof.get("source_groups_sha256"),
        "fold": oof.get("deployment_fold"),
        "partition": oof.get("deployment_partition"),
    }
    mismatched_run_binding = [
        key for key, value in run_binding.items() if run_manifest.get(key) != value
    ]
    if mismatched_run_binding:
        raise GateBValidationError(
            "base OOF evidence changed deployment run binding: "
            f"{mismatched_run_binding!r}"
        )
    completed = now()
    if not isinstance(completed, datetime) or completed.tzinfo is None:
        raise GateBValidationError("freeze timestamp must be timezone-aware")
    payload_without_hash = {
        "schema_version": _BASE_FREEZE_SCHEMA,
        "frozen_at_utc": completed.astimezone(UTC).isoformat(),
        "selection_frozen": True,
        "selection_basis": "development_evidence_only",
        "selection_kind": "base_only_complete_oof",
        "locked_holdout_accessed": False,
        "leaderboard_used_for_selection": False,
        "decision_note": decision_note,
        "source_manifest": source_manifest,
        "environment_lockfile": lockfile,
        "comparison_artifact": {
            "path": str(oof_path),
            "size_bytes": oof_path.stat().st_size,
            "sha256": oof_file_sha,
            "payload_sha256": oof["payload_sha256"],
        },
        "comparison_scope": "complete_out_of_fold_union",
        "model_id": oof.get("model_id"),
        "revision": oof.get("revision"),
        "split_version": oof.get("split_version"),
        "split_sha256": oof.get("split_sha256"),
        "source_groups_sha256": oof.get("source_groups_sha256"),
        "fold": oof.get("deployment_fold"),
        "development_eligibility_ids_sha256": oof.get("eligibility_ids_sha256"),
        "deployment_eligibility_ids_sha256": oof.get(
            "deployment_eligibility_ids_sha256"
        ),
        "data_provenance": oof.get("data_provenance"),
        "routing_policy": PRIMARY_ONLY_ROUTING_POLICY,
        "fallback_routing_evidence": None,
        "primary": {"label": primary, **dict(evidence)},
        "fallback": None,
    }
    return _write_hashed_json_noreplace(output_path, payload_without_hash)


def authorize_locked_holdout_once(
    freeze_artifact: str | Path,
    *,
    split_manifest: SplitManifest,
    excluded_ids: Iterable[str],
    acknowledgement: str,
    ledger_root: str | Path,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> GateBSelectionWriteResult:
    """Create one split-keyed claim, then its successful access receipt.

    The ledger key is the immutable split SHA rather than a caller-selected
    filename.  A failed attempt after claim publication remains consumed.  No
    holdout ID, question, or answer is serialized.
    """

    if acknowledgement != HOLDOUT_ACCESS_ACKNOWLEDGEMENT:
        raise GateBValidationError(
            "explicit one-time holdout acknowledgement is required: "
            f"{HOLDOUT_ACCESS_ACKNOWLEDGEMENT}"
        )
    if not isinstance(split_manifest, SplitManifest):
        raise TypeError("split_manifest must be a SplitManifest")
    split_manifest.validate()
    freeze_path, freeze, freeze_file_sha = _load_supported_freeze_artifact(
        freeze_artifact
    )
    if freeze.get("selection_frozen") is not True:
        raise GateBValidationError("selection is not frozen")
    if freeze.get("locked_holdout_accessed") is not False:
        raise GateBValidationError("freeze artifact does not represent a sealed holdout")
    if freeze.get("split_sha256") != split_manifest.sha256:
        raise GateBValidationError("freeze and split SHA-256 do not match")
    if freeze.get("source_groups_sha256") != split_manifest.source_groups_sha256:
        raise GateBValidationError("freeze and split group mapping do not match")
    completed = now()
    if not isinstance(completed, datetime) or completed.tzinfo is None:
        raise GateBValidationError("holdout authorization timestamp must be timezone-aware")
    exclusions = tuple(excluded_ids)
    ledger_raw = Path(ledger_root)
    if ledger_raw.is_symlink():
        raise GateBValidationError("holdout ledger root refuses symlinks")
    ledger = ledger_raw.resolve(strict=True)
    if not ledger.is_dir():
        raise GateBValidationError("holdout ledger root must be an existing directory")
    claim_target = ledger / f"{split_manifest.sha256}.claim.json"
    receipt_target = ledger / f"{split_manifest.sha256}.receipt.json"
    if receipt_target.exists() or receipt_target.is_symlink():
        raise GateBValidationError("locked holdout receipt already exists")
    claim_payload = {
        "schema_version": "gate-b-locked-holdout-claim-v1",
        "claimed_at_utc": completed.astimezone(UTC).isoformat(),
        "canonical_key": split_manifest.sha256,
        "freeze_artifact_sha256": freeze_file_sha,
        "split_sha256": split_manifest.sha256,
        "comparison_scope": "complete_out_of_fold_union",
        "status": "consumed_before_holdout_derivation",
    }
    claim = _write_hashed_json_noreplace(claim_target, claim_payload)
    # The claim is intentionally durable before this first holdout-ID derivation.
    # Any exception after this point consumes the one-time gate and requires an
    # explicit postmortem rather than a silent retry.
    expanded_excluded = expand_hard_group_exclusions(split_manifest, exclusions)
    excluded = set(expanded_excluded)
    eligible_ids = tuple(
        problem_id
        for problem_id in split_manifest.final_holdout_ids()
        if problem_id not in excluded
    )
    if not eligible_ids:
        raise GateBValidationError("eligible locked holdout must not be empty")
    payload_without_hash = {
        "schema_version": _HOLDOUT_SCHEMA,
        "authorized_at_utc": completed.astimezone(UTC).isoformat(),
        "authorization": "single_frozen_primary_fallback_holdout_evaluation",
        "selection_frozen": True,
        "model_selection_after_this_point_forbidden": True,
        "leaderboard_or_test_used": False,
        "ids_questions_answers_emitted": False,
        "freeze_artifact": {
            "path": str(freeze_path),
            "size_bytes": freeze_path.stat().st_size,
            "sha256": freeze_file_sha,
            "payload_sha256": freeze["payload_sha256"],
        },
        "claim_artifact": {
            "path": claim.path,
            "size_bytes": claim.size_bytes,
            "sha256": claim.sha256,
            "payload_sha256": claim.payload_sha256,
        },
        "model_id": freeze.get("model_id"),
        "revision": freeze.get("revision"),
        "split_version": freeze.get("split_version"),
        "split_sha256": split_manifest.sha256,
        "source_groups_sha256": split_manifest.source_groups_sha256,
        "comparison_scope": "complete_out_of_fold_union",
        "eligible_holdout_count": len(eligible_ids),
        "eligible_holdout_ids_sha256": _ids_sha256(eligible_ids),
        "primary": freeze.get("primary"),
        "fallback": freeze.get("fallback"),
        "routing_policy": freeze.get("routing_policy"),
    }
    return _write_hashed_json_noreplace(receipt_target, payload_without_hash)


def validate_frozen_selection_methods(
    freeze_artifact: str | Path,
    *,
    split_manifest: SplitManifest,
    train_file_sha256: str | None = None,
    exclusions_file_sha256: str | None = None,
    excluded_ids_sha256: str | None = None,
    split_artifact_sha256: str | None = None,
    development_shard_sha256: str | None = None,
    fold: int | None = None,
) -> FrozenSelectionMethods:
    """Validate frozen method identities without deriving any holdout ID."""

    if not isinstance(split_manifest, SplitManifest):
        raise TypeError("split_manifest must be a SplitManifest")
    split_manifest.validate()
    freeze_path, freeze, freeze_file_sha = _load_supported_freeze_artifact(
        freeze_artifact
    )
    expected = {
        "selection_frozen": True,
        "selection_basis": "development_evidence_only",
        "locked_holdout_accessed": False,
        "leaderboard_used_for_selection": False,
        "model_id": OFFICIAL_MODEL_ID,
        "revision": PINNED_MODEL_REVISION,
        "split_sha256": split_manifest.sha256,
        "source_groups_sha256": split_manifest.source_groups_sha256,
    }
    mismatched = [key for key, value in expected.items() if freeze.get(key) != value]
    if mismatched:
        raise GateBValidationError(
            f"freeze artifact violates the sealed selection contract: {mismatched!r}"
        )
    frozen_fold = freeze.get("fold")
    if isinstance(frozen_fold, bool) or not isinstance(frozen_fold, int) or frozen_fold < 0:
        raise GateBValidationError("freeze fold must be a non-negative integer")
    provenance = freeze.get("data_provenance")
    if not isinstance(provenance, Mapping) or set(provenance) != {
        "train_file_sha256",
        "exclusions_file_sha256",
        "excluded_ids_sha256",
        "split_artifact_sha256",
        "development_shard_sha256",
    }:
        raise GateBValidationError("freeze data_provenance has an unexpected schema")
    frozen_provenance = {
        key: _required_sha256(provenance.get(key), f"freeze {key}")
        for key in (
            "train_file_sha256",
            "exclusions_file_sha256",
            "excluded_ids_sha256",
            "split_artifact_sha256",
            "development_shard_sha256",
        )
    }
    supplied = {
        "train_file_sha256": train_file_sha256,
        "exclusions_file_sha256": exclusions_file_sha256,
        "excluded_ids_sha256": excluded_ids_sha256,
        "split_artifact_sha256": split_artifact_sha256,
        "development_shard_sha256": development_shard_sha256,
    }
    if any(value is not None for value in (*supplied.values(), fold)):
        if any(value is None for value in (*supplied.values(), fold)):
            raise GateBValidationError(
                "current freeze provenance comparison requires every file/ID SHA and fold"
            )
        current = {
            key: _required_sha256(value, f"current {key}")
            for key, value in supplied.items()
        }
        provenance_mismatches = [
            key for key, value in current.items() if frozen_provenance[key] != value
        ]
        if frozen_fold != fold:
            provenance_mismatches.append("fold")
        if provenance_mismatches:
            raise GateBValidationError(
                "current data/exclusion scope differs from the frozen development scope: "
                f"{provenance_mismatches!r}"
            )
    primary = _validated_frozen_method(freeze.get("primary"), "primary")
    fallback_value = freeze.get("fallback")
    fallback = (
        None
        if fallback_value is None
        else _validated_frozen_method(fallback_value, "fallback")
    )
    routing_policy = _validated_routing_policy(freeze, fallback is not None)
    return FrozenSelectionMethods(
        freeze_path=str(freeze_path),
        freeze_sha256=freeze_file_sha,
        split_sha256=split_manifest.sha256,
        fold=frozen_fold,
        train_file_sha256=frozen_provenance["train_file_sha256"],
        exclusions_file_sha256=frozen_provenance["exclusions_file_sha256"],
        excluded_ids_sha256=frozen_provenance["excluded_ids_sha256"],
        split_artifact_sha256=frozen_provenance["split_artifact_sha256"],
        development_shard_sha256=frozen_provenance["development_shard_sha256"],
        primary_label=primary[0],
        primary_checkpoint_sha256=primary[1],
        fallback_label=None if fallback is None else fallback[0],
        fallback_checkpoint_sha256=None if fallback is None else fallback[1],
        routing_policy=routing_policy,
    )


def validate_locked_holdout_access(
    receipt_artifact: str | Path,
    *,
    freeze_artifact: str | Path,
    split_manifest: SplitManifest,
    excluded_ids: Iterable[str],
) -> LockedHoldoutAccess:
    """Validate a receipt and derive its holdout IDs only in memory."""

    receipt_path, receipt, receipt_file_sha = _load_hashed_json_artifact(
        receipt_artifact, expected_schema=_HOLDOUT_SCHEMA
    )
    freeze_path, freeze, freeze_file_sha = _load_supported_freeze_artifact(
        freeze_artifact
    )
    if receipt.get("comparison_scope") != "complete_out_of_fold_union":
        raise GateBValidationError("holdout receipt is not bound to complete OOF evidence")
    freeze_evidence = receipt.get("freeze_artifact")
    if not isinstance(freeze_evidence, Mapping):
        raise GateBValidationError("holdout receipt lacks freeze evidence")
    if freeze_evidence.get("sha256") != freeze_file_sha:
        raise GateBValidationError("holdout receipt is not bound to the freeze bytes")
    if Path(str(freeze_evidence.get("path"))).resolve(strict=False) != freeze_path:
        raise GateBValidationError("holdout receipt freeze path does not match")
    claim_evidence = receipt.get("claim_artifact")
    if not isinstance(claim_evidence, Mapping):
        raise GateBValidationError("holdout receipt lacks canonical claim evidence")
    claim_path, claim, claim_file_sha = _load_hashed_json_artifact(
        _required_string(claim_evidence.get("path"), "claim path"),
        expected_schema="gate-b-locked-holdout-claim-v1",
    )
    if claim_evidence.get("sha256") != claim_file_sha:
        raise GateBValidationError("holdout receipt claim SHA-256 does not match")
    if claim.get("canonical_key") != split_manifest.sha256:
        raise GateBValidationError("holdout claim is not keyed by this split")
    if claim.get("freeze_artifact_sha256") != freeze_file_sha:
        raise GateBValidationError("holdout claim is not bound to this freeze")
    if claim.get("comparison_scope") != "complete_out_of_fold_union":
        raise GateBValidationError("holdout claim is not bound to complete OOF evidence")
    if claim_path.name != f"{split_manifest.sha256}.claim.json":
        raise GateBValidationError("holdout claim does not use the canonical filename")
    if receipt.get("split_sha256") != split_manifest.sha256:
        raise GateBValidationError("holdout receipt split SHA-256 does not match")
    if freeze.get("split_sha256") != split_manifest.sha256:
        raise GateBValidationError("freeze split SHA-256 does not match")
    excluded = set(expand_hard_group_exclusions(split_manifest, tuple(excluded_ids)))
    eligible_ids = tuple(
        problem_id
        for problem_id in split_manifest.final_holdout_ids()
        if problem_id not in excluded
    )
    if receipt.get("eligible_holdout_count") != len(eligible_ids):
        raise GateBValidationError("holdout receipt count does not match the split")
    if receipt.get("eligible_holdout_ids_sha256") != _ids_sha256(eligible_ids):
        raise GateBValidationError("holdout receipt ID digest does not match the split")
    if receipt.get("primary") != freeze.get("primary") or receipt.get(
        "fallback"
    ) != freeze.get("fallback"):
        raise GateBValidationError("holdout receipt changed the frozen methods")
    primary = _validated_frozen_method(freeze.get("primary"), "primary")
    fallback_value = freeze.get("fallback")
    fallback = (
        None
        if fallback_value is None
        else _validated_frozen_method(fallback_value, "fallback")
    )
    routing_policy = _validated_routing_policy(freeze, fallback is not None)
    if receipt.get("routing_policy") != routing_policy:
        raise GateBValidationError("holdout receipt changed the frozen routing policy")
    return LockedHoldoutAccess(
        receipt_path=str(receipt_path),
        receipt_sha256=receipt_file_sha,
        freeze_sha256=freeze_file_sha,
        split_sha256=split_manifest.sha256,
        eligible_ids=eligible_ids,
        primary_label=primary[0],
        primary_checkpoint_sha256=primary[1],
        fallback_label=None if fallback is None else fallback[0],
        fallback_checkpoint_sha256=None if fallback is None else fallback[1],
        routing_policy=routing_policy,
    )


def _validated_routing_policy(freeze: Mapping[str, Any], has_fallback: bool) -> str:
    expected = (
        INVALID_PRIMARY_FALLBACK_ROUTING_POLICY
        if has_fallback
        else PRIMARY_ONLY_ROUTING_POLICY
    )
    if freeze.get("routing_policy") != expected:
        raise GateBValidationError("freeze routing policy does not match its methods")
    evidence = freeze.get("fallback_routing_evidence")
    if not has_fallback:
        if evidence is not None:
            raise GateBValidationError("primary-only freeze must not contain fallback evidence")
        return expected
    if not isinstance(evidence, Mapping):
        raise GateBValidationError("fallback freeze lacks development routing evidence")
    primary = freeze.get("primary")
    fallback = freeze.get("fallback")
    if not isinstance(primary, Mapping) or not isinstance(fallback, Mapping):
        raise GateBValidationError("fallback freeze methods are invalid")
    if (
        evidence.get("primary_label") != primary.get("label")
        or evidence.get("fallback_label") != fallback.get("label")
        or evidence.get("routing_policy") != expected
    ):
        raise GateBValidationError("fallback evidence does not match the frozen route")
    return expected


def _require_complete_oof_freeze(freeze: Mapping[str, Any]) -> None:
    if freeze.get("comparison_scope") != "complete_out_of_fold_union":
        raise GateBValidationError("final selection must be frozen from complete OOF evidence")


def _require_base_oof_qualification(oof: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": _BASE_OOF_SCHEMA,
        "model_id": OFFICIAL_MODEL_ID,
        "revision": PINNED_MODEL_REVISION,
        "route": "direct_answer",
        "partition": "out_of_fold_cross_validation",
        "deployment_partition": "fold_validation",
        "locked_holdout_accessed": False,
        "leaderboard_or_test_used": False,
        "method_kind": FIXED_BASE_METHOD_KIND,
    }
    mismatched = [key for key, value in expected.items() if oof.get(key) != value]
    if mismatched:
        raise GateBValidationError(
            f"base OOF artifact violates its fixed-base contract: {mismatched!r}"
        )
    n_folds = oof.get("n_folds")
    deployment_fold = oof.get("deployment_fold")
    if (
        isinstance(n_folds, bool)
        or not isinstance(n_folds, int)
        or n_folds != _BASE_OOF_FOLD_COUNT
        or isinstance(deployment_fold, bool)
        or not isinstance(deployment_fold, int)
        or deployment_fold < 0
        or deployment_fold >= n_folds
        or oof.get("fold") != deployment_fold
        or oof.get("folds") != list(range(n_folds))
    ):
        raise GateBValidationError("base OOF fold coverage is invalid")
    for field_name in (
        "split_sha256",
        "source_groups_sha256",
        "eligibility_ids_sha256",
        "deployment_eligibility_ids_sha256",
    ):
        _required_sha256(oof.get(field_name), f"base OOF {field_name}")
    provenance = oof.get("data_provenance")
    provenance_keys = {
        "train_file_sha256",
        "exclusions_file_sha256",
        "excluded_ids_sha256",
        "split_artifact_sha256",
        "development_shard_sha256",
    }
    if not isinstance(provenance, Mapping) or set(provenance) != provenance_keys:
        raise GateBValidationError("base OOF data_provenance has an unexpected schema")
    for field_name in provenance_keys:
        _required_sha256(provenance.get(field_name), f"base OOF {field_name}")
    label = _validated_label(_required_string(oof.get("base_label"), "base label"))
    if oof.get("run_order") != [label]:
        raise GateBValidationError("base OOF run order is invalid")
    runs = oof.get("runs")
    if not isinstance(runs, Mapping) or set(runs) != {label}:
        raise GateBValidationError("base OOF runs must contain exactly the base label")
    evidence = runs[label]
    if not isinstance(evidence, Mapping):
        raise GateBValidationError("base OOF run evidence is invalid")
    fold_runs = evidence.get("fold_runs")
    if not isinstance(fold_runs, Mapping) or set(fold_runs) != {
        str(fold) for fold in range(n_folds)
    }:
        raise GateBValidationError("base OOF fold run evidence is incomplete")
    problem_count = oof.get("problem_count")
    if (
        isinstance(problem_count, bool)
        or not isinstance(problem_count, int)
        or problem_count <= 0
        or evidence.get("oof_problem_count") != problem_count
    ):
        raise GateBValidationError("base OOF problem count is invalid")
    total_fold_count = 0
    for fold in range(n_folds):
        fold_evidence = fold_runs[str(fold)]
        if not isinstance(fold_evidence, Mapping):
            raise GateBValidationError("base OOF fold evidence is invalid")
        if (
            fold_evidence.get("method_kind") != FIXED_BASE_METHOD_KIND
            or fold_evidence.get("adapter_artifact") is not None
            or fold_evidence.get("checkpoint_sha256")
            != BASE_MODEL_CHECKPOINT_SHA256
            or fold_evidence.get("training_method_fingerprint")
            != _base_method_fingerprint()
        ):
            raise GateBValidationError("base OOF fold method provenance is invalid")
        fold_count = fold_evidence.get("problem_count")
        if isinstance(fold_count, bool) or not isinstance(fold_count, int) or fold_count <= 0:
            raise GateBValidationError("base OOF fold problem count is invalid")
        total_fold_count += fold_count
    if total_fold_count != problem_count:
        raise GateBValidationError("base OOF fold counts do not match the OOF union")
    qualification = oof.get("qualification")
    if not isinstance(qualification, Mapping) or dict(qualification) != {
        "evidence_scope": "complete_out_of_fold_union",
        "fixed_base_only": True,
        "exact_fold_coverage": True,
        "validation_ids_unique": True,
        "run_evidence_unique": True,
        "checkpoint_sha256": BASE_MODEL_CHECKPOINT_SHA256,
        "training_method_fingerprint": _base_method_fingerprint(),
    }:
        raise GateBValidationError("base OOF qualification flags are invalid")


def _load_supported_freeze_artifact(
    path: str | Path,
) -> tuple[Path, Mapping[str, Any], str]:
    source, freeze, file_sha256 = _load_hashed_json_artifact(
        path,
        expected_schema=(_FREEZE_SCHEMA, _BASE_FREEZE_SCHEMA),
    )
    _require_complete_oof_freeze(freeze)
    if freeze.get("schema_version") == _FREEZE_SCHEMA:
        return source, freeze, file_sha256
    if (
        freeze.get("selection_kind") != "base_only_complete_oof"
        or freeze.get("fallback") is not None
        or freeze.get("routing_policy") != PRIMARY_ONLY_ROUTING_POLICY
        or freeze.get("fallback_routing_evidence") is not None
    ):
        raise GateBValidationError("base-only freeze shape is invalid")
    primary = freeze.get("primary")
    if not isinstance(primary, Mapping):
        raise GateBValidationError("base-only freeze primary is missing")
    if (
        primary.get("method_kind") != FIXED_BASE_METHOD_KIND
        or primary.get("adapter_artifact") is not None
        or primary.get("checkpoint_sha256") != BASE_MODEL_CHECKPOINT_SHA256
        or primary.get("training_method_fingerprint") != _base_method_fingerprint()
    ):
        raise GateBValidationError("base-only freeze primary provenance is invalid")
    comparison = freeze.get("comparison_artifact")
    if not isinstance(comparison, Mapping):
        raise GateBValidationError("base-only freeze lacks OOF evidence")
    oof_path, oof, oof_file_sha = _load_hashed_json_artifact(
        _required_string(comparison.get("path"), "base OOF path"),
        expected_schema=_BASE_OOF_SCHEMA,
    )
    _require_base_oof_qualification(oof)
    if (
        comparison.get("sha256") != oof_file_sha
        or comparison.get("payload_sha256") != oof.get("payload_sha256")
        or comparison.get("size_bytes") != oof_path.stat().st_size
    ):
        raise GateBValidationError("base-only freeze OOF binding is invalid")
    label = _validated_label(_required_string(primary.get("label"), "primary label"))
    runs = oof.get("runs")
    if (
        label != oof.get("base_label")
        or not isinstance(runs, Mapping)
        or dict(primary) != {"label": label, **dict(runs[label])}
    ):
        raise GateBValidationError("base-only freeze changed the qualified method")
    return source, freeze, file_sha256


def _revalidate_deployment_method_evidence(
    evidence: Mapping[str, Any], comparison: Mapping[str, Any], label: str
) -> None:
    method_kind = evidence.get("method_kind")
    checkpoint = _required_sha256(
        evidence.get("checkpoint_sha256"), f"run {label} checkpoint_sha256"
    )
    fingerprint = _required_sha256(
        evidence.get("training_method_fingerprint"),
        f"run {label} training_method_fingerprint",
    )
    adapter_evidence = evidence.get("adapter_artifact")
    if method_kind == FIXED_BASE_METHOD_KIND:
        if checkpoint != BASE_MODEL_CHECKPOINT_SHA256:
            raise GateBValidationError("frozen fixed-base method changed its checkpoint")
        if adapter_evidence is not None:
            raise GateBValidationError("frozen fixed-base method must not contain an adapter")
        if fingerprint != _base_method_fingerprint():
            raise GateBValidationError("frozen fixed-base method fingerprint is invalid")
        return
    if method_kind != ADAPTER_METHOD_KIND or not isinstance(adapter_evidence, Mapping):
        raise GateBValidationError("frozen candidate method lacks adapter provenance")
    base_adapter_keys = {
        "path",
        "artifact_sha256",
        "manifest_sha256",
        "checksums_sha256",
    }
    rationale_adapter_keys = {
        *base_adapter_keys,
        "training_target_kind",
        "rationale_candidate_config_sha256",
        "rationale_candidate_config_file_sha256",
        "rationale_corpus_records_sha256",
        "rationale_corpus_manifest_sha256",
        "rationale_corpus_audit_sha256",
    }
    actual_adapter_keys = frozenset(adapter_evidence)
    if actual_adapter_keys not in {
        frozenset(base_adapter_keys),
        frozenset(rationale_adapter_keys),
    }:
        raise GateBValidationError("frozen adapter evidence has an unexpected schema")
    adapter = validate_adapter_artifact(
        _required_string(adapter_evidence.get("path"), f"run {label} adapter path"),
        config=DEFAULT_GATE_B_CONFIG,
    )
    expected_adapter_identity = {
        "artifact_sha256": adapter.artifact_sha256,
        "manifest_sha256": adapter.manifest_sha256,
        "checksums_sha256": adapter.checksums_sha256,
    }
    expected_evidence_keys = (
        rationale_adapter_keys
        if adapter.training_target_kind == "verified_concise_rationale"
        else base_adapter_keys
    )
    if set(adapter_evidence) != expected_evidence_keys:
        raise GateBValidationError(
            "frozen adapter evidence does not match its training target schema"
        )
    if adapter.training_target_kind == "verified_concise_rationale":
        expected_adapter_identity.update(
            {
                "training_target_kind": adapter.training_target_kind,
                "rationale_candidate_config_sha256": (
                    adapter.rationale_candidate_config_sha256
                ),
                "rationale_candidate_config_file_sha256": (
                    adapter.rationale_candidate_config_file_sha256
                ),
                "rationale_corpus_records_sha256": (
                    adapter.rationale_corpus_records_sha256
                ),
                "rationale_corpus_manifest_sha256": (
                    adapter.rationale_corpus_manifest_sha256
                ),
                "rationale_corpus_audit_sha256": (
                    adapter.rationale_corpus_audit_sha256
                ),
            }
        )
    identity_mismatches = [
        field_name
        for field_name, actual_value in expected_adapter_identity.items()
        if adapter_evidence.get(field_name) != actual_value
    ]
    if Path(str(adapter_evidence["path"])).resolve(strict=True) != Path(adapter.path):
        identity_mismatches.append("path")
    if checkpoint != adapter.artifact_sha256:
        identity_mismatches.append("checkpoint_sha256")
    if fingerprint != _adapter_method_fingerprint(adapter):
        identity_mismatches.append("training_method_fingerprint")
    provenance = comparison.get("data_provenance")
    if not isinstance(provenance, Mapping):
        raise GateBValidationError("OOF comparison lacks data provenance")
    expected_scope = {
        "config_sha256": DEFAULT_GATE_B_CONFIG.sha256,
        "split_version": comparison.get("split_version"),
        "split_sha256": comparison.get("split_sha256"),
        "source_groups_sha256": comparison.get("source_groups_sha256"),
        "fold": comparison.get("deployment_fold"),
        "excluded_ids_sha256": provenance.get("excluded_ids_sha256"),
        "train_file_sha256": provenance.get("train_file_sha256"),
        "exclusions_file_sha256": provenance.get("exclusions_file_sha256"),
        "split_artifact_sha256": provenance.get("split_artifact_sha256"),
        "development_shard_sha256": provenance.get("development_shard_sha256"),
    }
    identity_mismatches.extend(
        field_name
        for field_name, expected_value in expected_scope.items()
        if getattr(adapter, field_name) != expected_value
    )
    if identity_mismatches:
        raise GateBValidationError(
            f"frozen adapter bytes/scope changed for {label}: "
            f"{sorted(set(identity_mismatches))!r}"
        )


def _validated_frozen_method(value: object, role: str) -> tuple[str, str]:
    if not isinstance(value, Mapping):
        raise GateBValidationError(f"frozen {role} method is missing")
    label = _validated_label(_required_string(value.get("label"), f"{role} label"))
    checkpoint = _required_sha256(
        value.get("checkpoint_sha256"), f"{role} checkpoint_sha256"
    )
    return label, checkpoint


def _load_development_run(
    named: NamedDevelopmentRun,
    *,
    expected_ids: Sequence[str],
    record_by_id: Mapping[str, MathRecord],
    split_manifest: SplitManifest,
    assignment_by_id: Mapping[str, Any],
    fold: int,
) -> _ValidatedRun:
    if not isinstance(named, NamedDevelopmentRun):
        raise TypeError("runs must contain NamedDevelopmentRun values")
    label = _validated_label(named.label)
    raw_path = Path(named.records_path)
    if raw_path.is_symlink():
        raise GateBValidationError(f"development JSONL refuses symlinks: {raw_path}")
    path = raw_path.resolve(strict=True)
    if not path.is_file():
        raise GateBValidationError(f"development JSONL must be a regular file: {path}")
    manifest_path, run_manifest = _load_run_manifest(named.manifest_path)
    _validate_execution_evidence(
        run_manifest.get("execution_evidence"),
        expected_config_sha256=DEFAULT_GATE_B_CONFIG.sha256,
    )
    records_sha256 = sha256_file(path)
    records_size = path.stat().st_size
    if run_manifest.get("records_file") != path.name:
        raise GateBValidationError(f"{label} run manifest points to another records file")
    if run_manifest.get("records_bytes") != records_size:
        raise GateBValidationError(f"{label} run manifest records size does not match")
    if run_manifest.get("records_sha256") != records_sha256:
        raise GateBValidationError(f"{label} run manifest records SHA-256 does not match")
    expected_set = set(expected_ids)
    rows: dict[str, Mapping[str, Any]] = {}
    with path.open("rb") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if len(raw_line) > _MAX_JSONL_LINE_BYTES:
                raise GateBValidationError(
                    f"{label} JSONL line {line_number} exceeds the safety limit"
                )
            if not raw_line.endswith(b"\n"):
                raise GateBValidationError(f"{label} JSONL must end every row with LF")
            if not raw_line.strip():
                raise GateBValidationError(f"{label} JSONL contains a blank row")
            try:
                row = json.loads(
                    raw_line.decode("utf-8", errors="strict"),
                    object_pairs_hook=_unique_json_object,
                    parse_constant=_reject_json_constant,
                )
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise GateBValidationError(
                    f"{label} JSONL line {line_number} is invalid: {exc}"
                ) from exc
            if not isinstance(row, Mapping) or set(row) != _BASELINE_RECORD_KEYS:
                raise GateBValidationError(
                    f"{label} JSONL line {line_number} has an unexpected schema"
                )
            problem_id = row.get("problem_id")
            if not isinstance(problem_id, str) or _TRAIN_ID_RE.fullmatch(problem_id) is None:
                raise GateBValidationError(f"{label} contains a non-train problem ID")
            if row.get("sample_index") != 0 or type(row.get("sample_index")) is not int:
                raise GateBValidationError(
                    f"{label} comparison requires exactly sample_index=0 per problem"
                )
            if problem_id in rows:
                raise GateBValidationError(f"{label} repeats problem ID {problem_id}")
            rows[problem_id] = row
    missing = sorted(expected_set - set(rows))
    extra = sorted(set(rows) - expected_set)
    if missing or extra:
        raise GateBValidationError(
            f"{label} rows do not match eligible validation IDs; "
            f"missing={missing[:5]!r}, extra={extra[:5]!r}"
        )

    eligibility_digest = _ids_sha256(expected_ids)
    checkpoint_values: set[str] = set()
    config_values: set[str] = set()
    predictions: dict[str, str] = {}
    exact_count = 0
    parser_counts: Counter[str] = Counter()
    finish_counts: Counter[str] = Counter()
    input_tokens_total = 0
    output_tokens_total = 0
    peaks: list[int] = []
    for problem_id in expected_ids:
        row = rows[problem_id]
        record = record_by_id[problem_id]
        assignment = assignment_by_id[problem_id]
        _validate_development_row(
            row,
            record=record,
            assignment=assignment,
            split_manifest=split_manifest,
            fold=fold,
            eligibility_digest=eligibility_digest,
        )
        checkpoint_values.add(str(row["checkpoint_sha256"]))
        config_values.add(str(row["config_sha256"]))
        parsed = parse_answer(str(row["raw_completion"]))
        parser_counts[parsed.status] += 1
        finish_counts[str(row["finish_reason"])] += 1
        input_tokens_total += int(row["input_token_count"])
        output_tokens_total += int(row["output_token_count"])
        if row["peak_vram_allocated_bytes"] is not None:
            peaks.append(int(row["peak_vram_allocated_bytes"]))
        if parsed.ok:
            assert parsed.value is not None
            predictions[problem_id] = str(parsed.value)
        exact_count += int(bool(row["exact_match"]))
    if len(checkpoint_values) != 1 or len(config_values) != 1:
        raise GateBValidationError(f"{label} has inconsistent run provenance")
    expected_manifest_values: dict[str, Any] = {
        "schema_version": "gate-b1-development-run-v2",
        "record_count": len(expected_ids),
        "problem_count": len(expected_ids),
        "samples_per_problem": 1,
        "model_id": OFFICIAL_MODEL_ID,
        "revision": PINNED_MODEL_REVISION,
        "route": "direct_answer",
        "checkpoint_sha256": next(iter(checkpoint_values)),
        "config_sha256": next(iter(config_values)),
        "decoding_policy": DEFAULT_GATE_B_CONFIG.decoding_policy.as_dict(),
        "decoding_policy_sha256": DEFAULT_GATE_B_CONFIG.decoding_policy.sha256,
        "split_version": split_manifest.version,
        "split_sha256": split_manifest.sha256,
        "source_groups_sha256": split_manifest.source_groups_sha256,
        "fold": fold,
        "partition": "fold_validation",
        "split_partition": SplitPartition.CROSS_VALIDATION.value,
        "eligibility_ids_sha256": eligibility_digest,
        "parser_status_counts": dict(sorted(parser_counts.items())),
        "finish_reason_counts": dict(sorted(finish_counts.items())),
        "exact_match_count": exact_count,
        "exact_match_accuracy": exact_count / len(expected_ids),
        "input_token_count_total": input_tokens_total,
        "output_token_count_total": output_tokens_total,
        "peak_vram_allocated_bytes_max": max(peaks) if peaks else None,
    }
    mismatched_manifest = [
        key for key, value in expected_manifest_values.items() if run_manifest.get(key) != value
    ]
    if mismatched_manifest:
        raise GateBValidationError(
            f"{label} run manifest summary mismatch: {mismatched_manifest!r}"
        )
    _validate_generation_evidence(
        run_manifest.get("generation_evidence"),
        tuple(rows[problem_id] for problem_id in expected_ids),
    )
    return _ValidatedRun(
        label=label,
        path=path,
        sha256=records_sha256,
        size_bytes=records_size,
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        manifest_size_bytes=manifest_path.stat().st_size,
        checkpoint_sha256=next(iter(checkpoint_values)),
        config_sha256=next(iter(config_values)),
        eligibility_ids_sha256=eligibility_digest,
        predictions=predictions,
        exact_match_count=exact_count,
    )


def _validate_development_row(
    row: Mapping[str, Any],
    *,
    record: MathRecord,
    assignment: Any,
    split_manifest: SplitManifest,
    fold: int,
    eligibility_digest: str,
) -> None:
    problem_id = record.id
    expected_scalars = {
        "schema_version": "gate-b1-development-baseline-v2",
        "model_id": OFFICIAL_MODEL_ID,
        "revision": PINNED_MODEL_REVISION,
        "route": "direct_answer",
        "sample_index": 0,
        "config_sha256": DEFAULT_GATE_B_CONFIG.sha256,
        "split_version": split_manifest.version,
        "split_sha256": split_manifest.sha256,
        "source_groups_sha256": split_manifest.source_groups_sha256,
        "fold": fold,
        "partition": "fold_validation",
        "split_partition": SplitPartition.CROSS_VALIDATION.value,
        "group_id": assignment.group_id,
        "eligibility_ids_sha256": eligibility_digest,
        "question_sha256": _sha256_text(record.question_raw),
        "decoding_policy": DEFAULT_GATE_B_CONFIG.decoding_policy.as_dict(),
        "decoding_policy_sha256": DEFAULT_GATE_B_CONFIG.decoding_policy.sha256,
    }
    mismatched = [key for key, value in expected_scalars.items() if row.get(key) != value]
    if mismatched:
        raise GateBValidationError(
            f"development row {problem_id} binding mismatch: {mismatched!r}"
        )
    if assignment.partition is not SplitPartition.CROSS_VALIDATION or assignment.fold != fold:
        raise GateBValidationError(f"development row {problem_id} is outside validation fold")
    for field in ("checkpoint_sha256", "raw_completion_sha256", "prompt_sha256"):
        _required_sha256(row.get(field), field)
    raw_completion = row.get("raw_completion")
    if not isinstance(raw_completion, str):
        raise GateBValidationError(f"development row {problem_id} raw completion is invalid")
    if row.get("raw_completion_sha256") != _sha256_text(raw_completion):
        raise GateBValidationError(f"development row {problem_id} raw hash mismatch")
    parsed = parse_answer(raw_completion)
    if row.get("parse") != asdict(parsed):
        raise GateBValidationError(f"development row {problem_id} parser result was changed")
    if type(record.answer) is not int or row.get("reference_answer") != record.answer:
        raise GateBValidationError(f"development row {problem_id} reference answer mismatch")
    expected_exact = parsed.ok and parsed.value == record.answer
    if type(row.get("exact_match")) is not bool or row.get("exact_match") != expected_exact:
        raise GateBValidationError(f"development row {problem_id} exact match was changed")
    prompt = {
        "messages": [
            {"role": "system", "content": DEFAULT_GATE_B_CONFIG.system_prompt},
            {"role": "user", "content": record.question_raw},
        ],
        "add_generation_prompt": True,
    }
    if row.get("prompt_sha256") != hashlib.sha256(canonical_json_bytes(prompt)).hexdigest():
        raise GateBValidationError(f"development row {problem_id} prompt hash mismatch")
    expected_seed = deterministic_seed(
        problem_id,
        0,
        salt=(
            f"gate-b1-v2:{DEFAULT_GATE_B_CONFIG.seed}:"
            f"{DEFAULT_GATE_B_CONFIG.sha256}"
        ),
    )
    if type(row.get("seed")) is not int or row.get("seed") != expected_seed:
        raise GateBValidationError(f"development row {problem_id} seed mismatch")
    for field in ("input_token_count", "output_token_count"):
        value = row.get(field)
        if type(value) is not int or value < 0:
            raise GateBValidationError(f"development row {problem_id} {field} is invalid")
    latency = row.get("latency_ms")
    if isinstance(latency, bool) or not isinstance(latency, (int, float)):
        raise GateBValidationError(f"development row {problem_id} latency is invalid")
    if not math.isfinite(float(latency)) or latency < 0:
        raise GateBValidationError(f"development row {problem_id} latency is invalid")
    peak = row.get("peak_vram_allocated_bytes")
    if peak is not None and (type(peak) is not int or peak < 0):
        raise GateBValidationError(f"development row {problem_id} peak VRAM is invalid")
    finish_reason = row.get("finish_reason")
    if not isinstance(finish_reason, str) or not finish_reason.strip():
        raise GateBValidationError(f"development row {problem_id} finish reason is invalid")


def _validated_exact_records(
    records: Iterable[MathRecord], expected_ids: Sequence[str]
) -> Mapping[str, MathRecord]:
    materialized = tuple(records)
    if any(not isinstance(record, MathRecord) for record in materialized):
        raise TypeError("records must contain MathRecord instances")
    ids = tuple(record.id for record in materialized)
    if len(set(ids)) != len(ids):
        raise GateBValidationError("records contain duplicate IDs")
    missing = sorted(set(expected_ids) - set(ids))
    extra = sorted(set(ids) - set(expected_ids))
    if missing or extra:
        raise GateBValidationError(
            "records must exactly match eligible validation IDs; "
            f"missing={missing[:5]!r}, extra={extra[:5]!r}"
        )
    by_id = {record.id: record for record in materialized}
    for problem_id in expected_ids:
        record = by_id[problem_id]
        if _TRAIN_ID_RE.fullmatch(problem_id) is None or type(record.answer) is not int:
            raise GateBValidationError("comparison accepts answered organizer-train rows only")
    return by_id


def _load_hashed_json_artifact(
    path: str | Path, *, expected_schema: str | Sequence[str]
) -> tuple[Path, Mapping[str, Any], str]:
    raw = Path(path)
    if raw.is_symlink():
        raise GateBValidationError(f"artifact refuses symlinks: {raw}")
    source = raw.resolve(strict=True)
    if not source.is_file():
        raise GateBValidationError(f"artifact must be a regular file: {source}")
    try:
        payload = json.loads(
            source.read_text(encoding="utf-8", errors="strict"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GateBValidationError(f"invalid JSON artifact {source}: {exc}") from exc
    schemas = (
        (expected_schema,)
        if isinstance(expected_schema, str)
        else tuple(expected_schema)
    )
    if (
        not schemas
        or any(not isinstance(schema, str) or not schema for schema in schemas)
        or not isinstance(payload, Mapping)
        or payload.get("schema_version") not in schemas
    ):
        raise GateBValidationError(f"artifact schema is not one of {schemas!r}")
    stored = _required_sha256(payload.get("payload_sha256"), "payload_sha256")
    without_hash = dict(payload)
    without_hash.pop("payload_sha256")
    computed = hashlib.sha256(canonical_json_bytes(without_hash)).hexdigest()
    if stored != computed:
        raise GateBValidationError("artifact payload_sha256 does not match its content")
    return source, payload, sha256_file(source)


def _load_run_manifest(path: str | Path) -> tuple[Path, Mapping[str, Any]]:
    raw = Path(path)
    if raw.is_symlink():
        raise GateBValidationError(f"development run manifest refuses symlinks: {raw}")
    source = raw.resolve(strict=True)
    if not source.is_file():
        raise GateBValidationError(f"development run manifest must be a file: {source}")
    try:
        payload = json.loads(
            source.read_text(encoding="utf-8", errors="strict"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GateBValidationError(f"invalid development run manifest: {exc}") from exc
    if (
        not isinstance(payload, Mapping)
        or set(payload) != _RUN_MANIFEST_KEYS
        or payload.get("schema_version") != "gate-b1-development-run-v2"
    ):
        raise GateBValidationError("development run manifest has an unexpected schema")
    return source, payload


def _validate_execution_evidence(
    value: object,
    *,
    expected_config_sha256: str,
) -> None:
    """Fail closed unless a run still binds to actual B0/source/config bytes."""

    if not isinstance(value, Mapping) or set(value) != _EXECUTION_EVIDENCE_KEYS:
        raise GateBValidationError("development run execution_evidence has an unexpected schema")
    if value.get("schema_version") != "gate-b1-execution-evidence-v1":
        raise GateBValidationError("development run execution_evidence schema is unsupported")

    source = value.get("source_manifest")
    if not isinstance(source, Mapping) or set(source) != _EXECUTION_SOURCE_KEYS:
        raise GateBValidationError("development source manifest evidence is invalid")
    source_path = _required_string(source.get("path"), "source manifest evidence path")
    source_file = _required_string(source.get("file"), "source manifest evidence file")
    if Path(source_file).name != source_file or source_file in {".", ".."}:
        raise GateBValidationError("source manifest evidence file must be one safe basename")
    source_actual = _regular_file_evidence(source_path, "source manifest")
    if source_actual["path"] != source_path or source_actual["sha256"] != _required_sha256(
        source.get("sha256"), "source manifest evidence sha256"
    ):
        raise GateBValidationError("source manifest evidence bytes changed after generation")
    if source_file != Path(source_path).name:
        raise GateBValidationError("source manifest evidence file does not match its path")
    if source_actual["tree_sha256"] != _required_sha256(
        source.get("tree_sha256"), "source manifest evidence tree_sha256"
    ):
        raise GateBValidationError("source manifest evidence tree digest changed")
    source_file_count = source.get("file_count")
    if (
        isinstance(source_file_count, bool)
        or not isinstance(source_file_count, int)
        or source_file_count < 1
        or source_actual["file_count"] != source_file_count
    ):
        raise GateBValidationError("source manifest evidence file_count changed")

    config = value.get("config_file")
    if not isinstance(config, Mapping) or set(config) != _EXECUTION_CONFIG_KEYS:
        raise GateBValidationError("development config-file evidence is invalid")
    _validate_execution_file_evidence(
        config,
        label="Gate B config",
        expected_sha256_key="sha256",
        expected_keys=_EXECUTION_CONFIG_KEYS,
    )
    if _required_sha256(config.get("config_sha256"), "config evidence config_sha256") != (
        expected_config_sha256
    ):
        raise GateBValidationError("development config evidence is not bound to run config")

    runtime = value.get("runtime_gate")
    if not isinstance(runtime, Mapping) or set(runtime) != _EXECUTION_RUNTIME_KEYS:
        raise GateBValidationError("development runtime-gate evidence is invalid")
    _validate_execution_file_evidence(
        runtime.get("preflight_report"),
        label="model preflight report",
        expected_sha256_key="sha256",
    )
    _validate_execution_file_evidence(
        runtime.get("gpu_smoke_report"),
        label="GPU smoke report",
        expected_sha256_key="sha256",
    )
    _required_string(runtime.get("gpu_device_name"), "runtime-gate GPU device name")


def _validate_execution_file_evidence(
    value: object,
    *,
    label: str,
    expected_sha256_key: str,
    expected_keys: frozenset[str] = _EXECUTION_RUNTIME_FILE_KEYS,
) -> None:
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise GateBValidationError(f"{label} evidence has an unexpected schema")
    path = _required_string(value.get("path"), f"{label} evidence path")
    expected_sha256 = _required_sha256(
        value.get(expected_sha256_key), f"{label} evidence sha256"
    )
    actual = _regular_file_evidence(path, label)
    if actual["path"] != path or actual["sha256"] != expected_sha256:
        raise GateBValidationError(f"{label} evidence bytes changed after generation")


def _validate_generation_evidence(
    value: object,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Recompute seed/prompt/latency summaries from validated private JSONL rows."""

    if not isinstance(value, Mapping) or set(value) != _GENERATION_EVIDENCE_KEYS:
        raise GateBValidationError("development generation_evidence has an unexpected schema")
    if value.get("schema_version") != "gate-b1-generation-evidence-v1":
        raise GateBValidationError("development generation_evidence schema is unsupported")
    ordered = tuple(
        sorted(rows, key=lambda row: (str(row["problem_id"]), int(row["sample_index"])))
    )
    expected_seed_sha = hashlib.sha256(
        canonical_json_bytes([int(row["seed"]) for row in ordered])
    ).hexdigest()
    expected_prompt_sha = hashlib.sha256(
        canonical_json_bytes([str(row["prompt_sha256"]) for row in ordered])
    ).hexdigest()
    if _required_sha256(value.get("seed_sequence_sha256"), "seed_sequence_sha256") != (
        expected_seed_sha
    ):
        raise GateBValidationError("development seed evidence does not match rows")
    if _required_sha256(
        value.get("prompt_sha256_sequence_sha256"), "prompt_sha256_sequence_sha256"
    ) != expected_prompt_sha:
        raise GateBValidationError("development prompt evidence does not match rows")
    latency = value.get("latency_ms")
    if not isinstance(latency, Mapping) or set(latency) != _LATENCY_EVIDENCE_KEYS:
        raise GateBValidationError("development latency evidence has an unexpected schema")
    values = tuple(float(row["latency_ms"]) for row in ordered)
    expected_latency = {
        "count": len(values),
        "total": math.fsum(values),
        "min": min(values),
        "max": max(values),
        "mean": math.fsum(values) / len(values),
    }
    for field_name, expected in expected_latency.items():
        actual = latency.get(field_name)
        if field_name == "count":
            if type(actual) is not int or actual != expected:
                raise GateBValidationError("development latency count does not match rows")
            continue
        if (
            isinstance(actual, bool)
            or not isinstance(actual, (int, float))
            or not math.isfinite(float(actual))
            or float(actual) != expected
        ):
            raise GateBValidationError(
                f"development latency evidence {field_name} does not match rows"
            )


def _regular_file_evidence(path: str | Path, label: str) -> dict[str, Any]:
    raw = Path(path)
    if raw.is_symlink():
        raise GateBValidationError(f"{label} refuses symlinks: {raw}")
    source = raw.resolve(strict=True)
    if not source.is_file():
        raise GateBValidationError(f"{label} must be a regular file: {source}")
    evidence: dict[str, Any] = {
        "path": str(source),
        "size_bytes": source.stat().st_size,
        "sha256": sha256_file(source),
    }
    if label == "source manifest":
        try:
            payload = json.loads(
                source.read_text(encoding="utf-8", errors="strict"),
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_json_constant,
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise GateBValidationError(f"invalid source manifest: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise GateBValidationError("source manifest must be a JSON object")
        evidence["tree_sha256"] = _required_sha256(
            payload.get("tree_sha256"), "source manifest tree_sha256"
        )
        file_count = payload.get("file_count")
        files = payload.get("files")
        if isinstance(file_count, bool) or not isinstance(file_count, int) or file_count < 1:
            if not isinstance(files, list) or not files:
                raise GateBValidationError("source manifest must contain file evidence")
            file_count = len(files)
        evidence["file_count"] = file_count
    return evidence


def _write_hashed_json_noreplace(
    path: str | Path, payload_without_hash: Mapping[str, Any]
) -> GateBSelectionWriteResult:
    raw_target = Path(path)
    if raw_target.is_symlink() or raw_target.parent.is_symlink():
        raise GateBValidationError("artifact target and parent refuse symlinks")
    target = raw_target.resolve(strict=False)
    if not target.parent.is_dir():
        raise GateBValidationError("artifact parent must be an existing real directory")
    if target.exists() or target.is_symlink():
        raise GateBValidationError(f"refusing to overwrite selection artifact: {target}")
    payload_sha = hashlib.sha256(canonical_json_bytes(payload_without_hash)).hexdigest()
    payload = {**dict(payload_without_hash), "payload_sha256": payload_sha}
    serialized = (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, raw_temp = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(raw_temp)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            raise GateBValidationError(
                f"refusing to overwrite selection artifact: {target}"
            ) from exc
        _fsync_directory(target.parent)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()
    return GateBSelectionWriteResult(
        path=str(target),
        size_bytes=len(serialized),
        sha256=hashlib.sha256(serialized).hexdigest(),
        payload_sha256=payload_sha,
    )


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise GateBValidationError(f"duplicate JSON key {key!r}")
        output[key] = value
    return output


def _reject_json_constant(value: str) -> None:
    raise GateBValidationError(f"non-finite JSON constant {value!r} is forbidden")


def _validated_label(value: str) -> str:
    if not isinstance(value, str) or _LABEL_RE.fullmatch(value) is None:
        raise GateBValidationError(
            "run label must be 1-64 ASCII letters, digits, dot, underscore, or hyphen"
        )
    return value


def _required_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise GateBValidationError(f"{field_name} must be a non-empty trimmed string")
    return value


def _required_sha256(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise GateBValidationError(f"{field_name} must be a lowercase SHA-256")
    return value


def _required_finite_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GateBValidationError(f"{field_name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise GateBValidationError(f"{field_name} must be finite")
    return result


def _validated_probe_holm_family(
    comparison: Mapping[str, Any],
    *,
    comparisons: list[object],
    alpha: float,
    family_size: int,
) -> dict[str, Any]:
    family_entries: list[tuple[str, Mapping[str, Any]]] = []
    for item in comparisons:
        if not isinstance(item, Mapping):
            raise GateBValidationError("candidate probe comparison must be an object")
        hypothesis = _validated_label(
            _required_string(item.get("candidate_label"), "candidate label")
        )
        family_entries.append((hypothesis, item))
    fallback_entries = comparison.get("fallback_routing_comparisons")
    if not isinstance(fallback_entries, list):
        raise GateBValidationError(
            "candidate probe fallback comparisons must be a list"
        )
    for item in fallback_entries:
        if not isinstance(item, Mapping):
            raise GateBValidationError(
                "candidate probe fallback comparison must be an object"
            )
        hypothesis = _required_string(
            item.get("hypothesis"), "fallback hypothesis"
        )
        family_entries.append((hypothesis, item))
    if len(family_entries) != family_size:
        raise GateBValidationError(
            "candidate probe family_size disagrees with comparison entries"
        )

    raw_p_values: dict[str, float] = {}
    stored_holm: dict[str, Mapping[str, Any]] = {}
    for hypothesis, item in family_entries:
        if hypothesis in raw_p_values:
            raise GateBValidationError(
                "candidate probe Holm hypotheses must be unique"
            )
        raw_p_values[hypothesis] = _required_probability(
            item.get("mcnemar_exact_p_value"),
            f"Holm raw p-value for {hypothesis}",
            open_interval=False,
        )
        holm = item.get("holm")
        if not isinstance(holm, Mapping) or holm.get("hypothesis") != hypothesis:
            raise GateBValidationError(
                "candidate probe lacks matching Holm evidence"
            )
        stored_holm[hypothesis] = holm

    expected = {
        item.hypothesis: item
        for item in holm_bonferroni(raw_p_values, alpha=alpha)
    }
    for hypothesis, holm in stored_holm.items():
        raw_p = _required_probability(
            holm.get("raw_p_value"),
            f"Holm stored raw p-value for {hypothesis}",
            open_interval=False,
        )
        adjusted_p = _required_probability(
            holm.get("adjusted_p_value"),
            f"Holm adjusted p-value for {hypothesis}",
            open_interval=False,
        )
        reject = holm.get("reject")
        if type(reject) is not bool:
            raise GateBValidationError("candidate Holm reject must be boolean")
        expected_item = expected[hypothesis]
        if (
            raw_p != expected_item.raw_p_value
            or not math.isclose(
                adjusted_p,
                expected_item.adjusted_p_value,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            or reject != expected_item.reject
        ):
            raise GateBValidationError(
                "candidate Holm evidence disagrees with the full comparison family"
            )
    return expected


def _required_probability(
    value: object, field_name: str, *, open_interval: bool
) -> float:
    result = _required_finite_number(value, field_name)
    valid = 0.0 < result < 1.0 if open_interval else 0.0 <= result <= 1.0
    if not valid:
        interval = (
            "strictly between zero and one"
            if open_interval
            else "between zero and one"
        )
        raise GateBValidationError(f"{field_name} must be {interval}")
    return result


def _ids_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256(canonical_json_bytes(list(values))).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "ADAPTER_METHOD_KIND",
    "FIXED_BASE_METHOD_KIND",
    "GateBSelectionWriteResult",
    "FrozenSelectionMethods",
    "FoldDevelopmentRun",
    "HOLDOUT_ACCESS_ACKNOWLEDGEMENT",
    "INVALID_PRIMARY_FALLBACK_ROUTING_POLICY",
    "LockedHoldoutAccess",
    "NamedDevelopmentRun",
    "PRIMARY_ONLY_ROUTING_POLICY",
    "authorize_locked_holdout_once",
    "compare_development_runs",
    "compare_cross_fold_development_runs",
    "decide_candidate_probe_promotion",
    "freeze_development_selection",
    "require_base_development_artifact",
    "validate_frozen_selection_methods",
    "validate_locked_holdout_access",
]

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
_OOF_COMPARISON_SCHEMA = "gate-b-development-oof-comparison-v1"
_FREEZE_SCHEMA = "gate-b-selection-freeze-v3"
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
    }
)


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
        "training_examples_sha256": plan.training_examples_sha256,
        "validation_examples_sha256": plan.validation_examples_sha256,
        "train_file_sha256": train_file_sha256,
        "exclusions_file_sha256": exclusions_file_sha256,
        "split_artifact_sha256": split_artifact_sha256,
        "development_shard_sha256": development_shard_sha256,
    }
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
    return hashlib.sha256(
        canonical_json_bytes(
            {
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
                "response_only_labels": True,
                "truncation": False,
                "tensor_contract": "qwen2.5-3b-all-linear-lora-504",
            }
        )
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
    freeze_path, freeze, freeze_file_sha = _load_hashed_json_artifact(
        freeze_artifact, expected_schema=_FREEZE_SCHEMA
    )
    _require_complete_oof_freeze(freeze)
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
    freeze_path, freeze, freeze_file_sha = _load_hashed_json_artifact(
        freeze_artifact, expected_schema=_FREEZE_SCHEMA
    )
    _require_complete_oof_freeze(freeze)
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
    freeze_path, freeze, freeze_file_sha = _load_hashed_json_artifact(
        freeze_artifact, expected_schema=_FREEZE_SCHEMA
    )
    _require_complete_oof_freeze(freeze)
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
    if set(adapter_evidence) != {
        "path",
        "artifact_sha256",
        "manifest_sha256",
        "checksums_sha256",
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
        "schema_version": "gate-b1-development-run-v1",
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
    if not isinstance(payload, Mapping) or set(payload) != _RUN_MANIFEST_KEYS:
        raise GateBValidationError("development run manifest has an unexpected schema")
    return source, payload


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
    "freeze_development_selection",
    "require_base_development_artifact",
    "validate_frozen_selection_methods",
    "validate_locked_holdout_access",
]

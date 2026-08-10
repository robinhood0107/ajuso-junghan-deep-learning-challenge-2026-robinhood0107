"""CPU-only real-tokenizer preflight for the locked Gate B SFT route."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .data import MathRecord
from .gate_b import (
    DEFAULT_GATE_B_CONFIG,
    ChatTokenizer,
    GateBConfig,
    GateBValidationError,
    encode_response_only_example,
)
from .gate_b_runtime import build_fold_sft_plan
from .model_preflight import OFFICIAL_MODEL_ID, OFFICIAL_REVISION
from .provenance import canonical_json_bytes
from .rationale_corpus import ConciseRationaleConfig, RationaleCorpusEvidence
from .splits import SplitManifest, eligible_training_ids, eligible_validation_ids


@dataclass(frozen=True, slots=True)
class SFTEncodingPreflightWriteResult:
    path: str
    size_bytes: int
    sha256: str
    payload_sha256: str


def run_sft_encoding_preflight(
    records: Iterable[MathRecord],
    *,
    split_manifest: SplitManifest,
    excluded_ids: Iterable[str],
    tokenizer: ChatTokenizer,
    tokenizer_provenance: Mapping[str, Any],
    train_file_sha256: str,
    exclusions_file_sha256: str,
    split_artifact_sha256: str,
    development_shard_sha256: str,
    output_path: str | Path,
    folds: Sequence[int] | None = None,
    rationale_corpus: RationaleCorpusEvidence | None = None,
    rationale_config: ConciseRationaleConfig | None = None,
    config: GateBConfig = DEFAULT_GATE_B_CONFIG,
) -> SFTEncodingPreflightWriteResult:
    """Encode exact fold partitions and prove response-only labels fit at 2K.

    The tokenizer is supplied by the caller so tests remain model-free.  The
    production CLI loads only the pinned local tokenizer snapshot; no weights,
    Torch, CUDA, leaderboard, or locked holdout records are touched.
    """

    if config != DEFAULT_GATE_B_CONFIG:
        raise GateBValidationError("SFT encoding preflight requires the locked config")
    if not isinstance(split_manifest, SplitManifest):
        raise TypeError("split_manifest must be a SplitManifest")
    split_manifest.validate()
    exclusions = tuple(excluded_ids)
    selected_folds = (
        tuple(range(split_manifest.n_folds)) if folds is None else tuple(folds)
    )
    if not selected_folds or len(set(selected_folds)) != len(selected_folds):
        raise GateBValidationError("folds must be a non-empty unique sequence")
    for fold in selected_folds:
        if isinstance(fold, bool) or not isinstance(fold, int):
            raise GateBValidationError("fold values must be integers")
        split_manifest.fold_ids(fold)
    if (rationale_corpus is None) != (rationale_config is None):
        raise GateBValidationError(
            "rationale_corpus and rationale_config must be supplied together"
        )
    if rationale_corpus is not None and selected_folds != (rationale_corpus.fold,):
        raise GateBValidationError(
            "rationale encoding preflight accepts exactly the corpus-bound fold"
        )

    required_ids: set[str] = set()
    for fold in selected_folds:
        required_ids.update(eligible_training_ids(split_manifest, fold, exclusions))
        required_ids.update(eligible_validation_ids(split_manifest, fold, exclusions))
    materialized = tuple(records)
    if not materialized or any(not isinstance(record, MathRecord) for record in materialized):
        raise GateBValidationError("records must contain organizer MathRecord values")
    record_by_id = {record.id: record for record in materialized}
    if len(record_by_id) != len(materialized):
        raise GateBValidationError("records contain duplicate IDs")
    missing = sorted(required_ids - set(record_by_id))
    extra = sorted(set(record_by_id) - required_ids)
    if missing or extra:
        raise GateBValidationError(
            "records must exactly match the requested eligible development-CV scope; "
            f"missing={missing[:5]!r}, extra={extra[:5]!r}"
        )

    if tokenizer_provenance.get("model_id") != OFFICIAL_MODEL_ID:
        raise GateBValidationError("tokenizer provenance does not use the fixed model")
    if tokenizer_provenance.get("resolved_commit") != OFFICIAL_REVISION:
        raise GateBValidationError("tokenizer provenance does not use the pinned revision")
    if tokenizer_provenance.get("local_files_only") is not True:
        raise GateBValidationError("tokenizer preflight must be local-files-only")
    for value, label in (
        (train_file_sha256, "train_file_sha256"),
        (exclusions_file_sha256, "exclusions_file_sha256"),
        (split_artifact_sha256, "split_artifact_sha256"),
        (development_shard_sha256, "development_shard_sha256"),
    ):
        if not isinstance(value, str) or len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise GateBValidationError(f"{label} must be a lowercase SHA-256")

    fold_reports: dict[str, Any] = {}
    all_development_ids: set[str] = set()
    all_validation_coverage_ids: set[str] = set()
    global_max: dict[str, Any] | None = None
    for fold in selected_folds:
        training_ids = eligible_training_ids(split_manifest, fold, exclusions)
        validation_ids = eligible_validation_ids(split_manifest, fold, exclusions)
        training_records = tuple(record_by_id[problem_id] for problem_id in training_ids)
        validation_records = tuple(record_by_id[problem_id] for problem_id in validation_ids)
        plan = build_fold_sft_plan(
            training_records,
            validation_records,
            split_manifest=split_manifest,
            fold=fold,
            excluded_ids=exclusions,
            rationale_corpus=rationale_corpus,
            rationale_config=rationale_config,
            config=config,
        )
        partitions: dict[str, Any] = {}
        for role, examples in (
            ("training", plan.training_examples),
            ("validation", plan.validation_examples),
        ):
            max_item: dict[str, Any] | None = None
            min_response_tokens: int | None = None
            over_limit = 0
            for example in examples:
                encoded = encode_response_only_example(example, tokenizer, config=config)
                response_tokens = encoded.sequence_token_count - encoded.prompt_token_count
                if response_tokens <= 0:
                    raise GateBValidationError(
                        f"{example.problem_id}: assistant target has no response tokens"
                    )
                min_response_tokens = (
                    response_tokens
                    if min_response_tokens is None
                    else min(min_response_tokens, response_tokens)
                )
                if encoded.sequence_token_count > config.max_sequence_length:
                    over_limit += 1  # pragma: no cover - encoder fails first
                item = {
                    "id": example.problem_id,
                    "prompt_tokens": encoded.prompt_token_count,
                    "sequence_tokens": encoded.sequence_token_count,
                    "response_tokens": response_tokens,
                }
                if max_item is None or (item["sequence_tokens"], item["id"]) > (
                    max_item["sequence_tokens"],
                    max_item["id"],
                ):
                    max_item = item
                if global_max is None or (item["sequence_tokens"], item["id"]) > (
                    global_max["sequence_tokens"],
                    global_max["id"],
                ):
                    global_max = {"fold": fold, "partition": role, **item}
            if max_item is None or min_response_tokens is None:  # pragma: no cover
                raise GateBValidationError(f"fold {fold} {role} is unexpectedly empty")
            partitions[role] = {
                "count": len(examples),
                "ids_sha256": (
                    plan.training_ids_sha256
                    if role == "training"
                    else plan.validation_ids_sha256
                ),
                "max": max_item,
                "min_response_tokens": min_response_tokens,
                "over_max_sequence_length_count": over_limit,
            }
        all_development_ids.update(training_ids)
        all_development_ids.update(validation_ids)
        all_validation_coverage_ids.update(validation_ids)
        fold_reports[str(fold)] = {
            "training": partitions["training"],
            "validation": partitions["validation"],
        }

    assert global_max is not None
    payload_without_hash = {
        "schema_version": (
            "gate-b-sft-encoding-preflight-v3"
            if rationale_corpus is None
            else "gate-b-sft-encoding-preflight-v4"
        ),
        "status": "green",
        "proof_scope": "cpu_only_pinned_tokenizer_response_only_encoding",
        "model_weights_loaded": False,
        "torch_or_cuda_used": False,
        "leaderboard_or_test_used": False,
        "locked_holdout_accessed": False,
        "model_id": OFFICIAL_MODEL_ID,
        "revision": OFFICIAL_REVISION,
        "config_sha256": config.sha256,
        "max_sequence_length": config.max_sequence_length,
        "response_only_loss": config.response_only_loss,
        "truncation_allowed": False,
        "training_target": (
            {"kind": "direct_answer"}
            if rationale_corpus is None
            else {
                "kind": "verified_concise_rationale",
                "candidate_config_sha256": rationale_corpus.candidate_config_sha256,
                "candidate_config_file_sha256": (
                    rationale_corpus.candidate_config_file_sha256
                ),
                "corpus_records_sha256": rationale_corpus.records_sha256,
                "corpus_manifest_sha256": rationale_corpus.manifest_sha256,
                "corpus_audit_sha256": rationale_corpus.audit_sha256,
            }
        ),
        "tokenizer_provenance": dict(tokenizer_provenance),
        "data_provenance": {
            "train_file_sha256": train_file_sha256,
            "exclusions_file_sha256": exclusions_file_sha256,
            "excluded_ids_sha256": _ids_sha256(tuple(sorted(exclusions))),
            "split_artifact_sha256": split_artifact_sha256,
            "development_shard_sha256": development_shard_sha256,
            "split_sha256": split_manifest.sha256,
            "source_groups_sha256": split_manifest.source_groups_sha256,
        },
        "folds_checked": list(selected_folds),
        "unique_development_cv_count": len(all_development_ids),
        "unique_development_cv_ids_sha256": _ids_sha256(
            tuple(sorted(all_development_ids))
        ),
        "unique_validation_coverage_count": len(all_validation_coverage_ids),
        "unique_validation_coverage_ids_sha256": _ids_sha256(
            tuple(sorted(all_validation_coverage_ids))
        ),
        "global_max": global_max,
        "folds": fold_reports,
    }
    return _write_hashed_json_noreplace(output_path, payload_without_hash)


def _ids_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256(canonical_json_bytes(list(values))).hexdigest()


def _write_hashed_json_noreplace(
    path: str | Path, payload_without_hash: Mapping[str, Any]
) -> SFTEncodingPreflightWriteResult:
    raw_target = Path(path)
    if raw_target.is_symlink() or raw_target.parent.is_symlink():
        raise GateBValidationError("SFT preflight output refuses symlinks")
    target = raw_target.resolve(strict=False)
    if not target.parent.is_dir():
        raise GateBValidationError("SFT preflight output parent must exist")
    if target.exists():
        raise GateBValidationError(f"refusing to overwrite SFT preflight: {target}")
    payload_sha = hashlib.sha256(canonical_json_bytes(payload_without_hash)).hexdigest()
    serialized = (
        json.dumps(
            {**dict(payload_without_hash), "payload_sha256": payload_sha},
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
                f"refusing to overwrite SFT preflight: {target}"
            ) from exc
        descriptor = os.open(
            target.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()
    return SFTEncodingPreflightWriteResult(
        path=str(target),
        size_bytes=len(serialized),
        sha256=hashlib.sha256(serialized).hexdigest(),
        payload_sha256=payload_sha,
    )


__all__ = ["SFTEncodingPreflightWriteResult", "run_sft_encoding_preflight"]

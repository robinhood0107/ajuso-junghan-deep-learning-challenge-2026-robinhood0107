"""Frozen-method offline leaderboard/test inference with strict parsed outputs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from .answers import parse_answer
from .data import MathRecord, load_leaderboard_csv
from .gate_b import (
    DEFAULT_GATE_B_CONFIG,
    ChatMessage,
    GateBConfig,
    GateBValidationError,
    GenerationRequest,
    GenerationResult,
)
from .gate_b_selection import (
    INVALID_PRIMARY_FALLBACK_ROUTING_POLICY,
    PRIMARY_ONLY_ROUTING_POLICY,
    validate_frozen_selection_methods,
)
from .inference import deterministic_seed
from .provenance import canonical_json_bytes, sha256_file
from .splits import SplitManifest

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class PredictionGenerationBackend(Protocol):
    @property
    def checkpoint_sha256(self) -> str: ...

    def generate(self, request: GenerationRequest) -> GenerationResult: ...


@dataclass(frozen=True, slots=True)
class PredictionArtifactWriteResult:
    artifact_path: str
    predictions_path: str
    artifact_sha256: str
    predictions_sha256: str
    problem_count: int
    invalid_count: int


def run_frozen_evaluation_inference(
    *,
    dataset_role: str,
    evaluation_file_path: str | Path,
    expected_evaluation_sha256: str,
    split_manifest: SplitManifest,
    excluded_ids: Iterable[str],
    train_file_sha256: str,
    exclusions_file_sha256: str,
    excluded_ids_sha256: str,
    split_artifact_sha256: str,
    development_shard_sha256: str,
    fold: int,
    freeze_artifact: str | Path,
    primary_backend: PredictionGenerationBackend,
    fallback_backend: PredictionGenerationBackend | None,
    artifact_path: str | Path,
    predictions_path: str | Path,
    config: GateBConfig = DEFAULT_GATE_B_CONFIG,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> PredictionArtifactWriteResult:
    """Run frozen primary/fallback inference without answer fabrication.

    The fallback is invoked only when the primary parser result is invalid.  If
    both are invalid, the ID is deliberately absent from the predictions map so
    the strict submission writer fails instead of silently inserting zero.
    """

    if config != DEFAULT_GATE_B_CONFIG:
        raise GateBValidationError("evaluation inference requires the locked Gate B config")
    if dataset_role not in {"leaderboard", "test"}:
        raise GateBValidationError("dataset_role must be leaderboard or test")
    if not isinstance(split_manifest, SplitManifest):
        raise TypeError("split_manifest must be a SplitManifest")
    split_manifest.validate()
    exclusions = tuple(excluded_ids)
    if len(exclusions) != len(set(exclusions)):
        raise GateBValidationError("excluded_ids must not contain duplicates")
    supplied_exclusion_digest = _required_sha256(
        excluded_ids_sha256, "excluded_ids_sha256"
    )
    if _ids_sha256(tuple(sorted(exclusions))) != supplied_exclusion_digest:
        raise GateBValidationError("excluded_ids do not match excluded_ids_sha256")
    expected_digest = _required_sha256(
        expected_evaluation_sha256, "expected_evaluation_sha256"
    )
    evaluation_source = _validated_regular_file(evaluation_file_path, "evaluation CSV")
    actual_digest = sha256_file(evaluation_source)
    if actual_digest != expected_digest:
        raise GateBValidationError("evaluation CSV SHA-256 does not match the expected contract")
    artifact_target, predictions_target = _validated_pair_targets(
        artifact_path, predictions_path
    )
    materialized = load_leaderboard_csv(evaluation_source).records
    _validate_evaluation_records(materialized)
    methods = validate_frozen_selection_methods(
        freeze_artifact,
        split_manifest=split_manifest,
        train_file_sha256=train_file_sha256,
        exclusions_file_sha256=exclusions_file_sha256,
        excluded_ids_sha256=excluded_ids_sha256,
        split_artifact_sha256=split_artifact_sha256,
        development_shard_sha256=development_shard_sha256,
        fold=fold,
    )
    _require_backend_checkpoint(
        primary_backend, methods.primary_checkpoint_sha256, "primary"
    )
    if methods.fallback_checkpoint_sha256 is None:
        if methods.routing_policy != PRIMARY_ONLY_ROUTING_POLICY:
            raise GateBValidationError("freeze has an invalid primary-only routing policy")
        if fallback_backend is not None:
            raise GateBValidationError("freeze has no fallback but a backend was supplied")
    else:
        if methods.routing_policy != INVALID_PRIMARY_FALLBACK_ROUTING_POLICY:
            raise GateBValidationError("freeze has an invalid fallback routing policy")
        if fallback_backend is None:
            raise GateBValidationError("freeze requires a fallback backend")
        if primary_backend is fallback_backend:
            raise GateBValidationError("primary and fallback backends must be distinct")
        _require_closeable_primary(primary_backend)
        _require_backend_checkpoint(
            fallback_backend, methods.fallback_checkpoint_sha256, "fallback"
        )

    primary_by_id: dict[str, dict[str, object]] = {}
    primary_parser_counts: Counter[str] = Counter()
    for record in materialized:
        primary = _generate_one(
            record,
            backend=primary_backend,
            checkpoint_sha256=methods.primary_checkpoint_sha256,
            method_label=methods.primary_label,
            dataset_role=dataset_role,
            config=config,
            clock_ns=clock_ns,
        )
        primary_parse = primary["parse"]
        assert isinstance(primary_parse, Mapping)
        primary_parser_counts[str(primary_parse["status"])] += 1
        primary_by_id[record.id] = primary

    invalid_primary_records = tuple(
        record
        for record in materialized
        if _parse_status(primary_by_id[record.id]) != "ok"
    )
    fallback_by_id: dict[str, dict[str, object]] = {}
    fallback_parser_counts: Counter[str] = Counter()
    if invalid_primary_records and fallback_backend is not None:
        _close_backend(primary_backend)
        for record in invalid_primary_records:
            fallback = _generate_one(
                record,
                backend=fallback_backend,
                checkpoint_sha256=str(methods.fallback_checkpoint_sha256),
                method_label=str(methods.fallback_label),
                dataset_role=dataset_role,
                config=config,
                clock_ns=clock_ns,
            )
            fallback_parse = fallback["parse"]
            assert isinstance(fallback_parse, Mapping)
            fallback_parser_counts[str(fallback_parse["status"])] += 1
            fallback_by_id[record.id] = fallback

    rows: list[dict[str, object]] = []
    predictions: dict[str, int] = {}
    for record in materialized:
        primary = primary_by_id[record.id]
        fallback = fallback_by_id.get(record.id)
        selected = primary
        if fallback is not None and _parse_status(fallback) == "ok":
            selected = fallback
        selected_parse = selected["parse"]
        assert isinstance(selected_parse, Mapping)
        selected_value = (
            selected_parse.get("value")
            if selected_parse.get("status") == "ok"
            else None
        )
        if type(selected_value) is int:
            predictions[record.id] = selected_value
        rows.append(
            {
                "problem_id": record.id,
                "question_sha256": hashlib.sha256(
                    record.question_raw.encode("utf-8")
                ).hexdigest(),
                "primary": primary,
                "fallback": fallback,
                "selected_method_label": selected["method_label"],
                "selected_answer": selected_value,
            }
        )
    invalid_ids = tuple(record.id for record in materialized if record.id not in predictions)
    predictions_bytes = (
        json.dumps(predictions, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
        + "\n"
    ).encode("utf-8")
    artifact_without_hash = {
        "schema_version": "gate-b-frozen-evaluation-inference-v1",
        "status": "complete" if not invalid_ids else "incomplete_invalid_answers",
        "dataset_role": dataset_role,
        "leaderboard_or_test_used_for_training": False,
        "leaderboard_or_test_used_for_method_selection": False,
        "internet_or_external_api_used": False,
        "model_id": config.model_id,
        "revision": config.revision,
        "config_sha256": config.sha256,
        "decoding_policy": config.decoding_policy.as_dict(),
        "split_sha256": split_manifest.sha256,
        "source_groups_sha256": split_manifest.source_groups_sha256,
        "development_shard_sha256": methods.development_shard_sha256,
        "freeze_sha256": methods.freeze_sha256,
        "routing_policy": methods.routing_policy,
        "evaluation_file": {
            "path": str(evaluation_source),
            "size_bytes": evaluation_source.stat().st_size,
            "sha256": actual_digest,
        },
        "problem_count": len(materialized),
        "expected_ids_sha256": _ids_sha256(tuple(record.id for record in materialized)),
        "valid_prediction_count": len(predictions),
        "invalid_prediction_count": len(invalid_ids),
        "invalid_ids": list(invalid_ids),
        "primary_parser_status_counts": dict(sorted(primary_parser_counts.items())),
        "fallback_invocation_count": len(fallback_by_id),
        "fallback_parser_status_counts": dict(sorted(fallback_parser_counts.items())),
        "predictions_file": {
            "name": predictions_target.name,
            "size_bytes": len(predictions_bytes),
            "sha256": hashlib.sha256(predictions_bytes).hexdigest(),
            "missing_invalid_answers_are_omitted": True,
            "silent_zero_fallback": False,
        },
        "model_residency_policy": "sequential_single_backend",
        "records": rows,
    }
    payload_sha = hashlib.sha256(canonical_json_bytes(artifact_without_hash)).hexdigest()
    artifact_bytes = (
        json.dumps(
            {**artifact_without_hash, "payload_sha256": payload_sha},
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")
    _publish_atomic_pair(
        predictions_target,
        predictions_bytes,
        artifact_target,
        artifact_bytes,
    )
    return PredictionArtifactWriteResult(
        artifact_path=str(artifact_target),
        predictions_path=str(predictions_target),
        artifact_sha256=hashlib.sha256(artifact_bytes).hexdigest(),
        predictions_sha256=hashlib.sha256(predictions_bytes).hexdigest(),
        problem_count=len(materialized),
        invalid_count=len(invalid_ids),
    )


def _generate_one(
    record: MathRecord,
    *,
    backend: PredictionGenerationBackend,
    checkpoint_sha256: str,
    method_label: str,
    dataset_role: str,
    config: GateBConfig,
    clock_ns: Callable[[], int],
) -> dict[str, object]:
    messages = (
        ChatMessage("system", config.system_prompt),
        ChatMessage("user", record.question_raw),
    )
    prompt_payload = {
        "messages": [message.as_dict() for message in messages],
        "add_generation_prompt": True,
    }
    prompt_sha256 = hashlib.sha256(canonical_json_bytes(prompt_payload)).hexdigest()
    seed = deterministic_seed(
        record.id,
        0,
        salt=f"gate-b-evaluation-v1:{dataset_role}:{config.seed}:{config.sha256}",
    )
    request = GenerationRequest(
        problem_id=record.id,
        messages=messages,
        seed=seed,
        sample_index=0,
        model_id=config.model_id,
        revision=config.revision,
        route=config.route,
        prompt_sha256=prompt_sha256,
        config_sha256=config.sha256,
        decoding_policy=config.decoding_policy,
    )
    started = clock_ns()
    generated = backend.generate(request)
    finished = clock_ns()
    if not isinstance(generated, GenerationResult):
        raise TypeError("evaluation backend must return GenerationResult")
    if finished < started:
        raise GateBValidationError("monotonic clock moved backwards during evaluation")
    latency_ms = (finished - started) / 1_000_000
    if not math.isfinite(latency_ms):  # pragma: no cover - integer clock invariant
        raise GateBValidationError("evaluation latency must be finite")
    parsed = parse_answer(generated.text)
    return {
        "method_label": method_label,
        "checkpoint_sha256": checkpoint_sha256,
        "seed": seed,
        "prompt_sha256": prompt_sha256,
        "raw_completion": generated.text,
        "raw_completion_sha256": hashlib.sha256(generated.text.encode("utf-8")).hexdigest(),
        "parse": asdict(parsed),
        "finish_reason": generated.finish_reason,
        "input_token_count": generated.input_token_count,
        "output_token_count": generated.output_token_count,
        "latency_ms": latency_ms,
        "peak_vram_allocated_bytes": generated.peak_vram_allocated_bytes,
    }


def _validate_evaluation_records(records: Sequence[MathRecord]) -> None:
    if not records or any(not isinstance(record, MathRecord) for record in records):
        raise GateBValidationError("evaluation records must contain MathRecord values")
    identifiers = tuple(record.id for record in records)
    if len(set(identifiers)) != len(identifiers):
        raise GateBValidationError("evaluation records contain duplicate IDs")
    for record in records:
        if not record.id or not record.question_raw.strip():
            raise GateBValidationError("evaluation IDs and questions must be non-empty")
        if record.answer is not None or record.answer_raw not in (None, ""):
            raise GateBValidationError("leaderboard/test answers must be absent during inference")


def _require_backend_checkpoint(
    backend: PredictionGenerationBackend, expected: str, role: str
) -> None:
    if getattr(backend, "checkpoint_sha256", None) != expected:
        raise GateBValidationError(f"{role} backend does not match the frozen checkpoint")


def _require_closeable_primary(backend: PredictionGenerationBackend) -> None:
    if not callable(getattr(backend, "close", None)):
        raise GateBValidationError(
            "fallback routing requires a closeable primary backend for single-model VRAM"
        )


def _close_backend(backend: PredictionGenerationBackend) -> None:
    close = getattr(backend, "close", None)
    if not callable(close):  # pragma: no cover - checked before generation
        raise GateBValidationError("primary backend cannot be released before fallback")
    close()


def _parse_status(result: Mapping[str, object]) -> str:
    parsed = result.get("parse")
    if not isinstance(parsed, Mapping):  # pragma: no cover - internal result invariant
        raise GateBValidationError("generation result lacks parser evidence")
    return str(parsed.get("status"))


def _validated_regular_file(path: str | Path, label: str) -> Path:
    raw = Path(path)
    if raw.is_symlink():
        raise GateBValidationError(f"{label} refuses symlinks")
    source = raw.resolve(strict=True)
    if not source.is_file():
        raise GateBValidationError(f"{label} must be a regular file")
    return source


def _validated_pair_targets(
    artifact_path: str | Path, predictions_path: str | Path
) -> tuple[Path, Path]:
    raw_values = (Path(artifact_path), Path(predictions_path))
    targets: list[Path] = []
    for raw in raw_values:
        if raw.is_symlink() or raw.exists() or raw.parent.is_symlink():
            raise GateBValidationError("evaluation outputs must be new non-symlink paths")
        target = raw.resolve(strict=False)
        if not target.parent.resolve(strict=True).is_dir():
            raise GateBValidationError("evaluation output parent must be a directory")
        targets.append(target)
    if targets[0] == targets[1]:
        raise GateBValidationError("artifact and predictions output paths must differ")
    if targets[0].parent != targets[1].parent:
        raise GateBValidationError("artifact and predictions must share one directory")
    return targets[0], targets[1]


def _publish_atomic_pair(
    first: Path, first_bytes: bytes, second: Path, second_bytes: bytes
) -> None:
    temporaries: list[Path] = []
    published: list[tuple[Path, Path]] = []
    try:
        for target, payload in ((first, first_bytes), (second, second_bytes)):
            descriptor, raw_temporary = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
            )
            temporary = Path(raw_temporary)
            temporaries.append(temporary)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        for target, temporary in zip((first, second), temporaries, strict=True):
            try:
                os.link(temporary, target)
            except FileExistsError as exc:
                raise GateBValidationError(
                    f"refusing to overwrite evaluation output: {target}"
                ) from exc
            published.append((target, temporary))
        descriptor = os.open(
            first.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except Exception:
        for target, temporary in reversed(published):
            with suppress(FileNotFoundError):
                if target.stat().st_ino == temporary.stat().st_ino:
                    target.unlink()
        raise
    finally:
        for temporary in temporaries:
            with suppress(FileNotFoundError):
                temporary.unlink()


def _required_sha256(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise GateBValidationError(f"{field_name} must be a lowercase SHA-256")
    return value


def _ids_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256(canonical_json_bytes(list(values))).hexdigest()


__all__ = ["PredictionArtifactWriteResult", "run_frozen_evaluation_inference"]

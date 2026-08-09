"""One-shot locked-holdout evaluation coupled to its durable claim."""

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
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from .answers import parse_answer
from .data import MathRecord
from .gate_b import (
    DEFAULT_GATE_B_CONFIG,
    ChatMessage,
    GateBConfig,
    GateBValidationError,
    GenerationRequest,
    GenerationResult,
)
from .gate_b_selection import (
    HOLDOUT_ACCESS_ACKNOWLEDGEMENT,
    GateBSelectionWriteResult,
    authorize_locked_holdout_once,
    validate_frozen_selection_methods,
    validate_locked_holdout_access,
)
from .inference import deterministic_seed
from .provenance import canonical_json_bytes
from .splits import SplitManifest

_TRAIN_ID_RE = re.compile(r"train-\d{6}\Z")
CANONICAL_HOLDOUT_LEDGER_ROOT = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "analysis"
    / "locked-holdout-access-v1"
)


class HoldoutGenerationBackend(Protocol):
    @property
    def checkpoint_sha256(self) -> str: ...

    def generate(self, request: GenerationRequest) -> GenerationResult: ...


def evaluate_locked_holdout_once(
    records_loader: Callable[[], Iterable[MathRecord]],
    *,
    split_manifest: SplitManifest,
    excluded_ids: Iterable[str],
    train_file_sha256: str,
    exclusions_file_sha256: str,
    excluded_ids_sha256: str,
    split_artifact_sha256: str,
    development_shard_sha256: str,
    fold: int,
    freeze_artifact: str | Path,
    primary_backend: HoldoutGenerationBackend,
    fallback_backend: HoldoutGenerationBackend | None,
    output_path: str | Path,
    holdout_acknowledgement: str,
    config: GateBConfig = DEFAULT_GATE_B_CONFIG,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> GateBSelectionWriteResult:
    """Claim, evaluate frozen method(s), and publish one aggregate/raw artifact.

    All fallible static inputs, output destinations, backend identities, and the
    frozen development decision are checked before the durable claim.  From the
    claim onward, any failure consumes the one permitted evaluation attempt.
    """

    if config != DEFAULT_GATE_B_CONFIG:
        raise GateBValidationError("locked holdout requires the locked Gate B config")
    if holdout_acknowledgement != HOLDOUT_ACCESS_ACKNOWLEDGEMENT:
        raise GateBValidationError(
            "explicit one-time locked-holdout acknowledgement is required"
        )
    if not callable(records_loader):
        raise TypeError("records_loader must be callable")
    if not isinstance(split_manifest, SplitManifest):
        raise TypeError("split_manifest must be a SplitManifest")
    split_manifest.validate()
    target = _validated_new_output(output_path)
    exclusions = tuple(excluded_ids)
    if len(exclusions) != len(set(exclusions)):
        raise GateBValidationError("excluded_ids must not contain duplicates")
    if _ids_sha256(tuple(sorted(exclusions))) != excluded_ids_sha256:
        raise GateBValidationError("excluded_ids do not match excluded_ids_sha256")
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
        primary_backend,
        methods.primary_checkpoint_sha256,
        "primary",
    )
    if methods.fallback_checkpoint_sha256 is None:
        if fallback_backend is not None:
            raise GateBValidationError("freeze has no fallback but a fallback backend was supplied")
    else:
        if fallback_backend is None:
            raise GateBValidationError("freeze requires a fallback backend")
        if primary_backend is fallback_backend:
            raise GateBValidationError("primary and fallback backends must be distinct")
        _require_closeable_primary(primary_backend)
        _require_backend_checkpoint(
            fallback_backend,
            methods.fallback_checkpoint_sha256,
            "fallback",
        )
    ledger = _prepare_canonical_ledger()

    receipt = authorize_locked_holdout_once(
        freeze_artifact,
        split_manifest=split_manifest,
        excluded_ids=exclusions,
        acknowledgement=HOLDOUT_ACCESS_ACKNOWLEDGEMENT,
        ledger_root=ledger,
        now=now,
    )
    access = validate_locked_holdout_access(
        receipt.path,
        freeze_artifact=freeze_artifact,
        split_manifest=split_manifest,
        excluded_ids=exclusions,
    )
    if (
        access.primary_label != methods.primary_label
        or access.primary_checkpoint_sha256 != methods.primary_checkpoint_sha256
        or access.fallback_label != methods.fallback_label
        or access.fallback_checkpoint_sha256 != methods.fallback_checkpoint_sha256
        or access.routing_policy != methods.routing_policy
    ):  # pragma: no cover - receipt/freeze validation already proves this
        raise GateBValidationError("holdout receipt changed the frozen method policy")

    materialized = tuple(records_loader())
    _validate_unopened_train_universe(materialized, split_manifest)
    by_id = {record.id: record for record in materialized}
    selected = tuple(by_id[problem_id] for problem_id in access.eligible_ids)
    if any(type(record.answer) is not int for record in selected):
        raise GateBValidationError("locked holdout contains a missing or non-integer answer")
    policy_result = _evaluate_frozen_policy(
        selected,
        primary_label=methods.primary_label,
        primary_checkpoint_sha256=methods.primary_checkpoint_sha256,
        primary_backend=primary_backend,
        fallback_label=methods.fallback_label,
        fallback_checkpoint_sha256=methods.fallback_checkpoint_sha256,
        fallback_backend=fallback_backend,
        split_manifest=split_manifest,
        receipt_sha256=access.receipt_sha256,
        config=config,
        clock_ns=clock_ns,
    )
    completed = now()
    if not isinstance(completed, datetime) or completed.tzinfo is None:
        raise GateBValidationError("holdout completion timestamp must be timezone-aware")
    payload_without_hash = {
        "schema_version": "gate-b-locked-holdout-evaluation-v2",
        "status": "complete",
        "evaluated_at_utc": completed.astimezone(UTC).isoformat(),
        "selection_frozen_before_access": True,
        "model_selection_after_this_evaluation_forbidden": True,
        "leaderboard_or_test_used": False,
        "model_id": config.model_id,
        "revision": config.revision,
        "config_sha256": config.sha256,
        "split_version": split_manifest.version,
        "split_sha256": split_manifest.sha256,
        "source_groups_sha256": split_manifest.source_groups_sha256,
        "development_shard_sha256": methods.development_shard_sha256,
        "eligible_holdout_count": len(access.eligible_ids),
        "eligible_holdout_ids_sha256": _ids_sha256(access.eligible_ids),
        "freeze_sha256": access.freeze_sha256,
        "routing_policy": methods.routing_policy,
        "claim_receipt": {
            "path": access.receipt_path,
            "sha256": access.receipt_sha256,
        },
        "frozen_methods": {
            "primary": {
                "label": methods.primary_label,
                "checkpoint_sha256": methods.primary_checkpoint_sha256,
            },
            "fallback": (
                None
                if methods.fallback_label is None
                else {
                    "label": methods.fallback_label,
                    "checkpoint_sha256": methods.fallback_checkpoint_sha256,
                }
            ),
        },
        "policy_result": policy_result,
    }
    return _write_hashed_json_noreplace(target, payload_without_hash)


def _evaluate_frozen_policy(
    records: Sequence[MathRecord],
    *,
    primary_label: str,
    primary_checkpoint_sha256: str,
    primary_backend: HoldoutGenerationBackend,
    fallback_label: str | None,
    fallback_checkpoint_sha256: str | None,
    fallback_backend: HoldoutGenerationBackend | None,
    split_manifest: SplitManifest,
    receipt_sha256: str,
    config: GateBConfig,
    clock_ns: Callable[[], int],
) -> dict[str, object]:
    primary_parser_counts: Counter[str] = Counter()
    primary_by_id: dict[str, dict[str, object]] = {}
    for record in records:
        assert type(record.answer) is int
        primary = _generate_method_result(
            record,
            label=primary_label,
            checkpoint_sha256=primary_checkpoint_sha256,
            backend=primary_backend,
            receipt_sha256=receipt_sha256,
            config=config,
            clock_ns=clock_ns,
        )
        primary_parse = primary["parse"]
        assert isinstance(primary_parse, Mapping)
        primary_parser_counts[str(primary_parse["status"])] += 1
        primary_by_id[record.id] = primary

    invalid_primary_records = tuple(
        record
        for record in records
        if _parse_status(primary_by_id[record.id]) != "ok"
    )
    fallback_parser_counts: Counter[str] = Counter()
    fallback_by_id: dict[str, dict[str, object]] = {}
    if invalid_primary_records and fallback_backend is not None:
        _close_backend(primary_backend)
        for record in invalid_primary_records:
            fallback = _generate_method_result(
                record,
                label=str(fallback_label),
                checkpoint_sha256=str(fallback_checkpoint_sha256),
                backend=fallback_backend,
                receipt_sha256=receipt_sha256,
                config=config,
                clock_ns=clock_ns,
            )
            fallback_parse = fallback["parse"]
            assert isinstance(fallback_parse, Mapping)
            fallback_parser_counts[str(fallback_parse["status"])] += 1
            fallback_by_id[record.id] = fallback

    rows: list[dict[str, object]] = []
    exact_count = 0
    latency_total = 0.0
    peak_values: list[int] = []
    assignments = split_manifest.assignment_by_id()
    for record in records:
        primary = primary_by_id[record.id]
        fallback = fallback_by_id.get(record.id)
        selected = primary
        if fallback is not None and _parse_status(fallback) == "ok":
            selected = fallback
        selected_parse = selected["parse"]
        assert isinstance(selected_parse, Mapping)
        selected_value = (
            selected_parse.get("value") if selected_parse.get("status") == "ok" else None
        )
        exact = type(selected_value) is int and selected_value == record.answer
        exact_count += int(exact)
        for result in (primary, fallback):
            if result is None:
                continue
            latency_total += float(result["latency_ms"])
            peak = result["peak_vram_allocated_bytes"]
            if type(peak) is int:
                peak_values.append(peak)
        rows.append(
            {
                "problem_id": record.id,
                "group_id": assignments[record.id].group_id,
                "question_sha256": hashlib.sha256(
                    record.question_raw.encode("utf-8")
                ).hexdigest(),
                "reference_answer": record.answer,
                "primary": primary,
                "fallback": fallback,
                "selected_method_label": selected["method_label"],
                "selected_answer": selected_value,
                "exact_match": exact,
            }
        )
    return {
        "problem_count": len(rows),
        "exact_match_count": exact_count,
        "exact_match_accuracy": exact_count / len(rows),
        "primary_parser_status_counts": dict(sorted(primary_parser_counts.items())),
        "fallback_invocation_count": len(fallback_by_id),
        "fallback_parser_status_counts": dict(sorted(fallback_parser_counts.items())),
        "model_residency_policy": "sequential_single_backend",
        "latency_ms_total": latency_total,
        "peak_vram_allocated_bytes_max": max(peak_values) if peak_values else None,
        "records": rows,
    }


def _generate_method_result(
    record: MathRecord,
    *,
    label: str,
    checkpoint_sha256: str,
    backend: HoldoutGenerationBackend,
    receipt_sha256: str,
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
        salt=f"gate-b-locked-holdout-v2:{config.seed}:{config.sha256}",
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
        raise TypeError("holdout backend must return GenerationResult")
    if finished < started:
        raise GateBValidationError("monotonic clock moved backwards during holdout")
    latency_ms = (finished - started) / 1_000_000
    if not math.isfinite(latency_ms):  # pragma: no cover - integer clock invariant
        raise GateBValidationError("holdout latency must be finite")
    parsed = parse_answer(generated.text)
    return {
        "method_label": label,
        "checkpoint_sha256": checkpoint_sha256,
        "seed": seed,
        "prompt_sha256": prompt_sha256,
        "raw_completion": generated.text,
        "raw_completion_sha256": hashlib.sha256(
            generated.text.encode("utf-8")
        ).hexdigest(),
        "parse": asdict(parsed),
        "finish_reason": generated.finish_reason,
        "input_token_count": generated.input_token_count,
        "output_token_count": generated.output_token_count,
        "latency_ms": latency_ms,
        "peak_vram_allocated_bytes": generated.peak_vram_allocated_bytes,
        "receipt_sha256": receipt_sha256,
    }


def _validate_unopened_train_universe(
    records: Sequence[MathRecord], split_manifest: SplitManifest
) -> None:
    if not records or any(not isinstance(record, MathRecord) for record in records):
        raise GateBValidationError("holdout evaluator requires organizer MathRecord values")
    identifiers = tuple(record.id for record in records)
    if len(set(identifiers)) != len(identifiers):
        raise GateBValidationError("organizer train universe contains duplicate IDs")
    split_ids = tuple(assignment.record_id for assignment in split_manifest.assignments)
    if tuple(sorted(identifiers)) != split_ids:
        raise GateBValidationError("organizer train universe does not match the locked split")
    if any(_TRAIN_ID_RE.fullmatch(problem_id) is None for problem_id in identifiers):
        raise GateBValidationError("holdout evaluator accepts organizer train IDs only")


def _require_backend_checkpoint(
    backend: HoldoutGenerationBackend, expected: str, role: str
) -> None:
    actual = getattr(backend, "checkpoint_sha256", None)
    if actual != expected:
        raise GateBValidationError(f"{role} backend does not match the frozen checkpoint")


def _require_closeable_primary(backend: HoldoutGenerationBackend) -> None:
    if not callable(getattr(backend, "close", None)):
        raise GateBValidationError(
            "fallback routing requires a closeable primary backend for single-model VRAM"
        )


def _close_backend(backend: HoldoutGenerationBackend) -> None:
    close = getattr(backend, "close", None)
    if not callable(close):  # pragma: no cover - checked before the holdout claim
        raise GateBValidationError("primary backend cannot be released before fallback")
    close()


def _parse_status(result: Mapping[str, object]) -> str:
    parsed = result.get("parse")
    if not isinstance(parsed, Mapping):  # pragma: no cover - internal result invariant
        raise GateBValidationError("generation result lacks parser evidence")
    return str(parsed.get("status"))


def _prepare_canonical_ledger() -> Path:
    root = CANONICAL_HOLDOUT_LEDGER_ROOT
    parent = root.parent
    if parent.is_symlink() or not parent.resolve(strict=True).is_dir():
        raise GateBValidationError("canonical holdout ledger parent is unsafe")
    if root.is_symlink():
        raise GateBValidationError("canonical holdout ledger refuses symlinks")
    root.mkdir(mode=0o700, exist_ok=True)
    resolved = root.resolve(strict=True)
    if resolved != root or not resolved.is_dir():
        raise GateBValidationError("canonical holdout ledger path changed unexpectedly")
    return resolved


def _validated_new_output(path: str | Path) -> Path:
    raw = Path(path)
    if raw.is_symlink() or raw.exists() or raw.parent.is_symlink():
        raise GateBValidationError("holdout evaluation output must be a new non-symlink path")
    target = raw.resolve(strict=False)
    if not target.parent.resolve(strict=True).is_dir():
        raise GateBValidationError("holdout evaluation output parent must be a directory")
    return target


def _ids_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256(canonical_json_bytes(list(values))).hexdigest()


def _write_hashed_json_noreplace(
    target: Path, payload_without_hash: Mapping[str, object]
) -> GateBSelectionWriteResult:
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
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            raise GateBValidationError(
                f"refusing to overwrite holdout evaluation: {target}"
            ) from exc
        parent_descriptor = os.open(
            target.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()
    return GateBSelectionWriteResult(
        path=str(target),
        size_bytes=len(serialized),
        sha256=hashlib.sha256(serialized).hexdigest(),
        payload_sha256=payload_sha,
    )


__all__ = ["CANONICAL_HOLDOUT_LEDGER_ROOT", "evaluate_locked_holdout_once"]

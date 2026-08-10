"""Private, redacted parser-golden evidence for Gate B development outputs.

The raw generation JSONL is an experiment artifact derived from organizer train
questions.  It must remain private.  This module validates such an atomically
published development bundle, re-parses every completion, and emits only safe
aggregate parser outcomes.  It never serializes IDs, answers, questions, raw
completions, completion hashes, or parsed integer values.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .answers import AnswerParseResult, parse_answer
from .gate_b import PINNED_MODEL_REVISION
from .model_preflight import OFFICIAL_MODEL_ID
from .provenance import canonical_json_bytes, sha256_file

PARSER_GOLDEN_AUDIT_SCHEMA = "gate-b-parser-golden-audit-v1"
PARSER_RESCORE_AUDIT_SCHEMA = "gate-b-parser-rescore-audit-v1"
_DEVELOPMENT_RECORD_SCHEMA = "gate-b1-development-baseline-v2"
_DEVELOPMENT_MANIFEST_SCHEMA = "gate-b1-development-run-v2"
_SAFE_PARSE_SOURCES = frozenset({"final_answer", "boxed", "hashes", "answer", "fallback"})
_SAFE_PARSE_STATUSES = frozenset({"ok", "invalid", "conflict"})
_SAFE_REASON_CODES = frozenset(
    {
        "parsed_unambiguous_integer",
        "empty_completion",
        "no_numeric_candidate",
        "empty_numeric_payload",
        "non_finite_value",
        "scientific_notation_not_allowed",
        "multiple_fraction_operators",
        "fraction_must_contain_two_plain_integers",
        "zero_denominator",
        "non_integral_fraction",
        "non_integral_decimal",
        "invalid_comma_grouping",
        "integer_outside_runtime_safety_limit",
        "multiple_values_in_payload",
        "unsupported_numeric_payload",
        "boxed_missing_open_brace",
        "boxed_unbalanced_braces",
        "empty_marker_payload",
    }
)


class ParserGoldenAuditError(ValueError):
    """Raised when a parser-golden input bundle is incomplete or unsafe."""


@dataclass(frozen=True, slots=True)
class ParserGoldenAuditWriteResult:
    """Metadata for a newly published redacted parser audit artifact."""

    path: str
    size_bytes: int
    sha256: str
    payload_sha256: str
    case_count: int


def audit_development_parser_golden(
    records_path: str | Path,
    manifest_path: str | Path,
    *,
    output_path: str | Path,
) -> ParserGoldenAuditWriteResult:
    """Write no-overwrite, redacted parser evidence from one development bundle.

    The caller supplies an existing atomic ``gate-b-development`` JSONL/manifest
    pair.  The pair must be a fold-validation, cross-validation run.  This
    intentionally rejects locked holdout and any other partition before a
    completion is parsed.
    """

    records_file = _require_regular_file(records_path, "development records JSONL")
    manifest_file = _require_regular_file(manifest_path, "development manifest")
    if records_file == manifest_file:
        raise ParserGoldenAuditError("records and manifest paths must be different")

    manifest = _load_json_object(manifest_file, "development manifest")
    _validate_manifest(manifest, records_file)
    rows = _load_and_validate_rows(records_file, manifest)
    payload = _redacted_payload(records_file, manifest_file, manifest, rows)
    payload_bytes = _json_bytes(payload)
    target = _write_json_noreplace(output_path, payload_bytes)
    return ParserGoldenAuditWriteResult(
        path=str(target),
        size_bytes=len(payload_bytes),
        sha256=hashlib.sha256(payload_bytes).hexdigest(),
        payload_sha256=hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
        case_count=len(payload["observed_parser_outcomes"]),
    )


def audit_development_parser_rescore(
    records_path: str | Path,
    manifest_path: str | Path,
    *,
    output_path: str | Path,
) -> ParserGoldenAuditWriteResult:
    """Compare stored and current parser outcomes without mutating run evidence.

    The source bundle must still be an intact development fold.  Unlike the
    golden audit, this diagnostic deliberately permits a stored/current parser
    mismatch, reports only aggregate counts, and marks itself ineligible for
    selection.  A current-source model run is still required before freeze.
    """

    records_file = _require_regular_file(records_path, "development records JSONL")
    manifest_file = _require_regular_file(manifest_path, "development manifest")
    if records_file == manifest_file:
        raise ParserGoldenAuditError("records and manifest paths must be different")
    manifest = _load_json_object(manifest_file, "development manifest")
    _validate_manifest(manifest, records_file)
    stored, current, stored_exact, current_exact = _load_parser_rescore_rows(
        records_file, manifest
    )
    changed = sum(before != after for before, after in zip(stored, current, strict=True))
    payload = {
        "schema_version": PARSER_RESCORE_AUDIT_SCHEMA,
        "input_evidence": _input_evidence(records_file, manifest_file, manifest),
        "stored": {
            "parser_status_counts": _status_counts(stored),
            "exact_match_count": stored_exact,
            "exact_match_accuracy": stored_exact / len(stored),
        },
        "current": {
            "parser_status_counts": _status_counts(current),
            "exact_match_count": current_exact,
            "exact_match_accuracy": current_exact / len(current),
        },
        "changed_parser_result_count": changed,
        "exact_match_delta": current_exact - stored_exact,
        "selection_eligible": False,
        "requires_current_source_run_before_freeze": True,
        "privacy_contract": _privacy_contract(),
    }
    payload_bytes = _json_bytes(payload)
    target = _write_json_noreplace(output_path, payload_bytes)
    return ParserGoldenAuditWriteResult(
        path=str(target),
        size_bytes=len(payload_bytes),
        sha256=hashlib.sha256(payload_bytes).hexdigest(),
        payload_sha256=hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
        case_count=changed,
    )


def _validate_manifest(manifest: dict[str, Any], records_file: Path) -> None:
    if manifest.get("schema_version") != _DEVELOPMENT_MANIFEST_SCHEMA:
        raise ParserGoldenAuditError("development manifest has an unsupported schema_version")
    if manifest.get("records_file") != records_file.name:
        raise ParserGoldenAuditError("development manifest records_file does not match input")
    if manifest.get("records_bytes") != records_file.stat().st_size:
        raise ParserGoldenAuditError("development manifest records_bytes does not match input")
    _require_sha256(manifest.get("records_sha256"), "development manifest records_sha256")
    if manifest["records_sha256"] != sha256_file(records_file):
        raise ParserGoldenAuditError("development manifest records_sha256 does not match input")
    for field_name in ("record_count", "problem_count", "samples_per_problem"):
        _require_positive_int(manifest.get(field_name), f"development manifest {field_name}")
    if manifest["record_count"] != manifest["problem_count"] * manifest["samples_per_problem"]:
        raise ParserGoldenAuditError("development manifest has inconsistent record accounting")
    if manifest.get("partition") != "fold_validation":
        raise ParserGoldenAuditError("parser golden audit accepts fold_validation artifacts only")
    if manifest.get("split_partition") != "cross_validation":
        raise ParserGoldenAuditError("parser golden audit refuses non-development partitions")
    _require_nonnegative_int(manifest.get("fold"), "development manifest fold")
    for field_name in (
        "model_id",
        "revision",
        "route",
        "config_sha256",
        "checkpoint_sha256",
        "split_version",
        "split_sha256",
        "source_groups_sha256",
        "eligibility_ids_sha256",
    ):
        _require_trimmed_string(manifest.get(field_name), f"development manifest {field_name}")
    if manifest["model_id"] != OFFICIAL_MODEL_ID:
        raise ParserGoldenAuditError(
            "development manifest does not bind the official model"
        )
    if manifest["revision"] != PINNED_MODEL_REVISION:
        raise ParserGoldenAuditError(
            "development manifest does not bind the pinned revision"
        )
    if manifest["route"] != "direct_answer":
        raise ParserGoldenAuditError(
            "development manifest route must be direct_answer"
        )
    for field_name in ("execution_evidence", "generation_evidence"):
        if not isinstance(manifest.get(field_name), dict):
            raise ParserGoldenAuditError(
                f"development manifest {field_name} is required for provenance-complete runs"
            )
    parser_status_counts = manifest.get("parser_status_counts")
    if not isinstance(parser_status_counts, dict) or not parser_status_counts:
        raise ParserGoldenAuditError(
            "development manifest parser_status_counts must be a non-empty object"
        )
    for status, count in parser_status_counts.items():
        if status not in _SAFE_PARSE_STATUSES:
            raise ParserGoldenAuditError(
                "development manifest parser_status_counts has an unsupported status"
            )
        _require_nonnegative_int(
            count, f"development manifest parser_status_counts[{status!r}]"
        )
    if sum(parser_status_counts.values()) != manifest["record_count"]:
        raise ParserGoldenAuditError(
            "development manifest parser_status_counts does not sum to record_count"
        )
    exact_match_count = manifest.get("exact_match_count")
    _require_nonnegative_int(
        exact_match_count, "development manifest exact_match_count"
    )
    if exact_match_count > manifest["record_count"]:
        raise ParserGoldenAuditError(
            "development manifest exact_match_count exceeds record_count"
        )
    accuracy = manifest.get("exact_match_accuracy")
    if (
        isinstance(accuracy, bool)
        or not isinstance(accuracy, (int, float))
        or not math.isfinite(float(accuracy))
        or float(accuracy) != exact_match_count / manifest["record_count"]
    ):
        raise ParserGoldenAuditError(
            "development manifest exact_match_accuracy is inconsistent"
        )
    for field_name in (
        "config_sha256",
        "checkpoint_sha256",
        "split_sha256",
        "source_groups_sha256",
        "eligibility_ids_sha256",
    ):
        _require_sha256(manifest[field_name], f"development manifest {field_name}")


def _load_and_validate_rows(
    records_file: Path, manifest: dict[str, Any]
) -> list[AnswerParseResult]:
    try:
        lines = records_file.read_text(encoding="utf-8", errors="strict").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ParserGoldenAuditError(f"could not read development records: {exc}") from exc
    if len(lines) != manifest["record_count"]:
        raise ParserGoldenAuditError("development records line count does not match manifest")

    parsed_results: list[AnswerParseResult] = []
    problem_ids: set[str] = set()
    observed_statuses: Counter[str] = Counter()
    observed_exact_matches = 0
    for index, line in enumerate(lines, start=1):
        row = _load_json_line(line, index)
        result = _validate_row(row, manifest, index)
        problem_id = row.get("problem_id")
        assert isinstance(problem_id, str)  # checked by _validate_row
        problem_ids.add(problem_id)
        parsed_results.append(result)
        observed_statuses[result.status] += 1
        observed_exact_matches += row["exact_match"]

    if len(problem_ids) != manifest["problem_count"]:
        raise ParserGoldenAuditError("development records problem_count does not match manifest")
    expected_statuses = manifest.get("parser_status_counts")
    if not isinstance(expected_statuses, dict):
        raise ParserGoldenAuditError("development manifest parser_status_counts must be an object")
    normalized_expected = {key: value for key, value in sorted(expected_statuses.items())}
    normalized_observed = dict(sorted(observed_statuses.items()))
    if normalized_expected != normalized_observed:
        raise ParserGoldenAuditError(
            "development manifest parser_status_counts does not match rows"
        )
    if manifest["exact_match_count"] != observed_exact_matches:
        raise ParserGoldenAuditError(
            "development manifest exact_match_count does not match rows"
        )
    return parsed_results


def _load_parser_rescore_rows(
    records_file: Path, manifest: dict[str, Any]
) -> tuple[list[AnswerParseResult], list[AnswerParseResult], int, int]:
    try:
        lines = records_file.read_text(encoding="utf-8", errors="strict").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ParserGoldenAuditError(f"could not read development records: {exc}") from exc
    if len(lines) != manifest["record_count"]:
        raise ParserGoldenAuditError("development records line count does not match manifest")

    stored_results: list[AnswerParseResult] = []
    current_results: list[AnswerParseResult] = []
    problem_ids: set[str] = set()
    stored_exact = 0
    current_exact = 0
    for index, line in enumerate(lines, start=1):
        row = _load_json_line(line, index)
        stored, current, reference_answer = _validate_row_pair(row, manifest, index)
        problem_id = row["problem_id"]
        assert isinstance(problem_id, str)
        problem_ids.add(problem_id)
        stored_results.append(stored)
        current_results.append(current)
        stored_match = stored.ok and stored.value == reference_answer
        current_match = current.ok and current.value == reference_answer
        if row.get("exact_match") is not stored_match:
            raise ParserGoldenAuditError(
                f"development row {index} stored exact_match is inconsistent"
            )
        stored_exact += stored_match
        current_exact += current_match

    if len(problem_ids) != manifest["problem_count"]:
        raise ParserGoldenAuditError("development records problem_count does not match manifest")
    if _status_counts(stored_results) != dict(
        sorted(manifest["parser_status_counts"].items())
    ):
        raise ParserGoldenAuditError(
            "development manifest parser_status_counts does not match stored rows"
        )
    if manifest.get("exact_match_count") != stored_exact:
        raise ParserGoldenAuditError(
            "development manifest exact_match_count does not match stored rows"
        )
    return stored_results, current_results, stored_exact, current_exact


def _validate_row(
    row: dict[str, Any], manifest: dict[str, Any], index: int
) -> AnswerParseResult:
    actual, expected, reference_answer = _validate_row_pair(row, manifest, index)
    if actual != expected:
        raise ParserGoldenAuditError(
            f"development row {index} stored parser result does not match completion"
        )
    stored_match = actual.ok and actual.value == reference_answer
    if row["exact_match"] is not stored_match:
        raise ParserGoldenAuditError(
            f"development row {index} stored exact_match is inconsistent"
        )
    return actual


def _validate_row_pair(
    row: dict[str, Any], manifest: dict[str, Any], index: int
) -> tuple[AnswerParseResult, AnswerParseResult, int]:
    prefix = f"development row {index}"
    if row.get("schema_version") != _DEVELOPMENT_RECORD_SCHEMA:
        raise ParserGoldenAuditError(f"{prefix} has an unsupported schema_version")
    _require_trimmed_string(row.get("problem_id"), f"{prefix} problem_id")
    if row.get("partition") != "fold_validation":
        raise ParserGoldenAuditError(f"{prefix} is not a fold_validation record")
    if row.get("split_partition") != "cross_validation":
        raise ParserGoldenAuditError(f"{prefix} is not a cross_validation record")
    if row.get("fold") != manifest["fold"]:
        raise ParserGoldenAuditError(f"{prefix} fold does not match manifest")
    for field_name in (
        "model_id",
        "revision",
        "route",
        "config_sha256",
        "checkpoint_sha256",
        "split_version",
        "split_sha256",
        "source_groups_sha256",
        "eligibility_ids_sha256",
    ):
        if row.get(field_name) != manifest[field_name]:
            raise ParserGoldenAuditError(f"{prefix} {field_name} does not match manifest")

    completion = row.get("raw_completion")
    if not isinstance(completion, str):
        raise ParserGoldenAuditError(f"{prefix} raw_completion must be a string")
    completion_sha256 = row.get("raw_completion_sha256")
    _require_sha256(completion_sha256, f"{prefix} raw_completion_sha256")
    if completion_sha256 != hashlib.sha256(completion.encode("utf-8")).hexdigest():
        raise ParserGoldenAuditError(f"{prefix} raw_completion_sha256 does not match")
    expected = parse_answer(completion)
    actual = _parse_result_from_row(row.get("parse"), prefix)
    reference_answer = row.get("reference_answer")
    if isinstance(reference_answer, bool) or not isinstance(reference_answer, int):
        raise ParserGoldenAuditError(f"{prefix} reference_answer must be an integer")
    if type(row.get("exact_match")) is not bool:
        raise ParserGoldenAuditError(f"{prefix} exact_match must be boolean")
    return actual, expected, reference_answer


def _parse_result_from_row(value: object, prefix: str) -> AnswerParseResult:
    if not isinstance(value, dict) or set(value) != {"status", "value", "source", "reason"}:
        raise ParserGoldenAuditError(f"{prefix} parse must be an exact parser-result object")
    status = value["status"]
    source = value["source"]
    reason = value["reason"]
    parsed_value = value["value"]
    if status not in _SAFE_PARSE_STATUSES:
        raise ParserGoldenAuditError(f"{prefix} parse status is unsupported")
    if source is not None and source not in _SAFE_PARSE_SOURCES:
        raise ParserGoldenAuditError(f"{prefix} parse source is unsupported")
    _require_trimmed_string(reason, f"{prefix} parse reason")
    if parsed_value is not None and (
        isinstance(parsed_value, bool) or not isinstance(parsed_value, int)
    ):
        raise ParserGoldenAuditError(f"{prefix} parse value must be an integer or null")
    if status == "ok" and parsed_value is None:
        raise ParserGoldenAuditError(f"{prefix} successful parse is missing its value")
    if status != "ok" and parsed_value is not None:
        raise ParserGoldenAuditError(f"{prefix} unsuccessful parse has a value")
    return AnswerParseResult(status, parsed_value, source, reason)


def _redacted_payload(
    records_file: Path,
    manifest_file: Path,
    manifest: dict[str, Any],
    results: list[AnswerParseResult],
) -> dict[str, Any]:
    counts: Counter[tuple[str, str | None, str]] = Counter(
        (result.status, result.source, _safe_reason_code(result.reason)) for result in results
    )
    outcomes = [
        {
            "status": status,
            "source": source,
            "reason_code": reason_code,
            "count": count,
        }
        for (status, source, reason_code), count in sorted(
            counts.items(), key=lambda item: (item[0][0], str(item[0][1]), item[0][2])
        )
    ]
    return {
        "schema_version": PARSER_GOLDEN_AUDIT_SCHEMA,
        "input_evidence": _input_evidence(records_file, manifest_file, manifest),
        "observed_parser_outcomes": outcomes,
        "privacy_contract": _privacy_contract(),
    }


def _input_evidence(
    records_file: Path, manifest_file: Path, manifest: dict[str, Any]
) -> dict[str, Any]:
    return {
        "records_sha256": sha256_file(records_file),
        "manifest_sha256": sha256_file(manifest_file),
        "record_count": manifest["record_count"],
        "problem_count": manifest["problem_count"],
        "fold": manifest["fold"],
        "model_id": manifest["model_id"],
        "revision": manifest["revision"],
        "route": manifest["route"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "config_sha256": manifest["config_sha256"],
        "split_sha256": manifest["split_sha256"],
        "eligibility_ids_sha256": manifest["eligibility_ids_sha256"],
        "partition": "fold_validation",
        "split_partition": "cross_validation",
    }


def _privacy_contract() -> dict[str, bool]:
    return {
        "raw_completion_serialized": False,
        "raw_completion_sha256_serialized": False,
        "problem_id_serialized": False,
        "question_serialized": False,
        "reference_answer_serialized": False,
        "parsed_integer_value_serialized": False,
        "locked_holdout_accessed": False,
        "leaderboard_or_test_used": False,
    }


def _status_counts(results: list[AnswerParseResult]) -> dict[str, int]:
    return dict(sorted(Counter(result.status for result in results).items()))


def _safe_reason_code(reason: str) -> str:
    if reason.startswith("conflicting_marker_values:"):
        return "conflicting_marker_values"
    if reason.startswith("marker_") and ":" in reason:
        _marker, candidate_reason = reason.split(":", 1)
        return (
            f"marker:{candidate_reason}"
            if candidate_reason in _SAFE_REASON_CODES
            else "marker:unknown"
        )
    return reason if reason in _SAFE_REASON_CODES else "unknown_parser_reason"


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8", errors="strict"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise ParserGoldenAuditError(f"invalid {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ParserGoldenAuditError(f"{label} must be a JSON object")
    return value


def _load_json_line(line: str, index: int) -> dict[str, Any]:
    try:
        value = json.loads(
            line,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except ValueError as exc:
        raise ParserGoldenAuditError(f"development row {index} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ParserGoldenAuditError(f"development row {index} must be a JSON object")
    return value


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} is not allowed")


def _require_regular_file(path: str | Path, label: str) -> Path:
    candidate = Path(path)
    try:
        metadata = os.lstat(candidate)
    except OSError as exc:
        raise ParserGoldenAuditError(f"{label} is unavailable: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ParserGoldenAuditError(f"{label} must be a regular non-symlink file")
    return candidate.resolve(strict=True)


def _write_json_noreplace(path: str | Path, payload: bytes) -> Path:
    target = Path(path)
    parent = target.parent
    try:
        parent_metadata = os.lstat(parent)
    except OSError as exc:
        raise ParserGoldenAuditError(f"parser golden output parent is unavailable: {exc}") from exc
    if stat.S_ISLNK(parent_metadata.st_mode) or not stat.S_ISDIR(parent_metadata.st_mode):
        raise ParserGoldenAuditError("parser golden output parent must be a non-symlink directory")
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to overwrite parser golden artifact: {target}")

    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_name = handle.name
        os.link(temporary_name, target)
        _fsync_directory(parent)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite parser golden artifact: {target}") from exc
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return target.resolve(strict=True)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _require_trimmed_string(value: object, label: str) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ParserGoldenAuditError(f"{label} must be a non-empty trimmed string")


def _require_positive_int(value: object, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ParserGoldenAuditError(f"{label} must be a positive integer")


def _require_nonnegative_int(value: object, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ParserGoldenAuditError(f"{label} must be a non-negative integer")


def _require_sha256(value: object, label: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ParserGoldenAuditError(f"{label} must be a 64-hex SHA-256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ParserGoldenAuditError(f"{label} must be a 64-hex SHA-256") from exc


__all__ = [
    "PARSER_GOLDEN_AUDIT_SCHEMA",
    "PARSER_RESCORE_AUDIT_SCHEMA",
    "ParserGoldenAuditError",
    "ParserGoldenAuditWriteResult",
    "audit_development_parser_golden",
    "audit_development_parser_rescore",
]

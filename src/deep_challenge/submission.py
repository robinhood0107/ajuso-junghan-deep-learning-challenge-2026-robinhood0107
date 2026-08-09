"""Strict validation and reproducible generation of competition CSV files."""

from __future__ import annotations

import csv
import hashlib
import os
import re
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

from deep_challenge.answers import AnswerParseResult

_CANONICAL_INTEGER_RE = re.compile(r"(?:0|-[1-9]\d*|[1-9]\d*)\Z")
_MISSING = object()

FallbackResolver: TypeAlias = Callable[[str, object | None, str], int | str]


@dataclass(frozen=True, slots=True)
class SubmissionSchema:
    """Externally supplied two-column submission schema."""

    # The 2026-08-04 competition Rules snapshot names the submission identifier
    # column with an uppercase ``ID``.  Input datasets continue to use lowercase
    # ``id``; only the emitted/validated submission schema changes here.
    id_column: str = "ID"
    answer_column: str = "answer"

    def __post_init__(self) -> None:
        columns = (self.id_column, self.answer_column)
        if any(not isinstance(column, str) or not column for column in columns):
            raise ValueError("schema column names must be non-empty strings")
        if self.id_column == self.answer_column:
            raise ValueError("id and answer column names must be distinct")
        if any("\r" in column or "\n" in column for column in columns):
            raise ValueError("schema column names must not contain newlines")

    @property
    def header(self) -> tuple[str, str]:
        """Return the exact required CSV header in column order."""

        return (self.id_column, self.answer_column)


DEFAULT_SUBMISSION_SCHEMA = SubmissionSchema()


@dataclass(frozen=True, slots=True)
class SubmissionIssue:
    """One deterministic submission validation error."""

    code: str
    message: str
    row_number: int | None = None
    problem_id: str | None = None


@dataclass(frozen=True, slots=True)
class SubmissionValidationReport:
    """Complete validation report for an in-memory or CSV submission."""

    row_count: int
    expected_count: int
    issues: tuple[SubmissionIssue, ...]

    @property
    def valid(self) -> bool:
        """Return whether all required invariants hold."""

        return not self.issues

    @property
    def errors(self) -> tuple[str, ...]:
        """Return compact error strings for logs and command-line output."""

        return tuple(f"{issue.code}: {issue.message}" for issue in self.issues)


PredictionProvenance = Literal["prediction", "emergency_fallback"]


@dataclass(frozen=True, slots=True)
class ResolvedPrediction:
    """Canonical answer plus auditable prediction provenance."""

    problem_id: str
    answer: int
    provenance: PredictionProvenance
    reason: str | None


@dataclass(frozen=True, slots=True)
class SubmissionWriteResult:
    """Result of atomically writing and round-trip validating a CSV."""

    path: Path
    report: SubmissionValidationReport
    predictions: tuple[ResolvedPrediction, ...]
    size_bytes: int = 0
    sha256: str = ""

    @property
    def fallback_count(self) -> int:
        """Return the number of rows filled by an explicit emergency policy."""

        return sum(item.provenance == "emergency_fallback" for item in self.predictions)


class SubmissionGenerationError(ValueError):
    """Raised when predictions cannot form a complete, canonical submission."""


def validate_submission_rows(
    rows: Iterable[Mapping[str, object]],
    expected_ids: Sequence[str],
    *,
    schema: SubmissionSchema = DEFAULT_SUBMISSION_SCHEMA,
) -> SubmissionValidationReport:
    """Validate rows against exact IDs, order, schema, and integer syntax.

    Args:
        rows: Mappings containing exactly the externally supplied schema keys.
        expected_ids: Required IDs in required output order.
        schema: Exact header names and column order.

    Returns:
        A report containing every detected structural error.

    Raises:
        ValueError: If ``expected_ids`` itself is malformed.
    """

    expected = _validated_expected_ids(expected_ids)
    materialized = list(rows)
    issues: list[SubmissionIssue] = []
    observed_ids: list[str] = []

    for row_number, row in enumerate(materialized, start=2):
        if not isinstance(row, Mapping):
            issues.append(
                SubmissionIssue("row_type", "row must be a mapping", row_number=row_number)
            )
            continue
        keys = tuple(row.keys())
        if keys != schema.header:
            issues.append(
                SubmissionIssue(
                    "row_schema",
                    f"expected columns {schema.header!r}, found {keys!r}",
                    row_number=row_number,
                )
            )

        raw_id = row.get(schema.id_column)
        if raw_id is None or raw_id == "":
            issues.append(
                SubmissionIssue("null_id", "ID must not be null or empty", row_number=row_number)
            )
            continue
        if not isinstance(raw_id, str):
            issues.append(
                SubmissionIssue("id_type", "ID must be a string", row_number=row_number)
            )
            continue
        observed_ids.append(raw_id)

        raw_answer = row.get(schema.answer_column)
        canonical = _canonical_integer_text(raw_answer)
        if canonical is None:
            issues.append(
                SubmissionIssue(
                    "noncanonical_answer",
                    "answer must be a canonical base-10 integer (for example -2, 0, or 17)",
                    row_number=row_number,
                    problem_id=raw_id,
                )
            )

    _append_collection_issues(issues, materialized, observed_ids, expected)
    return SubmissionValidationReport(len(materialized), len(expected), tuple(issues))


def validate_submission_csv(
    path: str | os.PathLike[str],
    expected_ids: Sequence[str],
    *,
    schema: SubmissionSchema = DEFAULT_SUBMISSION_SCHEMA,
    encoding: str = "utf-8-sig",
) -> SubmissionValidationReport:
    """Parse and validate a CSV with strict width and exact-header checks."""

    expected = _validated_expected_ids(expected_ids)
    csv_path = Path(path)
    issues: list[SubmissionIssue] = []
    rows: list[dict[str, object]] = []

    try:
        with csv_path.open("r", encoding=encoding, newline="") as handle:
            reader = csv.reader(handle, strict=True)
            try:
                header = next(reader)
            except StopIteration:
                return SubmissionValidationReport(
                    0,
                    len(expected),
                    (SubmissionIssue("empty_file", "CSV has no header"),),
                )
            if tuple(header) != schema.header:
                issues.append(
                    SubmissionIssue(
                        "header_schema",
                        f"expected header {schema.header!r}, found {tuple(header)!r}",
                        row_number=1,
                    )
                )

            for row_number, values in enumerate(reader, start=2):
                if len(values) != 2:
                    issues.append(
                        SubmissionIssue(
                            "row_width",
                            f"expected 2 fields, found {len(values)}",
                            row_number=row_number,
                        )
                    )
                    continue
                rows.append({schema.id_column: values[0], schema.answer_column: values[1]})
    except (csv.Error, UnicodeError) as exc:
        issues.append(SubmissionIssue("csv_parse", f"CSV could not be parsed: {exc}"))
    except OSError as exc:
        issues.append(SubmissionIssue("csv_io", f"CSV could not be read: {exc}"))

    row_report = validate_submission_rows(rows, expected, schema=schema)
    issues.extend(row_report.issues)
    physical_row_count = len(rows) + sum(issue.code == "row_width" for issue in issues)
    return SubmissionValidationReport(physical_row_count, len(expected), tuple(issues))


def resolve_predictions(
    predictions: Mapping[str, object],
    expected_ids: Sequence[str],
    *,
    fallback_value: int | str | None = None,
    fallback_resolver: FallbackResolver | None = None,
) -> tuple[ResolvedPrediction, ...]:
    """Resolve predictions in expected order with optional explicit fallback.

    Missing and invalid predictions fail by default.  An emergency fallback is
    enabled only when the caller explicitly supplies exactly one of
    ``fallback_value`` or ``fallback_resolver``.  Every fallback row records
    provenance and its triggering reason; parser failures are never silently
    converted to zero.

    The resolver receives ``(problem_id, raw_prediction, reason)``.  A missing
    prediction is passed as ``None`` with reason ``"missing_prediction"``.
    """

    expected = _validated_expected_ids(expected_ids)
    if not isinstance(predictions, Mapping):
        raise TypeError("predictions must be a mapping from ID to prediction")
    if fallback_value is not None and fallback_resolver is not None:
        raise ValueError("fallback_value and fallback_resolver are mutually exclusive")

    unexpected = sorted(set(predictions) - set(expected))
    if unexpected:
        raise SubmissionGenerationError(f"unexpected prediction IDs: {unexpected!r}")

    static_fallback: int | None = None
    if fallback_value is not None:
        static_text = _canonical_integer_text(fallback_value)
        if static_text is None:
            raise ValueError("fallback_value must be a canonical integer")
        static_fallback = int(static_text)

    resolved: list[ResolvedPrediction] = []
    unresolved: list[str] = []
    for problem_id in expected:
        raw = predictions.get(problem_id, _MISSING)
        answer, failure_reason = _prediction_to_integer(raw)
        if failure_reason is None:
            assert answer is not None
            resolved.append(ResolvedPrediction(problem_id, answer, "prediction", None))
            continue

        fallback_answer: int | None = None
        if static_fallback is not None:
            fallback_answer = static_fallback
        elif fallback_resolver is not None:
            resolver_raw = None if raw is _MISSING else raw
            candidate = fallback_resolver(problem_id, resolver_raw, failure_reason)
            candidate_text = _canonical_integer_text(candidate)
            if candidate_text is None:
                raise SubmissionGenerationError(
                    f"fallback_resolver returned a noncanonical answer for {problem_id!r}"
                )
            fallback_answer = int(candidate_text)

        if fallback_answer is None:
            unresolved.append(f"{problem_id}:{failure_reason}")
            continue
        resolved.append(
            ResolvedPrediction(
                problem_id,
                fallback_answer,
                "emergency_fallback",
                failure_reason,
            )
        )

    if unresolved:
        raise SubmissionGenerationError(
            "submission has unresolved predictions; configure an explicit fallback if intended: "
            + ", ".join(unresolved)
        )
    return tuple(resolved)


def write_submission_csv(
    path: str | os.PathLike[str],
    predictions: Mapping[str, object],
    expected_ids: Sequence[str],
    *,
    schema: SubmissionSchema = DEFAULT_SUBMISSION_SCHEMA,
    fallback_value: int | str | None = None,
    fallback_resolver: FallbackResolver | None = None,
    overwrite: bool = False,
) -> SubmissionWriteResult:
    """Atomically write, re-read, and validate a complete submission CSV.

    The output contains only the schema's ID and answer columns.  Fallback
    provenance is returned in :class:`SubmissionWriteResult` so the caller can
    persist it in an experiment manifest without corrupting the competition
    schema.
    """

    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing submission: {target}")
    resolved = resolve_predictions(
        predictions,
        expected_ids,
        fallback_value=fallback_value,
        fallback_resolver=fallback_resolver,
    )
    expected = _validated_expected_ids(expected_ids)
    target.parent.mkdir(parents=True, exist_ok=True)

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(schema.header)
            writer.writerows((item.problem_id, str(item.answer)) for item in resolved)
            handle.flush()
            os.fsync(handle.fileno())

        report = validate_submission_csv(temporary_path, expected, schema=schema)
        if not report.valid:
            raise SubmissionGenerationError(
                "generated CSV failed round-trip validation: " + "; ".join(report.errors)
            )
        if overwrite:
            os.replace(temporary_path, target)
            temporary_path = None
        else:
            try:
                os.link(temporary_path, target)
            except FileExistsError:
                raise FileExistsError(
                    f"refusing to overwrite existing submission: {target}"
                ) from None
            temporary_path.unlink()
            temporary_path = None
        _fsync_directory(target.parent)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    size_bytes = target.stat().st_size
    sha256 = _sha256_file(target)
    return SubmissionWriteResult(target, report, resolved, size_bytes, sha256)


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prediction_to_integer(raw: object) -> tuple[int | None, str | None]:
    if raw is _MISSING:
        return None, "missing_prediction"
    if isinstance(raw, AnswerParseResult):
        if raw.ok and raw.value is not None:
            return raw.value, None
        return None, f"parser_{raw.status}:{raw.reason}"
    canonical = _canonical_integer_text(raw)
    if canonical is None:
        return None, "invalid_or_noncanonical_prediction"
    return int(canonical), None


def _canonical_integer_text(value: object) -> str | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return str(value)
    if not isinstance(value, str) or _CANONICAL_INTEGER_RE.fullmatch(value) is None:
        return None
    return value


def _validated_expected_ids(expected_ids: Sequence[str]) -> tuple[str, ...]:
    if isinstance(expected_ids, (str, bytes)):
        raise ValueError("expected_ids must be a sequence of individual ID strings")
    expected = tuple(expected_ids)
    if any(not isinstance(problem_id, str) or not problem_id for problem_id in expected):
        raise ValueError("expected IDs must be non-empty strings")
    if len(set(expected)) != len(expected):
        raise ValueError("expected IDs must be unique")
    return expected


def _append_collection_issues(
    issues: list[SubmissionIssue],
    rows: Sequence[Mapping[str, object]],
    observed_ids: Sequence[str],
    expected: Sequence[str],
) -> None:
    if len(rows) != len(expected):
        issues.append(
            SubmissionIssue(
                "row_count",
                f"expected {len(expected)} rows, found {len(rows)}",
            )
        )

    seen: set[str] = set()
    duplicates: list[str] = []
    for problem_id in observed_ids:
        if problem_id in seen and problem_id not in duplicates:
            duplicates.append(problem_id)
        seen.add(problem_id)
    if duplicates:
        issues.append(
            SubmissionIssue("duplicate_ids", f"duplicate IDs: {sorted(duplicates)!r}")
        )

    expected_set = set(expected)
    observed_set = set(observed_ids)
    missing = sorted(expected_set - observed_set)
    extra = sorted(observed_set - expected_set)
    if missing:
        issues.append(SubmissionIssue("missing_ids", f"missing IDs: {missing!r}"))
    if extra:
        issues.append(SubmissionIssue("extra_ids", f"unexpected IDs: {extra!r}"))
    if list(observed_ids) != list(expected) and not missing and not extra and not duplicates:
        issues.append(SubmissionIssue("id_order", "rows are not in the expected ID order"))


__all__ = [
    "DEFAULT_SUBMISSION_SCHEMA",
    "FallbackResolver",
    "PredictionProvenance",
    "ResolvedPrediction",
    "SubmissionGenerationError",
    "SubmissionIssue",
    "SubmissionSchema",
    "SubmissionValidationReport",
    "SubmissionWriteResult",
    "resolve_predictions",
    "validate_submission_csv",
    "validate_submission_rows",
    "write_submission_csv",
]

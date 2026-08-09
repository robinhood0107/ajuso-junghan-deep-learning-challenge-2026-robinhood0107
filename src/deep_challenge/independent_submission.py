"""Independent, minimal cross-check for final competition CSV artifacts.

This module deliberately does not import the primary data or submission
implementations.  It provides a second parser and invariant check so a shared
bug in the production writer/validator is less likely to survive rehearsal.
"""

from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

_INTEGER_RE = re.compile(r"(?:0|-[1-9]\d*|[1-9]\d*)\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_EXPECTED_HEADERS = {("id", "question"), ("id", "question", "answer")}
_SUBMISSION_HEADER = ("ID", "answer")


@dataclass(frozen=True, slots=True)
class IndependentSubmissionReport:
    """Result from the standalone CSV cross-check."""

    valid: bool
    row_count: int
    expected_count: int
    submission_sha256: str
    expected_sha256: str
    issues: tuple[str, ...]


def verify_submission_independently(
    submission_path: str | Path,
    expected_path: str | Path,
    *,
    expected_file_sha256: str,
) -> IndependentSubmissionReport:
    """Cross-check exact header, IDs, order, and canonical integer answers."""

    expected_digest = _required_sha256(expected_file_sha256)
    expected_source = _regular_file(expected_path, "expected evaluation CSV")
    actual_expected_digest = _sha256_file(expected_source)
    if actual_expected_digest != expected_digest:
        raise ValueError("expected evaluation CSV SHA-256 does not match")
    submission_source = _regular_file(submission_path, "submission CSV")
    expected_ids, expected_issues = _read_expected_ids(expected_source)
    submitted_ids, answer_issues, row_count = _read_submission(submission_source)
    issues = [*expected_issues, *answer_issues]
    if len(submitted_ids) != len(set(submitted_ids)):
        issues.append("submission IDs are not unique")
    if tuple(submitted_ids) != tuple(expected_ids):
        issues.append("submission IDs or row order differ from the evaluation CSV")
    return IndependentSubmissionReport(
        valid=not issues,
        row_count=row_count,
        expected_count=len(expected_ids),
        submission_sha256=_sha256_file(submission_source),
        expected_sha256=actual_expected_digest,
        issues=tuple(issues),
    )


def _read_expected_ids(path: Path) -> tuple[list[str], list[str]]:
    identifiers: list[str] = []
    issues: list[str] = []
    with path.open("r", encoding="utf-8-sig", errors="strict", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        try:
            header = tuple(next(reader))
        except StopIteration:
            return identifiers, ["expected evaluation CSV is empty"]
        if header not in _EXPECTED_HEADERS:
            return identifiers, [f"unexpected evaluation header: {header!r}"]
        width = len(header)
        for row_number, row in enumerate(reader, start=2):
            if len(row) != width:
                issues.append(f"evaluation row {row_number} has width {len(row)}")
                continue
            problem_id, question = row[0], row[1]
            if not problem_id or not question.strip():
                issues.append(f"evaluation row {row_number} has empty ID or question")
            if width == 3 and row[2] != "":
                issues.append(f"evaluation row {row_number} unexpectedly contains an answer")
            identifiers.append(problem_id)
    if len(identifiers) != len(set(identifiers)):
        issues.append("evaluation IDs are not unique")
    return identifiers, issues


def _read_submission(path: Path) -> tuple[list[str], list[str], int]:
    identifiers: list[str] = []
    issues: list[str] = []
    with path.open("r", encoding="utf-8-sig", errors="strict", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        try:
            header = tuple(next(reader))
        except StopIteration:
            return identifiers, ["submission CSV is empty"], 0
        if header != _SUBMISSION_HEADER:
            issues.append(f"submission header must be {_SUBMISSION_HEADER!r}, found {header!r}")
        row_count = 0
        for row_number, row in enumerate(reader, start=2):
            row_count += 1
            if len(row) != 2:
                issues.append(f"submission row {row_number} has width {len(row)}")
                continue
            problem_id, answer = row
            identifiers.append(problem_id)
            if not problem_id:
                issues.append(f"submission row {row_number} has an empty ID")
            if _INTEGER_RE.fullmatch(answer) is None:
                issues.append(
                    f"submission row {row_number} answer is not a canonical signed integer"
                )
    return identifiers, issues, row_count


def _regular_file(path: str | Path, label: str) -> Path:
    raw = Path(path)
    if raw.is_symlink():
        raise ValueError(f"{label} refuses symlinks")
    source = raw.resolve(strict=True)
    if not source.is_file():
        raise ValueError(f"{label} must be a regular file")
    return source


def _required_sha256(value: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError("expected_file_sha256 must be a lowercase SHA-256")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = ["IndependentSubmissionReport", "verify_submission_independently"]

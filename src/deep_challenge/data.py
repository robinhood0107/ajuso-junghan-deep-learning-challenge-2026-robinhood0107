"""Strict, lossless CSV ingestion for the Deep Learning Challenge datasets.

The loader deliberately keeps the field text returned by :mod:`csv` untouched.
Consumers that need a comparison-friendly representation must opt in to the
``question_normalized`` view; training data should continue to use
``question_raw`` unless an experiment explicitly says otherwise.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final


class DatasetValidationError(ValueError):
    """Raised when an input file violates the competition CSV contract."""


class DatasetKind(StrEnum):
    """Supported competition dataset roles."""

    TRAIN = "train"
    LEADERBOARD = "leaderboard"


_TRAIN_HEADER: Final[tuple[str, ...]] = ("id", "question", "answer")
_TRAIN_EXCLUSION_HEADERS: Final[frozenset[tuple[str, ...]]] = frozenset(
    {
        ("id",),
        ("id", "answer", "question"),
    }
)
_LEADERBOARD_HEADERS: Final[frozenset[tuple[str, ...]]] = frozenset(
    {
        ("id", "question"),
        ("id", "question", "answer"),
        ("id", "question", " answer"),
    }
)
_INTEGER_RE: Final[re.Pattern[str]] = re.compile(r"(?:0|-?[1-9]\d*)\Z")


@dataclass(frozen=True, slots=True)
class FileManifest:
    """Content-addressed identity for a source file.

    ``path`` is informational; ``sha256`` and ``size_bytes`` are the fields that
    should be used to prove which bytes produced an experiment.
    """

    path: str
    size_bytes: int
    sha256: str

    @classmethod
    def from_path(cls, path: str | Path) -> FileManifest:
        source = Path(path)
        digest = hashlib.sha256()
        size = 0
        try:
            with source.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
                    size += len(chunk)
        except OSError as exc:
            raise DatasetValidationError(f"cannot read dataset {source}: {exc}") from exc
        return cls(path=str(source.resolve()), size_bytes=size, sha256=digest.hexdigest())


@dataclass(frozen=True, slots=True)
class MathRecord:
    """One validated row with both lossless and normalized question text."""

    id: str
    question_raw: str
    question_normalized: str
    answer_raw: str | None
    answer: int | None
    row_number: int

    @property
    def question(self) -> str:
        """Compatibility alias that intentionally returns the raw question."""

        return self.question_raw


@dataclass(frozen=True, slots=True)
class CsvDataset:
    """A completely validated competition CSV and its byte manifest."""

    kind: DatasetKind
    records: tuple[MathRecord, ...]
    manifest: FileManifest
    raw_header: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[MathRecord]:
        return iter(self.records)


@dataclass(frozen=True, slots=True)
class TrainExclusionSet:
    """Validated organizer exclusions with source and logical identities.

    ``manifest.sha256`` identifies the exact CSV bytes. ``ids_sha256`` instead
    identifies the sorted set of excluded IDs, so harmless source-row ordering
    differences do not change the logical exclusion policy.
    """

    ids: tuple[str, ...]
    manifest: FileManifest
    raw_header: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.ids)

    def __iter__(self) -> Iterator[str]:
        return iter(self.ids)

    def __contains__(self, record_id: object) -> bool:
        return record_id in self.ids

    @property
    def ids_sha256(self) -> str:
        """Return a stable hash of the sorted, unique exclusion ID set."""

        payload = json.dumps(
            self.ids,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def normalize_question(text: str) -> str:
    """Return a conservative comparison view without destroying the raw field.

    Only three transformations are made: line endings become LF, NBSP becomes a
    regular space, and Unicode NFKC is applied.  Whitespace is not stripped or
    collapsed because it can be meaningful inside code and LaTeX blocks.
    """

    line_normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFKC", line_normalized.replace("\u00a0", " "))


def load_train_csv(path: str | Path) -> CsvDataset:
    """Load a train CSV with an exact three-column contract."""

    return load_dataset(path, DatasetKind.TRAIN)


def load_leaderboard_csv(path: str | Path) -> CsvDataset:
    """Load a current two-column or documented legacy leaderboard CSV."""

    return load_dataset(path, DatasetKind.LEADERBOARD)


def load_train_exclusion_csv(
    path: str | Path,
    *,
    train_ids: Iterable[str],
) -> TrainExclusionSet:
    """Load the organizer train-exclusion CSV and validate it against train.

    The organizer's current file has the exact header
    ``id,answer,question``.  Only its ``id`` column is authoritative; the other
    two columns are retained solely as RFC 4180 fields and are not interpreted.
    A future minimal, semantically equivalent ``id``-only file is also accepted.
    No other header, row-width repair, or delimiter recovery is allowed.
    """

    known_train_ids = _validated_id_universe(train_ids)
    known_train_id_set = set(known_train_ids)
    source = Path(path)
    manifest = FileManifest.from_path(source)
    seen_ids: set[str] = set()

    try:
        with source.open("r", encoding="utf-8-sig", errors="strict", newline="") as stream:
            reader = csv.reader(
                stream,
                dialect="excel",
                strict=True,
                skipinitialspace=False,
            )
            try:
                header_row = next(reader)
            except StopIteration as exc:
                raise DatasetValidationError(f"empty train exclusion CSV: {source}") from exc

            header = tuple(header_row)
            if header not in _TRAIN_EXCLUSION_HEADERS:
                allowed = sorted(_TRAIN_EXCLUSION_HEADERS)
                raise DatasetValidationError(
                    f"train exclusion header must be one of {allowed!r}, got {header!r}"
                )

            for row in reader:
                row_number = reader.line_num
                if len(row) != len(header):
                    raise DatasetValidationError(
                        f"row {row_number}: expected exactly {len(header)} fields, got {len(row)}"
                    )
                if any("\x00" in field for field in row):
                    raise DatasetValidationError(f"row {row_number}: NUL byte is not allowed")
                record_id = row[0]
                if not record_id or not record_id.strip():
                    raise DatasetValidationError(f"row {row_number}: missing exclusion id")
                if record_id != record_id.strip():
                    raise DatasetValidationError(
                        f"row {row_number}: exclusion id has surrounding whitespace"
                    )
                if record_id in seen_ids:
                    raise DatasetValidationError(
                        f"row {row_number}: duplicate exclusion id {record_id!r}"
                    )
                seen_ids.add(record_id)
    except DatasetValidationError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise DatasetValidationError(f"invalid train exclusion CSV {source}: {exc}") from exc

    unknown_ids = seen_ids - known_train_id_set
    if unknown_ids:
        raise DatasetValidationError(
            f"train exclusions reference unknown train IDs: {sorted(unknown_ids)!r}"
        )

    return TrainExclusionSet(
        ids=tuple(sorted(seen_ids)),
        manifest=manifest,
        raw_header=header,
    )


def load_train_exclusions_csv(
    path: str | Path,
    *,
    train_ids: Iterable[str],
) -> TrainExclusionSet:
    """Plural spelling alias for :func:`load_train_exclusion_csv`."""

    return load_train_exclusion_csv(path, train_ids=train_ids)


def load_dataset(path: str | Path, kind: DatasetKind | str) -> CsvDataset:
    """Strictly parse and validate a competition CSV.

    The current filtered leaderboard has the exact header ``id,question`` and
    requires exactly two fields per row.  For either documented legacy
    three-column header, the sole row-width repair remains a missing trailing
    answer field: ``id,question`` is padded with ``""``.  No other truncation,
    padding, or delimiter recovery is attempted.
    """

    try:
        dataset_kind = DatasetKind(kind)
    except ValueError as exc:
        choices = ", ".join(member.value for member in DatasetKind)
        raise DatasetValidationError(f"unknown dataset kind {kind!r}; expected {choices}") from exc

    source = Path(path)
    manifest = FileManifest.from_path(source)
    records: list[MathRecord] = []
    seen_ids: set[str] = set()

    try:
        # utf-8-sig removes only a leading BOM.  newline="" is required by the
        # csv module for correct RFC 4180 handling, including embedded CRLF.
        with source.open("r", encoding="utf-8-sig", errors="strict", newline="") as stream:
            reader = csv.reader(
                stream,
                dialect="excel",
                strict=True,
                skipinitialspace=False,
            )
            try:
                header_row = next(reader)
            except StopIteration as exc:
                raise DatasetValidationError(f"empty CSV: {source}") from exc

            header = tuple(header_row)
            _validate_header(header, dataset_kind)

            for row in reader:
                row_number = reader.line_num
                normalized_row = _validate_width_and_pad(
                    row,
                    dataset_kind,
                    row_number,
                    header,
                )
                record = _record_from_row(normalized_row, dataset_kind, row_number)
                if record.id in seen_ids:
                    raise DatasetValidationError(
                        f"row {row_number}: duplicate id {record.id!r}"
                    )
                seen_ids.add(record.id)
                records.append(record)
    except DatasetValidationError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise DatasetValidationError(f"invalid CSV {source}: {exc}") from exc

    if not records:
        raise DatasetValidationError(f"dataset contains no records: {source}")

    return CsvDataset(
        kind=dataset_kind,
        records=tuple(records),
        manifest=manifest,
        raw_header=header,
    )


def _validate_header(header: tuple[str, ...], kind: DatasetKind) -> None:
    if kind is DatasetKind.TRAIN and header != _TRAIN_HEADER:
        raise DatasetValidationError(
            f"train header must be exactly {_TRAIN_HEADER!r}, got {header!r}"
        )
    if kind is DatasetKind.LEADERBOARD and header not in _LEADERBOARD_HEADERS:
        allowed = sorted(_LEADERBOARD_HEADERS)
        raise DatasetValidationError(
            f"leaderboard header must be one of {allowed!r}, got {header!r}"
        )


def _validate_width_and_pad(
    row: list[str],
    kind: DatasetKind,
    row_number: int,
    header: tuple[str, ...],
) -> tuple[str, str, str]:
    if kind is DatasetKind.LEADERBOARD and header == ("id", "question"):
        if len(row) != 2:
            raise DatasetValidationError(
                f"row {row_number}: expected exactly 2 fields, got {len(row)}"
            )
        return row[0], row[1], ""
    if kind is DatasetKind.LEADERBOARD and len(row) == 2:
        return row[0], row[1], ""
    if len(row) != 3:
        raise DatasetValidationError(
            f"row {row_number}: expected exactly 3 fields, got {len(row)}"
        )
    return row[0], row[1], row[2]


def _record_from_row(
    row: tuple[str, str, str], kind: DatasetKind, row_number: int
) -> MathRecord:
    record_id, question, answer_text = row
    if not record_id or not record_id.strip():
        raise DatasetValidationError(f"row {row_number}: missing id")
    if record_id != record_id.strip():
        raise DatasetValidationError(f"row {row_number}: id has surrounding whitespace")
    if "\x00" in record_id or "\x00" in question or "\x00" in answer_text:
        raise DatasetValidationError(f"row {row_number}: NUL byte is not allowed")
    if not question or not question.strip():
        raise DatasetValidationError(f"row {row_number}: missing question")

    if kind is DatasetKind.TRAIN:
        if not answer_text:
            raise DatasetValidationError(f"row {row_number}: missing train answer")
        if not _INTEGER_RE.fullmatch(answer_text):
            raise DatasetValidationError(
                f"row {row_number}: train answer must be an integer, got {answer_text!r}"
            )
        answer_raw: str | None = answer_text
        answer: int | None = int(answer_text)
    else:
        if answer_text != "":
            raise DatasetValidationError(
                f"row {row_number}: leaderboard answer must be empty, got {answer_text!r}"
            )
        answer_raw = None
        answer = None

    return MathRecord(
        id=record_id,
        question_raw=question,
        question_normalized=normalize_question(question),
        answer_raw=answer_raw,
        answer=answer,
        row_number=row_number,
    )


def _validated_id_universe(train_ids: Iterable[str]) -> tuple[str, ...]:
    ids = tuple(train_ids)
    for record_id in ids:
        if not isinstance(record_id, str) or not record_id or not record_id.strip():
            raise DatasetValidationError("every train ID must be a non-empty string")
        if record_id != record_id.strip():
            raise DatasetValidationError(
                f"train ID has surrounding whitespace: {record_id!r}"
            )
        if "\x00" in record_id:
            raise DatasetValidationError("train ID contains a NUL byte")
    if len(set(ids)) != len(ids):
        raise DatasetValidationError("train IDs contain duplicates")
    return ids


__all__ = [
    "CsvDataset",
    "DatasetKind",
    "DatasetValidationError",
    "FileManifest",
    "MathRecord",
    "TrainExclusionSet",
    "load_dataset",
    "load_leaderboard_csv",
    "load_train_exclusion_csv",
    "load_train_exclusions_csv",
    "load_train_csv",
    "normalize_question",
]

"""Validation, generation, and emergency-fallback tests for submissions."""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from deep_challenge.answers import parse_answer
from deep_challenge.submission import (
    SubmissionGenerationError,
    SubmissionSchema,
    resolve_predictions,
    validate_submission_csv,
    validate_submission_rows,
    write_submission_csv,
)

EXPECTED_IDS = ("val-000001", "val-000002", "val-000003")


def _codes(report: object) -> set[str]:
    return {issue.code for issue in report.issues}  # type: ignore[attr-defined]


def test_valid_rows_require_exact_schema_ids_order_and_answers() -> None:
    rows: list[Mapping[str, object]] = [
        {"ID": "val-000001", "answer": "-2"},
        {"ID": "val-000002", "answer": 0},
        {"ID": "val-000003", "answer": "17"},
    ]

    report = validate_submission_rows(rows, EXPECTED_IDS)

    assert report.valid
    assert report.row_count == report.expected_count == 3
    assert report.issues == ()
    assert report.errors == ()


@pytest.mark.parametrize("answer", ["", None, "+1", "-0", "01", " 1", "1 ", "1.0", "1e2", True])
def test_noncanonical_answers_are_rejected(answer: object) -> None:
    rows = [
        {"ID": "val-000001", "answer": answer},
        {"ID": "val-000002", "answer": "2"},
        {"ID": "val-000003", "answer": "3"},
    ]

    report = validate_submission_rows(rows, EXPECTED_IDS)

    assert not report.valid
    assert "noncanonical_answer" in _codes(report)


def test_duplicate_missing_extra_and_order_are_reported() -> None:
    duplicate = validate_submission_rows(
        [
            {"ID": "val-000001", "answer": "1"},
            {"ID": "val-000001", "answer": "2"},
            {"ID": "val-000003", "answer": "3"},
        ],
        EXPECTED_IDS,
    )
    extra = validate_submission_rows(
        [
            {"ID": "val-000001", "answer": "1"},
            {"ID": "unexpected", "answer": "2"},
            {"ID": "val-000003", "answer": "3"},
        ],
        EXPECTED_IDS,
    )
    reordered = validate_submission_rows(
        [
            {"ID": "val-000002", "answer": "2"},
            {"ID": "val-000001", "answer": "1"},
            {"ID": "val-000003", "answer": "3"},
        ],
        EXPECTED_IDS,
    )

    assert {"duplicate_ids", "missing_ids"} <= _codes(duplicate)
    assert {"missing_ids", "extra_ids"} <= _codes(extra)
    assert "id_order" in _codes(reordered)


def test_row_count_null_id_and_row_schema_are_reported() -> None:
    report = validate_submission_rows(
        [
            {"ID": None, "answer": "1"},
            {"answer": "2", "ID": "val-000002"},
        ],
        EXPECTED_IDS,
    )

    assert {"row_count", "null_id", "row_schema", "missing_ids"} <= _codes(report)


def test_csv_requires_exact_external_header_and_field_width(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("ID, answer\nval-000001,1,extra\n", encoding="utf-8")

    report = validate_submission_csv(path, ("val-000001",))

    assert not report.valid
    assert {"header_schema", "row_width", "row_count", "missing_ids"} <= _codes(report)


def test_custom_schema_is_honored_for_write_and_roundtrip(tmp_path: Path) -> None:
    schema = SubmissionSchema(id_column="problem_id", answer_column="integer_answer")
    path = tmp_path / "submission.csv"

    result = write_submission_csv(
        path,
        {"val-000001": 7, "val-000002": -3, "val-000003": "0"},
        EXPECTED_IDS,
        schema=schema,
    )

    assert result.report.valid
    assert result.fallback_count == 0
    payload = path.read_bytes()
    assert result.size_bytes == len(payload)
    assert result.sha256 == hashlib.sha256(payload).hexdigest()
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows == [
        ["problem_id", "integer_answer"],
        ["val-000001", "7"],
        ["val-000002", "-3"],
        ["val-000003", "0"],
    ]
    assert validate_submission_csv(path, EXPECTED_IDS, schema=schema).valid


def test_missing_prediction_fails_without_explicit_fallback(tmp_path: Path) -> None:
    path = tmp_path / "submission.csv"

    with pytest.raises(SubmissionGenerationError, match="missing_prediction"):
        write_submission_csv(
            path,
            {"val-000001": 1, "val-000003": 3},
            EXPECTED_IDS,
        )

    assert not path.exists()


def test_explicit_static_fallback_is_audited_not_silent(tmp_path: Path) -> None:
    path = tmp_path / "submission.csv"
    invalid_parse = parse_answer("Final answer: 1e3")

    result = write_submission_csv(
        path,
        {"val-000001": 1, "val-000002": invalid_parse},
        EXPECTED_IDS,
        fallback_value=0,
    )

    assert result.report.valid
    assert result.fallback_count == 2
    by_id = {item.problem_id: item for item in result.predictions}
    assert by_id["val-000001"].provenance == "prediction"
    assert by_id["val-000001"].reason is None
    assert by_id["val-000002"].provenance == "emergency_fallback"
    assert "parser_invalid" in (by_id["val-000002"].reason or "")
    assert by_id["val-000003"].provenance == "emergency_fallback"
    assert by_id["val-000003"].reason == "missing_prediction"
    assert path.read_text(encoding="utf-8").splitlines()[-2:] == [
        "val-000002,0",
        "val-000003,0",
    ]


def test_fallback_resolver_receives_id_raw_value_and_reason() -> None:
    calls: list[tuple[str, object | None, str]] = []

    def resolver(problem_id: str, raw: object | None, reason: str) -> int:
        calls.append((problem_id, raw, reason))
        return -1

    resolved = resolve_predictions(
        {"val-000001": "not-an-integer", "val-000003": 3},
        EXPECTED_IDS,
        fallback_resolver=resolver,
    )

    assert [item.answer for item in resolved] == [-1, -1, 3]
    assert calls == [
        ("val-000001", "not-an-integer", "invalid_or_noncanonical_prediction"),
        ("val-000002", None, "missing_prediction"),
    ]
    assert all(item.provenance == "emergency_fallback" for item in resolved[:2])


def test_fallback_configuration_is_strict() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_predictions(
            {},
            EXPECTED_IDS,
            fallback_value=0,
            fallback_resolver=lambda _id, _raw, _reason: 0,
        )
    with pytest.raises(ValueError, match="canonical"):
        resolve_predictions({}, EXPECTED_IDS, fallback_value="-0")
    with pytest.raises(SubmissionGenerationError, match="noncanonical"):
        resolve_predictions(
            {},
            EXPECTED_IDS,
            fallback_resolver=lambda _id, _raw, _reason: "+0",
        )


def test_unexpected_prediction_ids_always_fail_even_with_fallback() -> None:
    with pytest.raises(SubmissionGenerationError, match="unexpected prediction IDs"):
        resolve_predictions(
            {"unexpected": 1},
            EXPECTED_IDS,
            fallback_value=0,
        )


def test_writer_refuses_implicit_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "submission.csv"
    path.write_text("do not replace", encoding="utf-8")

    with pytest.raises(FileExistsError):
        write_submission_csv(
            path,
            dict.fromkeys(EXPECTED_IDS, 1),
            EXPECTED_IDS,
        )

    assert path.read_text(encoding="utf-8") == "do not replace"


def test_no_replace_publish_is_atomic_for_concurrent_writers(tmp_path: Path) -> None:
    path = tmp_path / "submission.csv"
    start = Barrier(2)

    def publish(answer: int) -> tuple[str, int]:
        start.wait()
        try:
            write_submission_csv(
                path,
                dict.fromkeys(EXPECTED_IDS, answer),
                EXPECTED_IDS,
            )
        except FileExistsError:
            return "exists", answer
        return "written", answer

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(publish, (11, 22)))

    assert sorted(status for status, _ in results) == ["exists", "written"]
    winning_answer = next(answer for status, answer in results if status == "written")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["answer"]) for row in rows] == [winning_answer] * len(EXPECTED_IDS)
    assert validate_submission_csv(path, EXPECTED_IDS).valid
    assert not list(tmp_path.glob(".submission.csv.*.tmp"))


def test_explicit_overwrite_atomically_replaces_existing_target(tmp_path: Path) -> None:
    path = tmp_path / "submission.csv"
    first = write_submission_csv(path, dict.fromkeys(EXPECTED_IDS, 1), EXPECTED_IDS)

    second = write_submission_csv(
        path,
        dict.fromkeys(EXPECTED_IDS, 2),
        EXPECTED_IDS,
        overwrite=True,
    )

    assert first.sha256 != second.sha256
    assert second.size_bytes == path.stat().st_size
    assert second.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["answer"] for row in rows] == ["2"] * len(EXPECTED_IDS)


def test_expected_ids_must_be_unique_nonempty_strings() -> None:
    with pytest.raises(ValueError, match="unique"):
        validate_submission_rows([], ("a", "a"))
    with pytest.raises(ValueError, match="non-empty"):
        validate_submission_rows([], ("a", ""))
    with pytest.raises(ValueError, match="sequence"):
        validate_submission_rows([], "abc")  # type: ignore[arg-type]

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pytest

from deep_challenge.cli import main
from deep_challenge.independent_submission import verify_submission_independently


def _write_csv(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(rows)


def test_independent_submission_cross_check_accepts_exact_uppercase_schema(
    tmp_path: Path,
) -> None:
    expected = tmp_path / "leaderboard.csv"
    submission = tmp_path / "submission.csv"
    _write_csv(
        expected,
        [["id", "question"], ["prob_1", "One?"], ["prob_2", "Two?"]],
    )
    _write_csv(
        submission,
        [["ID", "answer"], ["prob_1", "-2"], ["prob_2", "0"]],
    )
    expected_sha = hashlib.sha256(expected.read_bytes()).hexdigest()

    report = verify_submission_independently(
        submission,
        expected,
        expected_file_sha256=expected_sha,
    )

    assert report.valid is True
    assert report.row_count == report.expected_count == 2
    assert report.submission_sha256 == hashlib.sha256(submission.read_bytes()).hexdigest()
    assert (
        main(
            [
                "verify-submission-independent",
                "--submission",
                str(submission),
                "--expected",
                str(expected),
                "--expected-sha256",
                expected_sha,
            ]
        )
        == 0
    )


@pytest.mark.parametrize(
    ("submission_rows", "issue"),
    [
        ([['id', 'answer'], ['prob_1', '1'], ['prob_2', '2']], "header"),
        ([['ID', 'answer'], ['prob_2', '2'], ['prob_1', '1']], "row order"),
        ([['ID', 'answer'], ['prob_1', '01'], ['prob_2', '2']], "canonical"),
        ([['ID', 'answer'], ['prob_1', '1']], "row order"),
    ],
)
def test_independent_submission_cross_check_rejects_schema_order_and_answer_drift(
    tmp_path: Path, submission_rows: list[list[str]], issue: str
) -> None:
    expected = tmp_path / "leaderboard.csv"
    submission = tmp_path / "submission.csv"
    _write_csv(
        expected,
        [["id", "question"], ["prob_1", "One?"], ["prob_2", "Two?"]],
    )
    _write_csv(submission, submission_rows)

    report = verify_submission_independently(
        submission,
        expected,
        expected_file_sha256=hashlib.sha256(expected.read_bytes()).hexdigest(),
    )

    assert report.valid is False
    assert any(issue in value for value in report.issues)


def test_independent_submission_cross_check_binds_expected_bytes(tmp_path: Path) -> None:
    expected = tmp_path / "leaderboard.csv"
    submission = tmp_path / "submission.csv"
    _write_csv(expected, [["id", "question"], ["prob_1", "One?"]])
    _write_csv(submission, [["ID", "answer"], ["prob_1", "1"]])

    with pytest.raises(ValueError, match="SHA-256"):
        verify_submission_independently(
            submission,
            expected,
            expected_file_sha256="0" * 64,
        )

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from deep_challenge.audit import build_data_audit_report, quantile_r7


def _write_csv(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)


def test_quantile_r7_known_values() -> None:
    assert quantile_r7([0, 10], 0.25) == 2.5
    assert quantile_r7([1, 2, 3, 4, 5], 0.5) == 3.0
    assert quantile_r7([5], 0.99) == 5.0
    with pytest.raises(ValueError):
        quantile_r7([], 0.5)
    with pytest.raises(ValueError):
        quantile_r7([1], 1.1)


def test_build_report_preserves_manifests_and_finds_templates(tmp_path: Path) -> None:
    train = tmp_path / "train.csv"
    leaderboard = tmp_path / "leaderboard.csv"
    _write_csv(
        train,
        [
            ["id", "question", "answer"],
            ["train-0", "What is 2 + 3?", "5"],
            ["train-1", "What is 7+8?", "15"],
            ["train-2", "Return \\boxed{0}.", "0"],
            ["train-3", "What is 11+4?\n\n#", "15"],
        ],
    )
    # Deliberately write the competition's malformed three-field header/two-field rows.
    _write_csv(
        leaderboard,
        [
            ["id", "question", " answer"],
            ["val-0", "6. What is 11 + 4?"],
            ["val-1", "8."],
            ["val-2", "What is 9 + 4?"],
        ],
    )
    report = build_data_audit_report(train, leaderboard, source_tree_root=tmp_path)
    assert report["audit_version"] == "data-audit-v3"
    assert report["train"]["row_count"] == 4
    assert report["leaderboard"]["row_count"] == 3
    assert report["train"]["answers"]["zero_count"] == 1
    assert report["duplicates"]["train_number_masked_template"]["group_count"] == 1
    assert (
        report["duplicates"]["train_leaderboard_number_masked_template"][
            "leaderboard_row_count"
        ]
        == 1
    )
    assert (
        report["duplicates"]["train_leaderboard_source_format"][
            "leaderboard_row_count"
        ]
        == 1
    )
    assert report["quality_contract"]["flags_are_triage_not_auto_delete"] is True
    assert len(report["source_tree"]["tree_sha256"]) == 64

"""Tests for private, redacted parser-golden development evidence."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import pytest

from deep_challenge.answers import parse_answer
from deep_challenge.cli import main
from deep_challenge.parser_golden import (
    PARSER_GOLDEN_AUDIT_SCHEMA,
    ParserGoldenAuditError,
    audit_development_parser_golden,
)


def _write_bundle(tmp_path: Path, completions: list[str]) -> tuple[Path, Path]:
    rows: list[dict[str, object]] = []
    for index, completion in enumerate(completions):
        parsed = parse_answer(completion)
        rows.append(
            {
                "schema_version": "gate-b1-development-baseline-v2",
                "problem_id": f"synthetic-{index:03d}",
                "model_id": "Qwen/Qwen2.5-3B-Instruct",
                "revision": "a" * 40,
                "route": "direct-answer",
                "config_sha256": "c" * 64,
                "checkpoint_sha256": "b" * 64,
                "split_version": "v4",
                "split_sha256": "d" * 64,
                "source_groups_sha256": "e" * 64,
                "eligibility_ids_sha256": "f" * 64,
                "fold": 0,
                "partition": "fold_validation",
                "split_partition": "cross_validation",
                "raw_completion": completion,
                "raw_completion_sha256": hashlib.sha256(
                    completion.encode("utf-8")
                ).hexdigest(),
                "parse": asdict(parsed),
            }
        )
    records = tmp_path / "development.jsonl"
    records_bytes = (
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    ).encode("utf-8")
    records.write_bytes(records_bytes)
    manifest = tmp_path / "development.manifest.json"
    payload = {
        "schema_version": "gate-b1-development-run-v1",
        "records_file": records.name,
        "records_bytes": len(records_bytes),
        "records_sha256": hashlib.sha256(records_bytes).hexdigest(),
        "record_count": len(rows),
        "problem_count": len(rows),
        "samples_per_problem": 1,
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "revision": "a" * 40,
        "route": "direct-answer",
        "checkpoint_sha256": "b" * 64,
        "config_sha256": "c" * 64,
        "split_version": "v4",
        "split_sha256": "d" * 64,
        "source_groups_sha256": "e" * 64,
        "eligibility_ids_sha256": "f" * 64,
        "fold": 0,
        "partition": "fold_validation",
        "split_partition": "cross_validation",
        "parser_status_counts": dict(
            sorted(Counter(parse_answer(value).status for value in completions).items())
        ),
    }
    manifest.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return records, manifest


def _rewrite_manifest_for_records(records: Path, manifest: Path) -> None:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    contents = records.read_bytes()
    payload["records_bytes"] = len(contents)
    payload["records_sha256"] = hashlib.sha256(contents).hexdigest()
    manifest.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def test_parser_golden_audit_is_redacted_and_reparses_every_completion(tmp_path: Path) -> None:
    records, manifest = _write_bundle(
        tmp_path,
        [
            "Final answer: 5",
            "Final answer: 7391\nFinal answer: 8842",
            r"work \boxed{42}",
        ],
    )
    output = tmp_path / "parser-golden.json"

    result = audit_development_parser_golden(records, manifest, output_path=output)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert result.path == str(output.resolve())
    assert result.case_count == 3
    assert payload["schema_version"] == PARSER_GOLDEN_AUDIT_SCHEMA
    assert payload["observed_parser_outcomes"] == [
        {
            "count": 1,
            "reason_code": "conflicting_marker_values",
            "source": "final_answer",
            "status": "conflict",
        },
        {
            "count": 1,
            "reason_code": "parsed_unambiguous_integer",
            "source": "boxed",
            "status": "ok",
        },
        {
            "count": 1,
            "reason_code": "parsed_unambiguous_integer",
            "source": "final_answer",
            "status": "ok",
        },
    ]
    assert payload["privacy_contract"]["raw_completion_serialized"] is False
    assert payload["privacy_contract"]["problem_id_serialized"] is False
    assert payload["privacy_contract"]["reference_answer_serialized"] is False
    rendered = output.read_text(encoding="utf-8")
    assert "synthetic-000" not in rendered
    assert "Final answer: 7391" not in rendered
    assert "conflicting_marker_values:7391,8842" not in rendered
    assert "\\boxed{42}" not in rendered

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        audit_development_parser_golden(records, manifest, output_path=output)


def test_parser_golden_audit_rejects_parser_mismatch_without_output(tmp_path: Path) -> None:
    records, manifest = _write_bundle(tmp_path, ["Final answer: 5"])
    row = json.loads(records.read_text(encoding="utf-8"))
    row["parse"]["value"] = 6
    records.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    _rewrite_manifest_for_records(records, manifest)
    output = tmp_path / "must-not-exist.json"

    with pytest.raises(ParserGoldenAuditError, match="stored parser result"):
        audit_development_parser_golden(records, manifest, output_path=output)

    assert not output.exists()


def test_parser_golden_audit_rejects_non_development_partition(tmp_path: Path) -> None:
    records, manifest = _write_bundle(tmp_path, ["Final answer: 5"])
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["partition"] = "final_locked_holdout"
    payload["split_partition"] = "locked_holdout"
    manifest.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ParserGoldenAuditError, match="fold_validation"):
        audit_development_parser_golden(
            records,
            manifest,
            output_path=tmp_path / "must-not-exist.json",
        )


def test_parser_golden_cli_reports_only_safe_summary(tmp_path: Path, capsys) -> None:
    records, manifest = _write_bundle(tmp_path, ["Final answer: 5"])
    output = tmp_path / "parser-golden.json"

    assert (
        main(
            [
                "audit-parser-golden",
                "--records",
                str(records),
                "--manifest",
                str(manifest),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["case_count"] == 1
    assert report["raw_completion_serialized"] is False
    assert report["locked_holdout_accessed"] is False
    assert report["leaderboard_or_test_used"] is False

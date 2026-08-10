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
from deep_challenge.gate_b import PINNED_MODEL_REVISION
from deep_challenge.model_preflight import OFFICIAL_MODEL_ID
from deep_challenge.parser_golden import (
    PARSER_GOLDEN_AUDIT_SCHEMA,
    PARSER_RESCORE_AUDIT_SCHEMA,
    ParserGoldenAuditError,
    audit_development_parser_golden,
    audit_development_parser_rescore,
)


def _write_bundle(tmp_path: Path, completions: list[str]) -> tuple[Path, Path]:
    rows: list[dict[str, object]] = []
    for index, completion in enumerate(completions):
        parsed = parse_answer(completion)
        reference_answer = parsed.value if parsed.ok else 0
        rows.append(
            {
                "schema_version": "gate-b1-development-baseline-v2",
                "problem_id": f"synthetic-{index:03d}",
                "model_id": OFFICIAL_MODEL_ID,
                "revision": PINNED_MODEL_REVISION,
                "route": "direct_answer",
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
                "reference_answer": reference_answer,
                "exact_match": parsed.ok and parsed.value == reference_answer,
            }
        )
    records = tmp_path / "development.jsonl"
    records_bytes = (
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    ).encode("utf-8")
    records.write_bytes(records_bytes)
    manifest = tmp_path / "development.manifest.json"
    payload = {
        "schema_version": "gate-b1-development-run-v2",
        "records_file": records.name,
        "records_bytes": len(records_bytes),
        "records_sha256": hashlib.sha256(records_bytes).hexdigest(),
        "record_count": len(rows),
        "problem_count": len(rows),
        "samples_per_problem": 1,
        "model_id": OFFICIAL_MODEL_ID,
        "revision": PINNED_MODEL_REVISION,
        "route": "direct_answer",
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
        "exact_match_count": sum(parse_answer(value).ok for value in completions),
        "exact_match_accuracy": sum(parse_answer(value).ok for value in completions)
        / len(completions),
        "execution_evidence": {},
        "generation_evidence": {},
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
            "Final answer: 42.5",
        ],
    )
    output = tmp_path / "parser-golden.json"

    result = audit_development_parser_golden(records, manifest, output_path=output)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert result.path == str(output.resolve())
    assert result.case_count == 4
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
            "reason_code": "marker:non_integral_decimal",
            "source": "final_answer",
            "status": "invalid",
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


def test_parser_golden_audit_rejects_inconsistent_exact_match_without_output(
    tmp_path: Path,
) -> None:
    records, manifest = _write_bundle(tmp_path, ["Final answer: 5"])
    row = json.loads(records.read_text(encoding="utf-8"))
    row["exact_match"] = False
    records.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    _rewrite_manifest_for_records(records, manifest)
    output = tmp_path / "must-not-exist.json"

    with pytest.raises(ParserGoldenAuditError, match="stored exact_match"):
        audit_development_parser_golden(records, manifest, output_path=output)

    assert not output.exists()


def test_parser_golden_audit_rejects_manifest_exact_match_count_mismatch(
    tmp_path: Path,
) -> None:
    records, manifest = _write_bundle(tmp_path, ["Final answer: 5"])
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["exact_match_count"] = 0
    payload["exact_match_accuracy"] = 0.0
    manifest.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ParserGoldenAuditError, match="exact_match_count does not match"):
        audit_development_parser_golden(
            records,
            manifest,
            output_path=tmp_path / "must-not-exist.json",
        )


def test_parser_rescore_reports_aggregate_gain_without_promoting_selection(
    tmp_path: Path,
) -> None:
    records, manifest = _write_bundle(
        tmp_path, ["The final answer is: Final answer: 8"]
    )
    row = json.loads(records.read_text(encoding="utf-8"))
    row["parse"] = {
        "status": "invalid",
        "value": None,
        "source": "final_answer",
        "reason": "marker_1:unsupported_numeric_payload",
    }
    row["exact_match"] = False
    row["reference_answer"] = 8
    records.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    _rewrite_manifest_for_records(records, manifest)
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload["parser_status_counts"] = {"invalid": 1}
    manifest_payload["exact_match_count"] = 0
    manifest_payload["exact_match_accuracy"] = 0.0
    manifest.write_text(
        json.dumps(manifest_payload, sort_keys=True) + "\n", encoding="utf-8"
    )
    output = tmp_path / "parser-rescore.json"

    result = audit_development_parser_rescore(records, manifest, output_path=output)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert result.case_count == 1
    assert payload["schema_version"] == PARSER_RESCORE_AUDIT_SCHEMA
    assert payload["stored"]["exact_match_count"] == 0
    assert payload["current"]["exact_match_count"] == 1
    assert payload["exact_match_delta"] == 1
    assert payload["changed_parser_result_count"] == 1
    assert payload["selection_eligible"] is False
    assert payload["requires_current_source_run_before_freeze"] is True
    assert payload["privacy_contract"]["raw_completion_serialized"] is False
    rendered = output.read_text(encoding="utf-8")
    assert "Final answer: 8" not in rendered
    assert "synthetic-000" not in rendered

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        audit_development_parser_rescore(records, manifest, output_path=output)


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


@pytest.mark.parametrize(
    ("field_name", "bad_value", "message"),
    [
        ("model_id", "forbidden/model", "official model"),
        ("revision", "0" * 40, "pinned revision"),
        ("route", "chain_of_thought", "route must be direct_answer"),
    ],
)
def test_parser_golden_audit_enforces_fixed_model_contract(
    tmp_path: Path,
    field_name: str,
    bad_value: str,
    message: str,
) -> None:
    records, manifest = _write_bundle(tmp_path, ["Final answer: 5"])
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload[field_name] = bad_value
    manifest.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ParserGoldenAuditError, match=message):
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


def test_parser_rescore_cli_reports_only_safe_summary(tmp_path: Path, capsys) -> None:
    records, manifest = _write_bundle(tmp_path, ["Final answer: 5"])
    output = tmp_path / "parser-rescore.json"

    assert (
        main(
            [
                "audit-parser-rescore",
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
    assert report["changed_parser_result_count"] == 0
    assert report["selection_eligible"] is False
    assert report["raw_completion_serialized"] is False
    assert report["locked_holdout_accessed"] is False
    assert report["leaderboard_or_test_used"] is False


@pytest.mark.parametrize(
    ("completion", "status", "source", "reason_fragment"),
    [
        ("\\boxed{41}\n\\boxed{42}", "conflict", "boxed", "conflicting_marker_values"),
        (r"\boxed{7/2}", "invalid", "boxed", "non_integral_fraction"),
        (r"\boxed{a/2}", "invalid", "boxed", "fraction_must_contain_two_plain"),
        (r"\boxed{42", "invalid", "boxed", "unbalanced_braces"),
        ("Final answer: 2 or 3", "invalid", "final_answer", "multiple_values"),
        ("Final answer: 42.5", "invalid", "final_answer", "non_integral_decimal"),
        ("Final answer: the result is 42", "invalid", "final_answer", "unsupported_numeric"),
        ("#### 2 or 3", "invalid", "hashes", "multiple_values"),
        ("#### result is 42", "invalid", "hashes", "unsupported_numeric"),
        ("Result: 7/2", "invalid", "fallback", "non_integral_fraction"),
        ("Result: 42.5", "invalid", "fallback", "non_integral_decimal"),
    ],
)
def test_observed_generation_parser_golden_cases_are_safe_synthetic_structures(
    completion: str,
    status: str,
    source: str,
    reason_fragment: str,
) -> None:
    """Regression forms selected from redacted fold-0 outcome categories only."""

    result = parse_answer(completion)

    assert result.status == status
    assert result.source == source
    assert reason_fragment in result.reason

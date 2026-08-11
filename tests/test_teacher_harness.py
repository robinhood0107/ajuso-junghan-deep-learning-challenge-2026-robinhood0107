from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import deep_challenge.cli as cli_module
from deep_challenge.cli import main
from deep_challenge.data import MathRecord
from deep_challenge.provenance import (
    SourceTreeArtifactEvidence,
    build_source_tree_manifest,
    canonical_json_bytes,
)
from deep_challenge.teacher_harness import (
    HARNESS_CHUNK_SIZE,
    HARNESS_LIVE_SCHEMA,
    HARNESS_REPLAY_FAULT_COUNT,
    HARNESS_REPLAY_SCHEMA,
    FailureClassification,
    HarnessProfile,
    TeacherHarnessArtifactExistsError,
    TeacherHarnessValidationError,
    classify_codex_result,
    create_harness_authorization,
    diagnose_teacher_ledger,
    require_harness_live_execution_matches,
    run_harness_live,
    run_harness_replay,
    synthetic_fixture_rows,
    synthetic_fixture_sha256,
    validate_harness_evidence,
    verify_harness_authorization,
)
from deep_challenge.teacher_rationale import (
    CodexCommandResult,
    TeacherExecutionConfig,
    TeacherPromptPolicy,
    create_teacher_plan,
    load_teacher_attempts,
    run_teacher_plan,
)

_FIXTURE_SHA256 = "bc314a24ec872edf26bae13e296fb8fd500f80bcf6c00023518844d1b407e3b7"
_PROMPT_TEMPLATE_SHA256 = "cf56fc2c021410337f8be8f5f519912eabf6390aa8892ecd92cac1ced6175c72"
_HARNESS_CONFIG_FILE_SHA256 = "5e31d82fa7dcd5e3b2dcdcaa4cdcf7b0db1efd40009ed5c0ee53aa23cf4f300c"
_HARNESS_CONFIG_SEMANTIC_SHA256 = "628dd255ad33ca1995f409060e7daeb4f9ed8a26805ac9e4458b1cdfffbfe4b1"


def _profile() -> HarnessProfile:
    return HarnessProfile(
        label="codex-gpt-5.6-sol-teacher-harness-v1",
        version="harness-v1",
        seed=20_260_731,
        initial_chunk_size=32,
        chunk_count=2,
        max_workers=1,
        max_invocations=2,
        max_attempts=1,
        retry_count=0,
        repair_count=0,
        bank_output_count=0,
        initial_reasoning_effort="high",
    )


def _policy() -> TeacherPromptPolicy:
    return TeacherPromptPolicy(
        prompt_version="gate-b-codex-teacher-prompt-v3",
        prompt_template_sha256=_PROMPT_TEMPLATE_SHA256,
    )


def _hashes() -> dict[str, str]:
    return {
        "harness_config_sha256": "1" * 64,
        "harness_config_file_sha256": "2" * 64,
        "teacher_config_sha256": "3" * 64,
        "teacher_config_file_sha256": "4" * 64,
    }


def _events(items: list[dict[str, str]]) -> str:
    message = json.dumps({"items": items}, separators=(",", ":"))
    events = (
        {"type": "thread.started"},
        {"type": "turn.started"},
        {"type": "item.completed", "item": {"type": "agent_message", "text": message}},
        {"type": "turn.completed", "usage": {"input_tokens": 7, "output_tokens": 4}},
    )
    return "\n".join(json.dumps(event, separators=(",", ":")) for event in events)


def _target(answer: int) -> str:
    return "Compute the signed integer from the stated relation.\nFinal answer: " + str(answer)


def _safe_report_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    for forbidden in (
        "problem_id",
        "question",
        "target_text",
        "stderr",
        "INPUT_JSON",
        "train-900001",
        "Compute the signed integer",
        str(path.parent),
    ):
        assert forbidden not in text
    return text


def _source_evidence(tmp_path: Path) -> SourceTreeArtifactEvidence:
    return SourceTreeArtifactEvidence(
        path=str(tmp_path / "source-manifest.json"),
        sha256="5" * 64,
        tree_sha256="6" * 64,
        file_count=12,
    )


def _live_runner(calls: list[tuple[str, ...]]):
    answers = {row.problem_id: row.expected_answer for row in synthetic_fixture_rows()}

    def run(command: tuple[str, ...]) -> CodexCommandResult:
        calls.append(command)
        payload = json.loads(command[-1].rsplit("INPUT_JSON:\n", 1)[1])
        items = [
            {"problem_id": raw["problem_id"], "target_text": _target(answers[raw["problem_id"]])}
            for raw in payload["items"]
        ]
        return CodexCommandResult(stdout=_events(items), stderr="", returncode=0, latency_ms=1)

    return run


def _binary(tmp_path: Path) -> Path:
    binary = tmp_path / "codex"
    binary.write_text("synthetic test binary\n", encoding="utf-8")
    return binary

def _repo_config(name: str) -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "gate_b" / name


def test_fixture_is_fixed_and_replay_qualifies_without_raw_content(tmp_path: Path) -> None:
    assert len(synthetic_fixture_rows()) == 64
    assert synthetic_fixture_sha256() == _FIXTURE_SHA256
    report = tmp_path / "replay.json"
    result = run_harness_replay(
        report,
        **_hashes(),
        prompt_policy=_policy(),
        profile=_profile(),
    )
    assert result.qualified
    payload = json.loads(_safe_report_text(report))
    assert payload["schema_version"] == HARNESS_REPLAY_SCHEMA
    assert payload["qualified"] is True
    assert len(payload["classifications"]) == HARNESS_REPLAY_FAULT_COUNT
    assert any(
        entry["stage"] == "output_structure"
        and entry["code"] == "cardinality_mismatch"
        and entry["requested_count"] == 32
        and entry["returned_count"] == 33
        and entry["duplicate_count"] == 1
        for entry in payload["classifications"]
    )
    with pytest.raises(TeacherHarnessArtifactExistsError):
        run_harness_replay(
            report,
            **_hashes(),
            prompt_policy=_policy(),
            profile=_profile(),
        )


def test_harness_config_semantic_and_file_hashes_are_locked() -> None:
    config_path = _repo_config("codex-gpt-5.6-sol-teacher-harness-v1.json")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    stored = payload.pop("config_sha256")
    assert stored == _HARNESS_CONFIG_SEMANTIC_SHA256
    assert hashlib.sha256(canonical_json_bytes(payload)).hexdigest() == stored
    assert hashlib.sha256(config_path.read_bytes()).hexdigest() == _HARNESS_CONFIG_FILE_SHA256
    assert payload["fixture_sha256"] == _FIXTURE_SHA256


def test_classifier_prioritizes_cardinality_and_redacts_duplicate_ids() -> None:
    rows = synthetic_fixture_rows()[:HARNESS_CHUNK_SIZE]
    expected_ids = tuple(row.problem_id for row in rows)
    answers = {row.problem_id: row.expected_answer for row in rows}
    items = [
        {"problem_id": row.problem_id, "target_text": _target(row.expected_answer)} for row in rows
    ]
    items.append(dict(items[0]))
    classification = classify_codex_result(
        CodexCommandResult(stdout=_events(items), stderr="", returncode=0, latency_ms=1),
        expected_ids,
        prompt_policy=_policy(),
        expected_answers=answers,
    )
    assert classification.stage == "output_structure"
    assert classification.code == "cardinality_mismatch"
    assert classification.requested_count == 32
    assert classification.returned_count == 33
    assert classification.duplicate_count == 1
    assert classification.missing_count == 0
    assert classification.unexpected_count == 0
    assert classification.order_mismatch is True


def test_classifier_multi_fault_precedence_is_fixed() -> None:
    rows = synthetic_fixture_rows()[:2]
    expected_ids = tuple(row.problem_id for row in rows)
    answers = {row.problem_id: row.expected_answer for row in rows}
    items = [
        {"problem_id": row.problem_id, "target_text": _target(row.expected_answer)}
        for row in rows
    ]
    assert (
        classify_codex_result(
            CodexCommandResult(stdout="{malformed", stderr="", returncode=9, latency_ms=1),
            expected_ids,
            prompt_policy=_policy(),
            expected_answers=answers,
        ).code
        == "command_nonzero"
    )
    unsafe_then_bad_usage = "\n".join(
        json.dumps(event, separators=(",", ":"))
        for event in (
            {"type": "thread.started"},
            {"type": "item.completed", "item": {"type": "tool"}},
            {"type": "turn.completed", "usage": {"input_tokens": -1}},
        )
    )
    assert (
        classify_codex_result(
            CodexCommandResult(
                stdout=unsafe_then_bad_usage,
                stderr="",
                returncode=0,
                latency_ms=1,
            ),
            expected_ids,
            prompt_policy=_policy(),
            expected_answers=answers,
        ).code
        == "unsafe_item"
    )
    items[0] = {"problem_id": rows[0].problem_id, "target_text": "bad\nFinal answer: 1"}
    items.append(dict(items[1]))
    assert (
        classify_codex_result(
            CodexCommandResult(stdout=_events(items), stderr="", returncode=0, latency_ms=1),
            expected_ids,
            prompt_policy=_policy(),
            expected_answers=answers,
        ).code
        == "cardinality_mismatch"
    )
    with pytest.raises(TeacherHarnessValidationError, match="allowlisted"):
        FailureClassification(
            stage="unclassified",
            code="unknown",
            requested_count=2,
            returned_count=0,
            duplicate_count=0,
            missing_count=0,
            unexpected_count=0,
            order_mismatch=False,
        )


def test_profile_rejects_boolean_execution_cap() -> None:
    with pytest.raises(TeacherHarnessValidationError):
        HarnessProfile(
            label="codex-gpt-5.6-sol-teacher-harness-v1",
            version="harness-v1",
            seed=20_260_731,
            initial_chunk_size=32,
            chunk_count=2,
            max_workers=True,
            max_invocations=2,
            max_attempts=1,
            retry_count=0,
            repair_count=0,
            bank_output_count=0,
            initial_reasoning_effort="high",
        )


def test_diagnostic_is_read_only_and_reproduces_cardinality(tmp_path: Path) -> None:
    rows = synthetic_fixture_rows()[:HARNESS_CHUNK_SIZE]
    records = tuple(
        MathRecord(
            id=row.problem_id,
            question_raw=row.question,
            question_normalized=row.question,
            answer_raw=None,
            answer=None,
            row_number=index,
        )
        for index, row in enumerate(rows, start=1)
    )
    plan = create_teacher_plan(
        records,
        tuple(row.problem_id for row in rows),
        tmp_path / "private-plan",
        chunk_size=32,
        label="codex-gpt-5.6-sol-teacher-pilot-v3",
        version="pilot-v3",
        prompt_policy=_policy(),
    )
    items = [
        {"problem_id": row.problem_id, "target_text": _target(row.expected_answer)} for row in rows
    ]
    items.append(dict(items[0]))
    run_teacher_plan(
        plan.plan_dir,
        lambda _command: CodexCommandResult(
            stdout=_events(items), stderr="private", returncode=0, latency_ms=1
        ),
        max_attempts=1,
        max_chunks=1,
    )
    before = {
        path.relative_to(plan.plan_dir).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in plan.plan_dir.rglob("*")
        if path.is_file()
    }
    report = tmp_path / "diagnostic.json"
    result = diagnose_teacher_ledger(
        plan.plan_dir,
        report,
        teacher_config_sha256="7" * 64,
        teacher_config_file_sha256="8" * 64,
        prompt_policy=_policy(),
    )
    after = {
        path.relative_to(plan.plan_dir).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in plan.plan_dir.rglob("*")
        if path.is_file()
    }
    assert result.qualified is False
    assert before == after
    payload = json.loads(_safe_report_text(report))
    assert payload["classifications"] == [
        {
            "stage": "output_structure",
            "code": "cardinality_mismatch",
            "requested_count": 32,
            "returned_count": 33,
            "duplicate_count": 1,
            "missing_count": 0,
            "unexpected_count": 0,
            "order_mismatch": True,
        }
    ]


def test_live_harness_runs_exactly_two_chunks_and_authorization_revalidates(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, ...]] = []
    binary = _binary(tmp_path)
    execution = TeacherExecutionConfig(
        codex_binary=str(binary),
        codex_cli_version="codex synthetic-test",
        reasoning_effort="high",
    )
    live_report = tmp_path / "live.json"
    result = run_harness_live(
        tmp_path / "private-live-plan",
        live_report,
        **_hashes(),
        prompt_policy=_policy(),
        execution=execution,
        profile=_profile(),
        source_manifest=_source_evidence(tmp_path),
        command_runner=_live_runner(calls),
        working_directory=tmp_path,
    )
    assert result.qualified
    assert len(calls) == 2
    assert [
        len(json.loads(command[-1].rsplit("INPUT_JSON:\n", 1)[1])["items"]) for command in calls
    ] == [32, 32]
    payload = json.loads(_safe_report_text(live_report))
    assert payload["schema_version"] == HARNESS_LIVE_SCHEMA
    assert payload["qualified"] is True
    with pytest.raises(TeacherHarnessArtifactExistsError):
        run_harness_live(
            tmp_path / "private-live-plan",
            tmp_path / "second-live.json",
            **_hashes(),
            prompt_policy=_policy(),
            execution=execution,
            profile=_profile(),
            source_manifest=_source_evidence(tmp_path),
            command_runner=_live_runner(calls),
            working_directory=tmp_path,
        )
    replay_report = tmp_path / "replay.json"
    run_harness_replay(
        replay_report,
        **_hashes(),
        prompt_policy=_policy(),
        profile=_profile(),
    )
    validated_payload_sha = validate_harness_evidence(
        replay_report=replay_report,
        live_report=live_report,
        live_plan_dir=tmp_path / "private-live-plan",
        **_hashes(),
        prompt_policy=_policy(),
        source_manifest=_source_evidence(tmp_path),
    )
    require_harness_live_execution_matches(live_report, execution=execution)
    other_binary = tmp_path / "different-codex"
    other_binary.write_text("other synthetic binary\n", encoding="utf-8")
    with pytest.raises(TeacherHarnessValidationError, match="execution"):
        require_harness_live_execution_matches(
            live_report,
            execution=TeacherExecutionConfig(
                codex_binary=str(other_binary),
                codex_cli_version=execution.codex_cli_version,
                reasoning_effort="high",
            ),
        )
    authorization = tmp_path / "authorization.json"
    payload_sha = create_harness_authorization(
        authorization,
        replay_report=replay_report,
        live_report=live_report,
        live_plan_dir=tmp_path / "private-live-plan",
        **_hashes(),
        prompt_policy=_policy(),
        source_manifest=_source_evidence(tmp_path),
    )
    assert payload_sha == validated_payload_sha
    original_replay = replay_report.read_bytes()
    replay_payload = json.loads(original_replay)
    replay_payload["classifications"][0]["stage"] = "nonzero"
    replay_payload["classifications"][0]["code"] = "command_nonzero"
    raw = dict(replay_payload)
    raw.pop("payload_sha256")
    replay_payload["payload_sha256"] = hashlib.sha256(canonical_json_bytes(raw)).hexdigest()
    replay_report.write_text(json.dumps(replay_payload), encoding="utf-8")
    with pytest.raises(TeacherHarnessValidationError):
        create_harness_authorization(
            tmp_path / "rejected-authorization.json",
            replay_report=replay_report,
            live_report=live_report,
            live_plan_dir=tmp_path / "private-live-plan",
            **_hashes(),
            prompt_policy=_policy(),
            source_manifest=_source_evidence(tmp_path),
        )
    replay_report.write_bytes(original_replay)
    assert (
        verify_harness_authorization(
            authorization,
            replay_report=replay_report,
            live_report=live_report,
            live_plan_dir=tmp_path / "private-live-plan",
            **_hashes(),
            prompt_policy=_policy(),
            source_manifest=_source_evidence(tmp_path),
        )
        == payload_sha
    )
    original_live = live_report.read_bytes()
    live_payload = json.loads(original_live)
    live_payload["source_manifest"]["file"] = "different-manifest.json"
    raw = dict(live_payload)
    raw.pop("payload_sha256")
    live_payload["payload_sha256"] = hashlib.sha256(canonical_json_bytes(raw)).hexdigest()
    live_report.write_text(json.dumps(live_payload), encoding="utf-8")
    with pytest.raises(TeacherHarnessValidationError):
        verify_harness_authorization(
            authorization,
            replay_report=replay_report,
            live_report=live_report,
            live_plan_dir=tmp_path / "private-live-plan",
            **_hashes(),
            prompt_policy=_policy(),
            source_manifest=_source_evidence(tmp_path),
        )
    live_report.write_bytes(original_live)
    private_attempt = load_teacher_attempts(tmp_path / "private-live-plan")[0]
    original_event_stream = private_attempt.event_stream_path.read_bytes()
    private_attempt.event_stream_path.write_bytes(b"{}")
    with pytest.raises(ValueError):
        verify_harness_authorization(
            authorization,
            replay_report=replay_report,
            live_report=live_report,
            live_plan_dir=tmp_path / "private-live-plan",
            **_hashes(),
            prompt_policy=_policy(),
            source_manifest=_source_evidence(tmp_path),
        )
    private_attempt.event_stream_path.write_bytes(original_event_stream)
    tampered = json.loads(authorization.read_text(encoding="utf-8"))
    tampered["fixture_sha256"] = "0" * 64
    authorization.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(TeacherHarnessValidationError):
        verify_harness_authorization(
            authorization,
            replay_report=replay_report,
            live_report=live_report,
            live_plan_dir=tmp_path / "private-live-plan",
            **_hashes(),
            prompt_policy=_policy(),
            source_manifest=_source_evidence(tmp_path),
        )


def test_cli_replay_and_live_profile_failure_are_redacted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    harness_config = _repo_config("codex-gpt-5.6-sol-teacher-harness-v1.json")
    teacher_config = _repo_config("codex-gpt-5.6-sol-teacher-pilot-v3.json")
    replay = tmp_path / "replay.json"
    assert (
        main(
            [
                "gate-b-teacher-harness-replay",
                "--harness-config",
                str(harness_config),
                "--teacher-config",
                str(teacher_config),
                "--output",
                str(replay),
            ]
        )
        == 0
    )
    replay_stdout = capsys.readouterr().out
    assert "problem_id" not in replay_stdout
    assert "target_text" not in replay_stdout

    source_root = tmp_path / "frozen-source"
    source_root.mkdir()
    (source_root / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
    artifacts = source_root / "artifacts"
    artifacts.mkdir()
    source_manifest = artifacts / "source-manifest.json"
    source_manifest.write_text(
        json.dumps(
            build_source_tree_manifest(
                source_root,
                excluded_paths=(source_manifest,),
            ).as_dict()
        ),
        encoding="utf-8",
    )
    binary = _binary(tmp_path)
    monkeypatch.setattr(
        cli_module,
        "_probe_codex_chatgpt_cli",
        lambda: (str(binary), "codex synthetic-test"),
    )
    monkeypatch.setattr(
        cli_module,
        "_prepare_isolated_codex_home",
        lambda path: path,
    )
    calls: list[tuple[str, ...]] = []
    answers = {row.problem_id: row.expected_answer for row in synthetic_fixture_rows()}

    def failed_canary(
        command: tuple[str, ...],
        *,
        timeout_seconds: int,
        isolated_codex_home: Path | None = None,
    ) -> CodexCommandResult:
        del timeout_seconds, isolated_codex_home
        calls.append(command)
        payload = json.loads(command[-1].rsplit("INPUT_JSON:\n", 1)[1])
        items = [
            {"problem_id": raw["problem_id"], "target_text": _target(answers[raw["problem_id"]])}
            for raw in payload["items"]
        ]
        items.append(dict(items[0]))
        return CodexCommandResult(
            stdout=_events(items), stderr="private", returncode=0, latency_ms=1
        )

    monkeypatch.setattr(cli_module, "_run_codex_teacher_command", failed_canary)
    report = artifacts / "live.json"
    assert (
        main(
            [
                "gate-b-teacher-harness-live",
                "--harness-config",
                str(harness_config),
                "--teacher-config",
                str(teacher_config),
                "--source-root",
                str(source_root),
                "--source-manifest",
                str(source_manifest),
                "--plan-dir",
                str(artifacts / "private-plan"),
                "--report",
                str(report),
                "--acknowledge-synthetic-codex-canary",
            ]
        )
        == 1
    )
    assert len(calls) == 2
    live_stdout = capsys.readouterr().out
    assert "problem_id" not in live_stdout
    assert "target_text" not in live_stdout
    payload = json.loads(_safe_report_text(report))
    assert all(
        item["stage"] == "output_structure"
        and item["code"] == "cardinality_mismatch"
        and item["requested_count"] == 32
        and item["returned_count"] == 33
        and item["duplicate_count"] == 1
        for item in payload["classifications"]
    )

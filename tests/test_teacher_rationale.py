from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

import pytest

from deep_challenge.data import MathRecord
from deep_challenge.provenance import canonical_json_bytes
from deep_challenge.rationale_corpus import build_rationale_corpus
from deep_challenge.splits import eligible_training_ids, make_grouped_split_manifest
from deep_challenge.teacher_rationale import (
    CodexCommandResult,
    TeacherPlanLockError,
    TeacherPromptPolicy,
    TeacherRationaleArtifactExistsError,
    TeacherRationaleValidationError,
    build_codex_exec_command,
    build_teacher_logical_audit_prompt,
    build_teacher_prompt,
    create_teacher_logical_audit_plan,
    create_teacher_plan,
    finalize_teacher_bank,
    finalize_teacher_logical_audit,
    load_teacher_plan,
    run_teacher_logical_audit,
    run_teacher_plan,
    teacher_logical_audit_status,
    teacher_plan_lock,
    teacher_status,
    validate_codex_event_stream,
    validate_codex_logical_audit_event_stream,
)


def _record(index: int, answer: int) -> MathRecord:
    identifier = f"train-{index:06d}"
    question = f"Synthetic question {index}; determine its requested integer."
    return MathRecord(
        id=identifier,
        question_raw=question,
        question_normalized=question,
        answer_raw=str(answer),
        answer=answer,
        row_number=index + 1,
    )


def _target(answer: int) -> str:
    return f"The arithmetic condition uniquely fixes the integer.\nFinal answer: {answer}"


def _events(items: list[dict[str, str]]) -> str:
    message = json.dumps({"items": items}, ensure_ascii=False, separators=(",", ":"))
    rows = (
        {"type": "thread.started", "thread_id": "synthetic"},
        {"type": "turn.started"},
        {
            "type": "item.completed",
            "item": {"type": "agent_message", "text": message},
        },
        {
            "type": "turn.completed",
            "usage": {"input_tokens": 11, "output_tokens": 7},
        },
    )
    return "\n".join(json.dumps(row, separators=(",", ":")) for row in rows) + "\n"


def _items(ids: tuple[str, ...], answers: dict[str, int]) -> list[dict[str, str]]:
    return [
        {"problem_id": identifier, "target_text": _target(answers[identifier])}
        for identifier in ids
    ]


def _audit_events(ids: tuple[str, ...], consistent: dict[str, bool]) -> str:
    message = json.dumps(
        {
            "items": [
                {"problem_id": identifier, "consistent": consistent[identifier]}
                for identifier in ids
            ]
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    rows = (
        {"type": "thread.started", "thread_id": "synthetic-audit"},
        {"type": "turn.started"},
        {
            "type": "item.completed",
            "item": {"type": "agent_message", "text": message},
        },
        {
            "type": "turn.completed",
            "usage": {"input_tokens": 17, "output_tokens": 11},
        },
    )
    return "\n".join(json.dumps(row, separators=(",", ":")) for row in rows) + "\n"


def _make_plan(tmp_path: Path, *, count: int = 3, chunk_size: int = 2):
    records = tuple(_record(index, 900_000 + index) for index in range(1, count + 1))
    plan = create_teacher_plan(
        records,
        tuple(record.id for record in reversed(records)),
        tmp_path / "teacher-plan",
        chunk_size=chunk_size,
    )
    return records, plan


def _make_finalized_teacher_bank(tmp_path: Path, *, count: int = 64):
    records, plan = _make_plan(tmp_path, count=count, chunk_size=64)
    answers = {record.id: record.answer for record in records}

    def runner(command: tuple[str, ...]) -> CodexCommandResult:
        prompt = command[-1]
        input_ids = tuple(identifier for identifier in plan.problem_ids if identifier in prompt)
        return CodexCommandResult(
            stdout=_events(
                _items(
                    input_ids, {identifier: int(answers[identifier]) for identifier in input_ids}
                )
            ),
            stderr="",
            returncode=0,
            latency_ms=3,
        )

    run_teacher_plan(plan.plan_dir, runner)
    source = tmp_path / "private-source.jsonl"
    manifest = tmp_path / "private-bank-manifest.json"
    finalized = finalize_teacher_bank(
        plan.plan_dir,
        records,
        output_jsonl=source,
        output_manifest=manifest,
    )
    assert finalized.complete
    return records, plan, source, manifest


def _rewrite_self_hashed_attempt(path: Path, payload: dict[str, object]) -> None:
    """Write deliberately self-consistent private tampering for a loader test."""

    payload_without_hash = dict(payload)
    payload_without_hash.pop("payload_sha256", None)
    payload["payload_sha256"] = hashlib.sha256(
        canonical_json_bytes(payload_without_hash)
    ).hexdigest()
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def test_plan_is_question_only_immutable_and_command_is_generic(tmp_path: Path) -> None:
    records, plan = _make_plan(tmp_path)
    loaded = load_teacher_plan(plan.plan_dir)

    assert loaded.problem_ids == tuple(sorted(record.id for record in records))
    assert tuple(chunk.problem_ids for chunk in loaded.chunks) == (
        ("train-000001", "train-000002"),
        ("train-000003",),
    )
    prompt = build_teacher_prompt(loaded, 0)
    assert "900001" not in prompt
    assert "900002" not in prompt
    assert records[0].question_raw in prompt
    assert loaded.prompt_policy.as_dict() == {
        "prompt_version": "gate-b-codex-teacher-prompt-v1",
        "min_rationale_characters": 16,
        "max_rationale_characters": 1_500,
        "min_total_lines": 2,
        "max_total_lines": 12,
    }
    assert loaded.prompt_policy.sha256 == (
        "71f080314fb346390c473237ddac04d2ca5070febe31ac55346222bb8a9e8814"
    )
    assert prompt == (
        "You are a concise mathematical-reasoning teacher. Solve every supplied "
        "problem without tools, browsing, code execution, or external calls. "
        "For each item, write a self-contained 2 to 6 line rationale and end the "
        "target text with exactly `Final answer: N`, where N is one integer. "
        "Return only one JSON object matching this exact schema:\n"
        '{"items":[{"problem_id":"...","target_text":"..."}]}\n'
        "Keep the item order unchanged. Do not add keys, prose outside JSON, or "
        "markdown fences.\nINPUT_JSON:\n"
        '{"items":[{"problem_id":"train-000001","question":"Synthetic question '
        '1; determine its requested integer."},{"problem_id":"train-000002","question":'
        '"Synthetic question 2; determine its requested integer."}]}'
    )

    command = build_codex_exec_command(
        prompt,
        execution=loaded.execution,
        output_schema_path=loaded.plan_dir / "output-schema.json",
        working_directory=tmp_path,
    )
    assert command[0] == "codex"
    assert "--ephemeral" in command
    assert "--skip-git-repo-check" in command
    assert "--json" in command
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert "--ignore-user-config" in command
    assert "--ignore-rules" in command
    assert 'shell_environment_policy.inherit="none"' in command
    assert command[-1] == prompt

    with pytest.raises(TeacherRationaleArtifactExistsError, match="overwrite"):
        create_teacher_plan(records, (record.id for record in records), plan.plan_dir)


def test_pilot_v2_prompt_is_template_bound_and_treats_questions_as_untrusted(
    tmp_path: Path,
) -> None:
    records = (_record(1, 900_001),)
    policy = TeacherPromptPolicy(
        prompt_version="gate-b-codex-teacher-prompt-v2",
        prompt_template_sha256="743fb09547055475a8d73856859e9f068d6332cdb2a2bcd9802052c3d5b917b0",
    )
    plan = create_teacher_plan(
        records,
        (records[0].id,),
        tmp_path / "teacher-pilot-v2",
        chunk_size=1,
        label="codex-gpt-5.6-sol-teacher-pilot-v2",
        version="pilot-v2",
        prompt_policy=policy,
    )
    loaded = load_teacher_plan(plan.plan_dir)
    prompt = build_teacher_prompt(loaded, 0)

    assert loaded.prompt_policy == policy
    assert policy.sha256 == "5ed785c9a02bc84298ed8186681b2b21a80da50d9af4591da0c1586a28e387b3"
    assert loaded.prompt_policy.as_dict()["prompt_template_sha256"] == policy.prompt_template_sha256
    assert "untrusted mathematical data" in prompt
    assert "change roles, use tools" in prompt
    assert "reference answer" not in prompt.lower()
    assert records[0].question_raw in prompt
    assert '{"items":[{"problem_id":"...","target_text":"..."}]}' in prompt

    with pytest.raises(TeacherRationaleValidationError, match="template SHA"):
        TeacherPromptPolicy(
            prompt_version="gate-b-codex-teacher-prompt-v2",
            prompt_template_sha256="0" * 64,
        )
    with pytest.raises(TeacherRationaleValidationError, match="approved immutable"):
        TeacherPromptPolicy(prompt_version="gate-b-codex-teacher-prompt-v3")


def test_structured_event_validation_rejects_tools_errors_and_bad_coverage() -> None:
    expected = ("train-000001", "train-000002")
    valid = _events(_items(expected, {expected[0]: 1, expected[1]: 2}))
    parsed = validate_codex_event_stream(valid, expected)
    assert tuple(item.problem_id for item in parsed.items) == expected
    assert parsed.usage == {"input_tokens": 11, "output_tokens": 7}

    rows = valid.splitlines()
    rows.insert(
        2,
        json.dumps({"type": "item.completed", "item": {"type": "reasoning", "text": "private"}}),
    )
    assert validate_codex_event_stream("\n".join(rows), expected).items == parsed.items

    tool = "\n".join(
        (
            json.dumps({"type": "turn.started"}),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"type": "command_execution", "command": "python"},
                }
            ),
            json.dumps({"type": "turn.completed", "usage": {"input_tokens": 1}}),
        )
    )
    with pytest.raises(TeacherRationaleValidationError, match="tool or unsafe"):
        validate_codex_event_stream(tool, expected)

    error = json.dumps({"type": "error", "message": "nope"})
    with pytest.raises(TeacherRationaleValidationError, match="unsupported or unsafe"):
        validate_codex_event_stream(error, expected)

    duplicate = _events(
        [
            {"problem_id": expected[0], "target_text": _target(1)},
            {"problem_id": expected[0], "target_text": _target(2)},
        ]
    )
    with pytest.raises(TeacherRationaleValidationError, match="missing, reordered, or mismatched"):
        validate_codex_event_stream(duplicate, expected)

    missing = _events([{"problem_id": expected[0], "target_text": _target(1)}])
    with pytest.raises(TeacherRationaleValidationError, match="item count"):
        validate_codex_event_stream(missing, expected)


def test_run_resume_finalize_excludes_accepted_rows_and_writes_source_schema(
    tmp_path: Path,
) -> None:
    records, plan = _make_plan(tmp_path, count=2, chunk_size=2)
    expected = plan.chunks[0].problem_ids
    answers = {record.id: record.answer for record in records}
    assert all(isinstance(answer, int) for answer in answers.values())
    calls: list[tuple[str, ...]] = []

    def runner(command: tuple[str, ...]) -> CodexCommandResult:
        calls.append(command)
        prompt = command[-1]
        if len(calls) == 1:
            assert expected[0] in prompt and expected[1] in prompt
            generated = _items(expected, {expected[0]: 900001, expected[1]: 444})
        else:
            assert expected[0] not in prompt
            assert expected[1] in prompt
            generated = _items((expected[1],), {expected[1]: 900002})
        return CodexCommandResult(stdout=_events(generated), stderr="", returncode=0, latency_ms=23)

    first = run_teacher_plan(plan.plan_dir, runner, max_chunks=1)
    assert first.attempts_written == 1
    assert first.parsed_attempts == 1
    assert run_teacher_plan(plan.plan_dir, runner).attempts_written == 0

    before = teacher_status(plan.plan_dir).as_dict()
    raw_free = json.dumps(before, sort_keys=True)
    assert records[0].question_raw not in raw_free
    assert expected[0] not in raw_free
    assert "900001" not in raw_free
    assert before["unassessed_problem_count"] == 2

    incomplete = finalize_teacher_bank(
        plan.plan_dir,
        records,
        output_jsonl=tmp_path / "bank.jsonl",
        output_manifest=tmp_path / "bank-manifest.json",
    )
    assert not incomplete.complete
    assert incomplete.accepted_problem_count == 1
    assert incomplete.pending_problem_count == 1

    retry = run_teacher_plan(
        plan.plan_dir,
        runner,
        repair_chunk_size=1,
    )
    assert retry.attempts_written == 1
    assert retry.parsed_attempts == 1
    assert 'model_reasoning_effort="xhigh"' in calls[-1]
    complete = finalize_teacher_bank(
        plan.plan_dir,
        records,
        output_jsonl=tmp_path / "bank.jsonl",
        output_manifest=tmp_path / "bank-manifest.json",
    )
    assert complete.complete
    assert complete.source_jsonl_sha256 is not None
    rows = [json.loads(line) for line in (tmp_path / "bank.jsonl").read_text().splitlines()]
    assert [row["problem_id"] for row in rows] == list(expected)
    assert all(row["schema_version"] == "gate-b-concise-rationale-row-v1" for row in rows)
    assert all(row["teacher"]["reference_answer_in_prompt"] is False for row in rows)
    assert all(row["verification"]["status"] == "accepted" for row in rows)
    assert len(calls) == 2

    with pytest.raises(TeacherRationaleArtifactExistsError, match="overwrite"):
        finalize_teacher_bank(
            plan.plan_dir,
            records,
            output_jsonl=tmp_path / "bank.jsonl",
            output_manifest=tmp_path / "bank-manifest.json",
        )


def test_run_assigns_initial_and_repair_effort_per_job(tmp_path: Path) -> None:
    records, plan = _make_plan(tmp_path, count=2, chunk_size=1)
    answers = {record.id: record.answer for record in records}
    assert all(isinstance(answer, int) for answer in answers.values())
    calls: list[tuple[str, ...]] = []

    def runner(command: tuple[str, ...]) -> CodexCommandResult:
        calls.append(command)
        input_ids = tuple(
            identifier for identifier in plan.problem_ids if identifier in command[-1]
        )
        assert len(input_ids) == 1
        answer = answers[input_ids[0]]
        assert isinstance(answer, int)
        emitted = answer + 1 if len(calls) == 1 else answer
        return CodexCommandResult(
            stdout=_events(_items(input_ids, {input_ids[0]: emitted})),
            stderr="",
            returncode=0,
            latency_ms=1,
        )

    first = run_teacher_plan(plan.plan_dir, runner, max_chunks=1)
    assert first.attempts_written == 1
    assert 'model_reasoning_effort="high"' in calls[0]
    incomplete = finalize_teacher_bank(plan.plan_dir, records)
    assert not incomplete.complete

    second = run_teacher_plan(plan.plan_dir, runner)
    assert second.attempts_written == 2
    effort_by_id = {
        next(identifier for identifier in plan.problem_ids if identifier in command[-1]): next(
            value
            for value in command
            if value.startswith("model_reasoning_effort=")
        )
        for command in calls[1:]
    }
    assert effort_by_id[plan.problem_ids[0]] == 'model_reasoning_effort="xhigh"'
    assert effort_by_id[plan.problem_ids[1]] == 'model_reasoning_effort="high"'


def test_lock_tamper_and_exhaustion_fail_closed(tmp_path: Path) -> None:
    records, plan = _make_plan(tmp_path, count=2, chunk_size=1)
    expected = plan.chunks[0].problem_ids
    calls = 0

    def wrong_runner(_command: tuple[str, ...]) -> CodexCommandResult:
        nonlocal calls
        calls += 1
        return CodexCommandResult(
            stdout=_events(_items(expected, {expected[0]: -1})),
            stderr="",
            returncode=0,
            latency_ms=1,
        )

    with teacher_plan_lock(plan.plan_dir), pytest.raises(TeacherPlanLockError, match="lock"):
        run_teacher_plan(plan.plan_dir, wrong_runner)

    run_teacher_plan(plan.plan_dir, wrong_runner, max_attempts=1, max_chunks=1)
    with pytest.raises(TeacherRationaleValidationError, match="retries are exhausted"):
        finalize_teacher_bank(
            plan.plan_dir,
            records,
            output_jsonl=tmp_path / "bank.jsonl",
            output_manifest=tmp_path / "bank-manifest.json",
            max_attempts=1,
        )

    status = teacher_status(plan.plan_dir, max_attempts=1)
    assert status.exhausted_problem_count == 1
    assert status.retryable_problem_count == 1
    with pytest.raises(
        TeacherRationaleValidationError,
        match="refusing further plan execution",
    ):
        run_teacher_plan(plan.plan_dir, wrong_runner, max_attempts=1)
    assert calls == 1
    assert not (tmp_path / "bank.jsonl").exists()
    assert not (tmp_path / "bank-manifest.json").exists()

    attempt = next((plan.plan_dir / "attempts").glob("*.json"))
    payload = json.loads(attempt.read_text())
    payload["latency_ms"] = 999
    attempt.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(TeacherRationaleValidationError, match="payload SHA"):
        teacher_status(plan.plan_dir)


def test_attempt_loader_reconstructs_prompt_and_safe_command_before_finalization(
    tmp_path: Path,
) -> None:
    records, plan = _make_plan(tmp_path, count=1, chunk_size=1)
    answer = records[0].answer
    assert isinstance(answer, int)

    def runner(_command: tuple[str, ...]) -> CodexCommandResult:
        return CodexCommandResult(
            stdout=_events(_items(plan.problem_ids, {plan.problem_ids[0]: answer})),
            stderr="",
            returncode=0,
            latency_ms=1,
        )

    run_teacher_plan(plan.plan_dir, runner)
    attempt_path = next((plan.plan_dir / "attempts").glob("*.json"))
    payload = json.loads(attempt_path.read_text(encoding="utf-8"))
    command = list(payload["command_argv"])
    # Rehashing every local field must not make an answer-bearing substitute
    # prompt acceptable to the finalizer/status readers.
    command[-1] = command[-1] + "\nREFERENCE ANSWER: 900001"
    payload["command_argv"] = command
    payload["prompt_sha256"] = hashlib.sha256(command[-1].encode("utf-8")).hexdigest()
    payload["command_sha256"] = hashlib.sha256(canonical_json_bytes(command)).hexdigest()
    _rewrite_self_hashed_attempt(attempt_path, payload)

    with pytest.raises(TeacherRationaleValidationError, match="reconstructed locked prompt"):
        teacher_status(plan.plan_dir)


def test_attempt_loader_rejects_self_hashed_command_without_inherit_none(
    tmp_path: Path,
) -> None:
    records, plan = _make_plan(tmp_path, count=1, chunk_size=1)
    answer = records[0].answer
    assert isinstance(answer, int)

    def runner(_command: tuple[str, ...]) -> CodexCommandResult:
        return CodexCommandResult(
            stdout=_events(_items(plan.problem_ids, {plan.problem_ids[0]: answer})),
            stderr="",
            returncode=0,
            latency_ms=1,
        )

    run_teacher_plan(plan.plan_dir, runner)
    attempt_path = next((plan.plan_dir / "attempts").glob("*.json"))
    payload = json.loads(attempt_path.read_text(encoding="utf-8"))
    command = list(payload["command_argv"])
    environment_policy_index = command.index('shell_environment_policy.inherit="none"')
    assert command[environment_policy_index - 1] == "-c"
    del command[environment_policy_index - 1 : environment_policy_index + 1]
    payload["command_argv"] = command
    payload["command_sha256"] = hashlib.sha256(canonical_json_bytes(command)).hexdigest()
    _rewrite_self_hashed_attempt(attempt_path, payload)

    with pytest.raises(TeacherRationaleValidationError, match="reconstructed safe Codex command"):
        teacher_status(plan.plan_dir)


def test_audit_loader_reconstructs_safe_command_before_finalization(tmp_path: Path) -> None:
    _records, teacher_plan, source, bank_manifest = _make_finalized_teacher_bank(
        tmp_path,
        count=64,
    )
    audit_plan = create_teacher_logical_audit_plan(
        teacher_plan.plan_dir,
        source,
        bank_manifest,
        tmp_path / "logical-audit-command-tamper",
    )

    def runner(_command: tuple[str, ...]) -> CodexCommandResult:
        return CodexCommandResult(
            stdout=_audit_events(
                audit_plan.problem_ids,
                {problem_id: True for problem_id in audit_plan.problem_ids},
            ),
            stderr="",
            returncode=0,
            latency_ms=1,
        )

    run_teacher_logical_audit(audit_plan.audit_dir, runner)
    attempt_path = next((audit_plan.audit_dir / "audit-attempts").glob("*.json"))
    payload = json.loads(attempt_path.read_text(encoding="utf-8"))
    command = list(payload["command_argv"])
    command[command.index("read-only")] = "workspace-write"
    payload["command_argv"] = command
    payload["command_sha256"] = hashlib.sha256(canonical_json_bytes(command)).hexdigest()
    _rewrite_self_hashed_attempt(attempt_path, payload)

    with pytest.raises(TeacherRationaleValidationError, match="reconstructed safe Codex command"):
        teacher_logical_audit_status(audit_plan.audit_dir)


def test_two_workers_are_bounded_and_preallocate_distinct_attempt_evidence(
    tmp_path: Path,
) -> None:
    records, plan = _make_plan(tmp_path, count=4, chunk_size=1)
    answers = {record.id: record.answer for record in records}
    gate = threading.Barrier(2, timeout=5)
    active_lock = threading.Lock()
    active = 0
    peak_active = 0
    calls: list[tuple[str, ...]] = []

    def runner(command: tuple[str, ...]) -> CodexCommandResult:
        nonlocal active, peak_active
        prompt = command[-1]
        input_ids = tuple(identifier for identifier in plan.problem_ids if identifier in prompt)
        assert len(input_ids) == 1
        with active_lock:
            active += 1
            peak_active = max(peak_active, active)
            calls.append(command)
        try:
            gate.wait()
        finally:
            with active_lock:
                active -= 1
        return CodexCommandResult(
            stdout=_events(_items(input_ids, {input_ids[0]: int(answers[input_ids[0]])})),
            stderr="",
            returncode=0,
            latency_ms=4,
        )

    result = run_teacher_plan(plan.plan_dir, runner, max_workers=2)

    assert result.attempts_written == 4
    assert result.parsed_attempts == 4
    assert result.failed_attempts == 0
    assert peak_active == 2
    assert len(calls) == 4
    attempts = sorted((plan.plan_dir / "attempts").glob("*.json"))
    assert [path.name for path in attempts] == [
        "chunk-000000-attempt-000001.json",
        "chunk-000001-attempt-000001.json",
        "chunk-000002-attempt-000001.json",
        "chunk-000003-attempt-000001.json",
    ]


def test_worker_bound_cap_and_resume_remain_fail_closed(tmp_path: Path) -> None:
    records, plan = _make_plan(tmp_path, count=3, chunk_size=1)
    answers = {record.id: record.answer for record in records}
    calls: list[tuple[str, ...]] = []

    def runner(command: tuple[str, ...]) -> CodexCommandResult:
        calls.append(command)
        prompt = command[-1]
        input_ids = tuple(identifier for identifier in plan.problem_ids if identifier in prompt)
        assert len(input_ids) == 1
        return CodexCommandResult(
            stdout=_events(_items(input_ids, {input_ids[0]: int(answers[input_ids[0]])})),
            stderr="",
            returncode=0,
            latency_ms=2,
        )

    for invalid in (False, 0, 3):
        with pytest.raises(TeacherRationaleValidationError, match="max_workers"):
            run_teacher_plan(plan.plan_dir, runner, max_workers=invalid)
    assert not calls

    capped = run_teacher_plan(plan.plan_dir, runner, max_workers=2, max_chunks=2)
    assert capped.attempts_written == 2
    assert capped.parsed_attempts == 2
    assert len(calls) == 2
    assert teacher_status(plan.plan_dir).unassessed_problem_count == 2

    # A different runner cannot start a competing schedule while the plan is
    # already locked, even if the active schedule uses two workers.
    with teacher_plan_lock(plan.plan_dir), pytest.raises(TeacherPlanLockError, match="lock"):
        run_teacher_plan(plan.plan_dir, runner, max_workers=2)


def test_two_workers_preallocate_distinct_repair_attempt_numbers_within_one_chunk(
    tmp_path: Path,
) -> None:
    records, plan = _make_plan(tmp_path, count=3, chunk_size=3)
    answers = {record.id: record.answer for record in records}
    expected = plan.chunks[0].problem_ids

    def wrong_runner(_command: tuple[str, ...]) -> CodexCommandResult:
        return CodexCommandResult(
            stdout=_events(_items(expected, {identifier: -1 for identifier in expected})),
            stderr="",
            returncode=0,
            latency_ms=1,
        )

    run_teacher_plan(plan.plan_dir, wrong_runner)
    incomplete = finalize_teacher_bank(plan.plan_dir, records)
    assert not incomplete.complete
    assert incomplete.pending_problem_count == len(expected)

    repair_calls: list[tuple[str, ...]] = []

    def repair_runner(command: tuple[str, ...]) -> CodexCommandResult:
        repair_calls.append(command)
        prompt = command[-1]
        input_ids = tuple(identifier for identifier in expected if identifier in prompt)
        assert len(input_ids) == 1
        return CodexCommandResult(
            stdout=_events(_items(input_ids, {input_ids[0]: int(answers[input_ids[0]])})),
            stderr="",
            returncode=0,
            latency_ms=1,
        )

    repaired = run_teacher_plan(
        plan.plan_dir,
        repair_runner,
        repair_chunk_size=1,
        max_workers=2,
    )
    assert repaired.attempts_written == 3
    assert repaired.parsed_attempts == 3
    assert len(repair_calls) == 3
    assert all('model_reasoning_effort="xhigh"' in command for command in repair_calls)
    assert [path.name for path in sorted((plan.plan_dir / "attempts").glob("*.json"))] == [
        "chunk-000000-attempt-000001.json",
        "chunk-000000-attempt-000002.json",
        "chunk-000000-attempt-000003.json",
        "chunk-000000-attempt-000004.json",
    ]


def test_stale_lock_requires_explicit_recovery(tmp_path: Path) -> None:
    _records, plan = _make_plan(tmp_path, count=1, chunk_size=1)
    lock = plan.plan_dir / ".teacher-rationale.lock"
    lock.write_text(
        json.dumps(
            {
                "schema_version": "gate-b-codex-teacher-lock-v1",
                "pid": 999_999_999,
                "token": "synthetic",
            }
        ),
        encoding="utf-8",
    )
    assert teacher_status(plan.plan_dir).lock_state == "stale"
    with pytest.raises(TeacherPlanLockError, match="stale"), teacher_plan_lock(plan.plan_dir):
        pass
    with teacher_plan_lock(plan.plan_dir, allow_stale_recovery=True):
        assert lock.exists()
    assert not lock.exists()


def test_finalizer_rejects_records_outside_exact_plan_scope(tmp_path: Path) -> None:
    records, plan = _make_plan(tmp_path, count=2, chunk_size=2)
    with pytest.raises(TeacherRationaleValidationError, match="exactly match"):
        finalize_teacher_bank(plan.plan_dir, records[:1])


def test_finalized_bank_is_consumable_by_existing_rationale_corpus_builder(tmp_path: Path) -> None:
    all_records = tuple(_record(index, index) for index in range(1, 13))
    identifiers = tuple(record.id for record in all_records)
    split = make_grouped_split_manifest(
        identifiers,
        dict(zip(identifiers, identifiers, strict=True)),
        n_folds=2,
        holdout_fraction=0.25,
        seed=20_260_731,
        version="teacher-rationale-compat-test",
    )
    allowed = eligible_training_ids(split, 0, ())
    records = tuple(record for record in all_records if record.id in set(allowed))
    plan = create_teacher_plan(records, allowed, tmp_path / "compat-plan")
    answers = {record.id: record.answer for record in records}

    def runner(_command: tuple[str, ...]) -> CodexCommandResult:
        return CodexCommandResult(
            stdout=_events(
                _items(
                    plan.problem_ids,
                    {identifier: int(answers[identifier]) for identifier in plan.problem_ids},
                )
            ),
            stderr="",
            returncode=0,
            latency_ms=3,
        )

    run_teacher_plan(plan.plan_dir, runner)
    source = tmp_path / "private-source.jsonl"
    source_manifest = tmp_path / "private-bank-manifest.json"
    finalized = finalize_teacher_bank(
        plan.plan_dir,
        records,
        output_jsonl=source,
        output_manifest=source_manifest,
    )
    assert finalized.complete
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    corpus = build_rationale_corpus(
        source,
        records,
        split_manifest=split,
        fold=0,
        excluded_ids=(),
        candidate_config_file_sha256="c" * 64,
        output_jsonl=corpus_dir / "rationales.jsonl",
        output_manifest=corpus_dir / "manifest.json",
    )
    assert corpus.record_count == len(allowed)


def test_logical_audit_plan_is_deterministic_provenance_bound_and_candidate_only(
    tmp_path: Path,
) -> None:
    _records, teacher_plan, source, bank_manifest = _make_finalized_teacher_bank(
        tmp_path,
        count=65,
    )
    first = create_teacher_logical_audit_plan(
        teacher_plan.plan_dir,
        source,
        bank_manifest,
        tmp_path / "logical-audit-a",
    )
    second = create_teacher_logical_audit_plan(
        teacher_plan.plan_dir,
        source,
        bank_manifest,
        tmp_path / "logical-audit-b",
    )

    assert first.problem_ids == second.problem_ids
    assert len(first.problem_ids) == 64
    assert first.teacher_plan_sha256 == teacher_plan.plan_sha256
    assert first.source_jsonl_sha256
    prompt = build_teacher_logical_audit_prompt(first)
    assert first.items[0].question in prompt
    assert json.dumps(first.items[0].target_text, ensure_ascii=False)[1:-1] in prompt
    assert "answer key" in prompt
    assert "reference_answer" not in json.dumps(first.as_dict(), sort_keys=True)

    source.write_text("{}\n", encoding="utf-8")
    with pytest.raises(TeacherRationaleValidationError, match="source JSONL"):
        create_teacher_logical_audit_plan(
            teacher_plan.plan_dir,
            source,
            bank_manifest,
            tmp_path / "logical-audit-tampered",
        )


def test_logical_audit_safe_events_run_finalize_and_threshold_fail_closed(tmp_path: Path) -> None:
    records, teacher_plan, source, bank_manifest = _make_finalized_teacher_bank(tmp_path)
    audit_plan = create_teacher_logical_audit_plan(
        teacher_plan.plan_dir,
        source,
        bank_manifest,
        tmp_path / "logical-audit",
    )
    expected = audit_plan.problem_ids
    valid = _audit_events(
        expected,
        {identifier: index >= 4 for index, identifier in enumerate(expected)},
    )
    parsed = validate_codex_logical_audit_event_stream(valid, expected)
    assert sum(item.consistent for item in parsed.items) == 60

    tool = "\n".join(
        (
            json.dumps({"type": "turn.started"}),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"type": "command_execution", "command": "python"},
                }
            ),
            json.dumps({"type": "turn.completed", "usage": {"input_tokens": 1}}),
        )
    )
    with pytest.raises(TeacherRationaleValidationError, match="tool or unsafe"):
        validate_codex_logical_audit_event_stream(tool, expected)

    reordered = (expected[1], expected[0], *expected[2:])
    with pytest.raises(TeacherRationaleValidationError, match="missing, reordered, or mismatched"):
        validate_codex_logical_audit_event_stream(
            _audit_events(reordered, {identifier: True for identifier in reordered}),
            expected,
        )

    calls: list[tuple[str, ...]] = []

    def audit_runner(command: tuple[str, ...]) -> CodexCommandResult:
        calls.append(command)
        assert records[0].question_raw in command[-1] or records[-1].question_raw in command[-1]
        return CodexCommandResult(stdout=valid, stderr="", returncode=0, latency_ms=5)

    run = run_teacher_logical_audit(audit_plan.audit_dir, audit_runner)
    assert run.parsed_attempts == 1
    assert run_teacher_logical_audit(audit_plan.audit_dir, audit_runner).skipped_completed
    result = finalize_teacher_logical_audit(audit_plan.audit_dir)
    assert result.complete
    assert result.passed
    assert result.consistent_problem_count == 60
    assert result.inconsistent_problem_count == 4
    assert (
        finalize_teacher_logical_audit(audit_plan.audit_dir).manifest_sha256
        == result.manifest_sha256
    )
    status = teacher_logical_audit_status(audit_plan.audit_dir).as_dict()
    raw_free = json.dumps(status, sort_keys=True)
    assert records[0].question_raw not in raw_free
    assert expected[0] not in raw_free
    assert "900001" not in raw_free
    assert len(calls) == 1
    with pytest.raises(TeacherRationaleArtifactExistsError, match="overwrite"):
        create_teacher_logical_audit_plan(
            teacher_plan.plan_dir,
            source,
            bank_manifest,
            audit_plan.audit_dir,
        )

    failed_plan = create_teacher_logical_audit_plan(
        teacher_plan.plan_dir,
        source,
        bank_manifest,
        tmp_path / "logical-audit-fail",
    )
    failed_expected = failed_plan.problem_ids

    def failed_runner(_command: tuple[str, ...]) -> CodexCommandResult:
        return CodexCommandResult(
            stdout=_audit_events(
                failed_expected,
                {identifier: index >= 5 for index, identifier in enumerate(failed_expected)},
            ),
            stderr="",
            returncode=0,
            latency_ms=5,
        )

    run_teacher_logical_audit(failed_plan.audit_dir, failed_runner)
    failed = finalize_teacher_logical_audit(failed_plan.audit_dir)
    assert failed.complete
    assert not failed.passed
    assert failed.consistent_problem_count == 59

    retry_plan = create_teacher_logical_audit_plan(
        teacher_plan.plan_dir,
        source,
        bank_manifest,
        tmp_path / "logical-audit-retry",
    )
    retry_calls: list[tuple[str, ...]] = []

    def retry_runner(command: tuple[str, ...]) -> CodexCommandResult:
        retry_calls.append(command)
        if len(retry_calls) == 1:
            return CodexCommandResult(stdout="", stderr="", returncode=1, latency_ms=1)
        return CodexCommandResult(stdout=valid, stderr="", returncode=0, latency_ms=1)

    assert run_teacher_logical_audit(retry_plan.audit_dir, retry_runner).failed_attempts == 1
    assert run_teacher_logical_audit(retry_plan.audit_dir, retry_runner).parsed_attempts == 1
    assert 'model_reasoning_effort="high"' in retry_calls[0]
    assert 'model_reasoning_effort="xhigh"' in retry_calls[1]

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from deep_challenge.data import MathRecord
from deep_challenge.provenance import canonical_json_bytes
from deep_challenge.teacher_pilot_authorization import (
    FULL_V1_BANK_AUTHORIZATION_SCHEMA,
    FULL_V1_BANK_AUTHORIZATION_V2_SCHEMA,
    PILOT_AUTHORIZATION_PILOT_SIZE,
    PILOT_AUTHORIZATION_SCHEMA,
    PILOT_AUTHORIZATION_V2_SCHEMA,
    TeacherPilotAuthorizationContract,
    TeacherPilotAuthorizationError,
    create_teacher_pilot_authorization,
    load_teacher_full_v1_bank_authorization,
    verify_teacher_pilot_authorization,
    write_teacher_full_v1_bank_authorization,
)
from deep_challenge.teacher_rationale import (
    CodexCommandResult,
    TeacherPromptPolicy,
    create_teacher_logical_audit_plan,
    create_teacher_plan,
    finalize_teacher_bank,
    finalize_teacher_logical_audit,
    load_teacher_plan,
    run_teacher_logical_audit,
    run_teacher_plan,
)


def _record(index: int) -> MathRecord:
    identifier = f"train-{index:06d}"
    question = f"Synthetic pilot question {index}; determine the requested integer."
    return MathRecord(
        id=identifier,
        question_raw=question,
        question_normalized=question,
        answer_raw=str(80_000 + index),
        answer=80_000 + index,
        row_number=index + 1,
    )


def _teacher_events(items: list[dict[str, str]]) -> str:
    message = json.dumps({"items": items}, separators=(",", ":"))
    events = (
        {"type": "thread.started"},
        {"type": "turn.started"},
        {"type": "item.completed", "item": {"type": "agent_message", "text": message}},
        {"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 6}},
    )
    return "\n".join(json.dumps(event, separators=(",", ":")) for event in events)


def _audit_events(problem_ids: list[str]) -> str:
    message = json.dumps(
        {"items": [{"problem_id": problem_id, "consistent": True} for problem_id in problem_ids]},
        separators=(",", ":"),
    )
    events = (
        {"type": "thread.started"},
        {"type": "turn.started"},
        {"type": "item.completed", "item": {"type": "agent_message", "text": message}},
        {"type": "turn.completed", "usage": {"input_tokens": 11, "output_tokens": 7}},
    )
    return "\n".join(json.dumps(event, separators=(",", ":")) for event in events)


def _prompt_ids(command: tuple[str, ...]) -> list[str]:
    payload = json.loads(command[-1].rsplit("INPUT_JSON:\n", 1)[1])
    return [item["problem_id"] for item in payload["items"]]


def _contract(
    records: tuple[MathRecord, ...],
    *,
    teacher_plan_label: str = "codex-gpt-5.6-sol-teacher",
    teacher_plan_version: str = "v1",
    prompt_policy: TeacherPromptPolicy | None = None,
) -> TeacherPilotAuthorizationContract:
    policy = prompt_policy or TeacherPromptPolicy()
    return TeacherPilotAuthorizationContract(
        teacher_config_sha256="1" * 64,
        teacher_config_file_sha256="2" * 64,
        train_sha256="3" * 64,
        exclusions_sha256="4" * 64,
        exclusion_count=0,
        split_artifact_sha256="5" * 64,
        development_shard_sha256="6" * 64,
        split_version="v4",
        split_sha256="7" * 64,
        source_groups_sha256="8" * 64,
        fold=0,
        fold0_training_ids=tuple(record.id for record in records),
        pilot_ids=tuple(record.id for record in records),
        teacher_plan_label=teacher_plan_label,
        teacher_plan_version=teacher_plan_version,
        logical_audit_label="codex-gpt-5.6-sol-logical-audit",
        logical_audit_version="v1",
        teacher_prompt_policy_sha256=policy.sha256,
    )


def _finalized_pilot(
    tmp_path: Path,
    *,
    initial_wrong_ids: set[str] | None = None,
    chunk_size: int = 64,
    label: str = "codex-gpt-5.6-sol-teacher",
    version: str = "v1",
    prompt_policy: TeacherPromptPolicy | None = None,
) -> tuple[
    tuple[MathRecord, ...],
    Path,
    Path,
    Path,
    Path,
]:
    records = tuple(_record(index) for index in range(1, PILOT_AUTHORIZATION_PILOT_SIZE + 1))
    answers = {record.id: record.answer for record in records}
    assert all(answer is not None for answer in answers.values())
    plan = create_teacher_plan(
        records,
        tuple(record.id for record in records),
        tmp_path / "private-pilot-plan",
        chunk_size=chunk_size,
        label=label,
        version=version,
        prompt_policy=prompt_policy or TeacherPromptPolicy(),
    )
    first_full_chunk_calls = 0

    def teacher_runner(command: tuple[str, ...]) -> CodexCommandResult:
        nonlocal first_full_chunk_calls
        problem_ids = _prompt_ids(command)
        initial = len(problem_ids) == chunk_size and first_full_chunk_calls < len(records)
        if initial:
            first_full_chunk_calls += 1
        items: list[dict[str, str]] = []
        for problem_id in problem_ids:
            answer = answers[problem_id]
            assert isinstance(answer, int)
            emitted = answer
            if initial and initial_wrong_ids and problem_id in initial_wrong_ids:
                emitted += 1
            items.append(
                {
                    "problem_id": problem_id,
                    "target_text": (
                        "The stated relation determines a unique integer.\n"
                        f"Final answer: {emitted}"
                    ),
                }
            )
        return CodexCommandResult(
            stdout=_teacher_events(items), stderr="", returncode=0, latency_ms=1
        )

    run_teacher_plan(plan.plan_dir, teacher_runner)
    source = tmp_path / "private-source.jsonl"
    source_manifest = tmp_path / "private-source-manifest.json"
    first_finalized = finalize_teacher_bank(
        plan.plan_dir,
        records,
        output_jsonl=source,
        output_manifest=source_manifest,
    )
    if not first_finalized.complete:
        run_teacher_plan(plan.plan_dir, teacher_runner)
        final = finalize_teacher_bank(
            plan.plan_dir,
            records,
            output_jsonl=source,
            output_manifest=source_manifest,
        )
        assert final.complete

    audit = create_teacher_logical_audit_plan(
        plan.plan_dir,
        source,
        source_manifest,
        tmp_path / "private-logical-audit",
    )

    def audit_runner(command: tuple[str, ...]) -> CodexCommandResult:
        return CodexCommandResult(
            stdout=_audit_events(_prompt_ids(command)),
            stderr="",
            returncode=0,
            latency_ms=1,
        )

    run_teacher_logical_audit(audit.audit_dir, audit_runner)
    audit_final = finalize_teacher_logical_audit(audit.audit_dir)
    assert audit_final.complete and audit_final.passed
    return records, plan.plan_dir, source, source_manifest, audit.audit_dir


def _rewrite_audit_plan(path: Path, payload: dict[str, object]) -> None:
    unhashed = dict(payload)
    unhashed.pop("plan_sha256", None)
    payload["plan_sha256"] = hashlib.sha256(
        canonical_json_bytes(unhashed)
    ).hexdigest()
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def test_pilot_authorization_receipt_is_raw_free_immutable_and_reverified(
    tmp_path: Path,
) -> None:
    records, plan_dir, source, source_manifest, audit_dir = _finalized_pilot(tmp_path)
    receipt_path = tmp_path / "pilot-authorization.json"
    receipt = create_teacher_pilot_authorization(
        receipt_path,
        contract=_contract(records),
        pilot_plan_dir=plan_dir,
        source_jsonl=source,
        source_manifest=source_manifest,
        logical_audit_dir=audit_dir,
    )

    serialized = receipt_path.read_text(encoding="utf-8")
    assert "Synthetic pilot question" not in serialized
    assert "train-000001" not in serialized
    assert "80001" not in serialized
    assert receipt.initial_exact_match_accepted_count == 128
    assert receipt.initial_exact_match_total_count == 128
    assert receipt.logical_audit_consistent_problem_count == 64
    assert json.loads(serialized)["schema_version"] == PILOT_AUTHORIZATION_SCHEMA
    assert "teacher_prompt_policy_sha256" not in json.loads(serialized)
    assert verify_teacher_pilot_authorization(
        receipt_path,
        contract=_contract(records),
        pilot_plan_dir=plan_dir,
        source_jsonl=source,
        source_manifest=source_manifest,
        logical_audit_dir=audit_dir,
    ).file_sha256 == receipt.file_sha256
    sidecar_sha = write_teacher_full_v1_bank_authorization(plan_dir, receipt)
    sidecar = load_teacher_full_v1_bank_authorization(plan_dir)
    assert sidecar["schema_version"] == FULL_V1_BANK_AUTHORIZATION_SCHEMA
    assert "full_plan_prompt_policy_sha256" not in sidecar
    assert sidecar["payload_sha256"] == sidecar_sha
    with pytest.raises(TeacherPilotAuthorizationError, match="overwrite"):
        create_teacher_pilot_authorization(
            receipt_path,
            contract=_contract(records),
            pilot_plan_dir=plan_dir,
            source_jsonl=source,
            source_manifest=source_manifest,
            logical_audit_dir=audit_dir,
        )


@pytest.mark.parametrize(
    ("prompt_version", "template_sha256", "label", "version", "policy_sha256"),
    (
        (
            "gate-b-codex-teacher-prompt-v2",
            "743fb09547055475a8d73856859e9f068d6332cdb2a2bcd9802052c3d5b917b0",
            "codex-gpt-5.6-sol-teacher-pilot-v2",
            "pilot-v2",
            "5ed785c9a02bc84298ed8186681b2b21a80da50d9af4591da0c1586a28e387b3",
        ),
        (
            "gate-b-codex-teacher-prompt-v3",
            "cf56fc2c021410337f8be8f5f519912eabf6390aa8892ecd92cac1ced6175c72",
            "codex-gpt-5.6-sol-teacher-pilot-v3",
            "pilot-v3",
            "953d62e283d5237f29b2145b5ed513246d737acd7ec40879450d7bcc8d08402b",
        ),
        (
            "gate-b-codex-teacher-prompt-v4",
            "3029e9297bdda504e0f48e1ce4d57e363e5d3a5342edf18253b11c4f75ecd8a7",
            "codex-gpt-5.6-sol-teacher-pilot-v4",
            "pilot-v4",
            "8de961862f2cabf245753ee276d4b833d8917934d4ba84fa8f9caa20a64ab924",
        ),
    ),
)
def test_policy_bound_pilot_authorization_and_sidecar_bind_prompt_policy(
    tmp_path: Path,
    prompt_version: str,
    template_sha256: str,
    label: str,
    version: str,
    policy_sha256: str,
) -> None:
    policy = TeacherPromptPolicy(
        prompt_version=prompt_version,
        prompt_template_sha256=template_sha256,
    )
    assert policy.sha256 == policy_sha256
    records, plan_dir, source, source_manifest, audit_dir = _finalized_pilot(
        tmp_path,
        chunk_size=32,
        label=label,
        version=version,
        prompt_policy=policy,
    )
    receipt = create_teacher_pilot_authorization(
        tmp_path / "pilot-authorization.json",
        contract=_contract(
            records,
            teacher_plan_label=label,
            teacher_plan_version=version,
            prompt_policy=policy,
        ),
        pilot_plan_dir=plan_dir,
        source_jsonl=source,
        source_manifest=source_manifest,
        logical_audit_dir=audit_dir,
    )

    sidecar_sha = write_teacher_full_v1_bank_authorization(plan_dir, receipt)
    payload = load_teacher_full_v1_bank_authorization(plan_dir)
    receipt_payload = json.loads((tmp_path / "pilot-authorization.json").read_text())
    rendered = json.dumps(payload, sort_keys=True)
    assert receipt_payload["schema_version"] == PILOT_AUTHORIZATION_V2_SCHEMA
    assert receipt_payload["teacher_prompt_policy_sha256"] == policy.sha256
    assert payload["schema_version"] == FULL_V1_BANK_AUTHORIZATION_V2_SCHEMA
    assert payload["full_plan_prompt_policy_sha256"] == policy.sha256
    assert payload["pilot_plan_prompt_policy_sha256"] == policy.sha256
    assert payload["payload_sha256"] == sidecar_sha
    assert payload["pilot_authorization_file_sha256"] == receipt.file_sha256
    assert "Synthetic pilot question" not in rendered
    assert "train-000001" not in rendered
    assert "80001" not in rendered
    with pytest.raises(TeacherPilotAuthorizationError, match="overwrite"):
        write_teacher_full_v1_bank_authorization(plan_dir, receipt)


def test_pilot_authorization_rederives_audit_selection_and_candidate_bindings(
    tmp_path: Path,
) -> None:
    records, plan_dir, source, source_manifest, audit_dir = _finalized_pilot(tmp_path)
    audit_path = audit_dir / "audit-plan.json"
    teacher_plan = load_teacher_plan(plan_dir)
    source_rows = {
        row["problem_id"]: row
        for row in (
            json.loads(line)
            for line in source.read_text(encoding="utf-8").splitlines()
        )
    }
    question_by_id = {item.problem_id: item for item in teacher_plan.questions}
    alternate_ids = teacher_plan.problem_ids[64:]
    assert len(alternate_ids) == 64
    tampered = json.loads(audit_path.read_text(encoding="utf-8"))
    tampered["items"] = [
        {
            "problem_id": problem_id,
            "question": question_by_id[problem_id].question,
            "question_sha256": question_by_id[problem_id].question_sha256,
            "target_text": source_rows[problem_id]["target_text"],
            "target_sha256": source_rows[problem_id]["target_sha256"],
        }
        for problem_id in alternate_ids
    ]
    tampered["selected_ids_sha256"] = hashlib.sha256(
        canonical_json_bytes(list(alternate_ids))
    ).hexdigest()
    _rewrite_audit_plan(audit_path, tampered)

    with pytest.raises(
        TeacherPilotAuthorizationError,
        match="deterministic verified-bank sample",
    ):
        create_teacher_pilot_authorization(
            tmp_path / "must-not-publish.json",
            contract=_contract(records),
            pilot_plan_dir=plan_dir,
            source_jsonl=source,
            source_manifest=source_manifest,
            logical_audit_dir=audit_dir,
        )


def test_pilot_authorization_rejects_self_hashed_audit_candidate_swap(
    tmp_path: Path,
) -> None:
    records, plan_dir, source, source_manifest, audit_dir = _finalized_pilot(tmp_path)
    audit_path = audit_dir / "audit-plan.json"
    tampered = json.loads(audit_path.read_text(encoding="utf-8"))
    items = tampered["items"]
    assert isinstance(items, list) and len(items) == 64
    first, second = items[0], items[1]
    assert isinstance(first, dict) and isinstance(second, dict)
    first["target_text"] = second["target_text"]
    first["target_sha256"] = second["target_sha256"]
    _rewrite_audit_plan(audit_path, tampered)

    with pytest.raises(
        TeacherPilotAuthorizationError,
        match="deterministic verified-bank sample",
    ):
        create_teacher_pilot_authorization(
            tmp_path / "must-not-publish.json",
            contract=_contract(records),
            pilot_plan_dir=plan_dir,
            source_jsonl=source,
            source_manifest=source_manifest,
            logical_audit_dir=audit_dir,
        )


def test_pilot_authorization_rejects_below_80_percent_first_pass(
    tmp_path: Path,
) -> None:
    records, plan_dir, source, source_manifest, audit_dir = _finalized_pilot(
        tmp_path,
        initial_wrong_ids={f"train-{index:06d}" for index in range(101, 129)},
    )

    with pytest.raises(TeacherPilotAuthorizationError, match="first-pass"):
        create_teacher_pilot_authorization(
            tmp_path / "pilot-authorization.json",
            contract=_contract(records),
            pilot_plan_dir=plan_dir,
            source_jsonl=source,
            source_manifest=source_manifest,
            logical_audit_dir=audit_dir,
        )


def test_pilot_authorization_rejects_config_split_source_and_audit_mismatch(
    tmp_path: Path,
) -> None:
    records, plan_dir, source, source_manifest, audit_dir = _finalized_pilot(tmp_path)
    receipt_path = tmp_path / "pilot-authorization.json"
    contract = _contract(records)
    create_teacher_pilot_authorization(
        receipt_path,
        contract=contract,
        pilot_plan_dir=plan_dir,
        source_jsonl=source,
        source_manifest=source_manifest,
        logical_audit_dir=audit_dir,
    )
    for changed_contract in (
        replace(contract, teacher_config_sha256="9" * 64),
        replace(contract, teacher_prompt_policy_sha256="b" * 64),
        replace(contract, split_sha256="a" * 64),
    ):
        with pytest.raises(TeacherPilotAuthorizationError, match="does not"):
            verify_teacher_pilot_authorization(
                receipt_path,
                contract=changed_contract,
                pilot_plan_dir=plan_dir,
                source_jsonl=source,
                source_manifest=source_manifest,
                logical_audit_dir=audit_dir,
            )

    original_source = source.read_bytes()
    source.write_bytes(original_source + b"\n")
    with pytest.raises((TeacherPilotAuthorizationError, ValueError), match="source"):
        verify_teacher_pilot_authorization(
            receipt_path,
            contract=contract,
            pilot_plan_dir=plan_dir,
            source_jsonl=source,
            source_manifest=source_manifest,
            logical_audit_dir=audit_dir,
        )
    source.write_bytes(original_source)
    audit_manifest = audit_dir / "audit-manifest.json"
    audit_manifest.write_bytes(audit_manifest.read_bytes() + b"\n")
    with pytest.raises(TeacherPilotAuthorizationError, match="does not match"):
        verify_teacher_pilot_authorization(
            receipt_path,
            contract=contract,
            pilot_plan_dir=plan_dir,
            source_jsonl=source,
            source_manifest=source_manifest,
            logical_audit_dir=audit_dir,
        )

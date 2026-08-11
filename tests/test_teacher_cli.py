from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

import deep_challenge.cli as cli_module
from deep_challenge.cli import main
from deep_challenge.data import load_train_csv
from deep_challenge.provenance import (
    build_source_tree_manifest,
    validate_source_tree_manifest_artifact,
)
from deep_challenge.teacher_harness import (
    run_harness_live,
    run_harness_replay,
    synthetic_fixture_rows,
)
from deep_challenge.teacher_rationale import (
    CodexCommandResult,
    TeacherExecutionConfig,
    build_teacher_prompt,
    load_teacher_plan,
)


def _write_csv(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)


def _teacher_config_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "gate_b"
        / "codex-gpt-5.6-sol-teacher-v1.json"
    )


def _teacher_pilot_v2_config_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "gate_b"
        / "codex-gpt-5.6-sol-teacher-pilot-v2.json"
    )


def _teacher_pilot_v3_config_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "gate_b"
        / "codex-gpt-5.6-sol-teacher-pilot-v3.json"
    )


def _teacher_pilot_v4_config_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "gate_b"
        / "codex-gpt-5.6-sol-teacher-pilot-v4.json"
    )


def _harness_config_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "gate_b"
        / "codex-gpt-5.6-sol-teacher-harness-v1.json"
    )


def _teacher_contract(
    tmp_path: Path, *, row_count: int = 20
) -> tuple[list[str], Path]:
    def symbol(index: int) -> str:
        quotient, remainder = divmod(index - 1, 26)
        return f"{chr(97 + quotient)}{chr(97 + remainder)}"

    train = tmp_path / "train.csv"
    _write_csv(
        train,
        [
            ["id", "question", "answer"],
            *[
                [
                    f"train-{index:06d}",
                    f"Compute the unique expression with symbol letter {symbol(index)}.",
                    str(index - 10),
                ]
                for index in range(1, row_count + 1)
            ],
        ],
    )
    split = tmp_path / "split.json"
    assert (
        main(
            [
                "build-splits",
                "--train",
                str(train),
                "--folds",
                "2",
                "--seed",
                "7",
                "--output",
                str(split),
            ]
        )
        == 0
    )
    split_payload = json.loads(split.read_text(encoding="utf-8"))
    shard = tmp_path / "development-shard"
    assert (
        main(
            [
                "build-development-shard",
                "--train",
                str(train),
                "--split-artifact",
                str(split),
                "--expected-train-sha256",
                hashlib.sha256(train.read_bytes()).hexdigest(),
                "--expected-split-sha256",
                split_payload["split"]["sha256"],
                "--output-dir",
                str(shard),
            ]
        )
        == 0
    )
    exclusions = tmp_path / "exclusions.csv"
    _write_csv(exclusions, [["id"], ["train-000001"]])
    return (
        [
            "--train",
            str(train),
            "--train-exclusions",
            str(exclusions),
            "--split-artifact",
            str(split),
            "--fold",
            "0",
            "--expected-train-sha256",
            hashlib.sha256(train.read_bytes()).hexdigest(),
            "--expected-exclusions-sha256",
            hashlib.sha256(exclusions.read_bytes()).hexdigest(),
            "--expected-exclusion-count",
            "1",
            "--expected-split-sha256",
            split_payload["split"]["sha256"],
            "--development-shard",
            str(shard),
            "--expected-development-shard-sha256",
            hashlib.sha256((shard / "CHECKSUMS.sha256").read_bytes()).hexdigest(),
        ],
        train,
    )


def _event_stream_for_prompt(prompt: str, answer_by_id: dict[str, int]) -> str:
    input_payload = json.loads(prompt.rsplit("INPUT_JSON:\n", 1)[1])
    items = []
    for item in input_payload["items"]:
        problem_id = item["problem_id"]
        assert set(item) == {"problem_id", "question"}
        answer = answer_by_id[problem_id]
        items.append(
            {
                "problem_id": problem_id,
                "target_text": (
                    "Derive the requested integer from the stated mathematical relation.\n"
                    f"Final answer: {answer}"
                ),
            }
        )
    output = json.dumps({"items": items}, separators=(",", ":"))
    return "\n".join(
        json.dumps(event, separators=(",", ":"))
        for event in (
            {"type": "thread.started"},
            {"type": "turn.started"},
            {
                "type": "item.completed",
                "item": {"type": "agent_message", "text": output},
            },
            {"type": "turn.completed", "usage": {"input_tokens": 12, "output_tokens": 8}},
        )
    )


def _logical_audit_event_stream_for_prompt(prompt: str) -> str:
    input_payload = json.loads(prompt.rsplit("INPUT_JSON:\n", 1)[1])
    items = []
    for item in input_payload["items"]:
        assert set(item) == {
            "problem_id",
            "question",
            "candidate_rationale_and_final_answer",
        }
        items.append({"problem_id": item["problem_id"], "consistent": True})
    output = json.dumps({"items": items}, separators=(",", ":"))
    return "\n".join(
        json.dumps(event, separators=(",", ":"))
        for event in (
            {"type": "thread.started"},
            {"type": "turn.started"},
            {
                "type": "item.completed",
                "item": {"type": "agent_message", "text": output},
            },
            {
                "type": "turn.completed",
                "usage": {"input_tokens": 19, "output_tokens": 7},
            },
        )
    )


def _qualified_v4_harness_arguments(
    tmp_path: Path,
) -> tuple[list[str], Path, str]:
    """Create private, answer-free qualified evidence for one v4 CLI test."""

    source_root = tmp_path / "frozen-source"
    source_root.mkdir()
    (source_root / "frozen.py").write_text("VALUE = 4\n", encoding="utf-8")
    source_manifest = tmp_path / "source-manifest.json"
    source_manifest.write_text(
        json.dumps(build_source_tree_manifest(source_root).as_dict()),
        encoding="utf-8",
    )
    source_evidence = validate_source_tree_manifest_artifact(
        source_manifest,
        root=source_root,
    )
    harness_config, harness_file_sha256, profile = (
        cli_module._load_locked_codex_teacher_harness_config(_harness_config_path())
    )
    teacher_config, teacher_file_sha256 = cli_module._load_locked_codex_teacher_config(
        _teacher_pilot_v4_config_path()
    )
    replay_report = tmp_path / "replay.json"
    run_harness_replay(
        replay_report,
        harness_config_sha256=cli_module._teacher_config_sha256(harness_config),
        harness_config_file_sha256=harness_file_sha256,
        teacher_config_sha256=cli_module._teacher_config_sha256(teacher_config),
        teacher_config_file_sha256=teacher_file_sha256,
        prompt_policy=cli_module._teacher_prompt_policy_from_config(teacher_config),
        profile=profile,
    )
    binary = tmp_path / "codex-v4"
    binary.write_text("synthetic v4 Codex binary\n", encoding="utf-8")
    version = "codex-cli v4 synthetic"
    expected_answers = {
        row.problem_id: row.expected_answer for row in synthetic_fixture_rows()
    }

    def runner(command: tuple[str, ...]) -> CodexCommandResult:
        return CodexCommandResult(
            stdout=_event_stream_for_prompt(command[-1], expected_answers),
            stderr="",
            returncode=0,
            latency_ms=1,
        )

    live_plan_dir = tmp_path / "private-live-plan"
    live_report = tmp_path / "live.json"
    live = run_harness_live(
        live_plan_dir,
        live_report,
        harness_config_sha256=cli_module._teacher_config_sha256(harness_config),
        harness_config_file_sha256=harness_file_sha256,
        teacher_config_sha256=cli_module._teacher_config_sha256(teacher_config),
        teacher_config_file_sha256=teacher_file_sha256,
        prompt_policy=cli_module._teacher_prompt_policy_from_config(teacher_config),
        execution=TeacherExecutionConfig(
            codex_binary=str(binary),
            codex_cli_version=version,
            reasoning_effort="high",
            seed=20_260_731,
        ),
        profile=profile,
        source_manifest=source_evidence,
        command_runner=runner,
        working_directory=tmp_path,
    )
    assert live.qualified
    return (
        [
            "--harness-config",
            str(_harness_config_path()),
            "--harness-replay-report",
            str(replay_report),
            "--harness-live-report",
            str(live_report),
            "--harness-live-plan-dir",
            str(live_plan_dir),
            "--harness-source-root",
            str(source_root),
            "--harness-source-manifest",
            str(source_manifest),
        ],
        binary,
        version,
    )


def test_teacher_cli_plan_run_status_and_local_finalize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    contract, train = _teacher_contract(tmp_path, row_count=400)
    capsys.readouterr()
    plan_dir = tmp_path / "private-teacher-plan"
    monkeypatch.setattr(
        cli_module,
        "_probe_codex_chatgpt_cli",
        lambda: ("/private/codex", "codex-cli test"),
    )
    assert (
        main(
            [
                "gate-b-teacher-plan",
                *contract,
                "--teacher-config",
                str(_teacher_config_path()),
                "--pilot-size",
                "128",
                "--output-dir",
                str(plan_dir),
            ]
        )
        == 0
    )
    created = json.loads(capsys.readouterr().out)
    assert created["reference_answer_in_prompt"] is False
    assert created["locked_holdout_accessed"] is False
    assert "unique expression" not in json.dumps(created)
    plan = load_teacher_plan(plan_dir)
    assert len(plan.problem_ids) > 0

    calls: list[tuple[str, ...]] = []

    def runner(
        command: tuple[str, ...],
        *,
        timeout_seconds: int,
        isolated_codex_home: Path | None = None,
    ) -> CodexCommandResult:
        assert isolated_codex_home is not None
        workspace = Path(command[command.index("-C") + 1])
        assert not isolated_codex_home.is_relative_to(workspace)
        calls.append(command)
        answers = {
            record.id: record.answer
            for record in load_train_csv(train)
            if record.answer is not None
        }
        return CodexCommandResult(
            stdout=_event_stream_for_prompt(command[-1], answers),
            stderr="",
            returncode=0,
            latency_ms=7,
        )

    monkeypatch.setattr(cli_module, "_run_codex_teacher_command", runner)
    monkeypatch.setattr(
        cli_module,
        "_prepare_isolated_codex_home",
        lambda working_directory: working_directory,
    )
    assert (
        main(
            [
                "gate-b-teacher-run",
                "--plan-dir",
                str(plan_dir),
                "--teacher-config",
                str(_teacher_config_path()),
            ]
        )
        == 2
    )
    assert calls == []
    capsys.readouterr()

    assert (
        main(
            [
                "gate-b-teacher-run",
                "--plan-dir",
                str(plan_dir),
                "--teacher-config",
                str(_teacher_config_path()),
                "--acknowledge-codex-teacher",
                "--max-workers",
                "2",
            ]
        )
        == 2
    )
    assert calls == []
    assert "pilot requires --max-workers 1" in capsys.readouterr().err

    assert (
        main(
            [
                "gate-b-teacher-run",
                "--plan-dir",
                str(plan_dir),
                "--teacher-config",
                str(_teacher_config_path()),
                "--acknowledge-codex-teacher",
            ]
        )
        == 0
    )
    run = json.loads(capsys.readouterr().out)
    assert calls and all("--sandbox" in command for command in calls)
    assert all(
        'shell_environment_policy.inherit="none"' in command for command in calls
    )
    assert run["max_workers"] == 1
    assert run["reasoning_effort_policy"] == {"initial": "high", "repair": "xhigh"}
    assert run["status"]["unassessed_problem_count"] == len(plan.problem_ids)
    assert "unique expression" not in json.dumps(run)

    source = tmp_path / "private-source.jsonl"
    source_manifest = tmp_path / "private-source.manifest.json"
    assert (
        main(
            [
                "gate-b-teacher-finalize",
                *contract,
                "--teacher-config",
                str(_teacher_config_path()),
                "--plan-dir",
                str(plan_dir),
                "--pilot-size",
                "128",
                "--output-jsonl",
                str(source),
                "--output-manifest",
                str(source_manifest),
            ]
        )
        == 0
    )
    finalized = json.loads(capsys.readouterr().out)
    assert finalized["result"]["complete"] is True
    assert finalized["result"]["accepted_problem_count"] == len(plan.problem_ids)
    assert source.is_file() and source_manifest.is_file()
    assert "unique expression" not in json.dumps(finalized)

    assert (
        main(
            [
                "gate-b-teacher-status",
                "--plan-dir",
                str(plan_dir),
                "--teacher-config",
                str(_teacher_config_path()),
                "--output",
                str(tmp_path / "raw-free-status.json"),
            ]
        )
        == 0
    )
    status = json.loads(capsys.readouterr().out)
    assert status["status"]["accepted_problem_count"] == len(plan.problem_ids)
    assert json.loads((tmp_path / "raw-free-status.json").read_text(encoding="utf-8")) == status
    assert "unique expression" not in json.dumps(status)

    for foreign_config in (
        _teacher_pilot_v2_config_path(),
        _teacher_pilot_v3_config_path(),
    ):
        assert (
            main(
                [
                    "gate-b-teacher-status",
                    "--plan-dir",
                    str(plan_dir),
                    "--teacher-config",
                    str(foreign_config),
                ]
            )
            == 2
        )
        assert "does not match" in capsys.readouterr().err


def test_teacher_pilot_v2_profile_is_immutable_and_uses_four_safe_initial_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    contract, _ = _teacher_contract(tmp_path, row_count=400)
    capsys.readouterr()
    monkeypatch.setattr(
        cli_module,
        "_probe_codex_chatgpt_cli",
        lambda: ("/private/codex", "codex-cli test"),
    )
    plan_dir = tmp_path / "private-teacher-pilot-v2"

    assert (
        main(
            [
                "gate-b-teacher-plan",
                *contract,
                "--teacher-config",
                str(_teacher_pilot_v2_config_path()),
                "--pilot-size",
                "128",
                "--output-dir",
                str(plan_dir),
            ]
        )
        == 0
    )
    created = json.loads(capsys.readouterr().out)
    plan = load_teacher_plan(plan_dir)
    assert created["reference_answer_in_prompt"] is False
    assert created["allowed_problem_count"] == 128
    assert plan.label == "codex-gpt-5.6-sol-teacher-pilot-v2"
    assert plan.version == "pilot-v2"
    assert plan.prompt_policy.prompt_version == "gate-b-codex-teacher-prompt-v2"
    assert len(plan.chunks) == 4
    assert all(len(chunk.problem_ids) == 32 for chunk in plan.chunks)
    assert "untrusted mathematical data" in build_teacher_prompt(plan, 0)
    assert "unique expression" not in json.dumps(created)

    assert (
        main(
            [
                "gate-b-teacher-status",
                "--plan-dir",
                str(plan_dir),
                "--teacher-config",
                str(_teacher_pilot_v2_config_path()),
            ]
        )
        == 0
    )
    assert "unique expression" not in capsys.readouterr().out

    for foreign_config in (_teacher_config_path(), _teacher_pilot_v3_config_path()):
        assert (
            main(
                [
                    "gate-b-teacher-status",
                    "--plan-dir",
                    str(plan_dir),
                    "--teacher-config",
                    str(foreign_config),
                ]
            )
            == 2
        )
        assert "does not match" in capsys.readouterr().err

    drifted = tmp_path / "drifted-teacher-pilot-v2.json"
    payload = json.loads(_teacher_pilot_v2_config_path().read_text(encoding="utf-8"))
    payload["initial_chunk_size"] = 64
    drifted.write_text(json.dumps(payload), encoding="utf-8")
    assert (
        main(
            [
                "gate-b-teacher-plan",
                *contract,
                "--teacher-config",
                str(drifted),
                "--pilot-size",
                "128",
                "--output-dir",
                str(tmp_path / "must-not-exist"),
            ]
        )
        == 2
    )
    assert "differs from the locked no-API profile" in capsys.readouterr().err


def test_teacher_pilot_v3_profile_hashes_cross_loads_and_worker_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = _teacher_pilot_v3_config_path()
    config, file_sha256 = cli_module._load_locked_codex_teacher_config(config_path)
    assert cli_module._teacher_config_sha256(config) == (
        "deafe380e20079ef5e5fb2917c9f91d7a235d1135a23c64dcbd4ea7dddd38613"
    )
    assert file_sha256 == "54a2e31e716edfdd3d5a5d22a2d5124da14f552b4ceeab31fdf7c5ea11ddba01"
    assert config["initial_chunk_size"] == 32
    assert config["repair_chunk_size"] == 16
    assert config["max_attempts"] == 3
    assert config["initial_reasoning_effort"] == "high"
    assert config["repair_reasoning_effort"] == "xhigh"
    assert cli_module._teacher_prompt_policy_from_config(config).sha256 == (
        "953d62e283d5237f29b2145b5ed513246d737acd7ec40879450d7bcc8d08402b"
    )

    contract, _ = _teacher_contract(tmp_path, row_count=400)
    capsys.readouterr()
    probe_calls = 0

    def probe() -> tuple[str, str]:
        nonlocal probe_calls
        probe_calls += 1
        return "/private/codex", "codex-cli test"

    monkeypatch.setattr(cli_module, "_probe_codex_chatgpt_cli", probe)
    plan_dir = tmp_path / "private-teacher-pilot-v3"
    assert (
        main(
            [
                "gate-b-teacher-plan",
                *contract,
                "--teacher-config",
                str(config_path),
                "--pilot-size",
                "128",
                "--output-dir",
                str(plan_dir),
            ]
        )
        == 0
    )
    capsys.readouterr()
    plan = load_teacher_plan(plan_dir)
    assert plan.label == "codex-gpt-5.6-sol-teacher-pilot-v3"
    assert plan.version == "pilot-v3"
    assert len(plan.chunks) == 4
    assert all(len(chunk.problem_ids) == 32 for chunk in plan.chunks)
    assert plan.prompt_policy.prompt_version == "gate-b-codex-teacher-prompt-v3"
    assert "independently verify the candidate" in build_teacher_prompt(plan, 0)

    for foreign_config in (_teacher_config_path(), _teacher_pilot_v2_config_path()):
        assert (
            main(
                [
                    "gate-b-teacher-status",
                    "--plan-dir",
                    str(plan_dir),
                    "--teacher-config",
                    str(foreign_config),
                ]
            )
            == 2
        )
        assert "does not match" in capsys.readouterr().err

    assert (
        main(
            [
                "gate-b-teacher-run",
                "--plan-dir",
                str(plan_dir),
                "--teacher-config",
                str(config_path),
                "--acknowledge-codex-teacher",
                "--max-workers",
                "2",
            ]
        )
        == 2
    )
    assert "pilot requires --max-workers 1" in capsys.readouterr().err
    assert probe_calls == 1

    drifted = tmp_path / "drifted-teacher-pilot-v3.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["prompt_template_sha256"] = "0" * 64
    drifted.write_text(json.dumps(payload), encoding="utf-8")
    assert (
        main(
            [
                "gate-b-teacher-plan",
                *contract,
                "--teacher-config",
                str(drifted),
                "--pilot-size",
                "128",
                "--output-dir",
                str(tmp_path / "must-not-exist"),
            ]
        )
        == 2
    )
    assert "differs from the locked no-API profile" in capsys.readouterr().err


def test_teacher_pilot_v4_requires_reverified_harness_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = _teacher_pilot_v4_config_path()
    config, file_sha256 = cli_module._load_locked_codex_teacher_config(config_path)
    assert cli_module._teacher_config_sha256(config) == (
        "7a7f3e117a3f454c21dd9721ae432b9da6c0813e3b806a19af7f1d9a34d8adef"
    )
    assert file_sha256 == "063881f7d72a96e25202736a8fe729a0d64271376019b281094a5350c92a0d97"
    assert config["initial_chunk_size"] == 32
    assert config["repair_chunk_size"] == 16
    assert config["max_attempts"] == 3
    assert config["initial_reasoning_effort"] == "high"
    assert config["repair_reasoning_effort"] == "xhigh"
    assert cli_module._teacher_prompt_policy_from_config(config).sha256 == (
        "8de961862f2cabf245753ee276d4b833d8917934d4ba84fa8f9caa20a64ab924"
    )

    contract, _ = _teacher_contract(tmp_path, row_count=400)
    capsys.readouterr()
    harness_args, binary, version = _qualified_v4_harness_arguments(tmp_path)
    monkeypatch.setattr(
        cli_module,
        "_probe_codex_chatgpt_cli",
        lambda: (str(binary), version),
    )
    plan_dir = tmp_path / "private-teacher-pilot-v4"
    command = [
        "gate-b-teacher-plan",
        *contract,
        "--teacher-config",
        str(config_path),
        "--pilot-size",
        "128",
        "--output-dir",
        str(plan_dir),
    ]

    assert main(command) == 2
    assert "requires harness config" in capsys.readouterr().err
    assert not plan_dir.exists()

    assert main([*command, *harness_args]) == 0
    created = json.loads(capsys.readouterr().out)
    plan = load_teacher_plan(plan_dir)
    assert plan.label == "codex-gpt-5.6-sol-teacher-pilot-v4"
    assert plan.version == "pilot-v4"
    assert plan.prompt_policy.prompt_version == "gate-b-codex-teacher-prompt-v4"
    assert len(plan.chunks) == 4
    assert all(len(chunk.problem_ids) == 32 for chunk in plan.chunks)
    assert "no duplicate or omitted IDs" in build_teacher_prompt(plan, 0)
    assert created["harness_authorization_payload_sha256"] is not None
    sidecar = plan_dir / "harness-authorization-v1.json"
    assert sidecar.is_file()

    status_command = [
        "gate-b-teacher-status",
        "--plan-dir",
        str(plan_dir),
        "--teacher-config",
        str(config_path),
    ]
    assert main(status_command) == 2
    assert "requires harness config" in capsys.readouterr().err
    assert main([*status_command, *harness_args]) == 0
    assert "unique expression" not in capsys.readouterr().out

    original_sidecar = sidecar.read_bytes()
    sidecar.write_bytes(b"{}")
    assert main([*status_command, *harness_args]) == 2
    assert "harness authorization" in capsys.readouterr().err
    sidecar.write_bytes(original_sidecar)

    for foreign_config in (
        _teacher_config_path(),
        _teacher_pilot_v2_config_path(),
        _teacher_pilot_v3_config_path(),
    ):
        assert (
            main(
                [
                    "gate-b-teacher-status",
                    "--plan-dir",
                    str(plan_dir),
                    "--teacher-config",
                    str(foreign_config),
                ]
            )
            == 2
        )
        assert "does not match" in capsys.readouterr().err


def test_teacher_pilot_v4_initial_threshold_failure_blocks_all_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    contract, train = _teacher_contract(tmp_path, row_count=400)
    capsys.readouterr()
    harness_args, binary, version = _qualified_v4_harness_arguments(tmp_path)
    monkeypatch.setattr(
        cli_module,
        "_probe_codex_chatgpt_cli",
        lambda: (str(binary), version),
    )
    monkeypatch.setattr(
        cli_module,
        "_prepare_isolated_codex_home",
        lambda working_directory: working_directory,
    )
    config_path = _teacher_pilot_v4_config_path()
    plan_dir = tmp_path / "private-teacher-pilot-v4"
    assert (
        main(
            [
                "gate-b-teacher-plan",
                *contract,
                "--teacher-config",
                str(config_path),
                "--pilot-size",
                "128",
                "--output-dir",
                str(plan_dir),
                *harness_args,
            ]
        )
        == 0
    )
    capsys.readouterr()
    answers = {
        record.id: record.answer
        for record in load_train_csv(train)
        if record.answer is not None
    }
    calls: list[tuple[str, ...]] = []

    def wrong_runner(
        command: tuple[str, ...],
        *,
        timeout_seconds: int,
        isolated_codex_home: Path | None = None,
    ) -> CodexCommandResult:
        del timeout_seconds, isolated_codex_home
        calls.append(command)
        wrong_answers = {problem_id: answer + 1 for problem_id, answer in answers.items()}
        return CodexCommandResult(
            stdout=_event_stream_for_prompt(command[-1], wrong_answers),
            stderr="",
            returncode=0,
            latency_ms=1,
        )

    monkeypatch.setattr(cli_module, "_run_codex_teacher_command", wrong_runner)
    run_command = [
        "gate-b-teacher-run",
        "--plan-dir",
        str(plan_dir),
        "--teacher-config",
        str(config_path),
        "--acknowledge-codex-teacher",
        "--max-invocations",
        "4",
        "--max-workers",
        "1",
        *harness_args,
    ]
    short_initial_command = list(run_command)
    short_initial_command[
        short_initial_command.index("--max-invocations") + 1
    ] = "2"
    assert main(short_initial_command) == 2
    assert "requires exactly --max-invocations 4" in capsys.readouterr().err
    assert not calls
    assert main(run_command) == 0
    assert len(calls) == 4
    capsys.readouterr()

    source_jsonl = tmp_path / "must-not-exist.jsonl"
    source_manifest = tmp_path / "must-not-exist.manifest.json"
    finalize_command = [
        "gate-b-teacher-finalize",
        *contract,
        "--teacher-config",
        str(config_path),
        "--plan-dir",
        str(plan_dir),
        "--pilot-size",
        "128",
        "--output-jsonl",
        str(source_jsonl),
        "--output-manifest",
        str(source_manifest),
        *harness_args,
    ]
    assert main(finalize_command) == 1
    finalized = json.loads(capsys.readouterr().out)
    failure = finalized["initial_threshold_failure"]
    assert failure["accepted_problem_count"] == 0
    assert failure["minimum_initial_accepted_count"] == 103
    assert failure["failure_category"] == "initial_exact_match_below_threshold"
    assert not source_jsonl.exists()
    assert not source_manifest.exists()

    def unexpected_runner(*_args: object, **_kwargs: object) -> CodexCommandResult:
        pytest.fail("v4 threshold failure must block every later teacher call")

    monkeypatch.setattr(cli_module, "_run_codex_teacher_command", unexpected_runner)
    assert main(run_command) == 2
    assert "initial threshold failed" in capsys.readouterr().err
    assert (
        main(
            [
                "gate-b-teacher-status",
                "--plan-dir",
                str(plan_dir),
                "--teacher-config",
                str(config_path),
                *harness_args,
            ]
        )
        == 0
    )
    status = json.loads(capsys.readouterr().out)
    assert status["initial_threshold_failure"] == failure
    assert "unique expression" not in json.dumps(status)


def test_teacher_cli_rejects_nonzero_fold_and_config_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract, _ = _teacher_contract(tmp_path)
    bad_fold_contract = list(contract)
    bad_fold_contract[bad_fold_contract.index("--fold") + 1] = "1"
    probe_calls = 0

    def probe() -> tuple[str, str]:
        nonlocal probe_calls
        probe_calls += 1
        return "/private/codex", "codex-cli test"

    monkeypatch.setattr(cli_module, "_probe_codex_chatgpt_cli", probe)
    assert (
        main(
            [
                "gate-b-teacher-plan",
                *bad_fold_contract,
                "--teacher-config",
                str(_teacher_config_path()),
                "--output-dir",
                str(tmp_path / "not-created"),
            ]
        )
        == 2
    )
    assert probe_calls == 0

    drifted = tmp_path / "drifted-teacher-config.json"
    payload = json.loads(_teacher_config_path().read_text(encoding="utf-8"))
    payload["model_id"] = "forbidden-model"
    drifted.write_text(json.dumps(payload), encoding="utf-8")
    assert (
        main(
            [
                "gate-b-teacher-plan",
                *contract,
                "--teacher-config",
                str(drifted),
                "--output-dir",
                str(tmp_path / "drifted"),
            ]
        )
        == 2
    )
    assert probe_calls == 0


def test_teacher_cli_reprobes_the_exact_planned_codex_binary_and_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    contract, _ = _teacher_contract(tmp_path, row_count=400)
    capsys.readouterr()
    current = {"binary": "/private/codex", "version": "codex-cli test"}
    monkeypatch.setattr(
        cli_module,
        "_probe_codex_chatgpt_cli",
        lambda: (current["binary"], current["version"]),
    )
    plan_dir = tmp_path / "private-teacher-plan"
    assert (
        main(
            [
                "gate-b-teacher-plan",
                *contract,
                "--teacher-config",
                str(_teacher_config_path()),
                "--pilot-size",
                "128",
                "--output-dir",
                str(plan_dir),
            ]
        )
        == 0
    )
    capsys.readouterr()
    current["version"] = "codex-cli replaced"

    def unexpected_runner(*_args: object, **_kwargs: object) -> CodexCommandResult:
        pytest.fail("a stale or replaced Codex executable must not be run")

    monkeypatch.setattr(cli_module, "_run_codex_teacher_command", unexpected_runner)
    assert (
        main(
            [
                "gate-b-teacher-run",
                "--plan-dir",
                str(plan_dir),
                "--teacher-config",
                str(_teacher_config_path()),
                "--acknowledge-codex-teacher",
            ]
        )
        == 2
    )
    assert "binary/version does not match" in capsys.readouterr().err


@pytest.mark.parametrize(
    "command",
    (
        "gate-b-teacher-run",
        "gate-b-teacher-v2-run",
        "gate-b-teacher-logical-audit-run",
    ),
)
def test_teacher_run_commands_reject_reasoning_effort_override(command: str) -> None:
    """Initial high and retry xhigh are a ledger policy, never caller input."""

    parser = cli_module.build_parser()
    subcommands = next(
        action.choices
        for action in parser._actions
        if isinstance(getattr(action, "choices", None), dict)
    )
    options = {
        option
        for action in subcommands[command]._actions
        for option in action.option_strings
    }
    assert "--reasoning-effort" not in options


def test_logical_audit_reasoning_effort_is_selected_from_ledger_state() -> None:
    """The audit shares the no-override high/xhigh state policy."""

    config, _ = cli_module._load_locked_codex_teacher_config(_teacher_config_path())
    assert (
        cli_module._logical_audit_reasoning_effort(
            total_attempts=0,
            parsed_attempts=0,
            exhausted=False,
            config=config,
        )
        == "high"
    )
    assert (
        cli_module._logical_audit_reasoning_effort(
            total_attempts=1,
            parsed_attempts=0,
            exhausted=False,
            config=config,
        )
        == "xhigh"
    )


def test_teacher_cli_isolates_only_chatgpt_auth_and_drops_api_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    auth_dir = home / ".codex"
    auth_dir.mkdir(parents=True)
    source_auth = auth_dir / "auth.json"
    source_auth.write_text('{"synthetic":true}\n', encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("KAGGLE_API_TOKEN", "must-not-reach-codex")
    monkeypatch.setenv("CODEX_API_KEY", "must-not-reach-codex")
    parent = tmp_path / "runner"
    parent.mkdir()

    isolated = cli_module._prepare_isolated_codex_home(parent)
    environment = cli_module._codex_teacher_environment(
        isolated_codex_home=isolated
    )

    assert (isolated / "auth.json").read_text(encoding="utf-8") == source_auth.read_text(
        encoding="utf-8"
    )
    assert not (isolated / "skills").exists()
    assert environment["CODEX_HOME"] == str(isolated.resolve())
    assert "KAGGLE_API_TOKEN" not in environment
    assert "CODEX_API_KEY" not in environment


def test_teacher_logical_audit_cli_is_fixed_candidate_only_and_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Exercise the public audit boundary without exposing raw private text."""

    contract, train = _teacher_contract(tmp_path, row_count=400)
    capsys.readouterr()
    teacher_plan_dir = tmp_path / "private-teacher-plan"
    probe_calls: list[None] = []

    def probe() -> tuple[str, str]:
        probe_calls.append(None)
        return "/private/codex", "codex-cli test"

    monkeypatch.setattr(cli_module, "_probe_codex_chatgpt_cli", probe)
    assert (
        main(
            [
                "gate-b-teacher-plan",
                *contract,
                "--teacher-config",
                str(_teacher_config_path()),
                "--pilot-size",
                "128",
                "--output-dir",
                str(teacher_plan_dir),
            ]
        )
        == 0
    )
    capsys.readouterr()
    teacher_plan = load_teacher_plan(teacher_plan_dir)
    assert len(teacher_plan.problem_ids) >= 64

    calls: list[tuple[str, ...]] = []
    rationale_calls = 0
    audit_calls = 0

    def runner(
        command: tuple[str, ...],
        *,
        timeout_seconds: int,
        isolated_codex_home: Path | None = None,
    ) -> CodexCommandResult:
        nonlocal audit_calls, rationale_calls
        assert timeout_seconds > 0
        assert isolated_codex_home is not None
        workspace = Path(command[command.index("-C") + 1])
        assert not isolated_codex_home.is_relative_to(workspace)
        assert 'shell_environment_policy.inherit="none"' in command
        calls.append(command)
        prompt = command[-1]
        if "candidate_rationale_and_final_answer" in prompt:
            audit_calls += 1
            assert 'model_reasoning_effort="high"' in command
            return CodexCommandResult(
                stdout=_logical_audit_event_stream_for_prompt(prompt),
                stderr="",
                returncode=0,
                latency_ms=9,
            )
        rationale_calls += 1
        answers = {
            record.id: record.answer
            for record in load_train_csv(train)
            if record.answer is not None
        }
        return CodexCommandResult(
            stdout=_event_stream_for_prompt(prompt, answers),
            stderr="",
            returncode=0,
            latency_ms=8,
        )

    monkeypatch.setattr(cli_module, "_run_codex_teacher_command", runner)
    monkeypatch.setattr(
        cli_module,
        "_prepare_isolated_codex_home",
        lambda working_directory: working_directory,
    )
    assert (
        main(
            [
                "gate-b-teacher-run",
                "--plan-dir",
                str(teacher_plan_dir),
                "--teacher-config",
                str(_teacher_config_path()),
                "--acknowledge-codex-teacher",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert rationale_calls > 0

    source = tmp_path / "private-source.jsonl"
    source_manifest = tmp_path / "private-source.manifest.json"
    assert (
        main(
            [
                "gate-b-teacher-finalize",
                *contract,
                "--teacher-config",
                str(_teacher_config_path()),
                "--plan-dir",
                str(teacher_plan_dir),
                "--pilot-size",
                "128",
                "--output-jsonl",
                str(source),
                "--output-manifest",
                str(source_manifest),
            ]
        )
        == 0
    )
    capsys.readouterr()
    original_manifest = source_manifest.read_bytes()

    # A malformed bank manifest is rejected before a logical-audit directory
    # can be created; this command never accepts organizer reference answers.
    source_manifest.write_text("{}\n", encoding="utf-8")
    invalid_audit_dir = tmp_path / "invalid-logical-audit"
    assert (
        main(
            [
                "gate-b-teacher-logical-audit-plan",
                "--teacher-config",
                str(_teacher_config_path()),
                "--teacher-plan-dir",
                str(teacher_plan_dir),
                "--source-jsonl",
                str(source),
                "--source-manifest",
                str(source_manifest),
                "--output-dir",
                str(invalid_audit_dir),
            ]
        )
        == 2
    )
    assert not invalid_audit_dir.exists()
    source_manifest.write_bytes(original_manifest)
    capsys.readouterr()

    audit_dir = tmp_path / "private-logical-audit"
    assert (
        main(
            [
                "gate-b-teacher-logical-audit-plan",
                "--teacher-config",
                str(_teacher_config_path()),
                "--teacher-plan-dir",
                str(teacher_plan_dir),
                "--source-jsonl",
                str(source),
                "--source-manifest",
                str(source_manifest),
                "--output-dir",
                str(audit_dir),
            ]
        )
        == 0
    )
    planned = json.loads(capsys.readouterr().out)
    assert planned["sample_size"] == 64
    assert planned["min_consistent"] == 60
    assert planned["reference_answer_read"] is False
    assert planned["leaderboard_or_test_used"] is False
    assert planned["source_bank_provenance_reverified"] is True
    assert "unique expression" not in json.dumps(planned)

    calls_before_ack = len(calls)
    assert (
        main(
            [
                "gate-b-teacher-logical-audit-run",
                "--teacher-config",
                str(_teacher_config_path()),
                "--teacher-plan-dir",
                str(teacher_plan_dir),
                "--audit-dir",
                str(audit_dir),
            ]
        )
        == 2
    )
    assert len(calls) == calls_before_ack
    capsys.readouterr()

    drifted = tmp_path / "drifted-teacher-config.json"
    drifted_payload = json.loads(_teacher_config_path().read_text(encoding="utf-8"))
    drifted_payload["model_id"] = "forbidden-model"
    drifted.write_text(json.dumps(drifted_payload), encoding="utf-8")
    assert (
        main(
            [
                "gate-b-teacher-logical-audit-run",
                "--teacher-config",
                str(drifted),
                "--teacher-plan-dir",
                str(teacher_plan_dir),
                "--audit-dir",
                str(audit_dir),
                "--acknowledge-codex-teacher",
            ]
        )
        == 2
    )
    assert len(calls) == calls_before_ack
    capsys.readouterr()

    assert (
        main(
            [
                "gate-b-teacher-logical-audit-run",
                "--teacher-config",
                str(_teacher_config_path()),
                "--teacher-plan-dir",
                str(teacher_plan_dir),
                "--audit-dir",
                str(audit_dir),
                "--acknowledge-codex-teacher",
            ]
        )
        == 0
    )
    run = json.loads(capsys.readouterr().out)
    assert audit_calls == 1
    assert len(probe_calls) >= 3
    assert run["reasoning_effort"] == "high"
    assert run["status"]["completed_problem_count"] == 64
    assert run["status"]["consistent_problem_count"] == 64
    assert run["raw_generation_serialized"] is False
    assert "unique expression" not in json.dumps(run)

    status_path = tmp_path / "logical-audit-status.json"
    assert (
        main(
            [
                "gate-b-teacher-logical-audit-status",
                "--teacher-config",
                str(_teacher_config_path()),
                "--teacher-plan-dir",
                str(teacher_plan_dir),
                "--audit-dir",
                str(audit_dir),
                "--output",
                str(status_path),
            ]
        )
        == 0
    )
    status = json.loads(capsys.readouterr().out)
    assert status["status"]["sample_size"] == 64
    assert status["status"]["min_consistent"] == 60
    assert json.loads(status_path.read_text(encoding="utf-8")) == status
    assert "unique expression" not in json.dumps(status)

    assert (
        main(
            [
                "gate-b-teacher-logical-audit-finalize",
                "--teacher-config",
                str(_teacher_config_path()),
                "--teacher-plan-dir",
                str(teacher_plan_dir),
                "--audit-dir",
                str(audit_dir),
            ]
        )
        == 0
    )
    finalized = json.loads(capsys.readouterr().out)
    assert finalized["result"]["complete"] is True
    assert finalized["result"]["passed"] is True
    assert finalized["result"]["consistent_problem_count"] == 64
    assert finalized["reference_answer_read"] is False
    assert finalized["leaderboard_or_test_used"] is False
    assert "unique expression" not in json.dumps(finalized)
    assert len(probe_calls) >= 2

    # A complete v1 plan cannot be made merely because a pilot plan exists.
    # It needs the immutable authorization receipt and every live private
    # provenance path so post-receipt source/audit tampering is detectable.
    full_without_receipt = tmp_path / "full-without-receipt"
    assert (
        main(
            [
                "gate-b-teacher-plan",
                *contract,
                "--teacher-config",
                str(_teacher_config_path()),
                "--output-dir",
                str(full_without_receipt),
            ]
        )
        == 2
    )
    assert not full_without_receipt.exists()
    capsys.readouterr()

    authorization = tmp_path / "pilot-authorization.json"
    assert (
        main(
            [
                "gate-b-teacher-pilot-authorize",
                *contract,
                "--teacher-config",
                str(_teacher_config_path()),
                "--pilot-plan-dir",
                str(teacher_plan_dir),
                "--pilot-source-jsonl",
                str(source),
                "--pilot-source-manifest",
                str(source_manifest),
                "--pilot-logical-audit-dir",
                str(audit_dir),
                "--output",
                str(authorization),
            ]
        )
        == 0
    )
    authorized = json.loads(capsys.readouterr().out)
    assert authorized["receipt"]["initial_exact_match_accepted_count"] == 128
    assert authorized["receipt"]["logical_audit_consistent_problem_count"] == 64
    assert "unique expression" not in json.dumps(authorized)

    full_plan_dir = tmp_path / "authorized-full-v1-plan"
    assert (
        main(
            [
                "gate-b-teacher-plan",
                *contract,
                "--teacher-config",
                str(_teacher_config_path()),
                "--pilot-authorization",
                str(authorization),
                "--pilot-plan-dir",
                str(teacher_plan_dir),
                "--pilot-source-jsonl",
                str(source),
                "--pilot-source-manifest",
                str(source_manifest),
                "--pilot-logical-audit-dir",
                str(audit_dir),
                "--output-dir",
                str(full_plan_dir),
            ]
        )
        == 0
    )
    full_created = json.loads(capsys.readouterr().out)
    assert full_created["allowed_problem_count"] > 128
    assert full_created["pilot_authorization_file_sha256"]
    assert "unique expression" not in json.dumps(full_created)

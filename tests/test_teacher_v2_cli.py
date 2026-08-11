from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

import deep_challenge.cli as cli_module
from deep_challenge.cli import main
from deep_challenge.data import MathRecord, load_train_csv
from deep_challenge.provenance import canonical_json_bytes
from deep_challenge.splits import SplitManifest, eligible_training_ids, eligible_validation_ids
from deep_challenge.teacher_rationale import (
    CodexCommandResult,
    TeacherBankFinalizeResult,
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


def _teacher_contract(tmp_path: Path) -> tuple[list[str], list[str], SplitManifest]:
    train = tmp_path / "train.csv"
    _write_csv(
        train,
        [
            ["id", "question", "answer"],
            *[
                [
                    f"train-{index:06d}",
                    f"Compute the unique expression using symbol {chr(97 + index)}.",
                    str(index - 10),
                ]
                for index in range(1, 25)
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
    manifest = SplitManifest.from_dict(split_payload["split"])
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
                manifest.sha256,
                "--output-dir",
                str(shard),
            ]
        )
        == 0
    )
    exclusions = tmp_path / "exclusions.csv"
    _write_csv(exclusions, [["id"], ["train-000001"]])
    contract = [
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
        manifest.sha256,
        "--development-shard",
        str(shard),
        "--expected-development-shard-sha256",
        hashlib.sha256((shard / "CHECKSUMS.sha256").read_bytes()).hexdigest(),
    ]
    without_shard = list(contract)
    position = without_shard.index("--development-shard")
    del without_shard[position : position + 2]
    position = without_shard.index("--expected-development-shard-sha256")
    del without_shard[position : position + 2]
    return contract, without_shard, manifest


def _write_probe_decision(
    tmp_path: Path,
    manifest: SplitManifest,
    *,
    candidate_label: str = "qlora-rationale",
    authorized: bool = True,
) -> Path:
    comparison = tmp_path / "comparison.json"
    comparison.write_text("{}\n", encoding="utf-8")
    payload_without_hash: dict[str, object] = {
        "schema_version": "gate-b-candidate-probe-decision-v1",
        "decision_scope": "single_fold_gpu_cost_control_only",
        "policy": {"name": "synthetic"},
        "comparison_artifact": {
            "path": str(comparison.resolve()),
            "size_bytes": comparison.stat().st_size,
            "sha256": hashlib.sha256(comparison.read_bytes()).hexdigest(),
            "payload_sha256": "a" * 64,
        },
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        "split_version": manifest.version,
        "split_sha256": manifest.sha256,
        "source_groups_sha256": manifest.source_groups_sha256,
        "fold": 0,
        "reference_label": "base",
        "candidate_label": candidate_label,
        "evidence": {"synthetic": True},
        "significant_regression": not authorized,
        "candidate_action": (
            "continue_to_complete_oof" if authorized else "stop_before_remaining_folds"
        ),
        "candidate_full_oof_authorized": authorized,
        "final_selection_eligible": False,
        "complete_oof_required_before_freeze": True,
        "selection_frozen": False,
        "locked_holdout_accessed": False,
        "leaderboard_or_test_used": False,
    }
    payload = {
        **payload_without_hash,
        "payload_sha256": hashlib.sha256(
            canonical_json_bytes(payload_without_hash)
        ).hexdigest(),
    }
    output = tmp_path / "probe-decision.json"
    output.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def _install_probe_decision_recompute(
    monkeypatch: pytest.MonkeyPatch, decision: Path
) -> None:
    expected = json.loads(decision.read_text(encoding="utf-8"))

    def fake_decide(
        comparison_artifact: Path,
        *,
        candidate_label: str,
        output_path: Path,
    ) -> object:
        assert comparison_artifact == Path(expected["comparison_artifact"]["path"])
        assert candidate_label == expected["candidate_label"]
        output_path.write_text(
            json.dumps(expected, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return object()

    monkeypatch.setattr(cli_module, "decide_candidate_probe_promotion", fake_decide)


def _v2_plan_arguments(
    contract: list[str], decision: Path, plan_dir: Path
) -> list[str]:
    return [
        "gate-b-teacher-v2-plan",
        *contract,
        "--candidate-probe-decision",
        str(decision),
        "--candidate-label",
        "qlora-rationale",
        "--teacher-config",
        str(_teacher_config_path()),
        "--output-dir",
        str(plan_dir),
    ]


def test_teacher_v2_plan_and_finalize_use_only_remaining_development_cv_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    contract, without_shard, manifest = _teacher_contract(tmp_path)
    capsys.readouterr()
    decision = _write_probe_decision(tmp_path, manifest)
    _install_probe_decision_recompute(monkeypatch, decision)
    monkeypatch.setattr(
        cli_module,
        "_probe_codex_chatgpt_cli",
        lambda: ("/private/codex", "codex-cli test"),
    )
    plan_dir = tmp_path / "private-teacher-v2-plan"

    assert main(_v2_plan_arguments(contract, decision, plan_dir)) == 0
    created = json.loads(capsys.readouterr().out)
    plan = load_teacher_plan(plan_dir)
    remaining_ids = tuple(sorted(eligible_validation_ids(manifest, 0, ("train-000001",))))
    v1_training_ids = tuple(sorted(eligible_training_ids(manifest, 0, ("train-000001",))))
    assert plan.label == "codex-gpt-5.6-sol-teacher-development-v2"
    assert plan.version == "v2"
    assert plan.problem_ids == remaining_ids
    assert set(plan.problem_ids).isdisjoint(v1_training_ids)
    assert set(plan.problem_ids).isdisjoint(manifest.final_holdout_ids())
    assert created["allowed_problem_count"] == len(remaining_ids)
    assert created["locked_holdout_accessed"] is False
    assert "unique expression" not in json.dumps(created)

    status_arguments = [
        "gate-b-teacher-v2-status",
        *without_shard,
        "--candidate-probe-decision",
        str(decision),
        "--candidate-label",
        "qlora-rationale",
        "--teacher-config",
        str(_teacher_config_path()),
        "--plan-dir",
        str(plan_dir),
    ]
    assert main(status_arguments) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["scope"] == "remaining_development_cv_after_fold0_training"
    assert status["status"]["total_problem_count"] == len(remaining_ids)
    assert "unique expression" not in json.dumps(status)

    calls: list[tuple[str, ...]] = []

    def runner(
        command: tuple[str, ...],
        *,
        timeout_seconds: int,
        isolated_codex_home: Path | None = None,
    ) -> CodexCommandResult:
        assert timeout_seconds > 0
        assert isolated_codex_home is not None
        workspace = Path(command[command.index("-C") + 1])
        assert not isolated_codex_home.is_relative_to(workspace)
        assert 'shell_environment_policy.inherit="none"' in command
        calls.append(command)
        answers = {
            record.id: record.answer
            for record in load_train_csv(Path(contract[contract.index("--train") + 1]))
            if record.answer is not None
        }
        payload = json.loads(command[-1].rsplit("INPUT_JSON:\n", 1)[1])
        message = json.dumps(
            {
                "items": [
                    {
                        "problem_id": item["problem_id"],
                        "target_text": (
                            "The relation determines one integer.\n"
                            f"Final answer: {answers[item['problem_id']]}"
                        ),
                    }
                    for item in payload["items"]
                ]
            },
            separators=(",", ":"),
        )
        return CodexCommandResult(
            stdout="\n".join(
                json.dumps(event, separators=(",", ":"))
                for event in (
                    {"type": "thread.started"},
                    {"type": "turn.started"},
                    {
                        "type": "item.completed",
                        "item": {"type": "agent_message", "text": message},
                    },
                    {
                        "type": "turn.completed",
                        "usage": {"input_tokens": 12, "output_tokens": 8},
                    },
                )
            ),
            stderr="",
            returncode=0,
            latency_ms=1,
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
                "gate-b-teacher-v2-run",
                *without_shard,
                "--candidate-probe-decision",
                str(decision),
                "--candidate-label",
                "qlora-rationale",
                "--teacher-config",
                str(_teacher_config_path()),
                "--plan-dir",
                str(plan_dir),
                "--acknowledge-codex-teacher",
                "--max-workers",
                "2",
            ]
        )
        == 0
    )
    run = json.loads(capsys.readouterr().out)
    assert calls
    assert run["max_workers"] == 2
    assert run["reasoning_effort_policy"] == {"initial": "high", "repair": "xhigh"}
    assert run["status"]["unassessed_problem_count"] == len(remaining_ids)

    captured: dict[str, tuple[str, ...]] = {}

    def fake_finalize(
        _plan_dir: Path,
        records: tuple[MathRecord, ...],
        **_kwargs: object,
    ) -> TeacherBankFinalizeResult:
        captured["ids"] = tuple(record.id for record in records)
        return TeacherBankFinalizeResult(
            plan_sha256=plan.plan_sha256,
            total_problem_count=len(records),
            accepted_problem_count=0,
            rejected_problem_count=0,
            pending_problem_count=len(records),
            complete=False,
            source_jsonl=None,
            source_jsonl_sha256=None,
            manifest=None,
            manifest_sha256=None,
        )

    monkeypatch.setattr(cli_module, "finalize_teacher_bank", fake_finalize)
    assert (
        main(
            [
                "gate-b-teacher-v2-finalize",
                *contract,
                "--candidate-probe-decision",
                str(decision),
                "--candidate-label",
                "qlora-rationale",
                "--teacher-config",
                str(_teacher_config_path()),
                "--plan-dir",
                str(plan_dir),
                "--output-jsonl",
                str(tmp_path / "private-source.jsonl"),
                "--output-manifest",
                str(tmp_path / "private-source.manifest.json"),
            ]
        )
        == 0
    )
    assert captured["ids"] == remaining_ids


@pytest.mark.parametrize(
    ("candidate_label", "authorized", "expected_message"),
    [
        ("other-candidate", True, "does not authorize the requested candidate"),
        ("qlora-rationale", False, "does not authorize v2 teacher expansion"),
    ],
)
def test_teacher_v2_plan_rejects_unapproved_or_mismatched_probe_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    candidate_label: str,
    authorized: bool,
    expected_message: str,
) -> None:
    contract, _without_shard, manifest = _teacher_contract(tmp_path)
    capsys.readouterr()
    decision = _write_probe_decision(
        tmp_path,
        manifest,
        candidate_label=candidate_label,
        authorized=authorized,
    )
    _install_probe_decision_recompute(monkeypatch, decision)
    monkeypatch.setattr(
        cli_module,
        "_probe_codex_chatgpt_cli",
        lambda: ("/private/codex", "codex-cli test"),
    )
    plan_dir = tmp_path / "must-not-exist"

    assert main(_v2_plan_arguments(contract, decision, plan_dir)) == 2
    assert expected_message in capsys.readouterr().err
    assert not plan_dir.exists()


def test_teacher_v2_status_rejects_tampered_authorization_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    contract, without_shard, manifest = _teacher_contract(tmp_path)
    capsys.readouterr()
    decision = _write_probe_decision(tmp_path, manifest)
    _install_probe_decision_recompute(monkeypatch, decision)
    monkeypatch.setattr(
        cli_module,
        "_probe_codex_chatgpt_cli",
        lambda: ("/private/codex", "codex-cli test"),
    )
    plan_dir = tmp_path / "private-teacher-v2-plan"
    assert main(_v2_plan_arguments(contract, decision, plan_dir)) == 0
    capsys.readouterr()

    authorization_path = plan_dir / "v2-authorization.json"
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    authorization["allowed_problem_count"] = 0
    authorization_path.write_text(
        json.dumps(authorization, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    assert (
        main(
            [
                "gate-b-teacher-v2-status",
                *without_shard,
                "--candidate-probe-decision",
                str(decision),
                "--candidate-label",
                "qlora-rationale",
                "--teacher-config",
                str(_teacher_config_path()),
                "--plan-dir",
                str(plan_dir),
            ]
        )
        == 2
    )
    assert "authorization does not match" in capsys.readouterr().err

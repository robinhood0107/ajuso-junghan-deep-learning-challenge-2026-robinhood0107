from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from deep_challenge.gate_b import DevelopmentResumeStatus, GateBValidationError
from deep_challenge.gate_b_runtime import TrainingResumeStatus
from deep_challenge.workflow_status import (
    RUN_CONTEXT_SCHEMA,
    WORKFLOW_STATUS_SCHEMA,
    development_workflow_status,
    training_workflow_status,
    write_run_context,
)


def _development_status(state: str) -> DevelopmentResumeStatus:
    completed = 10 if state == "complete" else 4
    return DevelopmentResumeStatus(
        contract_sha256="a" * 64,
        state=state,
        process_id=123 if state == "running" else None,
        total_chunks=2,
        completed_chunks=2 if state == "complete" else 1,
        total_generations=10,
        completed_generations=completed,
        chunk_attempt_count=2,
        invalid_chunk_attempt_count=0,
        completed_latency_ms=12.5,
    )


@pytest.mark.parametrize(
    ("private_state", "state", "terminal", "resume_allowed", "next_action"),
    [
        ("planned", "ready", False, True, "start_development"),
        ("running", "running", False, False, "wait_for_development"),
        ("interrupted", "retryable", False, True, "resume_development"),
        ("complete", "complete", True, False, "verify_development_artifact"),
    ],
)
def test_development_status_is_raw_free_and_exact(
    private_state: str,
    state: str,
    terminal: bool,
    resume_allowed: bool,
    next_action: str,
) -> None:
    payload = development_workflow_status(_development_status(private_state)).as_dict()
    assert payload == {
        "schema_version": WORKFLOW_STATUS_SCHEMA,
        "stage": "development_generation",
        "state": state,
        "terminal": terminal,
        "resume_allowed": resume_allowed,
        "repair_allowed": False,
        "next_action": next_action,
        "blockers": [],
        "completed_count": 10 if private_state == "complete" else 4,
        "total_count": 10,
        "artifact_sha256": "a" * 64,
    }
    assert not ({"question", "answer", "problem_id", "prompt", "path"} & payload.keys())


@pytest.mark.parametrize(
    ("state", "terminal", "resume_allowed", "next_action", "blockers"),
    [
        ("ready", False, True, "start_training", []),
        ("running", False, False, "wait_for_training", []),
        ("retryable", False, True, "resume_training", []),
        (
            "terminal_failed",
            True,
            False,
            "record_terminal_training_failure",
            ["no_complete_resume_checkpoint"],
        ),
        ("complete", True, False, "verify_adapter_artifact", []),
    ],
)
def test_training_status_is_raw_free_and_exact(
    state: str,
    terminal: bool,
    resume_allowed: bool,
    next_action: str,
    blockers: list[str],
) -> None:
    completion = "b" * 64 if state == "complete" else None
    payload = training_workflow_status(
        TrainingResumeStatus(
            contract_sha256="a" * 64,
            state=state,
            process_id=123 if state == "running" else None,
            latest_checkpoint_step=7 if state == "retryable" else None,
            completion_artifact_sha256=completion,
        )
    ).as_dict()
    assert payload == {
        "schema_version": WORKFLOW_STATUS_SCHEMA,
        "stage": "fold_training",
        "state": state,
        "terminal": terminal,
        "resume_allowed": resume_allowed,
        "repair_allowed": False,
        "next_action": next_action,
        "blockers": blockers,
        "completed_count": 1 if state == "complete" else 0,
        "total_count": 1,
        "artifact_sha256": completion or "a" * 64,
    }


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_run_context_is_immutable_relative_and_hash_bound(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    config = _write(root / "configs" / "base.json", '{"base":true}\n')
    manifest = _write(
        root / "private" / "source-manifest.json",
        json.dumps({"tree_sha256": "1" * 64}) + "\n",
    )
    split = _write(root / "artifacts" / "split.json", '{"split":"v1"}\n')
    preflight = _write(root / "artifacts" / "preflight.json", '{"ready":true}\n')
    smoke = _write(root / "artifacts" / "smoke.json", '{"ready":true}\n')
    output = root / "private" / "run-context.json"

    result = write_run_context(
        output,
        source_root=root,
        source_commit="2" * 40,
        run_tag="base-oof-20260811-001",
        config_paths={"base": config},
        source_manifest_path=manifest,
        split_artifact_path=split,
        preflight_report_path=preflight,
        gpu_smoke_report_path=smoke,
        fold_output_paths={fold: root / "private" / f"fold-{fold}" for fold in range(5)},
        planned_stages=("gpu_gate", "base_oof", "teacher_v5"),
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == RUN_CONTEXT_SCHEMA
    assert payload["source_commit"] == "2" * 40
    assert payload["source_manifest"]["path"] == "private/source-manifest.json"
    assert payload["source_manifest"]["tree_sha256"] == "1" * 64
    assert payload["configs"]["base"]["path"] == "configs/base.json"
    assert payload["fold_output_paths"] == {
        str(fold): f"private/fold-{fold}" for fold in range(5)
    }
    assert "/mnt/" not in output.read_text(encoding="utf-8")
    unhashed = dict(payload)
    stored_payload_sha256 = unhashed.pop("payload_sha256")
    expected_payload_sha256 = hashlib.sha256(
        json.dumps(
            unhashed,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert stored_payload_sha256 == expected_payload_sha256 == result.payload_sha256
    assert hashlib.sha256(output.read_bytes()).hexdigest() == result.sha256

    with pytest.raises(GateBValidationError, match="overwrite"):
        write_run_context(
            output,
            source_root=root,
            source_commit="2" * 40,
            run_tag="base-oof-20260811-001",
            config_paths={"base": config},
            source_manifest_path=manifest,
            split_artifact_path=split,
            preflight_report_path=preflight,
            gpu_smoke_report_path=smoke,
            fold_output_paths={0: root / "private" / "fold-0"},
            planned_stages=("base_oof",),
        )

    outside = _write(tmp_path / "outside.json", "{}\n")
    with pytest.raises(GateBValidationError, match="source-root file"):
        write_run_context(
            root / "private" / "outside-context.json",
            source_root=root,
            source_commit="2" * 40,
            run_tag="outside",
            config_paths={"base": outside},
            source_manifest_path=manifest,
            split_artifact_path=split,
            preflight_report_path=preflight,
            gpu_smoke_report_path=smoke,
            fold_output_paths={0: root / "private" / "fold-0"},
            planned_stages=("base_oof",),
        )

    root_link = tmp_path / "source-link"
    root_link.symlink_to(root, target_is_directory=True)
    with pytest.raises(GateBValidationError, match="real directory"):
        write_run_context(
            root / "private" / "symlink-context.json",
            source_root=root_link,
            source_commit="2" * 40,
            run_tag="symlink",
            config_paths={"base": config},
            source_manifest_path=manifest,
            split_artifact_path=split,
            preflight_report_path=preflight,
            gpu_smoke_report_path=smoke,
            fold_output_paths={0: root / "private" / "fold-0"},
            planned_stages=("base_oof",),
        )

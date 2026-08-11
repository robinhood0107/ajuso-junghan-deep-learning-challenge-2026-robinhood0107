"""Raw-free workflow status and immutable run-context contracts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from .gate_b import DevelopmentResumeStatus, GateBValidationError
from .gate_b_runtime import TrainingResumeStatus
from .provenance import (
    canonical_json_bytes,
    sha256_file,
    validate_source_tree_manifest_artifact,
)

RUN_CONTEXT_SCHEMA = "gate-b-run-context-v1"
WORKFLOW_STATUS_SCHEMA = "gate-b-workflow-status-v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?\Z")
_RUN_TAG_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z")
_ALLOWED_STATES = frozenset(
    {"ready", "running", "qualified", "retryable", "terminal_failed", "complete"}
)


@dataclass(frozen=True, slots=True)
class WorkflowStatus:
    """One path-free status envelope shared by long Gate B jobs."""

    stage: str
    state: str
    terminal: bool
    resume_allowed: bool
    repair_allowed: bool
    next_action: str
    blockers: tuple[str, ...]
    completed_count: int
    total_count: int
    artifact_sha256: str

    def as_dict(self) -> dict[str, object]:
        if not self.stage or self.stage != self.stage.strip():
            raise GateBValidationError("workflow status stage is invalid")
        if self.state not in _ALLOWED_STATES:
            raise GateBValidationError("workflow status state is invalid")
        if not self.next_action or self.next_action != self.next_action.strip():
            raise GateBValidationError("workflow status next_action is invalid")
        if any(not item or item != item.strip() for item in self.blockers):
            raise GateBValidationError("workflow status blocker is invalid")
        if any(
            type(value) is not bool
            for value in (self.terminal, self.resume_allowed, self.repair_allowed)
        ):
            raise GateBValidationError("workflow status flags are invalid")
        if (
            isinstance(self.completed_count, bool)
            or not isinstance(self.completed_count, int)
            or self.completed_count < 0
            or isinstance(self.total_count, bool)
            or not isinstance(self.total_count, int)
            or self.total_count < 1
            or self.completed_count > self.total_count
        ):
            raise GateBValidationError("workflow status counts are invalid")
        _required_sha256(self.artifact_sha256, "workflow artifact_sha256")
        return {
            "schema_version": WORKFLOW_STATUS_SCHEMA,
            "stage": self.stage,
            "state": self.state,
            "terminal": self.terminal,
            "resume_allowed": self.resume_allowed,
            "repair_allowed": self.repair_allowed,
            "next_action": self.next_action,
            "blockers": list(self.blockers),
            "completed_count": self.completed_count,
            "total_count": self.total_count,
            "artifact_sha256": self.artifact_sha256,
        }


@dataclass(frozen=True, slots=True)
class RunContextWriteResult:
    path: str
    sha256: str
    payload_sha256: str


@dataclass(frozen=True, slots=True)
class ValidatedRunContext:
    path: str
    sha256: str
    payload_sha256: str
    source_commit: str
    run_tag: str


def development_workflow_status(status: DevelopmentResumeStatus) -> WorkflowStatus:
    """Translate a private development resume ledger to the shared envelope."""

    mapping = {
        "planned": ("ready", False, True, "start_development"),
        "running": ("running", False, False, "wait_for_development"),
        "interrupted": ("retryable", False, True, "resume_development"),
        "complete": ("complete", True, False, "verify_development_artifact"),
    }
    if status.state not in mapping:
        raise GateBValidationError("development resume state is unsupported")
    state, terminal, resume_allowed, next_action = mapping[status.state]
    return WorkflowStatus(
        stage="development_generation",
        state=state,
        terminal=terminal,
        resume_allowed=resume_allowed,
        repair_allowed=False,
        next_action=next_action,
        blockers=(),
        completed_count=status.completed_generations,
        total_count=status.total_generations,
        artifact_sha256=status.contract_sha256,
    )


def training_workflow_status(status: TrainingResumeStatus) -> WorkflowStatus:
    """Translate a private training resume ledger to the shared envelope."""

    mapping = {
        "ready": (False, True, "start_training", ()),
        "running": (False, False, "wait_for_training", ()),
        "retryable": (False, True, "resume_training", ()),
        "terminal_failed": (
            True,
            False,
            "record_terminal_training_failure",
            ("no_complete_resume_checkpoint",),
        ),
        "complete": (True, False, "verify_adapter_artifact", ()),
    }
    if status.state not in mapping:
        raise GateBValidationError("training resume state is unsupported")
    terminal, resume_allowed, next_action, blockers = mapping[status.state]
    artifact_sha256 = status.completion_artifact_sha256 or status.contract_sha256
    return WorkflowStatus(
        stage="fold_training",
        state=status.state,
        terminal=terminal,
        resume_allowed=resume_allowed,
        repair_allowed=False,
        next_action=next_action,
        blockers=blockers,
        completed_count=1 if status.state == "complete" else 0,
        total_count=1,
        artifact_sha256=artifact_sha256,
    )


def write_run_context(
    output_path: str | Path,
    *,
    source_root: str | Path,
    source_commit: str,
    run_tag: str,
    config_paths: Mapping[str, str | Path],
    source_manifest_path: str | Path,
    split_artifact_path: str | Path,
    preflight_report_path: str | Path,
    gpu_smoke_report_path: str | Path,
    fold_output_paths: Mapping[int, str | Path],
    planned_stages: Sequence[str],
) -> RunContextWriteResult:
    """Write one relative-path, no-overwrite execution context."""

    supplied_root = Path(source_root)
    if supplied_root.is_symlink():
        raise GateBValidationError("run context source_root must be a real directory")
    root = supplied_root.resolve(strict=True)
    if not root.is_dir():
        raise GateBValidationError("run context source_root must be a real directory")
    if _COMMIT_RE.fullmatch(source_commit) is None:
        raise GateBValidationError("run context source_commit is invalid")
    _require_clean_source_commit(root, source_commit)
    if _RUN_TAG_RE.fullmatch(run_tag) is None:
        raise GateBValidationError("run context run_tag is invalid")
    if not config_paths:
        raise GateBValidationError("run context requires at least one config")
    if not planned_stages or len(set(planned_stages)) != len(planned_stages):
        raise GateBValidationError("run context planned stages are invalid")
    if any(not stage or stage != stage.strip() for stage in planned_stages):
        raise GateBValidationError("run context planned stage is invalid")
    if not fold_output_paths or any(
        isinstance(fold, bool) or not isinstance(fold, int) or fold < 0
        for fold in fold_output_paths
    ):
        raise GateBValidationError("run context fold outputs are invalid")

    snapshots: list[tuple[Path, str, str]] = []

    def file_evidence(path: str | Path, label: str) -> dict[str, object]:
        source = Path(path)
        if source.is_symlink():
            raise GateBValidationError(f"run context {label} refuses symlinks")
        resolved = source.resolve(strict=True)
        if not resolved.is_file() or not resolved.is_relative_to(root):
            raise GateBValidationError(f"run context {label} must be a source-root file")
        digest = sha256_file(resolved)
        snapshots.append((resolved, digest, label))
        return {
            "path": resolved.relative_to(root).as_posix(),
            "sha256": digest,
        }

    configurations = {
        label: file_evidence(path, f"config {label}")
        for label, path in sorted(config_paths.items())
        if isinstance(label, str) and label and label == label.strip()
    }
    if set(configurations) != set(config_paths):
        raise GateBValidationError("run context config labels are invalid")
    source_manifest = file_evidence(source_manifest_path, "source manifest")
    try:
        validated_manifest = validate_source_tree_manifest_artifact(
            root / str(source_manifest["path"]),
            root=root,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise GateBValidationError("run context source manifest is invalid") from exc
    source_manifest["tree_sha256"] = validated_manifest.tree_sha256
    source_manifest["file_count"] = validated_manifest.file_count

    fold_paths = {
        str(fold): _relative_planned_path(path, root, f"fold {fold} output")
        for fold, path in sorted(fold_output_paths.items())
    }
    payload_without_hash = {
        "schema_version": RUN_CONTEXT_SCHEMA,
        "source_commit": source_commit,
        "run_tag": run_tag,
        "configs": configurations,
        "source_manifest": source_manifest,
        "split_artifact": file_evidence(split_artifact_path, "split artifact"),
        "runtime_gate": {
            "preflight_report": file_evidence(preflight_report_path, "preflight report"),
            "gpu_smoke_report": file_evidence(gpu_smoke_report_path, "GPU smoke report"),
        },
        "fold_output_paths": fold_paths,
        "planned_stages": list(planned_stages),
    }
    for source, expected_sha256, label in snapshots:
        if sha256_file(source) != expected_sha256:
            raise GateBValidationError(
                f"run context {label} changed during evidence snapshot"
            )
    try:
        final_manifest = validate_source_tree_manifest_artifact(
            root / str(source_manifest["path"]),
            root=root,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise GateBValidationError("run context source manifest changed during snapshot") from exc
    if (
        final_manifest.tree_sha256 != source_manifest["tree_sha256"]
        or final_manifest.file_count != source_manifest["file_count"]
    ):
        raise GateBValidationError("run context source tree changed during snapshot")
    _require_clean_source_commit(root, source_commit)
    payload_sha256 = hashlib.sha256(canonical_json_bytes(payload_without_hash)).hexdigest()
    payload = {**payload_without_hash, "payload_sha256": payload_sha256}
    target = Path(output_path).resolve(strict=False)
    if target.is_symlink() or target.parent.is_symlink() or not target.parent.is_dir():
        raise GateBValidationError("run context target is unsafe")
    if target.exists():
        raise GateBValidationError("refusing to overwrite run context")
    serialized = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            raise GateBValidationError("refusing to overwrite run context") from exc
        _fsync_directory(target.parent)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()
    validated = validate_run_context(target, source_root=root)
    return RunContextWriteResult(
        path=str(target),
        sha256=validated.sha256,
        payload_sha256=validated.payload_sha256,
    )


def validate_run_context(
    path: str | Path, *, source_root: str | Path
) -> ValidatedRunContext:
    """Revalidate every current input before any run-context consumer acts."""

    supplied_root = Path(source_root)
    if supplied_root.is_symlink():
        raise GateBValidationError("run context source_root must be a real directory")
    root = supplied_root.resolve(strict=True)
    supplied = Path(path)
    if supplied.is_symlink():
        raise GateBValidationError("run context artifact refuses symlinks")
    source = supplied.resolve(strict=True)
    if not source.is_file() or not source.is_relative_to(root):
        raise GateBValidationError("run context artifact must be a source-root file")
    try:
        payload = json.loads(
            source.read_text(encoding="utf-8", errors="strict"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GateBValidationError("run context artifact is invalid JSON") from exc
    expected_keys = {
        "schema_version",
        "source_commit",
        "run_tag",
        "configs",
        "source_manifest",
        "split_artifact",
        "runtime_gate",
        "fold_output_paths",
        "planned_stages",
        "payload_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise GateBValidationError("run context artifact schema is invalid")
    if payload.get("schema_version") != RUN_CONTEXT_SCHEMA:
        raise GateBValidationError("run context artifact schema_version is unsupported")
    stored_payload_sha256 = _required_sha256(
        payload.get("payload_sha256"), "run context payload_sha256"
    )
    unhashed = dict(payload)
    unhashed.pop("payload_sha256")
    if hashlib.sha256(canonical_json_bytes(unhashed)).hexdigest() != stored_payload_sha256:
        raise GateBValidationError("run context payload_sha256 is invalid")
    source_commit = payload.get("source_commit")
    run_tag = payload.get("run_tag")
    if not isinstance(source_commit, str) or _COMMIT_RE.fullmatch(source_commit) is None:
        raise GateBValidationError("run context source_commit is invalid")
    if not isinstance(run_tag, str) or _RUN_TAG_RE.fullmatch(run_tag) is None:
        raise GateBValidationError("run context run_tag is invalid")
    _require_clean_source_commit(root, source_commit)

    def verify_file_evidence(
        value: object, label: str, *, source_manifest: bool = False
    ) -> Path:
        required_keys = {"path", "sha256"}
        if source_manifest:
            required_keys |= {"tree_sha256", "file_count"}
        if not isinstance(value, Mapping) or set(value) != required_keys:
            raise GateBValidationError(f"run context {label} evidence is invalid")
        relative = value.get("path")
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise GateBValidationError(f"run context {label} path is invalid")
        candidate = root / relative
        if candidate.is_symlink():
            raise GateBValidationError(f"run context {label} path is unsafe")
        resolved = candidate.resolve(strict=True)
        if (
            not resolved.is_file()
            or not resolved.is_relative_to(root)
            or resolved.relative_to(root).as_posix() != relative
        ):
            raise GateBValidationError(f"run context {label} path is unsafe")
        expected_sha256 = _required_sha256(
            value.get("sha256"), f"run context {label} sha256"
        )
        if sha256_file(resolved) != expected_sha256:
            raise GateBValidationError(f"run context {label} bytes changed")
        return resolved

    configs = payload.get("configs")
    if not isinstance(configs, Mapping) or not configs:
        raise GateBValidationError("run context configs are invalid")
    for label, evidence in configs.items():
        if not isinstance(label, str) or not label or label != label.strip():
            raise GateBValidationError("run context config label is invalid")
        verify_file_evidence(evidence, f"config {label}")
    manifest_evidence = payload.get("source_manifest")
    manifest_path = verify_file_evidence(
        manifest_evidence, "source manifest", source_manifest=True
    )
    try:
        validated_manifest = validate_source_tree_manifest_artifact(
            manifest_path, root=root
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise GateBValidationError("run context source manifest is invalid") from exc
    assert isinstance(manifest_evidence, Mapping)
    if (
        manifest_evidence.get("tree_sha256") != validated_manifest.tree_sha256
        or manifest_evidence.get("file_count") != validated_manifest.file_count
    ):
        raise GateBValidationError("run context source manifest binding changed")
    verify_file_evidence(payload.get("split_artifact"), "split artifact")
    runtime_gate = payload.get("runtime_gate")
    if not isinstance(runtime_gate, Mapping) or set(runtime_gate) != {
        "preflight_report",
        "gpu_smoke_report",
    }:
        raise GateBValidationError("run context runtime gate is invalid")
    verify_file_evidence(runtime_gate.get("preflight_report"), "preflight report")
    verify_file_evidence(runtime_gate.get("gpu_smoke_report"), "GPU smoke report")
    fold_paths = payload.get("fold_output_paths")
    if not isinstance(fold_paths, Mapping) or not fold_paths:
        raise GateBValidationError("run context fold output paths are invalid")
    for fold, relative in fold_paths.items():
        if not isinstance(fold, str) or not fold.isdigit() or not isinstance(relative, str):
            raise GateBValidationError("run context fold output path is invalid")
        if _relative_planned_path(root / relative, root, f"fold {fold} output") != relative:
            raise GateBValidationError("run context fold output path is not canonical")
    stages = payload.get("planned_stages")
    if (
        not isinstance(stages, list)
        or not stages
        or len(set(stages)) != len(stages)
        or any(
            not isinstance(stage, str) or not stage or stage != stage.strip()
            for stage in stages
        )
    ):
        raise GateBValidationError("run context planned stages are invalid")
    return ValidatedRunContext(
        path=str(source),
        sha256=sha256_file(source),
        payload_sha256=stored_payload_sha256,
        source_commit=source_commit,
        run_tag=run_tag,
    )


def _relative_planned_path(path: str | Path, root: Path, label: str) -> str:
    supplied = Path(path)
    if supplied.is_symlink():
        raise GateBValidationError(f"run context {label} refuses symlinks")
    resolved = supplied.resolve(strict=False)
    if not resolved.is_relative_to(root) or resolved == root:
        raise GateBValidationError(f"run context {label} must be under source_root")
    return resolved.relative_to(root).as_posix()


def _required_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise GateBValidationError(f"{label} is invalid")
    return value


def _require_clean_source_commit(root: Path, source_commit: str) -> None:
    def run_git(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ("git", "-C", str(root), *arguments),
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )

    top_level = run_git("rev-parse", "--show-toplevel")
    head = run_git("rev-parse", "HEAD")
    if top_level.returncode != 0 or head.returncode != 0:
        raise GateBValidationError("run context source_root must be a Git checkout")
    try:
        resolved_top_level = Path(top_level.stdout.strip()).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise GateBValidationError("run context Git root is invalid") from exc
    if resolved_top_level != root or head.stdout.strip() != source_commit:
        raise GateBValidationError("run context source_commit does not match source_root HEAD")
    clean = run_git("status", "--porcelain=v1", "--untracked-files=all")
    if clean.returncode != 0 or clean.stdout:
        raise GateBValidationError("run context requires a clean source tree")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise GateBValidationError(f"duplicate JSON key {key!r}")
        output[key] = value
    return output


def _reject_json_constant(value: str) -> None:
    raise GateBValidationError(f"non-standard JSON numeric constant {value!r}")

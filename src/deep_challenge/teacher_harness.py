"""Raw-free synthetic live-evaluation harness for the Codex teacher boundary.

The production teacher ledger intentionally has a coarse, durable failure
reason.  This module never mutates that ledger.  Instead it reads already
verified private attempt evidence and emits only fixed categories and counts.
It also owns a contest-independent 64-row canary whose two immutable chunks
exercise the same question-only prompt and JSON validator before a new pilot
profile is permitted to send organizer questions to Codex.

No report produced here contains a problem ID, question, answer, rationale,
prompt, provider stderr, raw event text, or a machine-specific path.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path

from .answers import parse_answer
from .data import MathRecord
from .provenance import SourceTreeArtifactEvidence, canonical_json_bytes, sha256_file
from .teacher_rationale import (
    CodexCommandResult,
    TeacherAttempt,
    TeacherExecutionConfig,
    TeacherPlan,
    TeacherPromptPolicy,
    TeacherRationaleValidationError,
    create_teacher_plan,
    load_teacher_attempts,
    load_teacher_plan,
    run_teacher_plan,
    validate_codex_event_stream,
)

FAILURE_CLASSIFIER_SCHEMA = "gate-b-codex-teacher-failure-classifier-v1"
DIAGNOSTIC_SCHEMA = "gate-b-codex-teacher-diagnostic-v1"
HARNESS_REPLAY_SCHEMA = "gate-b-codex-teacher-harness-replay-v1"
HARNESS_LIVE_SCHEMA = "gate-b-codex-teacher-harness-live-v1"
HARNESS_AUTHORIZATION_SCHEMA = "gate-b-codex-teacher-harness-authorization-v1"

HARNESS_CONFIG_SCHEMA = "gate-b-codex-teacher-harness-config-v1"
HARNESS_FIXTURE_SCHEMA = "gate-b-codex-teacher-harness-fixture-v1"
HARNESS_LABEL = "codex-gpt-5.6-sol-teacher-harness-v1"
HARNESS_VERSION = "harness-v1"
HARNESS_CHUNK_SIZE = 32
HARNESS_CHUNK_COUNT = 2
HARNESS_FIXTURE_SIZE = HARNESS_CHUNK_SIZE * HARNESS_CHUNK_COUNT
HARNESS_REPLAY_FAULT_COUNT = 20

_SHA256_HEX_LENGTH = 64
_ALLOWED_EVENT_TYPES = frozenset(
    {
        "thread.started",
        "turn.started",
        "item.started",
        "item.updated",
        "item.completed",
        "turn.completed",
    }
)
_CLASSIFICATION_KEYS = frozenset(
    {
        "stage",
        "code",
        "requested_count",
        "returned_count",
        "duplicate_count",
        "missing_count",
        "unexpected_count",
        "order_mismatch",
    }
)
_ALLOWED_CLASSIFICATION_PAIRS = frozenset(
    {
        ("process", "runner_exception"),
        ("timeout_spawn", "timeout"),
        ("timeout_spawn", "spawn_failure"),
        ("nonzero", "command_nonzero"),
        ("event_json", "malformed_event_json"),
        ("event_json", "unreadable_event_stream"),
        ("unsafe_error_event", "unsafe_event_type"),
        ("unsafe_error_event", "error_event"),
        ("unsafe_error_event", "invalid_item_event"),
        ("unsafe_error_event", "unsafe_item"),
        ("terminal_usage", "multiple_terminal_events"),
        ("terminal_usage", "invalid_usage"),
        ("terminal_usage", "missing_terminal_event"),
        ("terminal_usage", "terminal_not_final"),
        ("agent_json", "missing_agent_message"),
        ("agent_json", "multiple_agent_messages"),
        ("agent_json", "malformed_agent_json"),
        ("output_schema", "invalid_output_schema"),
        ("output_schema", "invalid_item_schema"),
        ("output_structure", "cardinality_mismatch"),
        ("output_structure", "id_set_mismatch"),
        ("output_structure", "duplicate_ids"),
        ("output_structure", "order_mismatch"),
        ("target_policy", "target_policy_invalid"),
        ("target_policy", "wrong_target"),
        ("success", "qualified"),
    }
)

# These signed operands are a fixed, contest-independent test fixture.  The
# fixture deliberately stays small enough to be sent in exactly two 32-row
# canary calls, while using both signs and zero-like cancellation cases.
_FIXTURE_OPERANDS: tuple[tuple[int, int], ...] = (
    (17, -9),
    (-23, 8),
    (14, 6),
    (-11, -7),
    (31, -31),
    (42, -19),
    (-36, 15),
    (9, 27),
    (-48, 22),
    (55, -13),
    (-17, -18),
    (63, 4),
    (-29, 41),
    (18, -26),
    (72, -40),
    (-54, 9),
    (26, 35),
    (-64, 28),
    (11, -11),
    (39, -52),
    (-7, 44),
    (58, -34),
    (-45, -12),
    (24, 16),
    (-71, 29),
    (33, -47),
    (-16, 53),
    (67, -25),
    (-38, 38),
    (12, -49),
    (46, 21),
    (-59, -5),
    (28, -14),
    (-32, 45),
    (74, -18),
    (-26, -33),
    (5, 57),
    (-68, 24),
    (49, -49),
    (37, 12),
    (-13, 66),
    (81, -37),
    (-42, -21),
    (19, 48),
    (-57, 30),
    (64, -16),
    (-24, 24),
    (43, -58),
    (-35, 17),
    (22, 39),
    (-76, 31),
    (15, -63),
    (-51, -14),
    (69, 7),
    (-8, 36),
    (52, -27),
    (-44, 44),
    (30, -15),
    (-61, 20),
    (47, 26),
    (-28, -35),
    (6, 62),
    (-73, 19),
    (34, -56),
)


class TeacherHarnessValidationError(ValueError):
    """Raised for an invalid synthetic harness contract or evidence artifact."""


class TeacherHarnessArtifactExistsError(FileExistsError):
    """Raised when a raw-free harness artifact would overwrite prior evidence."""


@dataclass(frozen=True, slots=True)
class SyntheticFixtureRow:
    """One fixed synthetic arithmetic prompt and its local-only expected integer."""

    problem_id: str
    question: str
    expected_answer: int


@dataclass(frozen=True, slots=True)
class HarnessProfile:
    """Immutable execution caps for the offline and live synthetic harness."""

    label: str
    version: str
    seed: int
    initial_chunk_size: int
    chunk_count: int
    max_workers: int
    max_invocations: int
    max_attempts: int
    retry_count: int
    repair_count: int
    bank_output_count: int
    initial_reasoning_effort: str

    def __post_init__(self) -> None:
        for name in (
            "seed",
            "initial_chunk_size",
            "chunk_count",
            "max_workers",
            "max_invocations",
            "max_attempts",
            "retry_count",
            "repair_count",
            "bank_output_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TeacherHarnessValidationError(f"synthetic harness {name} is invalid")
        if self.label != HARNESS_LABEL or self.version != HARNESS_VERSION:
            raise TeacherHarnessValidationError("synthetic harness label/version is invalid")
        if self.seed != 20_260_731:
            raise TeacherHarnessValidationError("synthetic harness seed is invalid")
        if self.initial_chunk_size != HARNESS_CHUNK_SIZE:
            raise TeacherHarnessValidationError("synthetic harness chunk size is invalid")
        if self.chunk_count != HARNESS_CHUNK_COUNT:
            raise TeacherHarnessValidationError("synthetic harness chunk count is invalid")
        if self.max_workers != 1 or self.max_invocations != HARNESS_CHUNK_COUNT:
            raise TeacherHarnessValidationError("synthetic harness execution cap is invalid")
        if self.max_attempts != 1 or self.retry_count != 0 or self.repair_count != 0:
            raise TeacherHarnessValidationError("synthetic harness retry policy is invalid")
        if self.bank_output_count != 0:
            raise TeacherHarnessValidationError("synthetic harness must not create a bank")
        if self.initial_reasoning_effort != "high":
            raise TeacherHarnessValidationError("synthetic harness reasoning effort is invalid")


@dataclass(frozen=True, slots=True)
class FailureClassification:
    """One redacted classifier result, intentionally limited to fixed counts."""

    stage: str
    code: str
    requested_count: int
    returned_count: int
    duplicate_count: int
    missing_count: int
    unexpected_count: int
    order_mismatch: bool

    def __post_init__(self) -> None:
        if not self.stage or not self.code:
            raise TeacherHarnessValidationError("classifier stage and code must be non-empty")
        if (self.stage, self.code) not in _ALLOWED_CLASSIFICATION_PAIRS:
            raise TeacherHarnessValidationError("classifier stage/code is not allowlisted")
        for name in (
            "requested_count",
            "returned_count",
            "duplicate_count",
            "missing_count",
            "unexpected_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TeacherHarnessValidationError(f"classifier {name} is invalid")
        if not isinstance(self.order_mismatch, bool):
            raise TeacherHarnessValidationError("classifier order_mismatch is invalid")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class HarnessReportResult:
    """Raw-free result returned after an immutable report is written."""

    qualified: bool
    report_sha256: str
    payload_sha256: str
    plan_sha256: str | None = None


def synthetic_fixture_rows() -> tuple[SyntheticFixtureRow, ...]:
    """Return the fixed 64-row signed-integer fixture in canonical order."""

    rows = tuple(
        SyntheticFixtureRow(
            problem_id=f"train-{900_000 + index:06d}",
            question=(f"Compute the signed integer sum ({left}) + ({right}). Return the integer."),
            expected_answer=left + right,
        )
        for index, (left, right) in enumerate(_FIXTURE_OPERANDS, start=1)
    )
    if len(rows) != HARNESS_FIXTURE_SIZE:  # pragma: no cover - static tuple guard
        raise RuntimeError("synthetic harness fixture size drifted")
    return rows


def synthetic_fixture_sha256() -> str:
    """Hash the exact fixture words and expected local-only signed integers."""

    return hashlib.sha256(
        canonical_json_bytes(
            {
                "schema_version": HARNESS_FIXTURE_SCHEMA,
                "items": [
                    {
                        "problem_id": row.problem_id,
                        "question": row.question,
                        "expected_answer": row.expected_answer,
                    }
                    for row in synthetic_fixture_rows()
                ],
            }
        )
    ).hexdigest()


def profile_from_config(config: Mapping[str, object]) -> HarnessProfile:
    """Construct the immutable harness profile from an already locked config."""

    required = {
        "schema_version",
        "label",
        "version",
        "provider",
        "model_id",
        "model_revision",
        "seed",
        "initial_chunk_size",
        "chunk_count",
        "max_workers",
        "max_invocations",
        "max_attempts",
        "retry_count",
        "repair_count",
        "bank_output_count",
        "initial_reasoning_effort",
        "reference_answer_in_prompt",
        "allow_tool_use",
        "network_scope",
        "fixture_version",
        "fixture_sha256",
    }
    if set(config) != required:
        raise TeacherHarnessValidationError("synthetic harness config keys are invalid")
    if config["schema_version"] != HARNESS_CONFIG_SCHEMA:
        raise TeacherHarnessValidationError("synthetic harness config schema is invalid")
    for name in (
        "schema_version",
        "label",
        "version",
        "provider",
        "model_id",
        "model_revision",
        "initial_reasoning_effort",
        "network_scope",
        "fixture_version",
        "fixture_sha256",
    ):
        if not isinstance(config[name], str):
            raise TeacherHarnessValidationError("synthetic harness text field is invalid")
    for name in (
        "seed",
        "initial_chunk_size",
        "chunk_count",
        "max_workers",
        "max_invocations",
        "max_attempts",
        "retry_count",
        "repair_count",
        "bank_output_count",
    ):
        if isinstance(config[name], bool) or not isinstance(config[name], int):
            raise TeacherHarnessValidationError("synthetic harness integer field is invalid")
    for name in ("reference_answer_in_prompt", "allow_tool_use"):
        if not isinstance(config[name], bool):
            raise TeacherHarnessValidationError("synthetic harness boolean field is invalid")
    if (
        config["provider"] != "chatgpt_codex_cli"
        or config["model_id"] != "gpt-5.6-sol"
        or config["model_revision"] != "gpt-5.6-sol"
        or config["reference_answer_in_prompt"] is not False
        or config["allow_tool_use"] is not False
        or config["network_scope"] != "synthetic_canary_only"
        or config["fixture_version"] != HARNESS_FIXTURE_SCHEMA
        or config["fixture_sha256"] != synthetic_fixture_sha256()
    ):
        raise TeacherHarnessValidationError("synthetic harness config contract is invalid")
    values = {
        name: config[name]
        for name in (
            "label",
            "version",
            "seed",
            "initial_chunk_size",
            "chunk_count",
            "max_workers",
            "max_invocations",
            "max_attempts",
            "retry_count",
            "repair_count",
            "bank_output_count",
            "initial_reasoning_effort",
        )
    }
    return HarnessProfile(**values)  # type: ignore[arg-type]


def classify_codex_result(
    result: CodexCommandResult,
    expected_ids: Sequence[str],
    *,
    prompt_policy: TeacherPromptPolicy,
    expected_answers: Mapping[str, int] | None = None,
    failure_reason: str | None = None,
) -> FailureClassification:
    """Classify a Codex result without returning any untrusted content.

    The ordered checks intentionally mirror the documented fixed priority:
    process, timeout/spawn, nonzero, event JSON, unsafe/error event,
    terminal/usage, agent JSON, output schema, cardinality, ID coverage/order,
    target policy, then success.
    """

    if not isinstance(result, CodexCommandResult):
        raise TeacherHarnessValidationError("classifier result is invalid")
    expected = tuple(expected_ids)
    if not expected or len(set(expected)) != len(expected):
        raise TeacherHarnessValidationError("classifier expected IDs are invalid")
    if not isinstance(prompt_policy, TeacherPromptPolicy):
        raise TeacherHarnessValidationError("classifier prompt policy is invalid")
    counts = _empty_counts(len(expected))
    if failure_reason == "runner_exception":
        return _classification("process", "runner_exception", counts)
    if result.returncode == 124:
        return _classification("timeout_spawn", "timeout", counts)
    if result.returncode == 127:
        return _classification("timeout_spawn", "spawn_failure", counts)
    if result.returncode != 0:
        return _classification("nonzero", "command_nonzero", counts)

    decoded = _decode_event_stream(result.stdout)
    if decoded is None:
        return _classification("event_json", "malformed_event_json", counts)
    events = decoded
    agent_message: str | None = None
    terminal_index: int | None = None
    for index, event in enumerate(events):
        event_type = event.get("type")
        if not isinstance(event_type, str) or event_type not in _ALLOWED_EVENT_TYPES:
            return _classification("unsafe_error_event", "unsafe_event_type", counts)
        if "error" in event or event_type in {"turn.failed", "error"}:
            return _classification("unsafe_error_event", "error_event", counts)
        if event_type.startswith("item."):
            item = event.get("item")
            if not isinstance(item, Mapping):
                return _classification("unsafe_error_event", "invalid_item_event", counts)
            item_type = item.get("type")
            if item_type not in {"agent_message", "reasoning"}:
                return _classification("unsafe_error_event", "unsafe_item", counts)
            if event_type == "item.completed" and item_type == "agent_message":
                text = item.get("text")
                if not isinstance(text, str) or not text:
                    return _classification("agent_json", "missing_agent_message", counts)
                if agent_message is not None:
                    return _classification("agent_json", "multiple_agent_messages", counts)
                agent_message = text
        if event_type == "turn.completed":
            if terminal_index is not None:
                return _classification("terminal_usage", "multiple_terminal_events", counts)
            terminal_index = index
            if not _valid_usage(event.get("usage")):
                return _classification("terminal_usage", "invalid_usage", counts)
    if terminal_index is None:
        return _classification("terminal_usage", "missing_terminal_event", counts)
    if terminal_index != len(events) - 1:
        return _classification("terminal_usage", "terminal_not_final", counts)
    if agent_message is None:
        return _classification("agent_json", "missing_agent_message", counts)

    payload = _decode_json_object(agent_message)
    if payload is None:
        return _classification("agent_json", "malformed_agent_json", counts)
    if set(payload) != {"items"} or not isinstance(payload.get("items"), list):
        return _classification("output_schema", "invalid_output_schema", counts)
    raw_items = payload["items"]
    assert isinstance(raw_items, list)
    for raw in raw_items:
        if (
            not isinstance(raw, Mapping)
            or set(raw) != {"problem_id", "target_text"}
            or not isinstance(raw.get("problem_id"), str)
            or not isinstance(raw.get("target_text"), str)
        ):
            return _classification("output_schema", "invalid_item_schema", counts)
    observed_ids = tuple(str(raw["problem_id"]) for raw in raw_items)
    counts = _id_counts(expected, observed_ids)
    if counts["returned_count"] != len(expected):
        return _classification("output_structure", "cardinality_mismatch", counts)
    if counts["missing_count"] or counts["unexpected_count"]:
        return _classification("output_structure", "id_set_mismatch", counts)
    if counts["duplicate_count"]:
        return _classification("output_structure", "duplicate_ids", counts)
    if counts["order_mismatch"]:
        return _classification("output_structure", "order_mismatch", counts)
    try:
        output = validate_codex_event_stream(
            result.stdout,
            expected,
            prompt_policy=prompt_policy,
        )
    except TeacherRationaleValidationError:
        return _classification("target_policy", "target_policy_invalid", counts)
    if expected_answers is not None:
        if set(expected_answers) != set(expected):
            raise TeacherHarnessValidationError("synthetic expected-answer map is invalid")
        for item in output.items:
            parsed = parse_answer(item.target_text)
            if not parsed.ok or parsed.value != expected_answers[item.problem_id]:
                return _classification("target_policy", "wrong_target", counts)
    return _classification("success", "qualified", counts)


def diagnose_teacher_ledger(
    plan_dir: str | Path,
    output_path: str | Path,
    *,
    teacher_config_sha256: str,
    teacher_config_file_sha256: str,
    prompt_policy: TeacherPromptPolicy,
) -> HarnessReportResult:
    """Publish an immutable raw-free classification of an existing ledger.

    A before/after complete private file digest proves this read-only diagnostic
    did not resume, alter, or otherwise mutate the supplied ledger.
    """

    _require_sha256(teacher_config_sha256, "teacher config SHA")
    _require_sha256(teacher_config_file_sha256, "teacher config file SHA")
    plan = load_teacher_plan(plan_dir)
    if plan.prompt_policy != prompt_policy:
        raise TeacherHarnessValidationError("diagnostic plan prompt policy differs from config")
    before = _private_tree_snapshot(plan.plan_dir)
    classifications: list[FailureClassification] = []
    for attempt in load_teacher_attempts(plan.plan_dir):
        try:
            stdout = attempt.event_stream_path.read_text(encoding="utf-8", errors="strict")
        except (OSError, UnicodeError):
            classifications.append(
                _classification(
                    "event_json",
                    "unreadable_event_stream",
                    _empty_counts(len(attempt.input_ids)),
                )
            )
            continue
        result = CodexCommandResult(
            stdout=stdout,
            stderr="",
            returncode=attempt.returncode if attempt.returncode is not None else 127,
            latency_ms=attempt.latency_ms,
        )
        classifications.append(
            classify_codex_result(
                result,
                attempt.input_ids,
                prompt_policy=prompt_policy,
                failure_reason=attempt.failure_reason,
            )
        )
    after = _private_tree_snapshot(plan.plan_dir)
    if before != after:
        raise TeacherHarnessValidationError("diagnostic changed the supplied teacher ledger")
    payload_without_hash = {
        "schema_version": DIAGNOSTIC_SCHEMA,
        "failure_classifier_schema_version": FAILURE_CLASSIFIER_SCHEMA,
        "teacher_config_sha256": teacher_config_sha256,
        "teacher_config_file_sha256": teacher_config_file_sha256,
        "teacher_prompt_policy_sha256": prompt_policy.sha256,
        "prompt_template_sha256": _required_template_sha(prompt_policy),
        "plan_sha256": plan.plan_sha256,
        "ledger_tree_sha256": _snapshot_sha256(before),
        "classifications": [item.as_dict() for item in classifications],
    }
    payload = _with_payload_sha(payload_without_hash)
    target = _write_json_noreplace(output_path, payload)
    return HarnessReportResult(
        qualified=all(item.stage == "success" for item in classifications),
        report_sha256=sha256_file(target),
        payload_sha256=str(payload["payload_sha256"]),
        plan_sha256=plan.plan_sha256,
    )


def run_harness_replay(
    output_path: str | Path,
    *,
    harness_config_sha256: str,
    harness_config_file_sha256: str,
    teacher_config_sha256: str,
    teacher_config_file_sha256: str,
    prompt_policy: TeacherPromptPolicy,
    profile: HarnessProfile,
) -> HarnessReportResult:
    """Run the finite fault matrix locally, with no Codex process or network call."""

    _validate_profile_and_hashes(
        profile,
        harness_config_sha256=harness_config_sha256,
        harness_config_file_sha256=harness_config_file_sha256,
        teacher_config_sha256=teacher_config_sha256,
        teacher_config_file_sha256=teacher_config_file_sha256,
        prompt_policy=prompt_policy,
    )
    classifications, qualified = _replay_classifications_for_policy(prompt_policy)
    payload_without_hash = {
        "schema_version": HARNESS_REPLAY_SCHEMA,
        "failure_classifier_schema_version": FAILURE_CLASSIFIER_SCHEMA,
        "harness_config_sha256": harness_config_sha256,
        "harness_config_file_sha256": harness_config_file_sha256,
        "teacher_config_sha256": teacher_config_sha256,
        "teacher_config_file_sha256": teacher_config_file_sha256,
        "teacher_prompt_policy_sha256": prompt_policy.sha256,
        "prompt_template_sha256": _required_template_sha(prompt_policy),
        "fixture_sha256": synthetic_fixture_sha256(),
        "classifications": [item.as_dict() for item in classifications],
        "qualified": qualified,
    }
    payload = _with_payload_sha(payload_without_hash)
    target = _write_json_noreplace(output_path, payload)
    return HarnessReportResult(
        qualified=qualified,
        report_sha256=sha256_file(target),
        payload_sha256=str(payload["payload_sha256"]),
    )


def run_harness_live(
    plan_dir: str | Path,
    report_path: str | Path,
    *,
    harness_config_sha256: str,
    harness_config_file_sha256: str,
    teacher_config_sha256: str,
    teacher_config_file_sha256: str,
    prompt_policy: TeacherPromptPolicy,
    execution: TeacherExecutionConfig,
    profile: HarnessProfile,
    source_manifest: SourceTreeArtifactEvidence,
    command_runner: Callable[[tuple[str, ...]], CodexCommandResult],
    working_directory: str | Path,
) -> HarnessReportResult:
    """Execute exactly two immutable synthetic canary chunks through Codex.

    The generic teacher plan is only a private command/evidence container; it
    receives answer-free fixture records, is run once with one worker, and is
    never finalized into a bank.  Local expected answers are consulted only
    after the command completes while generating the redacted qualification.
    """

    _validate_profile_and_hashes(
        profile,
        harness_config_sha256=harness_config_sha256,
        harness_config_file_sha256=harness_config_file_sha256,
        teacher_config_sha256=teacher_config_sha256,
        teacher_config_file_sha256=teacher_config_file_sha256,
        prompt_policy=prompt_policy,
    )
    if not isinstance(execution, TeacherExecutionConfig):
        raise TeacherHarnessValidationError("synthetic live execution is invalid")
    if execution.reasoning_effort != "high":
        raise TeacherHarnessValidationError("synthetic live execution effort is invalid")
    if not isinstance(source_manifest, SourceTreeArtifactEvidence):
        raise TeacherHarnessValidationError("synthetic source manifest evidence is invalid")
    _validate_source_manifest_payload(source_manifest.as_dict())
    if not callable(command_runner):
        raise TeacherHarnessValidationError("synthetic live command runner is invalid")
    _new_harness_plan_directory_target(plan_dir)
    _new_file_target(report_path, "synthetic harness report")
    rows = synthetic_fixture_rows()
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
        plan_dir,
        chunk_size=HARNESS_CHUNK_SIZE,
        label=profile.label,
        version=profile.version,
        execution=execution,
        prompt_policy=prompt_policy,
    )
    result = run_teacher_plan(
        plan.plan_dir,
        command_runner,
        max_attempts=profile.max_attempts,
        repair_chunk_size=16,
        max_chunks=profile.max_invocations,
        max_workers=profile.max_workers,
        working_directory=working_directory,
    )
    attempts = load_teacher_attempts(plan.plan_dir)
    if result.attempts_written != HARNESS_CHUNK_COUNT or len(attempts) != HARNESS_CHUNK_COUNT:
        raise TeacherHarnessValidationError("synthetic live canary did not make exactly two calls")
    classifications = _classify_harness_live_attempts(plan, attempts, prompt_policy)
    qualified = len(classifications) == HARNESS_CHUNK_COUNT and all(
        item.stage == "success" and item.code == "qualified" for item in classifications
    )
    payload_without_hash = {
        "schema_version": HARNESS_LIVE_SCHEMA,
        "failure_classifier_schema_version": FAILURE_CLASSIFIER_SCHEMA,
        "harness_config_sha256": harness_config_sha256,
        "harness_config_file_sha256": harness_config_file_sha256,
        "teacher_config_sha256": teacher_config_sha256,
        "teacher_config_file_sha256": teacher_config_file_sha256,
        "teacher_prompt_policy_sha256": prompt_policy.sha256,
        "prompt_template_sha256": _required_template_sha(prompt_policy),
        "fixture_sha256": synthetic_fixture_sha256(),
        "source_manifest": source_manifest.as_dict(),
        "codex_binary_sha256": _codex_binary_sha256(execution),
        "codex_cli_version_sha256": _text_sha256(execution.codex_cli_version),
        "plan_sha256": plan.plan_sha256,
        "ledger_tree_sha256": _snapshot_sha256(_private_tree_snapshot(plan.plan_dir)),
        "classifications": [item.as_dict() for item in classifications],
        "qualified": qualified,
    }
    payload = _with_payload_sha(payload_without_hash)
    target = _write_json_noreplace(report_path, payload)
    return HarnessReportResult(
        qualified=qualified,
        report_sha256=sha256_file(target),
        payload_sha256=str(payload["payload_sha256"]),
        plan_sha256=plan.plan_sha256,
    )


def _classify_harness_live_attempts(
    plan: TeacherPlan,
    attempts: Sequence[TeacherAttempt],
    prompt_policy: TeacherPromptPolicy,
) -> tuple[FailureClassification, ...]:
    """Reclassify the two private canary attempts without exposing their text."""

    expected_ids = tuple(row.problem_id for row in synthetic_fixture_rows())
    expected_chunks = tuple(
        expected_ids[offset : offset + HARNESS_CHUNK_SIZE]
        for offset in range(0, len(expected_ids), HARNESS_CHUNK_SIZE)
    )
    if (
        plan.label != HARNESS_LABEL
        or plan.version != HARNESS_VERSION
        or plan.prompt_policy != prompt_policy
        or plan.problem_ids != expected_ids
        or tuple(
            (chunk.chunk_index, chunk.problem_ids)
            for chunk in plan.chunks
        )
        != tuple(enumerate(expected_chunks))
    ):
        raise TeacherHarnessValidationError("synthetic live plan differs from the fixed fixture")
    ordered_attempts = tuple(sorted(attempts, key=lambda item: item.key))
    if (
        len(ordered_attempts) != HARNESS_CHUNK_COUNT
        or tuple(
            (
                attempt.chunk_index,
                attempt.attempt_number,
                attempt.input_ids,
            )
            for attempt in ordered_attempts
        )
        != tuple((index, 1, chunk_ids) for index, chunk_ids in enumerate(expected_chunks))
    ):
        raise TeacherHarnessValidationError("synthetic live attempt layout is invalid")
    expected_answers = {row.problem_id: row.expected_answer for row in synthetic_fixture_rows()}
    classifications: list[FailureClassification] = []
    for attempt in ordered_attempts:
        try:
            stdout = attempt.event_stream_path.read_text(encoding="utf-8", errors="strict")
        except (OSError, UnicodeError):
            classifications.append(
                _classification(
                    "event_json",
                    "unreadable_event_stream",
                    _empty_counts(len(attempt.input_ids)),
                )
            )
            continue
        input_ids = attempt.input_ids
        classifications.append(
            classify_codex_result(
                CodexCommandResult(
                    stdout=stdout,
                    stderr="",
                    returncode=attempt.returncode if attempt.returncode is not None else 127,
                    latency_ms=attempt.latency_ms,
                ),
                input_ids,
                prompt_policy=prompt_policy,
                expected_answers={key: expected_answers[key] for key in input_ids},
                failure_reason=attempt.failure_reason,
            )
        )
    return tuple(classifications)


def _verify_harness_live_plan_evidence(
    live_plan_dir: str | Path,
    *,
    live_payload: Mapping[str, object],
    prompt_policy: TeacherPromptPolicy,
) -> str:
    """Re-derive a live report from its private no-bank plan before promotion."""

    plan = load_teacher_plan(live_plan_dir)
    if (
        plan.plan_sha256 != live_payload.get("plan_sha256")
        or plan.execution.reasoning_effort != "high"
        or _codex_binary_sha256(plan.execution) != live_payload.get("codex_binary_sha256")
        or _text_sha256(plan.execution.codex_cli_version)
        != live_payload.get("codex_cli_version_sha256")
    ):
        raise TeacherHarnessValidationError("live harness plan provenance does not match report")
    assessments_dir = plan.plan_dir / "assessments"
    if not assessments_dir.is_dir() or any(assessments_dir.iterdir()):
        raise TeacherHarnessValidationError("synthetic live harness must not contain assessments")
    classifications = _classify_harness_live_attempts(
        plan,
        load_teacher_attempts(plan.plan_dir),
        prompt_policy,
    )
    if [item.as_dict() for item in classifications] != live_payload.get("classifications"):
        raise TeacherHarnessValidationError(
            "live harness plan does not reproduce report classifications"
        )
    ledger_tree_sha256 = _snapshot_sha256(_private_tree_snapshot(plan.plan_dir))
    if ledger_tree_sha256 != live_payload.get("ledger_tree_sha256"):
        raise TeacherHarnessValidationError("live harness ledger digest does not match report")
    return ledger_tree_sha256


def create_harness_authorization(
    output_path: str | Path,
    *,
    replay_report: str | Path,
    live_report: str | Path,
    live_plan_dir: str | Path,
    harness_config_sha256: str,
    harness_config_file_sha256: str,
    teacher_config_sha256: str,
    teacher_config_file_sha256: str,
    prompt_policy: TeacherPromptPolicy,
    source_manifest: SourceTreeArtifactEvidence,
) -> str:
    """Bind qualified replay/live evidence for a later teacher-pilot plan."""

    payload = _build_harness_authorization_payload(
        replay_report=replay_report,
        live_report=live_report,
        live_plan_dir=live_plan_dir,
        harness_config_sha256=harness_config_sha256,
        harness_config_file_sha256=harness_config_file_sha256,
        teacher_config_sha256=teacher_config_sha256,
        teacher_config_file_sha256=teacher_config_file_sha256,
        prompt_policy=prompt_policy,
        source_manifest=source_manifest,
    )
    _write_json_noreplace(output_path, payload)
    return str(payload["payload_sha256"])


def _build_harness_authorization_payload(
    *,
    replay_report: str | Path,
    live_report: str | Path,
    live_plan_dir: str | Path,
    harness_config_sha256: str,
    harness_config_file_sha256: str,
    teacher_config_sha256: str,
    teacher_config_file_sha256: str,
    prompt_policy: TeacherPromptPolicy,
    source_manifest: SourceTreeArtifactEvidence,
) -> dict[str, object]:
    """Recompute the raw-free authorization payload without publishing it."""

    _validate_profile_and_hashes(
        HarnessProfile(
            label=HARNESS_LABEL,
            version=HARNESS_VERSION,
            seed=20_260_731,
            initial_chunk_size=HARNESS_CHUNK_SIZE,
            chunk_count=HARNESS_CHUNK_COUNT,
            max_workers=1,
            max_invocations=HARNESS_CHUNK_COUNT,
            max_attempts=1,
            retry_count=0,
            repair_count=0,
            bank_output_count=0,
            initial_reasoning_effort="high",
        ),
        harness_config_sha256=harness_config_sha256,
        harness_config_file_sha256=harness_config_file_sha256,
        teacher_config_sha256=teacher_config_sha256,
        teacher_config_file_sha256=teacher_config_file_sha256,
        prompt_policy=prompt_policy,
    )
    replay_payload, replay_file_sha256 = _load_harness_report(
        replay_report,
        HARNESS_REPLAY_SCHEMA,
        prompt_policy=prompt_policy,
    )
    live_payload, live_file_sha256 = _load_harness_report(live_report, HARNESS_LIVE_SCHEMA)
    expected = {
        "harness_config_sha256": harness_config_sha256,
        "harness_config_file_sha256": harness_config_file_sha256,
        "teacher_config_sha256": teacher_config_sha256,
        "teacher_config_file_sha256": teacher_config_file_sha256,
        "teacher_prompt_policy_sha256": prompt_policy.sha256,
        "prompt_template_sha256": _required_template_sha(prompt_policy),
        "fixture_sha256": synthetic_fixture_sha256(),
    }
    for key, value in expected.items():
        if replay_payload.get(key) != value or live_payload.get(key) != value:
            raise TeacherHarnessValidationError("harness reports do not match the current contract")
    if replay_payload["qualified"] is not True or live_payload["qualified"] is not True:
        raise TeacherHarnessValidationError("harness reports are not qualified")
    source_payload = source_manifest.as_dict()
    _validate_source_manifest_payload(source_payload)
    if live_payload.get("source_manifest") != source_payload:
        raise TeacherHarnessValidationError("live harness source manifest does not match")
    live_ledger_tree_sha256 = _verify_harness_live_plan_evidence(
        live_plan_dir,
        live_payload=live_payload,
        prompt_policy=prompt_policy,
    )
    payload_without_hash = {
        "schema_version": HARNESS_AUTHORIZATION_SCHEMA,
        **expected,
        "replay_report_file_sha256": replay_file_sha256,
        "replay_report_payload_sha256": replay_payload["payload_sha256"],
        "live_report_file_sha256": live_file_sha256,
        "live_report_payload_sha256": live_payload["payload_sha256"],
        "live_plan_sha256": live_payload["plan_sha256"],
        "live_plan_ledger_tree_sha256": live_ledger_tree_sha256,
        "source_manifest": source_payload,
    }
    return _with_payload_sha(payload_without_hash)


def verify_harness_authorization(
    authorization_path: str | Path,
    *,
    replay_report: str | Path,
    live_report: str | Path,
    live_plan_dir: str | Path,
    harness_config_sha256: str,
    harness_config_file_sha256: str,
    teacher_config_sha256: str,
    teacher_config_file_sha256: str,
    prompt_policy: TeacherPromptPolicy,
    source_manifest: SourceTreeArtifactEvidence,
) -> str:
    """Recompute and require an exact immutable harness-authorization sidecar."""

    path = _regular_file(authorization_path, "harness authorization")
    payload = _load_json_payload(path, "harness authorization")
    expected_payload = _build_harness_authorization_payload(
        replay_report=replay_report,
        live_report=live_report,
        live_plan_dir=live_plan_dir,
        harness_config_sha256=harness_config_sha256,
        harness_config_file_sha256=harness_config_file_sha256,
        teacher_config_sha256=teacher_config_sha256,
        teacher_config_file_sha256=teacher_config_file_sha256,
        prompt_policy=prompt_policy,
        source_manifest=source_manifest,
    )
    if payload != expected_payload:
        raise TeacherHarnessValidationError(
            "harness authorization does not match verified evidence"
        )
    return str(expected_payload["payload_sha256"])


def _fault_matrix(
    policy: TeacherPromptPolicy,
) -> tuple[
    tuple[
        CodexCommandResult,
        tuple[str, ...],
        Mapping[str, int] | None,
        str | None,
        str,
        str,
    ],
    ...,
]:
    """Return fixed local faults and their immutable expected classifications."""

    rows = synthetic_fixture_rows()[:HARNESS_CHUNK_SIZE]
    expected_ids = tuple(row.problem_id for row in rows)
    answers = {row.problem_id: row.expected_answer for row in rows}
    good_items = _output_items(expected_ids, answers)
    reordered = list(reversed(good_items))
    duplicate = [*good_items[:-1], dict(good_items[0])]
    unknown = [*good_items[:-1], {**good_items[-1], "problem_id": "train-999999"}]
    malformed_stream = "{not-json"
    tool_stream = _event_stream_from_items(good_items, item_type="tool")
    tool_with_bad_usage = _event_stream_from_items(
        good_items,
        item_type="tool",
        usage={"input_tokens": -1},
    )
    error_stream = "\n".join(
        (
            json.dumps({"type": "thread.started"}, separators=(",", ":")),
            json.dumps(
                {"type": "turn.started", "error": {"code": "synthetic"}},
                separators=(",", ":"),
            ),
        )
    )
    invalid_usage = _event_stream_from_items(good_items, usage={"input_tokens": -1})
    bad_agent = _event_stream_from_message("not-json")
    missing_terminal_bad_agent = "\n".join(bad_agent.splitlines()[:-1])
    bad_schema = _event_stream_from_message(json.dumps({"result": []}, separators=(",", ":")))
    invalid_target = [*good_items]
    invalid_target[0] = {**invalid_target[0], "target_text": "too short\nFinal answer: 8"}
    oversize_target = [*good_items]
    oversize_target[0] = {
        **oversize_target[0],
        "target_text": "x" * 1_501 + "\nFinal answer: 8",
    }
    return (
        (
            _result(_event_stream_from_items(good_items)),
            expected_ids,
            answers,
            None,
            "success",
            "qualified",
        ),
        (
            _result(_event_stream_from_items(good_items[:-1])),
            expected_ids,
            answers,
            None,
            "output_structure",
            "cardinality_mismatch",
        ),
        (
            _result(_event_stream_from_items([*good_items, dict(good_items[0])])),
            expected_ids,
            answers,
            None,
            "output_structure",
            "cardinality_mismatch",
        ),
        (
            _result(_event_stream_from_items(duplicate)),
            expected_ids,
            answers,
            None,
            "output_structure",
            "id_set_mismatch",
        ),
        (
            _result(_event_stream_from_items(unknown)),
            expected_ids,
            answers,
            None,
            "output_structure",
            "id_set_mismatch",
        ),
        (
            _result(_event_stream_from_items(reordered)),
            expected_ids,
            answers,
            None,
            "output_structure",
            "order_mismatch",
        ),
        (
            _result(malformed_stream),
            expected_ids,
            answers,
            None,
            "event_json",
            "malformed_event_json",
        ),
        (_result(tool_stream), expected_ids, answers, None, "unsafe_error_event", "unsafe_item"),
        (
            _result(tool_with_bad_usage),
            expected_ids,
            answers,
            None,
            "unsafe_error_event",
            "unsafe_item",
        ),
        (_result(error_stream), expected_ids, answers, None, "unsafe_error_event", "error_event"),
        (_result(invalid_usage), expected_ids, answers, None, "terminal_usage", "invalid_usage"),
        (
            _result(missing_terminal_bad_agent),
            expected_ids,
            answers,
            None,
            "terminal_usage",
            "missing_terminal_event",
        ),
        (_result(bad_agent), expected_ids, answers, None, "agent_json", "malformed_agent_json"),
        (
            _result(bad_schema),
            expected_ids,
            answers,
            None,
            "output_schema",
            "invalid_output_schema",
        ),
        (
            _result(_event_stream_from_items(invalid_target)),
            expected_ids,
            answers,
            None,
            "target_policy",
            "target_policy_invalid",
        ),
        (
            _result(_event_stream_from_items(oversize_target)),
            expected_ids,
            answers,
            None,
            "target_policy",
            "target_policy_invalid",
        ),
        (_result("", returncode=124), expected_ids, answers, None, "timeout_spawn", "timeout"),
        (
            _result("", returncode=127),
            expected_ids,
            answers,
            None,
            "timeout_spawn",
            "spawn_failure",
        ),
        (
            _result(malformed_stream, returncode=9),
            expected_ids,
            answers,
            None,
            "nonzero",
            "command_nonzero",
        ),
        (
            _result(malformed_stream),
            expected_ids,
            answers,
            "runner_exception",
            "process",
            "runner_exception",
        ),
    )


def _replay_classifications_for_policy(
    policy: TeacherPromptPolicy,
) -> tuple[tuple[FailureClassification, ...], bool]:
    """Recompute every fixed replay result for report verification.

    A replay report's self-hash only detects accidental mutation.  This pure
    reconstruction also rejects a self-consistently rehashed report whose
    allowlisted stage/code/count values were altered after publication.
    """

    classifications: list[FailureClassification] = []
    expected_outcomes: list[tuple[str, str]] = []
    for (
        result,
        expected_ids,
        answers,
        failure_reason,
        expected_stage,
        expected_code,
    ) in _fault_matrix(policy):
        classifications.append(
            classify_codex_result(
                result,
                expected_ids,
                prompt_policy=policy,
                expected_answers=answers,
                failure_reason=failure_reason,
            )
        )
        expected_outcomes.append((expected_stage, expected_code))
    qualified = all(
        (actual.stage, actual.code) == expected
        for actual, expected in zip(classifications, expected_outcomes, strict=True)
    )
    return tuple(classifications), qualified


def _result(stdout: str, *, returncode: int = 0) -> CodexCommandResult:
    return CodexCommandResult(stdout=stdout, stderr="", returncode=returncode, latency_ms=1)


def _output_items(expected_ids: Sequence[str], answers: Mapping[str, int]) -> list[dict[str, str]]:
    return [
        {
            "problem_id": problem_id,
            "target_text": (
                "Compute the signed integer using the stated arithmetic relation.\n"
                f"Final answer: {answers[problem_id]}"
            ),
        }
        for problem_id in expected_ids
    ]


def _event_stream_from_items(
    items: Sequence[Mapping[str, object]],
    *,
    item_type: str = "agent_message",
    usage: Mapping[str, int] | None = None,
) -> str:
    message = json.dumps({"items": list(items)}, separators=(",", ":"))
    if item_type == "agent_message":
        item: dict[str, object] = {"type": item_type, "text": message}
    else:
        item = {"type": item_type}
    events: tuple[dict[str, object], ...] = (
        {"type": "thread.started"},
        {"type": "turn.started"},
        {"type": "item.completed", "item": item},
        {"type": "turn.completed", "usage": dict(usage or {"input_tokens": 8, "output_tokens": 4})},
    )
    return "\n".join(json.dumps(event, separators=(",", ":")) for event in events)


def _event_stream_from_message(message: str) -> str:
    events = (
        {"type": "thread.started"},
        {"type": "turn.started"},
        {"type": "item.completed", "item": {"type": "agent_message", "text": message}},
        {"type": "turn.completed", "usage": {"input_tokens": 8, "output_tokens": 4}},
    )
    return "\n".join(json.dumps(event, separators=(",", ":")) for event in events)


def _decode_event_stream(value: str) -> list[Mapping[str, object]] | None:
    if not isinstance(value, str) or not value:
        return None
    lines = value.splitlines()
    if not lines or any(not line.strip() for line in lines):
        return None
    output: list[Mapping[str, object]] = []
    for line in lines:
        event = _decode_json_object(line)
        if event is None:
            return None
        output.append(event)
    return output


def _decode_json_object(value: str) -> dict[str, object] | None:
    try:
        payload = json.loads(
            value,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError("duplicate JSON key")
        payload[key] = value
    return payload


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _valid_usage(value: object) -> bool:
    if not isinstance(value, Mapping) or not value:
        return False
    return all(
        isinstance(key, str)
        and bool(key)
        and not isinstance(item, bool)
        and isinstance(item, int)
        and item >= 0
        for key, item in value.items()
    )


def _empty_counts(requested_count: int) -> dict[str, int | bool]:
    return {
        "requested_count": requested_count,
        "returned_count": 0,
        "duplicate_count": 0,
        "missing_count": 0,
        "unexpected_count": 0,
        "order_mismatch": False,
    }


def _id_counts(expected: Sequence[str], observed: Sequence[str]) -> dict[str, int | bool]:
    expected_set = set(expected)
    observed_set = set(observed)
    return {
        "requested_count": len(expected),
        "returned_count": len(observed),
        "duplicate_count": len(observed) - len(observed_set),
        "missing_count": len(expected_set - observed_set),
        "unexpected_count": len(observed_set - expected_set),
        "order_mismatch": tuple(observed) != tuple(expected),
    }


def _classification(
    stage: str, code: str, counts: Mapping[str, int | bool]
) -> FailureClassification:
    return FailureClassification(
        stage=stage,
        code=code,
        requested_count=_required_count(counts, "requested_count"),
        returned_count=_required_count(counts, "returned_count"),
        duplicate_count=_required_count(counts, "duplicate_count"),
        missing_count=_required_count(counts, "missing_count"),
        unexpected_count=_required_count(counts, "unexpected_count"),
        order_mismatch=_required_bool(counts, "order_mismatch"),
    )


def _required_count(values: Mapping[str, int | bool], key: str) -> int:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TeacherHarnessValidationError("classifier count is invalid")
    return value


def _required_bool(values: Mapping[str, int | bool], key: str) -> bool:
    value = values.get(key)
    if not isinstance(value, bool):
        raise TeacherHarnessValidationError("classifier boolean is invalid")
    return value


def _validate_profile_and_hashes(
    profile: HarnessProfile,
    *,
    harness_config_sha256: str,
    harness_config_file_sha256: str,
    teacher_config_sha256: str,
    teacher_config_file_sha256: str,
    prompt_policy: TeacherPromptPolicy,
) -> None:
    if not isinstance(profile, HarnessProfile):
        raise TeacherHarnessValidationError("synthetic harness profile is invalid")
    for label, value in (
        ("harness config SHA", harness_config_sha256),
        ("harness config file SHA", harness_config_file_sha256),
        ("teacher config SHA", teacher_config_sha256),
        ("teacher config file SHA", teacher_config_file_sha256),
    ):
        _require_sha256(value, label)
    if not isinstance(prompt_policy, TeacherPromptPolicy):
        raise TeacherHarnessValidationError("synthetic harness prompt policy is invalid")
    _required_template_sha(prompt_policy)


def _required_template_sha(prompt_policy: TeacherPromptPolicy) -> str:
    value = prompt_policy.prompt_template_sha256
    if value is None:
        raise TeacherHarnessValidationError("synthetic harness requires a policy-bound prompt")
    _require_sha256(value, "prompt template SHA")
    return value


def _codex_binary_sha256(execution: TeacherExecutionConfig) -> str:
    binary = Path(execution.codex_binary)
    if binary.is_symlink() or not binary.is_file():
        raise TeacherHarnessValidationError("synthetic Codex binary is invalid")
    try:
        return sha256_file(binary)
    except OSError as exc:
        raise TeacherHarnessValidationError("synthetic Codex binary could not be hashed") from exc


def _text_sha256(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise TeacherHarnessValidationError("Codex version text is invalid")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _private_tree_snapshot(root: Path) -> tuple[tuple[str, str], ...]:
    root = _regular_directory(root, "teacher ledger directory")
    entries: list[tuple[str, str]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            raise TeacherHarnessValidationError("teacher ledger contains a symbolic link")
        if path.is_file():
            entries.append((path.relative_to(root).as_posix(), sha256_file(path)))
    return tuple(entries)


def _snapshot_sha256(snapshot: Sequence[tuple[str, str]]) -> str:
    return hashlib.sha256(canonical_json_bytes(list(snapshot))).hexdigest()


def _with_payload_sha(payload_without_hash: Mapping[str, object]) -> dict[str, object]:
    payload = dict(payload_without_hash)
    payload["payload_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return payload


def _write_json_noreplace(path: str | Path, payload: Mapping[str, object]) -> Path:
    target = _new_file_target(path, "harness report")
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_json_bytes(payload))
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            raise TeacherHarnessArtifactExistsError("refusing to overwrite harness report") from exc
        _fsync_directory(target.parent)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()
    return target


def _new_file_target(path: str | Path, label: str) -> Path:
    raw = Path(path)
    if raw.is_symlink() or raw.parent.is_symlink():
        raise TeacherHarnessValidationError(f"{label} refuses symbolic links")
    target = raw.resolve(strict=False)
    if not target.parent.is_dir():
        raise TeacherHarnessValidationError(f"{label} parent does not exist")
    if target.exists():
        raise TeacherHarnessArtifactExistsError(f"refusing to overwrite {label}")
    return target


def _new_harness_plan_directory_target(path: str | Path) -> Path:
    """Preflight a no-overwrite plan directory before any live invocation."""

    raw = Path(path)
    if raw.is_symlink() or raw.parent.is_symlink() or not raw.parent.is_dir():
        raise TeacherHarnessValidationError("synthetic harness plan target is invalid")
    if raw.exists():
        raise TeacherHarnessArtifactExistsError("refusing to overwrite synthetic harness plan")
    return raw.resolve(strict=False)


def _regular_directory(path: str | Path, label: str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_dir():
        raise TeacherHarnessValidationError(f"{label} is invalid")
    return candidate.resolve(strict=True)


def _regular_file(path: str | Path, label: str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise TeacherHarnessValidationError(f"{label} is invalid")
    return candidate.resolve(strict=True)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _load_harness_report(
    path: str | Path,
    expected_schema: str,
    *,
    prompt_policy: TeacherPromptPolicy | None = None,
) -> tuple[dict[str, object], str]:
    target = _regular_file(path, "harness report")
    payload = _load_json_payload(target, "harness report")
    _validate_harness_report(payload, expected_schema, prompt_policy=prompt_policy)
    return payload, sha256_file(target)


def _load_json_payload(path: Path, label: str) -> dict[str, object]:
    try:
        raw = path.read_text(encoding="utf-8", errors="strict")
        payload = json.loads(
            raw,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise TeacherHarnessValidationError(f"{label} cannot be parsed") from exc
    if not isinstance(payload, dict):
        raise TeacherHarnessValidationError(f"{label} must contain one JSON object")
    return payload


def _validate_harness_report(
    payload: Mapping[str, object],
    expected_schema: str,
    *,
    prompt_policy: TeacherPromptPolicy | None = None,
) -> None:
    if expected_schema not in {DIAGNOSTIC_SCHEMA, HARNESS_REPLAY_SCHEMA, HARNESS_LIVE_SCHEMA}:
        raise TeacherHarnessValidationError("harness report schema is unsupported")
    schema = payload.get("schema_version")
    if schema != expected_schema:
        raise TeacherHarnessValidationError("harness report schema is invalid")
    common = {
        "schema_version",
        "failure_classifier_schema_version",
        "harness_config_sha256",
        "harness_config_file_sha256",
        "teacher_config_sha256",
        "teacher_config_file_sha256",
        "teacher_prompt_policy_sha256",
        "prompt_template_sha256",
        "fixture_sha256",
        "classifications",
        "qualified",
        "payload_sha256",
    }
    if expected_schema == DIAGNOSTIC_SCHEMA:
        common = {
            "schema_version",
            "failure_classifier_schema_version",
            "teacher_config_sha256",
            "teacher_config_file_sha256",
            "teacher_prompt_policy_sha256",
            "prompt_template_sha256",
            "plan_sha256",
            "ledger_tree_sha256",
            "classifications",
            "payload_sha256",
        }
    elif expected_schema == HARNESS_LIVE_SCHEMA:
        common.update(
            {
                "source_manifest",
                "codex_binary_sha256",
                "codex_cli_version_sha256",
                "plan_sha256",
                "ledger_tree_sha256",
            }
        )
    if set(payload) != common:
        raise TeacherHarnessValidationError("harness report keys are invalid")
    if payload.get("failure_classifier_schema_version") != FAILURE_CLASSIFIER_SCHEMA:
        raise TeacherHarnessValidationError("failure classifier schema is invalid")
    for key in (
        "harness_config_sha256",
        "harness_config_file_sha256",
        "teacher_config_sha256",
        "teacher_config_file_sha256",
        "teacher_prompt_policy_sha256",
        "prompt_template_sha256",
        "fixture_sha256",
    ):
        if key in payload:
            _require_sha256(payload[key], key)
    for key in (
        "plan_sha256",
        "ledger_tree_sha256",
        "codex_binary_sha256",
        "codex_cli_version_sha256",
    ):
        if key in payload:
            _require_sha256(payload[key], key)
    classifications = payload.get("classifications")
    if not isinstance(classifications, list):
        raise TeacherHarnessValidationError("harness report classifications are invalid")
    parsed_classifications: list[FailureClassification] = []
    for item in classifications:
        if not isinstance(item, Mapping) or set(item) != _CLASSIFICATION_KEYS:
            raise TeacherHarnessValidationError("harness report classification keys are invalid")
        parsed_classifications.append(
            FailureClassification(
                stage=_required_text(item.get("stage"), "classification stage"),
                code=_required_text(item.get("code"), "classification code"),
                requested_count=_required_nonnegative_int(item.get("requested_count")),
                returned_count=_required_nonnegative_int(item.get("returned_count")),
                duplicate_count=_required_nonnegative_int(item.get("duplicate_count")),
                missing_count=_required_nonnegative_int(item.get("missing_count")),
                unexpected_count=_required_nonnegative_int(item.get("unexpected_count")),
                order_mismatch=_required_boolean(item.get("order_mismatch")),
            )
        )
    if expected_schema != DIAGNOSTIC_SCHEMA and not isinstance(payload.get("qualified"), bool):
        raise TeacherHarnessValidationError("harness report qualification is invalid")
    if expected_schema == HARNESS_LIVE_SCHEMA:
        qualified = payload["qualified"]
        assert isinstance(qualified, bool)
        if qualified != all(
            item.stage == "success" and item.code == "qualified" for item in parsed_classifications
        ):
            raise TeacherHarnessValidationError(
                "harness report qualification does not match its classifications"
            )
    if (
        expected_schema == HARNESS_REPLAY_SCHEMA
        and len(parsed_classifications) != HARNESS_REPLAY_FAULT_COUNT
    ):
        raise TeacherHarnessValidationError("harness replay fault-matrix count is invalid")
    if expected_schema == HARNESS_REPLAY_SCHEMA:
        if not isinstance(prompt_policy, TeacherPromptPolicy):
            raise TeacherHarnessValidationError("harness replay prompt policy is required")
        expected_classifications, expected_qualified = _replay_classifications_for_policy(
            prompt_policy
        )
        if (
            tuple(parsed_classifications) != expected_classifications
            or payload["qualified"] is not expected_qualified
        ):
            raise TeacherHarnessValidationError(
                "harness replay report does not match the fixed fault matrix"
            )
    if expected_schema == HARNESS_LIVE_SCHEMA:
        if len(parsed_classifications) != HARNESS_CHUNK_COUNT:
            raise TeacherHarnessValidationError("harness live invocation count is invalid")
        if payload["qualified"] is True and any(
            item.requested_count != HARNESS_CHUNK_SIZE
            or item.returned_count != HARNESS_CHUNK_SIZE
            or item.duplicate_count != 0
            or item.missing_count != 0
            or item.unexpected_count != 0
            or item.order_mismatch
            for item in parsed_classifications
        ):
            raise TeacherHarnessValidationError("qualified live harness counts are invalid")
        _validate_source_manifest_payload(payload["source_manifest"])
    payload_sha256 = payload.get("payload_sha256")
    _require_sha256(payload_sha256, "harness report payload SHA")
    without_hash = dict(payload)
    without_hash.pop("payload_sha256")
    if hashlib.sha256(canonical_json_bytes(without_hash)).hexdigest() != payload_sha256:
        raise TeacherHarnessValidationError("harness report payload SHA is invalid")


def _validate_source_manifest_payload(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "file",
        "sha256",
        "tree_sha256",
        "file_count",
    }:
        raise TeacherHarnessValidationError("source manifest summary is invalid")
    file_name = value.get("file")
    if (
        not isinstance(file_name, str)
        or not file_name
        or file_name in {".", ".."}
        or "/" in file_name
        or "\\" in file_name
        or Path(file_name).name != file_name
    ):
        raise TeacherHarnessValidationError("source manifest file summary is invalid")
    _require_sha256(value.get("sha256"), "source manifest SHA")
    _require_sha256(value.get("tree_sha256"), "source tree SHA")
    _required_nonnegative_int(value.get("file_count"))


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TeacherHarnessValidationError(f"{label} is invalid")
    return value


def _required_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TeacherHarnessValidationError(f"{label} is invalid")
    return value


def _required_nonnegative_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TeacherHarnessValidationError("count is invalid")
    return value


def _required_boolean(value: object) -> bool:
    if not isinstance(value, bool):
        raise TeacherHarnessValidationError("boolean is invalid")
    return value

"""Fail-closed authorization for expanding a Codex teacher pilot to v1.

The teacher ledger intentionally keeps raw prompts, responses, questions, and
reference answers private.  This module reads only the already hash-validated
private ledger artifacts needed to establish an *aggregate* authorization
receipt.  The receipt itself is raw-free and immutable: it contains hashes,
counts, and policy flags, never an ID, question, answer, or rationale.

It is deliberately separate from :mod:`teacher_rationale`.  The latter owns
the evidence formats; this module composes their verified readers into the
strict pilot gate used before the complete fold-0 bank can even be planned.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from .provenance import canonical_json_bytes, sha256_file
from .teacher_rationale import (
    TeacherAttempt,
    TeacherLogicalAuditPlan,
    TeacherPlan,
    TeacherRationaleValidationError,
    _accepted_items,
    _expected_teacher_logical_audit_items,
    _load_assessments,
    _load_attempts,
    _verify_teacher_bank_for_logical_audit,
    finalize_teacher_logical_audit,
    load_teacher_logical_audit_plan,
    load_teacher_plan,
    teacher_logical_audit_status,
    teacher_status,
)

PILOT_AUTHORIZATION_SCHEMA = "gate-b-codex-teacher-pilot-authorization-v1"
PILOT_AUTHORIZATION_V2_SCHEMA = "gate-b-codex-teacher-pilot-authorization-v2"
PILOT_AUTHORIZATION_PILOT_SIZE = 128
PILOT_AUTHORIZATION_MAX_ATTEMPTS = 3
PILOT_AUTHORIZATION_INITIAL_EXACT_MATCH_PERCENT = 80
PILOT_AUTHORIZATION_LOGICAL_AUDIT_SAMPLE_SIZE = 64
PILOT_AUTHORIZATION_LOGICAL_AUDIT_MIN_CONSISTENT = 60

# A complete v1 plan is created only after a passed 128-row pilot receipt.
# Keep that promotion evidence beside the otherwise question-only full plan so
# later private materialization cannot accept a hand-supplied plan/source tuple
# that bypassed the pilot gate.
FULL_V1_BANK_AUTHORIZATION_SCHEMA = "gate-b-codex-teacher-v1-bank-authorization-v1"
FULL_V1_BANK_AUTHORIZATION_V2_SCHEMA = "gate-b-codex-teacher-v1-bank-authorization-v2"
FULL_V1_BANK_AUTHORIZATION_FILENAME = "v1-pilot-authorization.json"

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_TRAIN_ID_RE = re.compile(r"train-\d{6}\Z")
_TEACHER_PROMPT_V1 = "gate-b-codex-teacher-prompt-v1"
_TEACHER_PROMPT_V2 = "gate-b-codex-teacher-prompt-v2"
_TEACHER_PROMPT_V3 = "gate-b-codex-teacher-prompt-v3"
_TEACHER_PROMPT_V4 = "gate-b-codex-teacher-prompt-v4"
_POLICY_BOUND_TEACHER_PROMPTS = frozenset(
    {_TEACHER_PROMPT_V2, _TEACHER_PROMPT_V3, _TEACHER_PROMPT_V4}
)


class TeacherPilotAuthorizationError(ValueError):
    """Raised when a pilot receipt or its private evidence is not promotable."""


@dataclass(frozen=True, slots=True)
class TeacherPilotAuthorizationContract:
    """Current non-secret data/config contract used to re-verify a receipt.

    The ID tuples are held only while validation runs.  They are condensed to
    SHA-256 values and counts before any receipt is serialized.
    """

    teacher_config_sha256: str
    teacher_config_file_sha256: str
    train_sha256: str
    exclusions_sha256: str
    exclusion_count: int
    split_artifact_sha256: str
    development_shard_sha256: str
    split_version: str
    split_sha256: str
    source_groups_sha256: str
    fold: int
    fold0_training_ids: tuple[str, ...]
    pilot_ids: tuple[str, ...]
    teacher_plan_label: str
    teacher_plan_version: str
    logical_audit_label: str
    logical_audit_version: str
    # The historic v1 receipt schema did not serialize this binding.  Keep it
    # optional for exact v1 replay, while every v2 prompt plan requires it.
    teacher_prompt_policy_sha256: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "teacher_config_sha256",
            "teacher_config_file_sha256",
            "train_sha256",
            "exclusions_sha256",
            "split_artifact_sha256",
            "development_shard_sha256",
            "split_sha256",
            "source_groups_sha256",
        ):
            _required_sha256(getattr(self, field_name), field_name)
        if self.teacher_prompt_policy_sha256 is not None:
            _required_sha256(
                self.teacher_prompt_policy_sha256,
                "teacher_prompt_policy_sha256",
            )
        _required_nonempty_text(self.split_version, "split_version")
        _required_nonempty_text(self.teacher_plan_label, "teacher_plan_label")
        _required_nonempty_text(self.teacher_plan_version, "teacher_plan_version")
        _required_nonempty_text(self.logical_audit_label, "logical_audit_label")
        _required_nonempty_text(self.logical_audit_version, "logical_audit_version")
        if self.fold != 0:
            raise TeacherPilotAuthorizationError("pilot authorization is locked to fold 0")
        if isinstance(self.exclusion_count, bool) or not isinstance(self.exclusion_count, int):
            raise TeacherPilotAuthorizationError("exclusion_count must be an integer")
        if self.exclusion_count < 0:
            raise TeacherPilotAuthorizationError("exclusion_count must be non-negative")
        _validate_canonical_train_ids(
            self.fold0_training_ids, "fold0_training_ids", allow_empty=False
        )
        _validate_canonical_train_ids(self.pilot_ids, "pilot_ids", allow_empty=False)
        if len(self.pilot_ids) != PILOT_AUTHORIZATION_PILOT_SIZE:
            raise TeacherPilotAuthorizationError(
                "pilot_ids must contain the locked 128-problem pilot exactly"
            )
        if not set(self.pilot_ids).issubset(self.fold0_training_ids):
            raise TeacherPilotAuthorizationError(
                "pilot_ids must be a subset of fold0_training_ids"
            )


@dataclass(frozen=True, slots=True)
class TeacherPilotAuthorizationReceipt:
    """One verified, immutable, raw-free pilot promotion receipt."""

    path: Path
    file_sha256: str
    payload_sha256: str
    pilot_plan_sha256: str
    audit_plan_sha256: str
    audit_manifest_sha256: str
    initial_exact_match_accepted_count: int
    initial_exact_match_total_count: int
    logical_audit_consistent_problem_count: int
    teacher_prompt_policy_sha256: str | None = None

    def as_dict(self) -> dict[str, object]:
        """Return a monitor-safe receipt summary with no private text."""

        payload: dict[str, object] = {
            "payload_sha256": self.payload_sha256,
            "file_sha256": self.file_sha256,
            "pilot_plan_sha256": self.pilot_plan_sha256,
            "audit_plan_sha256": self.audit_plan_sha256,
            "audit_manifest_sha256": self.audit_manifest_sha256,
            "initial_exact_match_accepted_count": self.initial_exact_match_accepted_count,
            "initial_exact_match_total_count": self.initial_exact_match_total_count,
            "logical_audit_consistent_problem_count": self.logical_audit_consistent_problem_count,
        }
        if self.teacher_prompt_policy_sha256 is not None:
            payload["teacher_prompt_policy_sha256"] = self.teacher_prompt_policy_sha256
        return payload


def _uses_policy_bound_prompt(plan: TeacherPlan) -> bool:
    """Classify a plan without silently downgrading versioned prompt evidence."""

    prompt_version = plan.prompt_policy.prompt_version
    if prompt_version == _TEACHER_PROMPT_V1:
        return False
    if prompt_version in _POLICY_BOUND_TEACHER_PROMPTS:
        return True
    raise TeacherPilotAuthorizationError(
        "pilot authorization does not recognize the teacher prompt policy"
    )


def _receipt_uses_v2_schema(payload: Mapping[str, object]) -> bool:
    """Return the receipt schema generation, rejecting unknown variants."""

    schema_version = payload.get("schema_version")
    if schema_version == PILOT_AUTHORIZATION_SCHEMA:
        return False
    if schema_version == PILOT_AUTHORIZATION_V2_SCHEMA:
        return True
    raise TeacherPilotAuthorizationError("pilot authorization receipt schema is invalid")


def create_teacher_pilot_authorization(
    output_path: str | Path,
    *,
    contract: TeacherPilotAuthorizationContract,
    pilot_plan_dir: str | Path,
    source_jsonl: str | Path,
    source_manifest: str | Path,
    logical_audit_dir: str | Path,
) -> TeacherPilotAuthorizationReceipt:
    """Verify all pilot gates and publish one new receipt without overwrite."""

    payload_without_hash = _verified_payload(
        contract=contract,
        pilot_plan_dir=pilot_plan_dir,
        source_jsonl=source_jsonl,
        source_manifest=source_manifest,
        logical_audit_dir=logical_audit_dir,
    )
    payload = {
        **payload_without_hash,
        "payload_sha256": _payload_sha256(payload_without_hash),
    }
    target = _write_new_receipt(output_path, canonical_json_bytes(payload))
    return _receipt_from_payload(
        target,
        payload,
        file_sha256=sha256_file(target),
    )


def verify_teacher_pilot_authorization(
    authorization_path: str | Path,
    *,
    contract: TeacherPilotAuthorizationContract,
    pilot_plan_dir: str | Path,
    source_jsonl: str | Path,
    source_manifest: str | Path,
    logical_audit_dir: str | Path,
) -> TeacherPilotAuthorizationReceipt:
    """Recompute every gate and require an exact immutable receipt match.

    Recomputing from private artifacts prevents a receipt from authorizing a
    later-tampered source bank, audit, split, or config binding.
    """

    receipt_path = _regular_file(authorization_path, "pilot authorization receipt")
    stored = _load_receipt_payload(receipt_path)
    expected_without_hash = _verified_payload(
        contract=contract,
        pilot_plan_dir=pilot_plan_dir,
        source_jsonl=source_jsonl,
        source_manifest=source_manifest,
        logical_audit_dir=logical_audit_dir,
    )
    expected = {
        **expected_without_hash,
        "payload_sha256": _payload_sha256(expected_without_hash),
    }
    if stored != expected:
        raise TeacherPilotAuthorizationError(
            "pilot authorization receipt does not match the current verified evidence"
        )
    return _receipt_from_payload(
        receipt_path,
        stored,
        file_sha256=sha256_file(receipt_path),
    )


def write_teacher_full_v1_bank_authorization(
    plan_dir: str | Path,
    receipt: TeacherPilotAuthorizationReceipt,
) -> str:
    """Bind a complete v1 plan to an already re-verified pilot receipt.

    The caller must have used :func:`verify_teacher_pilot_authorization` at
    the CLI promotion boundary.  This function reopens and hashes the receipt
    nevertheless, then atomically writes a raw-free, no-overwrite sidecar in
    the new full-plan directory.  The sidecar has no local path, ID, question,
    answer, or rationale text.
    """

    if not isinstance(receipt, TeacherPilotAuthorizationReceipt):
        raise TeacherPilotAuthorizationError("v1 bank authorization receipt is invalid")
    plan = load_teacher_plan(plan_dir)
    receipt_path = _regular_file(receipt.path, "v1 bank authorization receipt")
    if sha256_file(receipt_path) != receipt.file_sha256:
        raise TeacherPilotAuthorizationError(
            "v1 bank authorization receipt file SHA does not match the verified receipt"
        )
    receipt_payload = _load_receipt_payload(receipt_path)
    if (
        receipt_payload["payload_sha256"] != receipt.payload_sha256
        or receipt_payload["pilot_plan_sha256"] != receipt.pilot_plan_sha256
        or receipt_payload["audit_plan_sha256"] != receipt.audit_plan_sha256
        or receipt_payload["audit_manifest_sha256"] != receipt.audit_manifest_sha256
        or receipt_payload.get("teacher_prompt_policy_sha256")
        != receipt.teacher_prompt_policy_sha256
    ):
        raise TeacherPilotAuthorizationError(
            "v1 bank authorization receipt object does not match its immutable file"
        )
    if (
        receipt_payload["all_pilot_rows_accepted"] is not True
        or receipt_payload["reference_answer_in_prompt"] is not False
        or receipt_payload["locked_holdout_accessed"] is not False
        or receipt_payload["leaderboard_or_test_used"] is not False
        or receipt_payload["raw_generation_serialized"] is not False
        or receipt_payload["initial_exact_match_accepted_count"] * 100
        < PILOT_AUTHORIZATION_INITIAL_EXACT_MATCH_PERCENT
        * PILOT_AUTHORIZATION_PILOT_SIZE
        or receipt_payload["initial_exact_match_total_count"]
        != PILOT_AUTHORIZATION_PILOT_SIZE
        or receipt_payload["audit_sample_size"]
        != PILOT_AUTHORIZATION_LOGICAL_AUDIT_SAMPLE_SIZE
        or receipt_payload["audit_min_consistent"]
        != PILOT_AUTHORIZATION_LOGICAL_AUDIT_MIN_CONSISTENT
        or receipt_payload["audit_consistent_problem_count"]
        < PILOT_AUTHORIZATION_LOGICAL_AUDIT_MIN_CONSISTENT
    ):
        raise TeacherPilotAuthorizationError(
            "v1 bank authorization receipt does not pass the locked pilot gates"
        )
    receipt_uses_policy_bound_schema = _receipt_uses_v2_schema(receipt_payload)
    if (
        plan.allowed_ids_sha256 != receipt_payload["fold0_training_ids_sha256"]
        or len(plan.problem_ids) != receipt_payload["fold0_training_problem_count"]
        or plan.label != receipt_payload["pilot_plan_label"]
        or plan.version != receipt_payload["pilot_plan_version"]
        or (
            receipt_uses_policy_bound_schema
            and plan.prompt_policy.sha256
            != receipt_payload["teacher_prompt_policy_sha256"]
        )
        or (
            not receipt_uses_policy_bound_schema
            and plan.prompt_policy.prompt_version != _TEACHER_PROMPT_V1
        )
    ):
        raise TeacherPilotAuthorizationError(
            "complete v1 plan does not exactly match the receipt's fold-0 training scope"
        )
    payload_without_hash: dict[str, object] = {
        "schema_version": (
            FULL_V1_BANK_AUTHORIZATION_V2_SCHEMA
            if receipt_uses_policy_bound_schema
            else FULL_V1_BANK_AUTHORIZATION_SCHEMA
        ),
        "full_plan_sha256": plan.plan_sha256,
        "full_plan_label": plan.label,
        "full_plan_version": plan.version,
        "full_plan_allowed_ids_sha256": plan.allowed_ids_sha256,
        "full_plan_problem_count": len(plan.problem_ids),
        "pilot_authorization_file_sha256": receipt.file_sha256,
        "pilot_authorization_payload_sha256": receipt.payload_sha256,
        "pilot_plan_sha256": receipt.pilot_plan_sha256,
        "pilot_plan_label": receipt_payload["pilot_plan_label"],
        "pilot_plan_version": receipt_payload["pilot_plan_version"],
        "pilot_plan_allowed_ids_sha256": receipt_payload["pilot_plan_allowed_ids_sha256"],
        "pilot_problem_count": receipt_payload["pilot_problem_count"],
        "fold0_training_ids_sha256": receipt_payload["fold0_training_ids_sha256"],
        "fold0_training_problem_count": receipt_payload["fold0_training_problem_count"],
        "all_pilot_rows_accepted": True,
        "initial_exact_match_accepted_count": receipt_payload[
            "initial_exact_match_accepted_count"
        ],
        "initial_exact_match_total_count": receipt_payload[
            "initial_exact_match_total_count"
        ],
        "audit_plan_sha256": receipt.audit_plan_sha256,
        "audit_manifest_sha256": receipt.audit_manifest_sha256,
        "audit_sample_size": receipt_payload["audit_sample_size"],
        "audit_min_consistent": receipt_payload["audit_min_consistent"],
        "audit_consistent_problem_count": receipt_payload[
            "audit_consistent_problem_count"
        ],
        "reference_answer_in_prompt": False,
        "locked_holdout_accessed": False,
        "leaderboard_or_test_used": False,
        "raw_generation_serialized": False,
    }
    if receipt_uses_policy_bound_schema:
        payload_without_hash["full_plan_prompt_policy_sha256"] = plan.prompt_policy.sha256
        payload_without_hash["pilot_plan_prompt_policy_sha256"] = receipt_payload[
            "teacher_prompt_policy_sha256"
        ]
    payload = {
        **payload_without_hash,
        "payload_sha256": _payload_sha256(payload_without_hash),
    }
    _write_new_sidecar(
        plan.plan_dir / FULL_V1_BANK_AUTHORIZATION_FILENAME,
        canonical_json_bytes(payload),
        label="v1 bank authorization",
    )
    return _required_sha256(payload["payload_sha256"], "v1 bank authorization payload_sha256")


def load_teacher_full_v1_bank_authorization(
    plan_dir: str | Path,
) -> dict[str, object]:
    """Load one raw-free v1 promotion sidecar and bind it to its plan.

    This intentionally verifies only artifact-local evidence.  The live pilot
    receipt was fully re-verified before this immutable sidecar was published;
    materializers additionally bind the sidecar to the current split scope.
    """

    plan = load_teacher_plan(plan_dir)
    path = _regular_file(
        plan.plan_dir / FULL_V1_BANK_AUTHORIZATION_FILENAME,
        "v1 bank authorization",
    )
    payload = _load_json_object(path, "v1 bank authorization")
    schema_version = payload.get("schema_version")
    if schema_version == FULL_V1_BANK_AUTHORIZATION_SCHEMA:
        sidecar_uses_v2 = False
    elif schema_version == FULL_V1_BANK_AUTHORIZATION_V2_SCHEMA:
        sidecar_uses_v2 = True
    else:
        raise TeacherPilotAuthorizationError("v1 bank authorization schema is invalid")
    expected_keys = {
        "schema_version",
        "full_plan_sha256",
        "full_plan_label",
        "full_plan_version",
        "full_plan_allowed_ids_sha256",
        "full_plan_problem_count",
        "pilot_authorization_file_sha256",
        "pilot_authorization_payload_sha256",
        "pilot_plan_sha256",
        "pilot_plan_label",
        "pilot_plan_version",
        "pilot_plan_allowed_ids_sha256",
        "pilot_problem_count",
        "fold0_training_ids_sha256",
        "fold0_training_problem_count",
        "all_pilot_rows_accepted",
        "initial_exact_match_accepted_count",
        "initial_exact_match_total_count",
        "audit_plan_sha256",
        "audit_manifest_sha256",
        "audit_sample_size",
        "audit_min_consistent",
        "audit_consistent_problem_count",
        "reference_answer_in_prompt",
        "locked_holdout_accessed",
        "leaderboard_or_test_used",
        "raw_generation_serialized",
        "payload_sha256",
    }
    if sidecar_uses_v2:
        expected_keys.update(
            {
                "full_plan_prompt_policy_sha256",
                "pilot_plan_prompt_policy_sha256",
            }
        )
    if set(payload) != expected_keys:
        raise TeacherPilotAuthorizationError(
            "v1 bank authorization keys differ from the locked schema"
        )
    stored_sha = _required_sha256(payload["payload_sha256"], "v1 bank authorization payload_sha256")
    without_hash = dict(payload)
    without_hash.pop("payload_sha256")
    if _payload_sha256(without_hash) != stored_sha:
        raise TeacherPilotAuthorizationError(
            "v1 bank authorization payload SHA does not match content"
        )
    for field_name in (
        "full_plan_sha256",
        "full_plan_allowed_ids_sha256",
        "pilot_authorization_file_sha256",
        "pilot_authorization_payload_sha256",
        "pilot_plan_sha256",
        "pilot_plan_allowed_ids_sha256",
        "fold0_training_ids_sha256",
        "audit_plan_sha256",
        "audit_manifest_sha256",
    ):
        _required_sha256(payload[field_name], f"v1 bank authorization {field_name}")
    if sidecar_uses_v2:
        for field_name in (
            "full_plan_prompt_policy_sha256",
            "pilot_plan_prompt_policy_sha256",
        ):
            _required_sha256(payload[field_name], f"v1 bank authorization {field_name}")
    for field_name in (
        "full_plan_label",
        "full_plan_version",
        "pilot_plan_label",
        "pilot_plan_version",
    ):
        _required_nonempty_text(payload[field_name], f"v1 bank authorization {field_name}")
    for field_name in (
        "full_plan_problem_count",
        "pilot_problem_count",
        "fold0_training_problem_count",
        "initial_exact_match_accepted_count",
        "initial_exact_match_total_count",
        "audit_sample_size",
        "audit_min_consistent",
        "audit_consistent_problem_count",
    ):
        value = payload[field_name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise TeacherPilotAuthorizationError(
                f"v1 bank authorization {field_name} is invalid"
            )
    if (
        payload["full_plan_sha256"] != plan.plan_sha256
        or payload["full_plan_label"] != plan.label
        or payload["full_plan_version"] != plan.version
        or payload["full_plan_allowed_ids_sha256"] != plan.allowed_ids_sha256
        or payload["full_plan_problem_count"] != len(plan.problem_ids)
        or payload["pilot_plan_label"] != plan.label
        or payload["pilot_plan_version"] != plan.version
        or payload["fold0_training_ids_sha256"] != plan.allowed_ids_sha256
        or payload["fold0_training_problem_count"] != len(plan.problem_ids)
        or payload["all_pilot_rows_accepted"] is not True
        or payload["pilot_problem_count"] != PILOT_AUTHORIZATION_PILOT_SIZE
        or payload["initial_exact_match_total_count"] != PILOT_AUTHORIZATION_PILOT_SIZE
        or payload["initial_exact_match_accepted_count"] * 100
        < PILOT_AUTHORIZATION_INITIAL_EXACT_MATCH_PERCENT
        * PILOT_AUTHORIZATION_PILOT_SIZE
        or payload["audit_sample_size"]
        != PILOT_AUTHORIZATION_LOGICAL_AUDIT_SAMPLE_SIZE
        or payload["audit_min_consistent"]
        != PILOT_AUTHORIZATION_LOGICAL_AUDIT_MIN_CONSISTENT
        or payload["audit_consistent_problem_count"]
        < PILOT_AUTHORIZATION_LOGICAL_AUDIT_MIN_CONSISTENT
        or payload["reference_answer_in_prompt"] is not False
        or payload["locked_holdout_accessed"] is not False
        or payload["leaderboard_or_test_used"] is not False
        or payload["raw_generation_serialized"] is not False
        or (
            sidecar_uses_v2
            and payload["full_plan_prompt_policy_sha256"] != plan.prompt_policy.sha256
        )
        or (
            sidecar_uses_v2
            and payload["pilot_plan_prompt_policy_sha256"] != plan.prompt_policy.sha256
        )
        or (not sidecar_uses_v2 and plan.prompt_policy.prompt_version != _TEACHER_PROMPT_V1)
    ):
        raise TeacherPilotAuthorizationError(
            "v1 bank authorization does not match the locked full-plan promotion contract"
        )
    return payload


def _verified_payload(
    *,
    contract: TeacherPilotAuthorizationContract,
    pilot_plan_dir: str | Path,
    source_jsonl: str | Path,
    source_manifest: str | Path,
    logical_audit_dir: str | Path,
) -> dict[str, object]:
    """Return a receipt payload only when every locked promotion rule passes."""

    plan = load_teacher_plan(pilot_plan_dir)
    _require_exact_pilot_plan(plan, contract)
    plan_uses_policy_bound_prompt = _uses_policy_bound_prompt(plan)

    # These readers rehash all linked attempt, parsed-output, and assessment
    # files.  The private helper also re-derives the source JSONL/manifest from
    # exactly those accepted rows, without reading organizer reference answers.
    plan_status = teacher_status(
        plan.plan_dir,
        max_attempts=PILOT_AUTHORIZATION_MAX_ATTEMPTS,
    )
    if (
        plan_status.total_problem_count != PILOT_AUTHORIZATION_PILOT_SIZE
        or plan_status.accepted_problem_count != PILOT_AUTHORIZATION_PILOT_SIZE
        or plan_status.retryable_problem_count != 0
        or plan_status.exhausted_problem_count != 0
        or plan_status.unassessed_problem_count != 0
        or plan_status.lock_state != "unlocked"
    ):
        raise TeacherPilotAuthorizationError(
            "pilot ledger is not completely accepted and unlocked"
        )
    verified_bank = _verify_teacher_bank_for_logical_audit(
        plan.plan_dir,
        source_jsonl,
        source_manifest,
    )
    if verified_bank.teacher_plan.plan_sha256 != plan.plan_sha256:
        raise TeacherPilotAuthorizationError("verified source bank is bound to another pilot")

    attempts = _load_attempts(plan)
    assessments = _load_assessments(plan, attempts)
    accepted = _accepted_items(plan, attempts, assessments)
    if set(accepted) != set(plan.problem_ids):
        raise TeacherPilotAuthorizationError("pilot accepted IDs are incomplete")
    _require_attempt_budget(plan, attempts)
    initial_accepted = _initial_exact_match_accepted_count(plan, attempts, assessments)
    if (
        initial_accepted * 100
        < PILOT_AUTHORIZATION_INITIAL_EXACT_MATCH_PERCENT
        * PILOT_AUTHORIZATION_PILOT_SIZE
    ):
        raise TeacherPilotAuthorizationError(
            "pilot first-pass local exact-match rate is below the locked 80 percent gate"
        )

    audit_plan = load_teacher_logical_audit_plan(logical_audit_dir)
    _require_exact_logical_audit_plan(audit_plan, plan, verified_bank, contract)
    audit_status = teacher_logical_audit_status(
        audit_plan.audit_dir,
        max_attempts=PILOT_AUTHORIZATION_MAX_ATTEMPTS,
    )
    if not audit_status.manifest_published or audit_status.lock_state != "unlocked":
        raise TeacherPilotAuthorizationError(
            "pilot logical audit must be finalized and unlocked before authorization"
        )
    audit_result = finalize_teacher_logical_audit(
        audit_plan.audit_dir,
        max_attempts=PILOT_AUTHORIZATION_MAX_ATTEMPTS,
    )
    if (
        audit_result.complete is not True
        or audit_result.passed is not True
        or audit_result.sample_size != PILOT_AUTHORIZATION_LOGICAL_AUDIT_SAMPLE_SIZE
        or audit_result.min_consistent
        != PILOT_AUTHORIZATION_LOGICAL_AUDIT_MIN_CONSISTENT
        or audit_result.completed_problem_count
        != PILOT_AUTHORIZATION_LOGICAL_AUDIT_SAMPLE_SIZE
        or audit_result.consistent_problem_count
        < PILOT_AUTHORIZATION_LOGICAL_AUDIT_MIN_CONSISTENT
        or audit_result.manifest_sha256 is None
    ):
        raise TeacherPilotAuthorizationError(
            "pilot logical audit did not pass the locked 64/60 gate"
        )

    payload: dict[str, object] = {
        "schema_version": (
            PILOT_AUTHORIZATION_V2_SCHEMA
            if plan_uses_policy_bound_prompt
            else PILOT_AUTHORIZATION_SCHEMA
        ),
        "teacher_config_sha256": contract.teacher_config_sha256,
        "teacher_config_file_sha256": contract.teacher_config_file_sha256,
        "train_sha256": contract.train_sha256,
        "exclusions_sha256": contract.exclusions_sha256,
        "exclusion_count": contract.exclusion_count,
        "split_artifact_sha256": contract.split_artifact_sha256,
        "development_shard_sha256": contract.development_shard_sha256,
        "split_version": contract.split_version,
        "split_sha256": contract.split_sha256,
        "source_groups_sha256": contract.source_groups_sha256,
        "fold": contract.fold,
        "fold0_training_ids_sha256": _ids_sha256(contract.fold0_training_ids),
        "fold0_training_problem_count": len(contract.fold0_training_ids),
        "pilot_ids_sha256": _ids_sha256(contract.pilot_ids),
        "pilot_problem_count": PILOT_AUTHORIZATION_PILOT_SIZE,
        "pilot_plan_sha256": plan.plan_sha256,
        "pilot_plan_label": plan.label,
        "pilot_plan_version": plan.version,
        "pilot_plan_allowed_ids_sha256": plan.allowed_ids_sha256,
        "source_jsonl_sha256": verified_bank.source_jsonl_sha256,
        "source_manifest_sha256": verified_bank.source_manifest_sha256,
        "audit_plan_sha256": audit_plan.plan_sha256,
        "audit_manifest_sha256": audit_result.manifest_sha256,
        "audit_sample_size": PILOT_AUTHORIZATION_LOGICAL_AUDIT_SAMPLE_SIZE,
        "audit_min_consistent": PILOT_AUTHORIZATION_LOGICAL_AUDIT_MIN_CONSISTENT,
        "audit_consistent_problem_count": audit_result.consistent_problem_count,
        "audit_inconsistent_problem_count": audit_result.inconsistent_problem_count,
        "initial_exact_match_accepted_count": initial_accepted,
        "initial_exact_match_total_count": PILOT_AUTHORIZATION_PILOT_SIZE,
        "initial_exact_match_threshold_percent": (
            PILOT_AUTHORIZATION_INITIAL_EXACT_MATCH_PERCENT
        ),
        "max_attempts": PILOT_AUTHORIZATION_MAX_ATTEMPTS,
        "all_pilot_rows_accepted": True,
        "reference_answer_used_locally": True,
        "reference_answer_in_prompt": False,
        "locked_holdout_accessed": False,
        "leaderboard_or_test_used": False,
        "raw_generation_serialized": False,
    }
    if plan_uses_policy_bound_prompt:
        # _require_exact_pilot_plan already rejected a missing or mismatched
        # binding, so this cast remains a fail-closed invariant.
        assert contract.teacher_prompt_policy_sha256 is not None
        payload["teacher_prompt_policy_sha256"] = contract.teacher_prompt_policy_sha256
    return payload


def _require_exact_pilot_plan(
    plan: TeacherPlan, contract: TeacherPilotAuthorizationContract
) -> None:
    plan_uses_policy_bound_prompt = _uses_policy_bound_prompt(plan)
    if (
        plan.label != contract.teacher_plan_label
        or plan.version != contract.teacher_plan_version
        or (
            plan_uses_policy_bound_prompt
            and (
                contract.teacher_prompt_policy_sha256 is None
                or plan.prompt_policy.sha256 != contract.teacher_prompt_policy_sha256
            )
        )
        or (
            not plan_uses_policy_bound_prompt
            and contract.teacher_prompt_policy_sha256 is not None
            and plan.prompt_policy.sha256 != contract.teacher_prompt_policy_sha256
        )
        or plan.problem_ids != contract.pilot_ids
        or plan.allowed_ids_sha256 != _ids_sha256(contract.pilot_ids)
        or len(plan.problem_ids) != PILOT_AUTHORIZATION_PILOT_SIZE
    ):
        raise TeacherPilotAuthorizationError(
            "pilot plan does not exactly match the deterministic locked pilot scope"
        )


def _require_attempt_budget(
    plan: TeacherPlan, attempts: Sequence[TeacherAttempt]
) -> None:
    """Require every accepted pilot row to have at most three total attempts."""

    counts: Counter[str] = Counter()
    first_by_chunk: dict[int, TeacherAttempt] = {}
    for attempt in attempts:
        # ``_load_attempts`` has already verified the immutable ledger schema.
        counts.update(attempt.input_ids)
        if attempt.attempt_number == 1:
            if attempt.chunk_index in first_by_chunk:
                raise TeacherPilotAuthorizationError("pilot ledger has duplicate first attempts")
            first_by_chunk[attempt.chunk_index] = attempt
    if set(counts) != set(plan.problem_ids) or any(
        count > PILOT_AUTHORIZATION_MAX_ATTEMPTS for count in counts.values()
    ):
        raise TeacherPilotAuthorizationError(
            "pilot rows were not all accepted within the locked three-attempt budget"
        )
    for chunk in plan.chunks:
        first = first_by_chunk.get(chunk.chunk_index)
        if first is None or first.input_ids != chunk.problem_ids:
            raise TeacherPilotAuthorizationError(
                "pilot ledger is missing an exact full-chunk first attempt"
            )


def _initial_exact_match_accepted_count(
    plan: TeacherPlan,
    attempts: Sequence[TeacherAttempt],
    assessments: Mapping[tuple[int, int], object],
) -> int:
    """Count accepted local assessments from the exact first attempt only."""

    first_attempts = {
        (attempt.chunk_index, attempt.attempt_number): attempt
        for attempt in attempts
        if attempt.attempt_number == 1
    }
    accepted_count = 0
    for chunk in plan.chunks:
        key = (chunk.chunk_index, 1)
        attempt = first_attempts.get(key)
        if attempt is None or attempt.input_ids != chunk.problem_ids:
            raise TeacherPilotAuthorizationError(
                "pilot ledger does not preserve its full-chunk first-pass evidence"
            )
        assessment = assessments.get(key)
        if assessment is None:
            # A failed/invalid initial command contributes zero exact matches,
            # while a later repair can still make the full pilot complete.
            continue
        results = getattr(assessment, "results", None)
        if not isinstance(results, tuple) or len(results) != len(chunk.problem_ids):
            raise TeacherPilotAuthorizationError("pilot first-pass assessment is invalid")
        accepted_count += sum(
            result.get("status") == "accepted"
            for result in results
            if isinstance(result, Mapping)
        )
    if not 0 <= accepted_count <= PILOT_AUTHORIZATION_PILOT_SIZE:
        raise TeacherPilotAuthorizationError("pilot first-pass exact-match count is invalid")
    return accepted_count


def _require_exact_logical_audit_plan(
    audit_plan: TeacherLogicalAuditPlan,
    teacher_plan: TeacherPlan,
    verified_bank: object,
    contract: TeacherPilotAuthorizationContract,
) -> None:
    source_jsonl_sha256 = getattr(verified_bank, "source_jsonl_sha256", None)
    source_manifest_sha256 = getattr(verified_bank, "source_manifest_sha256", None)
    if (
        audit_plan.label != contract.logical_audit_label
        or audit_plan.version != contract.logical_audit_version
        or audit_plan.teacher_plan_sha256 != teacher_plan.plan_sha256
        or audit_plan.source_jsonl_sha256 != source_jsonl_sha256
        or audit_plan.source_manifest_sha256 != source_manifest_sha256
        or audit_plan.sample_size != PILOT_AUTHORIZATION_LOGICAL_AUDIT_SAMPLE_SIZE
        or audit_plan.min_consistent
        != PILOT_AUTHORIZATION_LOGICAL_AUDIT_MIN_CONSISTENT
    ):
        raise TeacherPilotAuthorizationError(
            "pilot logical audit is not bound to the locked plan/source/64-60 contract"
        )
    accepted = getattr(verified_bank, "accepted", None)
    if not isinstance(accepted, Mapping):
        raise TeacherPilotAuthorizationError(
            "pilot logical audit has no verified accepted-bank binding"
        )
    try:
        expected_items = _expected_teacher_logical_audit_items(
            teacher_plan,
            accepted,
            sample_size=PILOT_AUTHORIZATION_LOGICAL_AUDIT_SAMPLE_SIZE,
        )
    except (TypeError, TeacherRationaleValidationError) as exc:
        raise TeacherPilotAuthorizationError(
            "pilot logical audit cannot rederive its deterministic bank sample"
        ) from exc
    expected_ids = tuple(item.problem_id for item in expected_items)
    if (
        audit_plan.selection_algorithm != "sha256-plan-id-v1"
        or audit_plan.problem_ids != expected_ids
        or audit_plan.selected_ids_sha256 != _ids_sha256(expected_ids)
        or audit_plan.items != expected_items
    ):
        raise TeacherPilotAuthorizationError(
            "pilot logical audit does not match the deterministic verified-bank sample"
        )


def _load_receipt_payload(path: Path) -> dict[str, object]:
    payload = _load_json_object(path, "pilot authorization receipt")
    receipt_uses_v2 = _receipt_uses_v2_schema(payload)
    expected_keys = {
        "schema_version",
        "teacher_config_sha256",
        "teacher_config_file_sha256",
        "train_sha256",
        "exclusions_sha256",
        "exclusion_count",
        "split_artifact_sha256",
        "development_shard_sha256",
        "split_version",
        "split_sha256",
        "source_groups_sha256",
        "fold",
        "fold0_training_ids_sha256",
        "fold0_training_problem_count",
        "pilot_ids_sha256",
        "pilot_problem_count",
        "pilot_plan_sha256",
        "pilot_plan_label",
        "pilot_plan_version",
        "pilot_plan_allowed_ids_sha256",
        "source_jsonl_sha256",
        "source_manifest_sha256",
        "audit_plan_sha256",
        "audit_manifest_sha256",
        "audit_sample_size",
        "audit_min_consistent",
        "audit_consistent_problem_count",
        "audit_inconsistent_problem_count",
        "initial_exact_match_accepted_count",
        "initial_exact_match_total_count",
        "initial_exact_match_threshold_percent",
        "max_attempts",
        "all_pilot_rows_accepted",
        "reference_answer_used_locally",
        "reference_answer_in_prompt",
        "locked_holdout_accessed",
        "leaderboard_or_test_used",
        "raw_generation_serialized",
        "payload_sha256",
    }
    if receipt_uses_v2:
        expected_keys.add("teacher_prompt_policy_sha256")
    if set(payload) != expected_keys:
        raise TeacherPilotAuthorizationError(
            "pilot authorization receipt keys differ from the locked schema"
        )
    stored_sha = payload["payload_sha256"]
    _required_sha256(stored_sha, "pilot authorization payload_sha256")
    without_hash = dict(payload)
    without_hash.pop("payload_sha256")
    if _payload_sha256(without_hash) != stored_sha:
        raise TeacherPilotAuthorizationError(
            "pilot authorization receipt payload SHA does not match content"
        )
    _validate_receipt_value_types(
        without_hash,
        requires_prompt_policy=receipt_uses_v2,
    )
    return payload


def _validate_receipt_value_types(
    payload: Mapping[str, object], *, requires_prompt_policy: bool
) -> None:
    for field_name in (
        "teacher_config_sha256",
        "teacher_config_file_sha256",
        "train_sha256",
        "exclusions_sha256",
        "split_artifact_sha256",
        "development_shard_sha256",
        "split_sha256",
        "source_groups_sha256",
        "fold0_training_ids_sha256",
        "pilot_ids_sha256",
        "pilot_plan_sha256",
        "pilot_plan_allowed_ids_sha256",
        "source_jsonl_sha256",
        "source_manifest_sha256",
        "audit_plan_sha256",
        "audit_manifest_sha256",
    ):
        _required_sha256(payload.get(field_name), field_name)
    if requires_prompt_policy:
        _required_sha256(
            payload.get("teacher_prompt_policy_sha256"),
            "teacher_prompt_policy_sha256",
        )
    for field_name in ("split_version", "pilot_plan_label", "pilot_plan_version"):
        _required_nonempty_text(payload.get(field_name), field_name)
    for field_name in (
        "exclusion_count",
        "fold",
        "fold0_training_problem_count",
        "pilot_problem_count",
        "audit_sample_size",
        "audit_min_consistent",
        "audit_consistent_problem_count",
        "audit_inconsistent_problem_count",
        "initial_exact_match_accepted_count",
        "initial_exact_match_total_count",
        "initial_exact_match_threshold_percent",
        "max_attempts",
    ):
        value = payload.get(field_name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise TeacherPilotAuthorizationError(
                f"pilot authorization {field_name} must be a non-negative integer"
            )
    for field_name in (
        "all_pilot_rows_accepted",
        "reference_answer_used_locally",
        "reference_answer_in_prompt",
        "locked_holdout_accessed",
        "leaderboard_or_test_used",
        "raw_generation_serialized",
    ):
        if type(payload.get(field_name)) is not bool:
            raise TeacherPilotAuthorizationError(
                f"pilot authorization {field_name} must be a boolean"
            )


def _receipt_from_payload(
    path: Path, payload: Mapping[str, object], *, file_sha256: str
) -> TeacherPilotAuthorizationReceipt:
    return TeacherPilotAuthorizationReceipt(
        path=path.resolve(strict=True),
        file_sha256=file_sha256,
        payload_sha256=str(payload["payload_sha256"]),
        pilot_plan_sha256=str(payload["pilot_plan_sha256"]),
        audit_plan_sha256=str(payload["audit_plan_sha256"]),
        audit_manifest_sha256=str(payload["audit_manifest_sha256"]),
        initial_exact_match_accepted_count=int(
            payload["initial_exact_match_accepted_count"]
        ),
        initial_exact_match_total_count=int(payload["initial_exact_match_total_count"]),
        logical_audit_consistent_problem_count=int(
            payload["audit_consistent_problem_count"]
        ),
        teacher_prompt_policy_sha256=(
            str(payload["teacher_prompt_policy_sha256"])
            if "teacher_prompt_policy_sha256" in payload
            else None
        ),
    )


def _write_new_receipt(path: str | Path, payload: bytes) -> Path:
    raw = Path(path)
    if raw.is_symlink() or raw.parent.is_symlink():
        raise TeacherPilotAuthorizationError("pilot authorization receipt refuses symbolic links")
    target = raw.resolve(strict=False)
    if not target.parent.is_dir():
        raise TeacherPilotAuthorizationError(
            "pilot authorization receipt parent must be an existing directory"
        )
    if target.exists():
        raise TeacherPilotAuthorizationError(
            "refusing to overwrite pilot authorization receipt"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            raise TeacherPilotAuthorizationError(
                "refusing to overwrite pilot authorization receipt"
            ) from exc
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()
    return target.resolve(strict=True)


def _write_new_sidecar(path: str | Path, payload: bytes, *, label: str) -> Path:
    """Atomically publish an immutable raw-free promotion sidecar."""

    raw = Path(path)
    if raw.is_symlink() or raw.parent.is_symlink():
        raise TeacherPilotAuthorizationError(f"{label} refuses symbolic links")
    target = raw.resolve(strict=False)
    if not target.parent.is_dir():
        raise TeacherPilotAuthorizationError(f"{label} parent must be an existing directory")
    if target.exists():
        raise TeacherPilotAuthorizationError(f"refusing to overwrite {label}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            raise TeacherPilotAuthorizationError(
                f"refusing to overwrite {label}"
            ) from exc
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()
    return target.resolve(strict=True)


def _regular_file(path: str | Path, label: str) -> Path:
    raw = Path(path)
    if raw.is_symlink():
        raise TeacherPilotAuthorizationError(f"{label} refuses symbolic links")
    resolved = raw.resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise TeacherPilotAuthorizationError(f"{label} must be a regular file")
    return resolved


def _load_json_object(path: Path, label: str) -> dict[str, object]:
    try:
        text = path.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeError) as exc:
        raise TeacherPilotAuthorizationError(f"cannot read {label}: {exc}") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, UnicodeError, ValueError) as exc:
        raise TeacherPilotAuthorizationError(f"cannot parse {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise TeacherPilotAuthorizationError(f"{label} must contain one JSON object")
    return value


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"unsupported JSON constant: {value}")


def _payload_sha256(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(dict(payload))).hexdigest()


def _ids_sha256(ids: Sequence[str]) -> str:
    return hashlib.sha256(canonical_json_bytes(list(ids))).hexdigest()


def _required_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise TeacherPilotAuthorizationError(f"{label} must be a lowercase SHA-256")
    return value


def _required_nonempty_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip() or "\x00" in value:
        raise TeacherPilotAuthorizationError(f"{label} must be non-empty safe text")
    return value


def _validate_canonical_train_ids(
    identifiers: Sequence[str], label: str, *, allow_empty: bool
) -> None:
    if not isinstance(identifiers, tuple):
        raise TeacherPilotAuthorizationError(f"{label} must be an immutable tuple")
    if not identifiers and not allow_empty:
        raise TeacherPilotAuthorizationError(f"{label} must not be empty")
    if tuple(sorted(identifiers)) != identifiers or len(set(identifiers)) != len(identifiers):
        raise TeacherPilotAuthorizationError(f"{label} must be sorted and unique")
    if any(
        not isinstance(identifier, str)
        or _TRAIN_ID_RE.fullmatch(identifier) is None
        for identifier in identifiers
    ):
        raise TeacherPilotAuthorizationError(f"{label} contains an invalid organizer train ID")


__all__ = [
    "FULL_V1_BANK_AUTHORIZATION_FILENAME",
    "FULL_V1_BANK_AUTHORIZATION_SCHEMA",
    "PILOT_AUTHORIZATION_INITIAL_EXACT_MATCH_PERCENT",
    "PILOT_AUTHORIZATION_LOGICAL_AUDIT_MIN_CONSISTENT",
    "PILOT_AUTHORIZATION_LOGICAL_AUDIT_SAMPLE_SIZE",
    "PILOT_AUTHORIZATION_MAX_ATTEMPTS",
    "PILOT_AUTHORIZATION_PILOT_SIZE",
    "PILOT_AUTHORIZATION_SCHEMA",
    "TeacherPilotAuthorizationContract",
    "TeacherPilotAuthorizationError",
    "TeacherPilotAuthorizationReceipt",
    "create_teacher_pilot_authorization",
    "load_teacher_full_v1_bank_authorization",
    "verify_teacher_pilot_authorization",
    "write_teacher_full_v1_bank_authorization",
]

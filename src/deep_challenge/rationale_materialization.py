"""Fail-closed private materialization of finalized Codex teacher banks.

The Codex teacher ledger publishes one private source JSONL per immutable plan.
That is convenient while producing the fold-0 bank (v1), but a later CV fold
needs the exact union of the v1 and separately authorized remaining-CV (v2)
banks.  This module selects only ``training_ids(fold)`` from one or more
already-finalized banks and writes a new private source JSONL that
``build_rationale_corpus`` can consume.

The companion manifest is deliberately raw-free: it records hashes and counts
only, never problem IDs, questions, rationale text, parsed integers, or local
paths.  It is a no-overwrite pair with the private JSONL, so a later command
can bind a corpus build to a deterministic, split-scoped source selection.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .answers import parse_answer
from .data import MathRecord
from .provenance import canonical_json_bytes
from .splits import (
    SplitManifest,
    SplitValidationError,
    eligible_training_ids,
    eligible_validation_ids,
)
from .teacher_pilot_authorization import (
    FULL_V1_BANK_AUTHORIZATION_FILENAME,
    TeacherPilotAuthorizationError,
    load_teacher_full_v1_bank_authorization,
)
from .teacher_rationale import (
    TeacherRationaleValidationError,
    _verify_teacher_bank_for_logical_audit,
)

_MATERIALIZATION_SCHEMA = "gate-b-teacher-bank-materialization-v1"
_SOURCE_ROW_SCHEMA = "gate-b-concise-rationale-row-v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_V2_BANK_AUTHORIZATION_SCHEMA = "gate-b-codex-teacher-v2-authorization-v1"
_V2_BANK_AUTHORIZATION_FILENAME = "v2-authorization.json"
_V2_BANK_SCOPE = "remaining_development_cv_after_fold0_training"
_V2_PLAN_LABEL = "codex-gpt-5.6-sol-teacher-development-v2"
_V2_PLAN_VERSION = "v2"


class TeacherBankMaterializationValidationError(ValueError):
    """Raised when a private bank cannot safely enter a fold corpus."""


class TeacherBankMaterializationArtifactExistsError(FileExistsError):
    """Raised when an immutable materialized pair would be overwritten."""


@dataclass(frozen=True, slots=True)
class FinalizedTeacherBank:
    """One private, finalized teacher source together with its ledger plan."""

    plan_dir: str | Path
    source_jsonl: str | Path
    source_manifest: str | Path


@dataclass(frozen=True, slots=True)
class TeacherBankMaterializationResult:
    """Portable identity of a private exact-fold source selection."""

    records_path: str
    records_sha256: str
    manifest_path: str
    manifest_sha256: str
    record_count: int
    source_bank_count: int
    training_ids_sha256: str
    promotion_authorization_verified: bool = True

    def as_dict(self) -> dict[str, object]:
        """Return a raw-free result suitable for CLI status output."""

        return {
            "records_file": Path(self.records_path).name,
            "records_sha256": self.records_sha256,
            "manifest_file": Path(self.manifest_path).name,
            "manifest_sha256": self.manifest_sha256,
            "record_count": self.record_count,
            "source_bank_count": self.source_bank_count,
            "training_ids_sha256": self.training_ids_sha256,
            "promotion_authorization_verified": self.promotion_authorization_verified,
        }


@dataclass(frozen=True, slots=True)
class _VerifiedBank:
    """Internal raw-bearing bank evidence; never serialized into a manifest."""

    plan_sha256: str
    allowed_ids_sha256: str
    questions_sha256: str
    output_schema_sha256: str
    source_jsonl_sha256: str
    source_manifest_sha256: str
    promotion_authorization_kind: str | None
    promotion_authorization_payload_sha256: str | None
    problem_ids: tuple[str, ...]
    rows_by_id: Mapping[str, bytes]


def materialize_teacher_bank_source(
    banks: Iterable[FinalizedTeacherBank],
    records: Iterable[MathRecord],
    *,
    split_manifest: SplitManifest,
    fold: int,
    excluded_ids: Iterable[str],
    output_jsonl: str | Path,
    output_manifest: str | Path,
    allow_unqualified_synthetic_banks: bool = False,
) -> TeacherBankMaterializationResult:
    """Write the exact private ``training_ids(fold)`` subset from finalized banks.

    Every input bank is independently re-derived from its original immutable
    teacher ledger before any row is selected.  A bank may contain rows for a
    different development fold, which is necessary for v1+v2 reuse, but it may
    never contain a locked-holdout, excluded, leaderboard/test, or other
    out-of-split ID.  A complete v1 bank must carry the immutable pilot-receipt
    sidecar and a v2 bank must carry its positive probe-decision sidecar;
    direct unqualified tuples fail closed.  ``allow_unqualified_synthetic_banks``
    exists solely for isolated module fixtures and is never used by the public
    CLI.  Duplicate IDs across banks are rejected even when their raw rows
    happen to be byte-identical.
    """

    exclusions = tuple(excluded_ids)
    expected_ids, record_by_id, excluded_digest = _fold_training_contract(
        records,
        split_manifest=split_manifest,
        fold=fold,
        excluded_ids=exclusions,
    )
    if type(allow_unqualified_synthetic_banks) is not bool:
        raise TeacherBankMaterializationValidationError(
            "allow_unqualified_synthetic_banks must be a boolean"
        )
    development_cv_ids = _eligible_development_cv_ids(split_manifest, exclusions)
    verified_banks = _verify_banks(
        tuple(banks),
        development_cv_ids,
        record_by_id,
        split_manifest=split_manifest,
        excluded_ids=exclusions,
        allow_unqualified_synthetic_banks=allow_unqualified_synthetic_banks,
    )

    rows_by_id: dict[str, bytes] = {}
    for bank in verified_banks:
        for problem_id, raw_line in bank.rows_by_id.items():
            if problem_id in rows_by_id:
                raise TeacherBankMaterializationValidationError(
                    "duplicate or conflicting teacher problem_id across finalized banks"
                )
            rows_by_id[problem_id] = raw_line

    missing = set(expected_ids) - set(rows_by_id)
    if missing:
        raise TeacherBankMaterializationValidationError(
            "finalized teacher banks do not cover the exact eligible fold-training scope"
        )
    selected_bytes = b"".join(rows_by_id[problem_id] for problem_id in expected_ids)
    selected_sha256 = hashlib.sha256(selected_bytes).hexdigest()
    records_target, manifest_target = _new_pair_targets(output_jsonl, output_manifest)
    manifest_bytes = _materialization_manifest_bytes(
        records_target=records_target,
        records_bytes=selected_bytes,
        records_sha256=selected_sha256,
        verified_banks=verified_banks,
        split_manifest=split_manifest,
        fold=fold,
        excluded_ids_sha256=excluded_digest,
        training_ids_sha256=_ids_sha256(expected_ids),
        development_cv_ids_sha256=_ids_sha256(development_cv_ids),
    )
    _publish_pair_noreplace(
        records_target,
        selected_bytes,
        manifest_target,
        manifest_bytes,
    )
    return TeacherBankMaterializationResult(
        records_path=str(records_target),
        records_sha256=selected_sha256,
        manifest_path=str(manifest_target),
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        record_count=len(expected_ids),
        source_bank_count=len(verified_banks),
        training_ids_sha256=_ids_sha256(expected_ids),
        promotion_authorization_verified=all(
            bank.promotion_authorization_kind is not None for bank in verified_banks
        ),
    )


def load_teacher_bank_materialization_manifest(
    path: str | Path,
) -> dict[str, object]:
    """Validate and return the raw-free immutable materialization manifest.

    This loader deliberately does not read the private source JSONL.  It is
    useful for provenance inspection without widening access to rationales.
    """

    source = _regular_file(path, "teacher-bank materialization manifest")
    try:
        payload = json.loads(
            source.read_text(encoding="utf-8", errors="strict"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise TeacherBankMaterializationValidationError(
            f"cannot load teacher-bank materialization manifest: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise TeacherBankMaterializationValidationError(
            "teacher-bank materialization manifest must be one JSON object"
        )
    expected_keys = {
        "schema_version",
        "status",
        "partition",
        "split_partition",
        "fold",
        "split_version",
        "split_sha256",
        "source_groups_sha256",
        "excluded_ids_sha256",
        "development_cv_ids_sha256",
        "training_ids_sha256",
        "record_count",
        "records_file",
        "records_bytes",
        "records_sha256",
        "source_bank_count",
        "source_banks",
        "promotion_authorization_verified",
        "raw_rationale_serialized",
        "problem_id_serialized",
        "question_serialized",
        "reference_answer_serialized",
        "leaderboard_or_test_used",
        "locked_holdout_accessed",
        "python_or_sympy_tool_used",
        "payload_sha256",
    }
    if set(payload) != expected_keys:
        raise TeacherBankMaterializationValidationError(
            "teacher-bank materialization manifest keys differ from the locked schema"
        )
    if payload["schema_version"] != _MATERIALIZATION_SCHEMA or payload["status"] != "green":
        raise TeacherBankMaterializationValidationError(
            "teacher-bank materialization manifest schema or status is invalid"
        )
    stored = _required_sha256(payload["payload_sha256"], "payload_sha256")
    unhashed = dict(payload)
    unhashed.pop("payload_sha256")
    if hashlib.sha256(canonical_json_bytes(unhashed)).hexdigest() != stored:
        raise TeacherBankMaterializationValidationError(
            "teacher-bank materialization manifest payload SHA is invalid"
        )
    _validate_raw_free_manifest(payload)
    return payload


def _fold_training_contract(
    records: Iterable[MathRecord],
    *,
    split_manifest: SplitManifest,
    fold: int,
    excluded_ids: Iterable[str],
) -> tuple[tuple[str, ...], dict[str, MathRecord], str]:
    if not isinstance(split_manifest, SplitManifest):
        raise TeacherBankMaterializationValidationError(
            "split_manifest must be a SplitManifest"
        )
    try:
        split_manifest.validate()
        exclusions = _canonical_train_ids(excluded_ids, "excluded_ids", allow_empty=True)
        expected_ids = eligible_training_ids(split_manifest, fold, exclusions)
    except SplitValidationError as exc:
        raise TeacherBankMaterializationValidationError(
            f"invalid split boundary: {exc}"
        ) from exc
    materialized = tuple(records)
    if not materialized or any(not isinstance(record, MathRecord) for record in materialized):
        raise TeacherBankMaterializationValidationError(
            "records must contain organizer fold-training MathRecord values"
        )
    record_by_id = {record.id: record for record in materialized}
    if len(record_by_id) != len(materialized):
        raise TeacherBankMaterializationValidationError(
            "organizer fold-training records contain duplicate IDs"
        )
    if set(record_by_id) != set(expected_ids):
        raise TeacherBankMaterializationValidationError(
            "organizer records must exactly match eligible fold-training IDs"
        )
    for record in record_by_id.values():
        if isinstance(record.answer, bool) or not isinstance(record.answer, int):
            raise TeacherBankMaterializationValidationError(
                "organizer fold-training records must have integer reference answers"
            )
    return expected_ids, record_by_id, _ids_sha256(exclusions)


def _eligible_development_cv_ids(
    split_manifest: SplitManifest,
    excluded_ids: Iterable[str],
) -> tuple[str, ...]:
    """Return all eligible CV IDs without ever adding the locked holdout."""

    try:
        train_ids = eligible_training_ids(split_manifest, 0, excluded_ids)
        validation_ids = eligible_validation_ids(split_manifest, 0, excluded_ids)
    except SplitValidationError as exc:
        raise TeacherBankMaterializationValidationError(
            f"cannot derive eligible development-CV scope: {exc}"
        ) from exc
    union = tuple(sorted(set(train_ids) | set(validation_ids)))
    if len(union) != len(train_ids) + len(validation_ids):
        raise TeacherBankMaterializationValidationError(
            "eligible development-CV scope unexpectedly overlaps itself"
        )
    return union


def _verify_banks(
    banks: Sequence[FinalizedTeacherBank],
    development_cv_ids: Sequence[str],
    record_by_id: Mapping[str, MathRecord],
    *,
    split_manifest: SplitManifest,
    excluded_ids: Sequence[str],
    allow_unqualified_synthetic_banks: bool,
) -> tuple[_VerifiedBank, ...]:
    if not banks:
        raise TeacherBankMaterializationValidationError(
            "at least one finalized teacher bank is required"
        )
    development_cv_set = set(development_cv_ids)
    seen_plan_sha256: set[str] = set()
    verified_banks: list[_VerifiedBank] = []
    for bank in banks:
        if not isinstance(bank, FinalizedTeacherBank):
            raise TeacherBankMaterializationValidationError(
                "banks must contain FinalizedTeacherBank values"
            )
        try:
            verified = _verify_teacher_bank_for_logical_audit(
                bank.plan_dir,
                bank.source_jsonl,
                bank.source_manifest,
            )
        except (OSError, TeacherRationaleValidationError) as exc:
            raise TeacherBankMaterializationValidationError(
                "teacher bank plan/source/manifest provenance validation failed"
            ) from exc
        plan = verified.teacher_plan
        if plan.plan_sha256 in seen_plan_sha256:
            raise TeacherBankMaterializationValidationError(
                "duplicate finalized teacher plan input"
            )
        seen_plan_sha256.add(plan.plan_sha256)
        outside_scope = set(plan.problem_ids) - development_cv_set
        if outside_scope:
            raise TeacherBankMaterializationValidationError(
                "finalized teacher bank crosses the eligible development-CV scope"
            )
        promotion_kind, promotion_payload_sha256 = _verify_bank_promotion_binding(
            plan,
            split_manifest=split_manifest,
            excluded_ids=excluded_ids,
            allow_unqualified_synthetic_banks=allow_unqualified_synthetic_banks,
        )
        rows_by_id = _load_verified_source_rows(
            bank.source_jsonl,
            expected_ids=plan.problem_ids,
            record_by_id=record_by_id,
        )
        verified_banks.append(
            _VerifiedBank(
                plan_sha256=plan.plan_sha256,
                allowed_ids_sha256=plan.allowed_ids_sha256,
                questions_sha256=plan.questions_sha256,
                output_schema_sha256=plan.output_schema_sha256,
                source_jsonl_sha256=verified.source_jsonl_sha256,
                source_manifest_sha256=verified.source_manifest_sha256,
                promotion_authorization_kind=promotion_kind,
                promotion_authorization_payload_sha256=promotion_payload_sha256,
                problem_ids=plan.problem_ids,
                rows_by_id=rows_by_id,
            )
        )
    return tuple(sorted(verified_banks, key=lambda item: item.plan_sha256))


def _verify_bank_promotion_binding(
    plan: object,
    *,
    split_manifest: SplitManifest,
    excluded_ids: Sequence[str],
    allow_unqualified_synthetic_banks: bool,
) -> tuple[str | None, str | None]:
    """Require the promotion evidence appropriate to a v1 or v2 bank.

    The finalized source/manifest prove that a bank was locally assessed, but
    they do not prove that it was *allowed to be generated*.  Complete v1
    banks therefore need their immutable pilot-receipt sidecar, while v2 banks
    need the positive candidate-probe sidecar written during v2 planning.
    ``allow_unqualified_synthetic_banks`` is deliberately an explicit private
    test-fixture escape hatch; the public CLI never supplies it.
    """

    # ``_verify_teacher_bank_for_logical_audit`` already returned a loaded
    # TeacherPlan.  Avoid importing its class solely for an assertion here;
    # the attributes below are the locked plan contract we need.
    plan_ids = getattr(plan, "problem_ids", None)
    plan_dir = getattr(plan, "plan_dir", None)
    if not isinstance(plan_ids, tuple) or not isinstance(plan_dir, Path):
        raise TeacherBankMaterializationValidationError(
            "finalized teacher bank has an invalid plan promotion binding"
        )
    v1_ids = eligible_training_ids(split_manifest, 0, excluded_ids)
    v2_ids = eligible_validation_ids(split_manifest, 0, excluded_ids)
    if plan_ids == v1_ids:
        sidecar = plan_dir / FULL_V1_BANK_AUTHORIZATION_FILENAME
        if not sidecar.exists():
            if allow_unqualified_synthetic_banks:
                return None, None
            raise TeacherBankMaterializationValidationError(
                "full v1 teacher bank is missing its pilot promotion authorization"
            )
        try:
            payload = load_teacher_full_v1_bank_authorization(plan_dir)
        except (OSError, TeacherPilotAuthorizationError, TeacherRationaleValidationError) as exc:
            raise TeacherBankMaterializationValidationError(
                "full v1 teacher bank pilot promotion authorization is invalid"
            ) from exc
        if (
            payload.get("fold0_training_ids_sha256") != _ids_sha256(v1_ids)
            or payload.get("fold0_training_problem_count") != len(v1_ids)
        ):
            raise TeacherBankMaterializationValidationError(
                "full v1 teacher bank promotion authorization does not match split scope"
            )
        return "v1_pilot_receipt", str(payload["payload_sha256"])
    if plan_ids == v2_ids:
        sidecar = plan_dir / _V2_BANK_AUTHORIZATION_FILENAME
        if not sidecar.exists():
            if allow_unqualified_synthetic_banks:
                return None, None
            raise TeacherBankMaterializationValidationError(
                "v2 teacher bank is missing its positive probe authorization"
            )
        payload = _load_v2_bank_authorization(
            sidecar,
            plan=plan,
            split_manifest=split_manifest,
            excluded_ids=excluded_ids,
            v1_ids=v1_ids,
            v2_ids=v2_ids,
        )
        return "v2_positive_probe", str(payload["payload_sha256"])
    if allow_unqualified_synthetic_banks:
        return None, None
    raise TeacherBankMaterializationValidationError(
        "teacher bank plan does not match an authorized v1 or v2 promotion scope"
    )


def _load_v2_bank_authorization(
    path: Path,
    *,
    plan: object,
    split_manifest: SplitManifest,
    excluded_ids: Sequence[str],
    v1_ids: Sequence[str],
    v2_ids: Sequence[str],
) -> dict[str, object]:
    """Validate the plan-local immutable positive-probe authorization."""

    source = _regular_file(path, "v2 teacher bank promotion authorization")
    payload = _load_json_object(source, "v2 teacher bank promotion authorization")
    expected_keys = {
        "schema_version",
        "plan_sha256",
        "plan_label",
        "plan_version",
        "scope",
        "fold",
        "candidate_label",
        "candidate_full_oof_authorized",
        "candidate_action",
        "candidate_probe_decision_sha256",
        "candidate_probe_decision_payload_sha256",
        "split_sha256",
        "source_groups_sha256",
        "eligible_development_cv_ids_sha256",
        "fold0_training_ids_sha256",
        "allowed_ids_sha256",
        "allowed_problem_count",
        "payload_sha256",
    }
    if set(payload) != expected_keys:
        raise TeacherBankMaterializationValidationError(
            "v2 teacher bank promotion authorization keys differ from the locked schema"
        )
    if payload.get("schema_version") != _V2_BANK_AUTHORIZATION_SCHEMA:
        raise TeacherBankMaterializationValidationError(
            "v2 teacher bank promotion authorization schema is invalid"
        )
    stored_sha = _required_sha256(
        payload.get("payload_sha256"),
        "v2 teacher bank promotion authorization payload_sha256",
    )
    without_hash = dict(payload)
    without_hash.pop("payload_sha256")
    if hashlib.sha256(canonical_json_bytes(without_hash)).hexdigest() != stored_sha:
        raise TeacherBankMaterializationValidationError(
            "v2 teacher bank promotion authorization payload SHA is invalid"
        )
    for field_name in (
        "plan_sha256",
        "candidate_probe_decision_sha256",
        "candidate_probe_decision_payload_sha256",
        "split_sha256",
        "source_groups_sha256",
        "eligible_development_cv_ids_sha256",
        "fold0_training_ids_sha256",
        "allowed_ids_sha256",
    ):
        _required_sha256(payload.get(field_name), f"v2 authorization {field_name}")
    for field_name in ("plan_label", "plan_version", "candidate_label"):
        value = payload.get(field_name)
        if not isinstance(value, str) or not value or value != value.strip() or "\x00" in value:
            raise TeacherBankMaterializationValidationError(
                f"v2 authorization {field_name} is invalid"
            )
    allowed_problem_count = payload.get("allowed_problem_count")
    if (
        isinstance(allowed_problem_count, bool)
        or not isinstance(allowed_problem_count, int)
        or allowed_problem_count < 1
    ):
        raise TeacherBankMaterializationValidationError(
            "v2 authorization allowed_problem_count is invalid"
        )
    if (
        payload.get("scope") != _V2_BANK_SCOPE
        or payload.get("fold") != 0
        or payload.get("candidate_full_oof_authorized") is not True
        or payload.get("candidate_action") != "continue_to_complete_oof"
        or payload.get("plan_sha256") != getattr(plan, "plan_sha256", None)
        or payload.get("plan_label") != _V2_PLAN_LABEL
        or payload.get("plan_version") != _V2_PLAN_VERSION
        or getattr(plan, "label", None) != _V2_PLAN_LABEL
        or getattr(plan, "version", None) != _V2_PLAN_VERSION
        or payload.get("allowed_ids_sha256") != getattr(plan, "allowed_ids_sha256", None)
        or allowed_problem_count != len(getattr(plan, "problem_ids", ()))
        or payload.get("split_sha256") != split_manifest.sha256
        or payload.get("source_groups_sha256") != split_manifest.source_groups_sha256
        or payload.get("eligible_development_cv_ids_sha256")
        != _ids_sha256(_eligible_development_cv_ids(split_manifest, excluded_ids))
        or payload.get("fold0_training_ids_sha256") != _ids_sha256(v1_ids)
        or tuple(getattr(plan, "problem_ids", ())) != tuple(v2_ids)
    ):
        raise TeacherBankMaterializationValidationError(
            "v2 teacher bank promotion authorization does not match its positive split scope"
        )
    return payload


def _load_verified_source_rows(
    source_jsonl: str | Path,
    *,
    expected_ids: Sequence[str],
    record_by_id: Mapping[str, MathRecord],
) -> dict[str, bytes]:
    source = _regular_file(source_jsonl, "teacher bank source JSONL")
    try:
        payload = source.read_bytes()
        text = payload.decode("utf-8", errors="strict")
    except (OSError, UnicodeError) as exc:
        raise TeacherBankMaterializationValidationError(
            f"cannot read teacher bank source JSONL: {exc}"
        ) from exc
    if not text.endswith("\n"):
        raise TeacherBankMaterializationValidationError(
            "teacher bank source JSONL is not newline-terminated"
        )
    lines = text.splitlines(keepends=True)
    if len(lines) != len(expected_ids):
        raise TeacherBankMaterializationValidationError(
            "teacher bank source JSONL count does not match its finalized plan"
        )
    rows_by_id: dict[str, bytes] = {}
    for problem_id, line in zip(expected_ids, lines, strict=True):
        if not line.endswith("\n"):
            raise TeacherBankMaterializationValidationError(
                "teacher bank source JSONL line is not newline-terminated"
            )
        try:
            row = json.loads(
                line[:-1],
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_json_constant,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise TeacherBankMaterializationValidationError(
                "teacher bank source JSONL has an invalid row"
            ) from exc
        _validate_selected_source_row(row, problem_id, record_by_id)
        if problem_id in rows_by_id:
            raise TeacherBankMaterializationValidationError(
                "teacher bank source JSONL contains duplicate problem IDs"
            )
        rows_by_id[problem_id] = line.encode("utf-8")
    return rows_by_id


def _validate_selected_source_row(
    row: object,
    expected_problem_id: str,
    record_by_id: Mapping[str, MathRecord],
) -> None:
    if not isinstance(row, dict):
        raise TeacherBankMaterializationValidationError(
            "teacher bank source row must be an object"
        )
    required = {
        "schema_version",
        "problem_id",
        "question_sha256",
        "target_text",
        "target_sha256",
        "teacher",
        "verification",
    }
    if set(row) != required or row.get("schema_version") != _SOURCE_ROW_SCHEMA:
        raise TeacherBankMaterializationValidationError(
            "teacher bank source row schema is invalid"
        )
    problem_id = row.get("problem_id")
    if problem_id != expected_problem_id or not isinstance(problem_id, str):
        raise TeacherBankMaterializationValidationError(
            "teacher bank source row order or problem ID is invalid"
        )
    record = record_by_id.get(problem_id)
    if record is None:
        # This is allowed for a source-bank row outside the *selected* fold;
        # its question bytes were already bound by the immutable teacher plan.
        return
    question_sha256 = _required_sha256(row.get("question_sha256"), "question_sha256")
    if question_sha256 != hashlib.sha256(record.question_raw.encode("utf-8")).hexdigest():
        raise TeacherBankMaterializationValidationError(
            "teacher bank source question does not match current organizer train bytes"
        )
    target_text = row.get("target_text")
    target_sha256 = _required_sha256(row.get("target_sha256"), "target_sha256")
    if not isinstance(target_text, str) or (
        hashlib.sha256(target_text.encode("utf-8")).hexdigest() != target_sha256
    ):
        raise TeacherBankMaterializationValidationError(
            "teacher bank source target SHA is invalid"
        )
    parsed = parse_answer(target_text)
    if (
        not parsed.ok
        or parsed.source != "final_answer"
        or parsed.value != record.answer
    ):
        raise TeacherBankMaterializationValidationError(
            "teacher bank source target does not match current organizer answer"
        )
    teacher = row.get("teacher")
    verification = row.get("verification")
    if not isinstance(teacher, dict) or not isinstance(verification, dict):
        raise TeacherBankMaterializationValidationError(
            "teacher bank source evidence is invalid"
        )
    if (
        teacher.get("reference_answer_in_prompt") is not False
        or teacher.get("network_scope") != "training_only"
        or verification.get("status") != "accepted"
        or verification.get("leaderboard_or_test_used") is not False
        or verification.get("locked_holdout_accessed") is not False
        or verification.get("tool_used") is not False
    ):
        raise TeacherBankMaterializationValidationError(
            "teacher bank source evidence crosses the locked safety contract"
        )


def _materialization_manifest_bytes(
    *,
    records_target: Path,
    records_bytes: bytes,
    records_sha256: str,
    verified_banks: Sequence[_VerifiedBank],
    split_manifest: SplitManifest,
    fold: int,
    excluded_ids_sha256: str,
    training_ids_sha256: str,
    development_cv_ids_sha256: str,
) -> bytes:
    source_banks = [
        {
            "plan_sha256": bank.plan_sha256,
            "allowed_ids_sha256": bank.allowed_ids_sha256,
            "questions_sha256": bank.questions_sha256,
            "output_schema_sha256": bank.output_schema_sha256,
            "source_jsonl_sha256": bank.source_jsonl_sha256,
            "source_manifest_sha256": bank.source_manifest_sha256,
            "record_count": len(bank.problem_ids),
            "promotion_authorization_kind": bank.promotion_authorization_kind,
            "promotion_authorization_payload_sha256": (
                bank.promotion_authorization_payload_sha256
            ),
        }
        for bank in verified_banks
    ]
    payload_without_hash: dict[str, object] = {
        "schema_version": _MATERIALIZATION_SCHEMA,
        "status": "green",
        "partition": "fold_training",
        "split_partition": "cross_validation",
        "fold": fold,
        "split_version": split_manifest.version,
        "split_sha256": split_manifest.sha256,
        "source_groups_sha256": split_manifest.source_groups_sha256,
        "excluded_ids_sha256": excluded_ids_sha256,
        "development_cv_ids_sha256": development_cv_ids_sha256,
        "training_ids_sha256": training_ids_sha256,
        "record_count": len(records_bytes.splitlines()),
        "records_file": records_target.name,
        "records_bytes": len(records_bytes),
        "records_sha256": records_sha256,
        "source_bank_count": len(source_banks),
        "source_banks": source_banks,
        "promotion_authorization_verified": all(
            bank.promotion_authorization_kind is not None for bank in verified_banks
        ),
        "raw_rationale_serialized": False,
        "problem_id_serialized": False,
        "question_serialized": False,
        "reference_answer_serialized": False,
        "leaderboard_or_test_used": False,
        "locked_holdout_accessed": False,
        "python_or_sympy_tool_used": False,
    }
    payload = {
        **payload_without_hash,
        "payload_sha256": hashlib.sha256(
            canonical_json_bytes(payload_without_hash)
        ).hexdigest(),
    }
    return (
        json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
        + "\n"
    ).encode("utf-8")


def _validate_raw_free_manifest(payload: Mapping[str, object]) -> None:
    required_false = (
        "raw_rationale_serialized",
        "problem_id_serialized",
        "question_serialized",
        "reference_answer_serialized",
        "leaderboard_or_test_used",
        "locked_holdout_accessed",
        "python_or_sympy_tool_used",
    )
    if any(payload.get(key) is not False for key in required_false):
        raise TeacherBankMaterializationValidationError(
            "teacher-bank materialization manifest crosses the raw-free safety contract"
        )
    if (
        payload.get("partition") != "fold_training"
        or payload.get("split_partition") != "cross_validation"
    ):
        raise TeacherBankMaterializationValidationError(
            "teacher-bank materialization manifest scope is invalid"
        )
    for key in (
        "split_sha256",
        "source_groups_sha256",
        "excluded_ids_sha256",
        "development_cv_ids_sha256",
        "training_ids_sha256",
        "records_sha256",
    ):
        _required_sha256(payload.get(key), key)
    if (
        isinstance(payload.get("fold"), bool)
        or not isinstance(payload.get("fold"), int)
        or payload["fold"] < 0
    ):
        raise TeacherBankMaterializationValidationError(
            "teacher-bank materialization manifest fold is invalid"
        )
    for key in ("record_count", "records_bytes", "source_bank_count"):
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise TeacherBankMaterializationValidationError(
                f"teacher-bank materialization manifest {key} is invalid"
            )
    records_file = payload.get("records_file")
    if (
        not isinstance(records_file, str)
        or Path(records_file).name != records_file
    ):
        raise TeacherBankMaterializationValidationError(
            "teacher-bank materialization manifest records_file is invalid"
        )
    source_banks = payload.get("source_banks")
    if not isinstance(source_banks, list) or len(source_banks) != payload["source_bank_count"]:
        raise TeacherBankMaterializationValidationError(
            "teacher-bank materialization manifest source-bank count is invalid"
        )
    required_bank_keys = {
        "plan_sha256",
        "allowed_ids_sha256",
        "questions_sha256",
        "output_schema_sha256",
        "source_jsonl_sha256",
        "source_manifest_sha256",
        "record_count",
        "promotion_authorization_kind",
        "promotion_authorization_payload_sha256",
    }
    plan_hashes: list[str] = []
    for bank in source_banks:
        if not isinstance(bank, dict) or set(bank) != required_bank_keys:
            raise TeacherBankMaterializationValidationError(
                "teacher-bank materialization manifest source-bank schema is invalid"
            )
        for key in (
            "plan_sha256",
            "allowed_ids_sha256",
            "questions_sha256",
            "output_schema_sha256",
            "source_jsonl_sha256",
            "source_manifest_sha256",
        ):
            _required_sha256(bank.get(key), f"source_banks.{key}")
        bank_record_count = bank.get("record_count")
        if (
            isinstance(bank_record_count, bool)
            or not isinstance(bank_record_count, int)
            or bank_record_count <= 0
        ):
            raise TeacherBankMaterializationValidationError(
                "teacher-bank materialization manifest source-bank record count is invalid"
            )
        authorization_kind = bank.get("promotion_authorization_kind")
        authorization_payload_sha256 = bank.get("promotion_authorization_payload_sha256")
        if authorization_kind is None:
            if authorization_payload_sha256 is not None:
                raise TeacherBankMaterializationValidationError(
                    "unqualified synthetic source bank has authorization SHA evidence"
                )
        elif authorization_kind in {"v1_pilot_receipt", "v2_positive_probe"}:
            _required_sha256(
                authorization_payload_sha256,
                "source_banks.promotion_authorization_payload_sha256",
            )
        else:
            raise TeacherBankMaterializationValidationError(
                "teacher-bank materialization source-bank promotion kind is invalid"
            )
        plan_hashes.append(bank["plan_sha256"])
    if plan_hashes != sorted(plan_hashes) or len(set(plan_hashes)) != len(plan_hashes):
        raise TeacherBankMaterializationValidationError(
            "teacher-bank materialization manifest source-bank plans are invalid"
        )
    if type(payload.get("promotion_authorization_verified")) is not bool:
        raise TeacherBankMaterializationValidationError(
            "teacher-bank materialization promotion authorization flag is invalid"
        )
    expected_authorization_verified = all(
        bank["promotion_authorization_kind"] is not None for bank in source_banks
    )
    if payload["promotion_authorization_verified"] is not expected_authorization_verified:
        raise TeacherBankMaterializationValidationError(
            "teacher-bank materialization promotion authorization flag is inconsistent"
        )


def _new_pair_targets(
    records_path: str | Path,
    manifest_path: str | Path,
) -> tuple[Path, Path]:
    records_target = _new_file_target(records_path, "materialized teacher source JSONL")
    manifest_target = _new_file_target(
        manifest_path, "teacher-bank materialization manifest"
    )
    if records_target == manifest_target:
        raise TeacherBankMaterializationValidationError(
            "materialized teacher source and manifest paths must differ"
        )
    if records_target.parent != manifest_target.parent:
        raise TeacherBankMaterializationValidationError(
            "materialized teacher source and manifest must share one directory"
        )
    return records_target, manifest_target


def _new_file_target(path: str | Path, label: str) -> Path:
    raw = Path(path)
    if raw.is_symlink() or raw.parent.is_symlink():
        raise TeacherBankMaterializationValidationError(f"{label} refuses symbolic links")
    target = raw.resolve(strict=False)
    if not target.parent.is_dir():
        raise TeacherBankMaterializationValidationError(f"{label} parent does not exist")
    if target.exists():
        raise TeacherBankMaterializationArtifactExistsError(
            f"refusing to overwrite {label}: {target}"
        )
    return target


def _publish_pair_noreplace(
    records_target: Path,
    records_bytes: bytes,
    manifest_target: Path,
    manifest_bytes: bytes,
) -> None:
    temporaries: list[Path] = []
    published_records = False
    try:
        for target, payload in (
            (records_target, records_bytes),
            (manifest_target, manifest_bytes),
        ):
            descriptor, raw_temporary = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
            )
            temporary = Path(raw_temporary)
            temporaries.append(temporary)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        try:
            os.link(temporaries[0], records_target)
            published_records = True
            os.link(temporaries[1], manifest_target)
        except FileExistsError as exc:
            if published_records:
                with suppress(FileNotFoundError):
                    records_target.unlink()
                _fsync_directory(records_target.parent)
            raise TeacherBankMaterializationArtifactExistsError(
                "refusing to overwrite materialized teacher source/manifest pair"
            ) from exc
        except BaseException:
            if published_records:
                with suppress(FileNotFoundError):
                    records_target.unlink()
                _fsync_directory(records_target.parent)
            raise
        _fsync_directory(records_target.parent)
    finally:
        for temporary in temporaries:
            with suppress(FileNotFoundError):
                temporary.unlink()


def _regular_file(path: str | Path, label: str) -> Path:
    raw = Path(path)
    if raw.is_symlink() or not raw.is_file():
        raise TeacherBankMaterializationValidationError(f"{label} must be a regular file")
    return raw.resolve(strict=True)


def _load_json_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8", errors="strict"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise TeacherBankMaterializationValidationError(
            f"cannot load {label}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise TeacherBankMaterializationValidationError(
            f"{label} must contain one JSON object"
        )
    return value


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON numeric constant {value!r}")


def _required_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise TeacherBankMaterializationValidationError(
            f"{label} must be a lowercase 64-character SHA-256"
        )
    return value


def _canonical_train_ids(
    values: Iterable[str],
    label: str,
    *,
    allow_empty: bool,
) -> tuple[str, ...]:
    materialized = tuple(values)
    if not allow_empty and not materialized:
        raise TeacherBankMaterializationValidationError(f"{label} must not be empty")
    if any(
        not isinstance(value, str)
        or len(value) != 12
        or not value.startswith("train-")
        or not value[6:].isdigit()
        for value in materialized
    ):
        raise TeacherBankMaterializationValidationError(
            f"{label} contains a non-train ID"
        )
    if len(set(materialized)) != len(materialized):
        raise TeacherBankMaterializationValidationError(f"{label} contains duplicate IDs")
    return tuple(sorted(materialized))


def _ids_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256(canonical_json_bytes(list(values))).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "FinalizedTeacherBank",
    "TeacherBankMaterializationArtifactExistsError",
    "TeacherBankMaterializationResult",
    "TeacherBankMaterializationValidationError",
    "load_teacher_bank_materialization_manifest",
    "materialize_teacher_bank_source",
]

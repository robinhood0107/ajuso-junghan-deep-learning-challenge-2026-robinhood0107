"""CPU-only contracts for private, answer-verified concise-rationale corpora.

The module never imports model runtimes.  It turns an ignored, private teacher
JSONL into a split-bound canonical corpus and a raw-free audit.  Leaderboard,
test, and locked-holdout inputs are structurally impossible because every row
must match the exact organizer fold-training ID set derived from ``split v4``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .answers import parse_answer
from .data import MathRecord
from .model_preflight import OFFICIAL_MODEL_ID, OFFICIAL_REVISION
from .provenance import canonical_json_bytes, sha256_file
from .splits import SplitManifest, SplitValidationError, eligible_training_ids

_ROW_SCHEMA = "gate-b-concise-rationale-row-v1"
_MANIFEST_SCHEMA = "gate-b-concise-rationale-corpus-v1"
_AUDIT_SCHEMA = "gate-b-concise-rationale-audit-v1"
_CONFIG_SCHEMA = "gate-b-concise-rationale-candidate-v1"
_TRAIN_ID_RE = re.compile(r"train-\d{6}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_FINAL_LINE_RE = re.compile(r"(?:\A|\n)Final answer: (0|-?[1-9]\d*)\Z")
_FINAL_MARKER_RE = re.compile(r"(?i)final\s+answer\s*:")
_ALLOWED_CONTROL_CHARACTERS = frozenset({"\n", "\t"})


class RationaleCorpusValidationError(ValueError):
    """Raised when a rationale artifact crosses a safety boundary."""


class RationaleCorpusArtifactExistsError(FileExistsError):
    """Raised when a corpus or audit target would be overwritten."""


@dataclass(frozen=True, slots=True)
class ConciseRationaleConfig:
    """Versioned candidate policy, separate from the inference config."""

    schema_version: str = _CONFIG_SCHEMA
    candidate_label: str = "qlora-concise-rationale-v1"
    training_target_kind: str = "verified_concise_rationale"
    student_model_id: str = OFFICIAL_MODEL_ID
    student_revision: str = OFFICIAL_REVISION
    inference_route: str = "direct_answer"
    min_rationale_characters: int = 16
    max_rationale_characters: int = 1_500
    max_rationale_lines: int = 12
    required_final_marker: str = "Final answer:"
    require_exact_fold_training_coverage: bool = True
    require_reference_answer_exact_match: bool = True
    require_reference_answer_hidden_from_teacher: bool = True
    require_training_only_teacher_scope: bool = True
    allow_leaderboard_or_test: bool = False
    allow_locked_holdout: bool = False
    allow_python_or_sympy_tool: bool = False
    allowed_verification_methods: tuple[str, ...] = (
        "reference_answer_exact_match",
        "human_reviewed_reference_answer_exact_match",
    )

    def __post_init__(self) -> None:
        expected: tuple[tuple[str, object, object], ...] = (
            ("schema_version", self.schema_version, _CONFIG_SCHEMA),
            ("candidate_label", self.candidate_label, "qlora-concise-rationale-v1"),
            (
                "training_target_kind",
                self.training_target_kind,
                "verified_concise_rationale",
            ),
            ("student_model_id", self.student_model_id, OFFICIAL_MODEL_ID),
            ("student_revision", self.student_revision, OFFICIAL_REVISION),
            ("inference_route", self.inference_route, "direct_answer"),
            ("min_rationale_characters", self.min_rationale_characters, 16),
            ("max_rationale_characters", self.max_rationale_characters, 1_500),
            ("max_rationale_lines", self.max_rationale_lines, 12),
            ("required_final_marker", self.required_final_marker, "Final answer:"),
            (
                "require_exact_fold_training_coverage",
                self.require_exact_fold_training_coverage,
                True,
            ),
            (
                "require_reference_answer_exact_match",
                self.require_reference_answer_exact_match,
                True,
            ),
            (
                "require_reference_answer_hidden_from_teacher",
                self.require_reference_answer_hidden_from_teacher,
                True,
            ),
            (
                "require_training_only_teacher_scope",
                self.require_training_only_teacher_scope,
                True,
            ),
            ("allow_leaderboard_or_test", self.allow_leaderboard_or_test, False),
            ("allow_locked_holdout", self.allow_locked_holdout, False),
            ("allow_python_or_sympy_tool", self.allow_python_or_sympy_tool, False),
            (
                "allowed_verification_methods",
                self.allowed_verification_methods,
                (
                    "reference_answer_exact_match",
                    "human_reviewed_reference_answer_exact_match",
                ),
            ),
        )
        for field_name, value, locked in expected:
            if value != locked or type(value) is not type(locked):
                raise RationaleCorpusValidationError(
                    f"{field_name} is locked to {locked!r} for this candidate"
                )

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["allowed_verification_methods"] = list(
            self.allowed_verification_methods
        )
        return payload

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.as_dict())).hexdigest()


DEFAULT_CONCISE_RATIONALE_CONFIG = ConciseRationaleConfig()


@dataclass(frozen=True, slots=True)
class RationaleTeacherEvidence:
    provider: str
    model_id: str
    model_revision: str
    prompt_sha256: str
    generation_config_sha256: str
    seed: int
    sample_index: int
    raw_generation_sha256: str
    reference_answer_in_prompt: bool
    network_scope: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RationaleVerification:
    status: str
    method: str
    leaderboard_or_test_used: bool
    locked_holdout_accessed: bool
    tool_used: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RationaleCorpusRow:
    problem_id: str
    question_sha256: str
    target_text: str
    target_sha256: str
    teacher: RationaleTeacherEvidence
    verification: RationaleVerification

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": _ROW_SCHEMA,
            "problem_id": self.problem_id,
            "question_sha256": self.question_sha256,
            "target_text": self.target_text,
            "target_sha256": self.target_sha256,
            "teacher": self.teacher.as_dict(),
            "verification": self.verification.as_dict(),
        }

    def to_json_line(self) -> str:
        return json.dumps(
            self.as_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )


@dataclass(frozen=True, slots=True)
class RationaleCorpusEvidence:
    records_path: str
    records_sha256: str
    manifest_path: str
    manifest_sha256: str
    candidate_config_sha256: str
    candidate_config_file_sha256: str
    fold: int
    training_ids_sha256: str
    record_count: int
    rows: tuple[RationaleCorpusRow, ...]
    audit_path: str | None = None
    audit_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class RationaleArtifactWriteResult:
    records_path: str
    records_sha256: str
    manifest_path: str
    manifest_sha256: str
    record_count: int


@dataclass(frozen=True, slots=True)
class RationaleAuditWriteResult:
    path: str
    sha256: str
    payload_sha256: str
    size_bytes: int
    record_count: int


def load_concise_rationale_config(path: str | Path) -> tuple[ConciseRationaleConfig, str]:
    """Load one exact public candidate spec and validate its semantic hash."""

    source, payload, digest = _load_json_object(path, "concise-rationale config")
    del source
    stored_sha256 = payload.pop("config_sha256", None)
    if not isinstance(stored_sha256, str):
        raise RationaleCorpusValidationError(
            "concise-rationale config is missing config_sha256"
        )
    try:
        allowed = payload.get("allowed_verification_methods")
        if isinstance(allowed, list):
            payload["allowed_verification_methods"] = tuple(allowed)
        config = ConciseRationaleConfig(**payload)
    except (TypeError, RationaleCorpusValidationError) as exc:
        raise RationaleCorpusValidationError(
            f"concise-rationale config schema is invalid: {exc}"
        ) from exc
    if stored_sha256 != config.sha256:
        raise RationaleCorpusValidationError(
            "concise-rationale config semantic SHA does not match its content"
        )
    return config, digest


def build_rationale_corpus(
    source_jsonl: str | Path,
    records: Iterable[MathRecord],
    *,
    split_manifest: SplitManifest,
    fold: int,
    excluded_ids: Iterable[str],
    candidate_config_file_sha256: str,
    output_jsonl: str | Path,
    output_manifest: str | Path,
    config: ConciseRationaleConfig = DEFAULT_CONCISE_RATIONALE_CONFIG,
) -> RationaleArtifactWriteResult:
    """Validate private teacher rows and atomically publish a canonical pair."""

    expected_ids, record_by_id, excluded_digest = _training_contract(
        records, split_manifest=split_manifest, fold=fold, excluded_ids=excluded_ids
    )
    rows = _load_rows(
        source_jsonl,
        expected_ids=expected_ids,
        record_by_id=record_by_id,
        config=config,
    )
    config_file_digest = _required_sha256(
        candidate_config_file_sha256, "candidate_config_file_sha256"
    )
    records_target, manifest_target = _artifact_targets(
        output_jsonl, output_manifest
    )
    records_bytes = (
        "".join(f"{row.to_json_line()}\n" for row in rows).encode("utf-8")
    )
    records_digest = hashlib.sha256(records_bytes).hexdigest()
    rationale_lengths = [_rationale_text(row.target_text, config)[1] for row in rows]
    teacher_counts = Counter(
        (row.teacher.provider, row.teacher.model_id, row.teacher.model_revision)
        for row in rows
    )
    manifest = {
        "schema_version": _MANIFEST_SCHEMA,
        "row_schema_version": _ROW_SCHEMA,
        "candidate_config_sha256": config.sha256,
        "candidate_config_file_sha256": config_file_digest,
        "student_model_id": OFFICIAL_MODEL_ID,
        "student_revision": OFFICIAL_REVISION,
        "training_target_kind": config.training_target_kind,
        "partition": "fold_training",
        "split_partition": "cross_validation",
        "fold": fold,
        "split_version": split_manifest.version,
        "split_sha256": split_manifest.sha256,
        "source_groups_sha256": split_manifest.source_groups_sha256,
        "excluded_ids_sha256": excluded_digest,
        "training_ids_sha256": _ids_sha256(expected_ids),
        "problem_count": len(expected_ids),
        "record_count": len(rows),
        "records_file": records_target.name,
        "records_bytes": len(records_bytes),
        "records_sha256": records_digest,
        "answer_verified_count": len(rows),
        "reference_answer_in_prompt_true_count": sum(
            row.teacher.reference_answer_in_prompt for row in rows
        ),
        "parser_status_counts": {"ok": len(rows)},
        "verification_method_counts": dict(
            sorted(Counter(row.verification.method for row in rows).items())
        ),
        "teacher_identity_counts": [
            {
                "provider": identity[0],
                "model_id": identity[1],
                "model_revision": identity[2],
                "count": count,
            }
            for identity, count in sorted(teacher_counts.items())
        ],
        "teacher_evidence_sequence_sha256": hashlib.sha256(
            canonical_json_bytes([row.teacher.as_dict() for row in rows])
        ).hexdigest(),
        "target_sha256_sequence_sha256": hashlib.sha256(
            canonical_json_bytes([row.target_sha256 for row in rows])
        ).hexdigest(),
        "rationale_character_count": _integer_summary(rationale_lengths),
        "leaderboard_or_test_used": False,
        "locked_holdout_accessed": False,
        "python_or_sympy_tool_used": False,
    }
    manifest_bytes = (
        json.dumps(
            manifest,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")
    _publish_pair_noreplace(
        records_target,
        records_bytes,
        manifest_target,
        manifest_bytes,
    )
    return RationaleArtifactWriteResult(
        records_path=str(records_target),
        records_sha256=records_digest,
        manifest_path=str(manifest_target),
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        record_count=len(rows),
    )


def load_verified_rationale_corpus(
    records_path: str | Path,
    manifest_path: str | Path,
    records: Iterable[MathRecord],
    *,
    split_manifest: SplitManifest,
    fold: int,
    excluded_ids: Iterable[str],
    candidate_config_file_sha256: str,
    audit_path: str | Path | None = None,
    config: ConciseRationaleConfig = DEFAULT_CONCISE_RATIONALE_CONFIG,
) -> RationaleCorpusEvidence:
    """Recompute every corpus invariant and optionally require its redacted audit."""

    expected_ids, record_by_id, excluded_digest = _training_contract(
        records, split_manifest=split_manifest, fold=fold, excluded_ids=excluded_ids
    )
    source, manifest, manifest_digest = _load_json_object(
        manifest_path, "rationale corpus manifest"
    )
    records_source = _regular_file(records_path, "rationale corpus records")
    expected_manifest = {
        "schema_version": _MANIFEST_SCHEMA,
        "row_schema_version": _ROW_SCHEMA,
        "candidate_config_sha256": config.sha256,
        "candidate_config_file_sha256": _required_sha256(
            candidate_config_file_sha256, "candidate_config_file_sha256"
        ),
        "student_model_id": OFFICIAL_MODEL_ID,
        "student_revision": OFFICIAL_REVISION,
        "training_target_kind": config.training_target_kind,
        "partition": "fold_training",
        "split_partition": "cross_validation",
        "fold": fold,
        "split_version": split_manifest.version,
        "split_sha256": split_manifest.sha256,
        "source_groups_sha256": split_manifest.source_groups_sha256,
        "excluded_ids_sha256": excluded_digest,
        "training_ids_sha256": _ids_sha256(expected_ids),
        "problem_count": len(expected_ids),
        "record_count": len(expected_ids),
        "records_file": records_source.name,
        "records_bytes": records_source.stat().st_size,
        "records_sha256": sha256_file(records_source),
        "answer_verified_count": len(expected_ids),
        "reference_answer_in_prompt_true_count": 0,
        "parser_status_counts": {"ok": len(expected_ids)},
        "leaderboard_or_test_used": False,
        "locked_holdout_accessed": False,
        "python_or_sympy_tool_used": False,
    }
    mismatched = [
        key for key, expected in expected_manifest.items() if manifest.get(key) != expected
    ]
    if mismatched:
        raise RationaleCorpusValidationError(
            f"rationale corpus manifest binding mismatch: {mismatched!r}"
        )
    if source.parent != records_source.parent:
        raise RationaleCorpusValidationError(
            "rationale corpus records and manifest must share one directory"
        )
    rows = _load_rows(
        records_source,
        expected_ids=expected_ids,
        record_by_id=record_by_id,
        config=config,
    )
    computed_methods = dict(
        sorted(Counter(row.verification.method for row in rows).items())
    )
    computed_teacher_counts = Counter(
        (row.teacher.provider, row.teacher.model_id, row.teacher.model_revision)
        for row in rows
    )
    expected_teacher_counts = [
        {
            "provider": identity[0],
            "model_id": identity[1],
            "model_revision": identity[2],
            "count": count,
        }
        for identity, count in sorted(computed_teacher_counts.items())
    ]
    rationale_lengths = [_rationale_text(row.target_text, config)[1] for row in rows]
    expected_teacher_sequence = hashlib.sha256(
        canonical_json_bytes([row.teacher.as_dict() for row in rows])
    ).hexdigest()
    expected_target_sequence = hashlib.sha256(
        canonical_json_bytes([row.target_sha256 for row in rows])
    ).hexdigest()
    derived_manifest = {
        "verification_method_counts": computed_methods,
        "teacher_identity_counts": expected_teacher_counts,
        "teacher_evidence_sequence_sha256": expected_teacher_sequence,
        "target_sha256_sequence_sha256": expected_target_sequence,
        "rationale_character_count": _integer_summary(rationale_lengths),
    }
    derived_mismatches = [
        key for key, expected in derived_manifest.items() if manifest.get(key) != expected
    ]
    if derived_mismatches:
        raise RationaleCorpusValidationError(
            "rationale corpus derived manifest fields do not match rows: "
            f"{derived_mismatches!r}"
        )
    expected_manifest_keys = set(expected_manifest) | set(derived_manifest)
    if set(manifest) != expected_manifest_keys:
        raise RationaleCorpusValidationError(
            "rationale corpus manifest keys differ from the locked schema"
        )
    evidence = RationaleCorpusEvidence(
        records_path=str(records_source),
        records_sha256=expected_manifest["records_sha256"],
        manifest_path=str(source),
        manifest_sha256=manifest_digest,
        candidate_config_sha256=config.sha256,
        candidate_config_file_sha256=expected_manifest[
            "candidate_config_file_sha256"
        ],
        fold=fold,
        training_ids_sha256=_ids_sha256(expected_ids),
        record_count=len(rows),
        rows=rows,
    )
    if audit_path is None:
        return evidence
    audit_source, audit, audit_digest = _load_json_object(
        audit_path, "rationale corpus audit"
    )
    line_counts = [
        _rationale_text(row.target_text, config)[0].count("\n") + 1
        for row in rows
    ]
    expected_audit = {
        "schema_version": _AUDIT_SCHEMA,
        "status": "green",
        "records_sha256": evidence.records_sha256,
        "manifest_sha256": evidence.manifest_sha256,
        "candidate_config_sha256": config.sha256,
        "candidate_config_file_sha256": evidence.candidate_config_file_sha256,
        "fold": fold,
        "partition": "fold_training",
        "split_partition": "cross_validation",
        "split_sha256": split_manifest.sha256,
        "source_groups_sha256": split_manifest.source_groups_sha256,
        "record_count": len(rows),
        "training_ids_sha256": evidence.training_ids_sha256,
        "answer_verified_count": len(rows),
        "reference_answer_in_prompt_true_count": sum(
            row.teacher.reference_answer_in_prompt for row in rows
        ),
        "verification_method_counts": computed_methods,
        "teacher_identity_count": len(computed_teacher_counts),
        "rationale_character_count": _integer_summary(rationale_lengths),
        "rationale_line_count": _integer_summary(line_counts),
        "raw_rationale_serialized": False,
        "problem_id_serialized": False,
        "question_serialized": False,
        "reference_answer_serialized": False,
        "parsed_integer_value_serialized": False,
        "teacher_prompt_serialized": False,
        "leaderboard_or_test_used": False,
        "locked_holdout_accessed": False,
        "python_or_sympy_tool_used": False,
    }
    audit_mismatches = [
        key for key, expected in expected_audit.items() if audit.get(key) != expected
    ]
    if audit_mismatches:
        raise RationaleCorpusValidationError(
            f"rationale corpus audit binding mismatch: {audit_mismatches!r}"
        )
    if set(audit) != {*expected_audit, "payload_sha256"}:
        raise RationaleCorpusValidationError(
            "rationale corpus audit keys differ from the locked schema"
        )
    payload_sha256 = audit.get("payload_sha256")
    if not isinstance(payload_sha256, str):
        raise RationaleCorpusValidationError("rationale audit is missing payload_sha256")
    payload_without_hash = dict(audit)
    payload_without_hash.pop("payload_sha256")
    if hashlib.sha256(canonical_json_bytes(payload_without_hash)).hexdigest() != payload_sha256:
        raise RationaleCorpusValidationError("rationale audit payload SHA is invalid")
    return RationaleCorpusEvidence(
        records_path=evidence.records_path,
        records_sha256=evidence.records_sha256,
        manifest_path=evidence.manifest_path,
        manifest_sha256=evidence.manifest_sha256,
        candidate_config_sha256=evidence.candidate_config_sha256,
        candidate_config_file_sha256=evidence.candidate_config_file_sha256,
        fold=evidence.fold,
        training_ids_sha256=evidence.training_ids_sha256,
        record_count=evidence.record_count,
        rows=evidence.rows,
        audit_path=str(audit_source),
        audit_sha256=audit_digest,
    )


def audit_rationale_corpus(
    records_path: str | Path,
    manifest_path: str | Path,
    records: Iterable[MathRecord],
    *,
    split_manifest: SplitManifest,
    fold: int,
    excluded_ids: Iterable[str],
    candidate_config_file_sha256: str,
    output_path: str | Path,
    config: ConciseRationaleConfig = DEFAULT_CONCISE_RATIONALE_CONFIG,
) -> RationaleAuditWriteResult:
    """Emit a no-raw audit that a later training command must re-hash."""

    evidence = load_verified_rationale_corpus(
        records_path,
        manifest_path,
        records,
        split_manifest=split_manifest,
        fold=fold,
        excluded_ids=excluded_ids,
        candidate_config_file_sha256=candidate_config_file_sha256,
        config=config,
    )
    rationale_lengths = [
        _rationale_text(row.target_text, config)[1] for row in evidence.rows
    ]
    line_counts = [
        _rationale_text(row.target_text, config)[0].count("\n") + 1
        for row in evidence.rows
    ]
    payload = {
        "schema_version": _AUDIT_SCHEMA,
        "status": "green",
        "records_sha256": evidence.records_sha256,
        "manifest_sha256": evidence.manifest_sha256,
        "candidate_config_sha256": config.sha256,
        "candidate_config_file_sha256": evidence.candidate_config_file_sha256,
        "fold": fold,
        "partition": "fold_training",
        "split_partition": "cross_validation",
        "split_sha256": split_manifest.sha256,
        "source_groups_sha256": split_manifest.source_groups_sha256,
        "record_count": evidence.record_count,
        "training_ids_sha256": evidence.training_ids_sha256,
        "answer_verified_count": evidence.record_count,
        "reference_answer_in_prompt_true_count": sum(
            row.teacher.reference_answer_in_prompt for row in evidence.rows
        ),
        "verification_method_counts": dict(
            sorted(Counter(row.verification.method for row in evidence.rows).items())
        ),
        "teacher_identity_count": len(
            {
                (
                    row.teacher.provider,
                    row.teacher.model_id,
                    row.teacher.model_revision,
                )
                for row in evidence.rows
            }
        ),
        "rationale_character_count": _integer_summary(rationale_lengths),
        "rationale_line_count": _integer_summary(line_counts),
        "raw_rationale_serialized": False,
        "problem_id_serialized": False,
        "question_serialized": False,
        "reference_answer_serialized": False,
        "parsed_integer_value_serialized": False,
        "teacher_prompt_serialized": False,
        "leaderboard_or_test_used": False,
        "locked_holdout_accessed": False,
        "python_or_sympy_tool_used": False,
    }
    return _write_hashed_json_noreplace(output_path, payload)


def _training_contract(
    records: Iterable[MathRecord],
    *,
    split_manifest: SplitManifest,
    fold: int,
    excluded_ids: Iterable[str],
) -> tuple[tuple[str, ...], dict[str, MathRecord], str]:
    if not isinstance(split_manifest, SplitManifest):
        raise TypeError("split_manifest must be a SplitManifest")
    try:
        split_manifest.validate()
        exclusions = _validated_train_ids(excluded_ids, "excluded_ids", allow_empty=True)
        expected_ids = eligible_training_ids(split_manifest, fold, exclusions)
    except SplitValidationError as exc:
        raise RationaleCorpusValidationError(f"invalid split boundary: {exc}") from exc
    materialized = tuple(records)
    if not materialized or any(not isinstance(record, MathRecord) for record in materialized):
        raise RationaleCorpusValidationError(
            "records must contain organizer fold-training MathRecord values"
        )
    record_by_id = {record.id: record for record in materialized}
    if len(record_by_id) != len(materialized):
        raise RationaleCorpusValidationError("organizer records contain duplicate IDs")
    missing = sorted(set(expected_ids) - set(record_by_id))
    extra = sorted(set(record_by_id) - set(expected_ids))
    if missing or extra:
        raise RationaleCorpusValidationError(
            "organizer records must exactly match eligible fold-training IDs; "
            f"missing={missing[:5]!r}, extra={extra[:5]!r}"
        )
    for problem_id, record in record_by_id.items():
        if record.answer is None or isinstance(record.answer, bool):
            raise RationaleCorpusValidationError(
                f"{problem_id}: organizer reference answer is missing"
            )
    return expected_ids, record_by_id, _ids_sha256(exclusions)


def _load_rows(
    path: str | Path,
    *,
    expected_ids: Sequence[str],
    record_by_id: Mapping[str, MathRecord],
    config: ConciseRationaleConfig,
) -> tuple[RationaleCorpusRow, ...]:
    source = _regular_file(path, "rationale JSONL")
    rows_by_id: dict[str, RationaleCorpusRow] = {}
    try:
        with source.open("r", encoding="utf-8", errors="strict", newline="") as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                if not raw_line.endswith("\n"):
                    raise RationaleCorpusValidationError(
                        f"rationale JSONL line {line_number} is not newline-terminated"
                    )
                line = raw_line[:-1]
                if not line:
                    raise RationaleCorpusValidationError(
                        f"rationale JSONL line {line_number} is empty"
                    )
                try:
                    payload = json.loads(
                        line,
                        object_pairs_hook=_unique_json_object,
                        parse_constant=_reject_json_constant,
                    )
                except (json.JSONDecodeError, ValueError) as exc:
                    raise RationaleCorpusValidationError(
                        f"invalid rationale JSONL line {line_number}: {exc}"
                    ) from exc
                row = _parse_row(payload, record_by_id=record_by_id, config=config)
                if row.problem_id in rows_by_id:
                    raise RationaleCorpusValidationError(
                        f"duplicate rationale problem_id: {row.problem_id}"
                    )
                rows_by_id[row.problem_id] = row
    except (OSError, UnicodeError) as exc:
        raise RationaleCorpusValidationError(f"cannot read rationale JSONL: {exc}") from exc
    missing = sorted(set(expected_ids) - set(rows_by_id))
    extra = sorted(set(rows_by_id) - set(expected_ids))
    if missing or extra:
        raise RationaleCorpusValidationError(
            "rationale rows must exactly cover eligible fold-training IDs; "
            f"missing={missing[:5]!r}, extra={extra[:5]!r}"
        )
    return tuple(rows_by_id[problem_id] for problem_id in expected_ids)


def _parse_row(
    payload: object,
    *,
    record_by_id: Mapping[str, MathRecord],
    config: ConciseRationaleConfig,
) -> RationaleCorpusRow:
    if not isinstance(payload, dict):
        raise RationaleCorpusValidationError("rationale JSONL rows must be objects")
    required_keys = {
        "schema_version",
        "problem_id",
        "question_sha256",
        "target_text",
        "target_sha256",
        "teacher",
        "verification",
    }
    if set(payload) != required_keys:
        raise RationaleCorpusValidationError(
            "rationale row keys differ from the locked schema"
        )
    if payload["schema_version"] != _ROW_SCHEMA:
        raise RationaleCorpusValidationError("rationale row schema_version is invalid")
    problem_id = _trimmed_string(payload["problem_id"], "problem_id")
    if _TRAIN_ID_RE.fullmatch(problem_id) is None or problem_id not in record_by_id:
        raise RationaleCorpusValidationError(
            f"rationale row is not an eligible organizer train ID: {problem_id!r}"
        )
    record = record_by_id[problem_id]
    expected_question_sha = hashlib.sha256(record.question_raw.encode("utf-8")).hexdigest()
    question_sha = _required_sha256(payload["question_sha256"], "question_sha256")
    if question_sha != expected_question_sha:
        raise RationaleCorpusValidationError(
            f"{problem_id}: question SHA does not match organizer bytes"
        )
    target_text = _trimmed_string(payload["target_text"], "target_text")
    _validate_text_controls(target_text, problem_id)
    rationale, _ = _rationale_text(target_text, config)
    target_sha = _required_sha256(payload["target_sha256"], "target_sha256")
    if target_sha != hashlib.sha256(target_text.encode("utf-8")).hexdigest():
        raise RationaleCorpusValidationError(f"{problem_id}: target SHA mismatch")
    answer = record.answer
    assert isinstance(answer, int) and not isinstance(answer, bool)
    final_match = _FINAL_LINE_RE.search(target_text)
    if final_match is None or final_match.group(1) != str(answer):
        raise RationaleCorpusValidationError(
            f"{problem_id}: final line does not exactly match the organizer answer"
        )
    if len(_FINAL_MARKER_RE.findall(target_text)) != 1:
        raise RationaleCorpusValidationError(
            f"{problem_id}: target must contain exactly one final-answer marker"
        )
    parsed = parse_answer(target_text)
    if not parsed.ok or parsed.value != answer or parsed.source != "final_answer":
        raise RationaleCorpusValidationError(
            f"{problem_id}: target parser result is not the exact organizer answer"
        )
    if not rationale:
        raise RationaleCorpusValidationError(f"{problem_id}: rationale is empty")
    teacher = _parse_teacher(payload["teacher"], config=config)
    verification = _parse_verification(payload["verification"], config=config)
    return RationaleCorpusRow(
        problem_id=problem_id,
        question_sha256=question_sha,
        target_text=target_text,
        target_sha256=target_sha,
        teacher=teacher,
        verification=verification,
    )


def _parse_teacher(
    value: object, *, config: ConciseRationaleConfig
) -> RationaleTeacherEvidence:
    if not isinstance(value, dict):
        raise RationaleCorpusValidationError("teacher must be an object")
    expected_keys = {
        "provider",
        "model_id",
        "model_revision",
        "prompt_sha256",
        "generation_config_sha256",
        "seed",
        "sample_index",
        "raw_generation_sha256",
        "reference_answer_in_prompt",
        "network_scope",
    }
    if set(value) != expected_keys:
        raise RationaleCorpusValidationError("teacher keys differ from the locked schema")
    seed = _non_negative_integer(value["seed"], "teacher.seed")
    sample_index = _non_negative_integer(
        value["sample_index"], "teacher.sample_index"
    )
    network_scope = _trimmed_string(value["network_scope"], "teacher.network_scope")
    if network_scope != "training_only":
        raise RationaleCorpusValidationError(
            "teacher network_scope must be exactly 'training_only'"
        )
    reference_answer_in_prompt = value["reference_answer_in_prompt"]
    if type(reference_answer_in_prompt) is not bool:
        raise RationaleCorpusValidationError(
            "teacher.reference_answer_in_prompt must be a boolean"
        )
    if config.require_reference_answer_hidden_from_teacher and reference_answer_in_prompt:
        raise RationaleCorpusValidationError(
            "teacher reference answer must remain hidden from the generation prompt"
        )
    return RationaleTeacherEvidence(
        provider=_trimmed_string(value["provider"], "teacher.provider"),
        model_id=_trimmed_string(value["model_id"], "teacher.model_id"),
        model_revision=_trimmed_string(
            value["model_revision"], "teacher.model_revision"
        ),
        prompt_sha256=_required_sha256(
            value["prompt_sha256"], "teacher.prompt_sha256"
        ),
        generation_config_sha256=_required_sha256(
            value["generation_config_sha256"],
            "teacher.generation_config_sha256",
        ),
        seed=seed,
        sample_index=sample_index,
        raw_generation_sha256=_required_sha256(
            value["raw_generation_sha256"], "teacher.raw_generation_sha256"
        ),
        reference_answer_in_prompt=reference_answer_in_prompt,
        network_scope=network_scope,
    )


def _parse_verification(
    value: object, *, config: ConciseRationaleConfig
) -> RationaleVerification:
    if not isinstance(value, dict):
        raise RationaleCorpusValidationError("verification must be an object")
    expected_keys = {
        "status",
        "method",
        "leaderboard_or_test_used",
        "locked_holdout_accessed",
        "tool_used",
    }
    if set(value) != expected_keys:
        raise RationaleCorpusValidationError(
            "verification keys differ from the locked schema"
        )
    status = _trimmed_string(value["status"], "verification.status")
    method = _trimmed_string(value["method"], "verification.method")
    if status != "accepted":
        raise RationaleCorpusValidationError("verification.status must be accepted")
    if method not in config.allowed_verification_methods:
        raise RationaleCorpusValidationError(
            f"unsupported rationale verification method: {method!r}"
        )
    booleans: dict[str, bool] = {}
    for field_name in (
        "leaderboard_or_test_used",
        "locked_holdout_accessed",
        "tool_used",
    ):
        item = value[field_name]
        if type(item) is not bool:
            raise RationaleCorpusValidationError(
                f"verification.{field_name} must be a boolean"
            )
        if item:
            raise RationaleCorpusValidationError(
                f"verification.{field_name} must remain false"
            )
        booleans[field_name] = item
    return RationaleVerification(status=status, method=method, **booleans)


def _rationale_text(
    target_text: str, config: ConciseRationaleConfig
) -> tuple[str, int]:
    final_match = _FINAL_LINE_RE.search(target_text)
    if final_match is None:
        raise RationaleCorpusValidationError(
            "target must end with a canonical 'Final answer: <integer>' line"
        )
    rationale = target_text[: final_match.start()].rstrip("\n")
    length = len(rationale)
    if not config.min_rationale_characters <= length <= config.max_rationale_characters:
        raise RationaleCorpusValidationError(
            "rationale character count is outside the locked concise range"
        )
    if rationale.count("\n") + 1 > config.max_rationale_lines:
        raise RationaleCorpusValidationError(
            "rationale line count exceeds the locked concise limit"
        )
    return rationale, length


def _integer_summary(values: Sequence[int]) -> dict[str, int | float]:
    if not values:
        raise RationaleCorpusValidationError("cannot summarize an empty corpus")
    return {
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def _artifact_targets(
    records_path: str | Path, manifest_path: str | Path
) -> tuple[Path, Path]:
    records = _new_file_target(records_path, "rationale corpus records")
    manifest = _new_file_target(manifest_path, "rationale corpus manifest")
    if records == manifest:
        raise RationaleCorpusValidationError(
            "rationale records and manifest paths must differ"
        )
    if records.parent != manifest.parent:
        raise RationaleCorpusValidationError(
            "rationale records and manifest must share one directory"
        )
    return records, manifest


def _new_file_target(path: str | Path, label: str) -> Path:
    raw = Path(path)
    if raw.is_symlink() or raw.parent.is_symlink():
        raise RationaleCorpusValidationError(f"{label} refuses symbolic links")
    target = raw.resolve(strict=False)
    if not target.parent.is_dir():
        raise RationaleCorpusValidationError(f"{label} parent does not exist")
    if target.exists():
        raise RationaleCorpusArtifactExistsError(f"refusing to overwrite {label}: {target}")
    return target


def _publish_pair_noreplace(
    first_target: Path,
    first_payload: bytes,
    second_target: Path,
    second_payload: bytes,
) -> None:
    descriptor_one, raw_one = tempfile.mkstemp(
        prefix=f".{first_target.name}.", suffix=".tmp", dir=first_target.parent
    )
    descriptor_two, raw_two = tempfile.mkstemp(
        prefix=f".{second_target.name}.", suffix=".tmp", dir=second_target.parent
    )
    temporary_one = Path(raw_one)
    temporary_two = Path(raw_two)
    published_first = False
    try:
        for descriptor, _temporary, payload in (
            (descriptor_one, temporary_one, first_payload),
            (descriptor_two, temporary_two, second_payload),
        ):
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        try:
            os.link(temporary_one, first_target)
            published_first = True
            os.link(temporary_two, second_target)
        except FileExistsError as exc:
            if published_first:
                first_target.unlink()
                _fsync_directory(first_target.parent)
            raise RationaleCorpusArtifactExistsError(
                "refusing to overwrite rationale corpus pair"
            ) from exc
        except BaseException:
            if published_first:
                with suppress(FileNotFoundError):
                    first_target.unlink()
                _fsync_directory(first_target.parent)
            raise
        _fsync_directory(first_target.parent)
    finally:
        for temporary in (temporary_one, temporary_two):
            with suppress(FileNotFoundError):
                temporary.unlink()


def _write_hashed_json_noreplace(
    path: str | Path, payload_without_hash: Mapping[str, Any]
) -> RationaleAuditWriteResult:
    target = _new_file_target(path, "rationale corpus audit")
    payload_sha256 = hashlib.sha256(
        canonical_json_bytes(payload_without_hash)
    ).hexdigest()
    serialized = (
        json.dumps(
            {**dict(payload_without_hash), "payload_sha256": payload_sha256},
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, raw_temp = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(raw_temp)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            raise RationaleCorpusArtifactExistsError(
                f"refusing to overwrite rationale audit: {target}"
            ) from exc
        _fsync_directory(target.parent)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()
    return RationaleAuditWriteResult(
        path=str(target),
        sha256=hashlib.sha256(serialized).hexdigest(),
        payload_sha256=payload_sha256,
        size_bytes=len(serialized),
        record_count=int(payload_without_hash["record_count"]),
    )


def _regular_file(path: str | Path, label: str) -> Path:
    raw = Path(path)
    if raw.is_symlink():
        raise RationaleCorpusValidationError(f"{label} must not be a symbolic link")
    source = raw.resolve(strict=True)
    if not source.is_file():
        raise RationaleCorpusValidationError(f"{label} must be a regular file")
    return source


def _load_json_object(
    path: str | Path, label: str
) -> tuple[Path, dict[str, Any], str]:
    source = _regular_file(path, label)
    try:
        payload = json.loads(
            source.read_text(encoding="utf-8", errors="strict"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RationaleCorpusValidationError(f"cannot load {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RationaleCorpusValidationError(f"{label} must contain one JSON object")
    return source, payload, sha256_file(source)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON key {key!r}")
        payload[key] = value
    return payload


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON numeric constant {value!r}")


def _trimmed_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise RationaleCorpusValidationError(f"{label} must be a non-empty trimmed string")
    return value


def _required_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RationaleCorpusValidationError(
            f"{label} must be a lowercase 64-character SHA-256"
        )
    return value


def _non_negative_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RationaleCorpusValidationError(f"{label} must be a non-negative integer")
    return value


def _validated_train_ids(
    values: Iterable[str], label: str, *, allow_empty: bool = False
) -> tuple[str, ...]:
    materialized = tuple(values)
    if not allow_empty and not materialized:
        raise RationaleCorpusValidationError(f"{label} must not be empty")
    if any(
        not isinstance(value, str) or _TRAIN_ID_RE.fullmatch(value) is None
        for value in materialized
    ):
        raise RationaleCorpusValidationError(f"{label} contains a non-train ID")
    if len(set(materialized)) != len(materialized):
        raise RationaleCorpusValidationError(f"{label} contains duplicate IDs")
    return tuple(sorted(materialized))


def _validate_text_controls(value: str, problem_id: str) -> None:
    for character in value:
        if ord(character) < 32 and character not in _ALLOWED_CONTROL_CHARACTERS:
            raise RationaleCorpusValidationError(
                f"{problem_id}: rationale contains a forbidden control character"
            )


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
    "ConciseRationaleConfig",
    "DEFAULT_CONCISE_RATIONALE_CONFIG",
    "RationaleArtifactWriteResult",
    "RationaleAuditWriteResult",
    "RationaleCorpusArtifactExistsError",
    "RationaleCorpusEvidence",
    "RationaleCorpusRow",
    "RationaleCorpusValidationError",
    "audit_rationale_corpus",
    "build_rationale_corpus",
    "load_concise_rationale_config",
    "load_verified_rationale_corpus",
]

"""Command-line entry points for the offline competition toolkit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .answers import parse_answer
from .audit import (
    AUDIT_VERSION,
    FILTERED_AUDIT_VERSION,
    build_data_audit_report_from_datasets,
)
from .data import (
    CsvDataset,
    MathRecord,
    TrainExclusionSet,
    load_leaderboard_csv,
    load_train_csv,
    load_train_exclusions_csv,
)
from .development_shard import (
    build_development_cv_shard,
    load_development_cv_shard,
)
from .gate_b import (
    DEFAULT_GATE_B_CONFIG,
    GateBConfig,
    create_development_execution_evidence,
    read_development_resume_status,
    run_development_baseline,
    write_development_artifacts,
)
from .gate_b_holdout import evaluate_locked_holdout_once
from .gate_b_prediction import run_frozen_evaluation_inference
from .gate_b_runtime import (
    BASE_MODEL_CHECKPOINT_SHA256,
    GPU_EXECUTION_ACKNOWLEDGEMENT,
    RuntimeGateEvidence,
    create_adapted_development_backend,
    create_base_development_backend,
    read_training_resume_status,
    train_qlora_fold,
)
from .gate_b_selection import (
    ADAPTER_METHOD_KIND,
    FIXED_BASE_METHOD_KIND,
    HOLDOUT_ACCESS_ACKNOWLEDGEMENT,
    FoldDevelopmentRun,
    NamedDevelopmentRun,
    compare_cross_fold_development_runs,
    compare_development_runs,
    decide_candidate_probe_promotion,
    freeze_base_development_selection,
    freeze_development_selection,
    require_base_development_artifact,
    verify_base_development_oof,
)
from .gate_b_sft_preflight import run_sft_encoding_preflight
from .gpu_smoke import run_final_gpu_smoke
from .independent_submission import verify_submission_independently
from .model_preflight import OFFICIAL_MODEL_ID, OFFICIAL_REVISION, run_model_preflight
from .parser_golden import (
    audit_development_parser_golden,
    audit_development_parser_rescore,
)
from .provenance import (
    SourceTreeArtifactEvidence,
    build_source_tree_manifest,
    canonical_json_bytes,
    sha256_file,
    validate_source_tree_manifest_artifact,
    write_json_atomic,
)
from .quality import (
    assess_question,
    math_aware_fingerprint,
    source_format_fingerprint,
)
from .rationale_corpus import (
    ConciseRationaleConfig,
    RationaleCorpusEvidence,
    audit_rationale_corpus,
    build_rationale_corpus,
    load_concise_rationale_config,
    load_verified_rationale_corpus,
)
from .rationale_materialization import (
    FinalizedTeacherBank,
    materialize_teacher_bank_source,
)
from .splits import (
    SplitManifest,
    build_group_clusters,
    eligible_training_ids,
    eligible_validation_ids,
    expand_hard_group_exclusions,
    make_grouped_split_manifest,
)
from .submission import SubmissionSchema, validate_submission_csv, write_submission_csv
from .teacher_harness import (
    HARNESS_AUTHORIZATION_FILENAME,
    HARNESS_CONFIG_SCHEMA,
    HARNESS_CONFIG_V2_SCHEMA,
    HarnessProfile,
    create_harness_authorization,
    diagnose_teacher_ledger,
    profile_from_config,
    require_harness_live_execution_matches,
    run_harness_live,
    run_harness_replay,
    validate_harness_evidence,
    verify_harness_authorization,
)
from .teacher_pilot_authorization import (
    PILOT_AUTHORIZATION_INITIAL_EXACT_MATCH_PERCENT,
    PILOT_AUTHORIZATION_LOGICAL_AUDIT_MIN_CONSISTENT,
    PILOT_AUTHORIZATION_LOGICAL_AUDIT_SAMPLE_SIZE,
    TeacherPilotAuthorizationContract,
    create_teacher_pilot_authorization,
    verify_teacher_pilot_authorization,
    write_teacher_full_v1_bank_authorization,
)
from .teacher_rationale import (
    CodexCommandResult,
    TeacherBankFinalizeResult,
    TeacherExecutionConfig,
    TeacherLogicalAuditPlan,
    TeacherPlan,
    TeacherPromptPolicy,
    create_teacher_logical_audit_plan,
    create_teacher_plan,
    finalize_teacher_bank,
    finalize_teacher_logical_audit,
    load_teacher_logical_audit_plan,
    load_teacher_plan,
    run_teacher_logical_audit,
    run_teacher_plan,
    teacher_logical_audit_status,
    teacher_status,
)
from .tokenizer_profile import (
    DEFAULT_SYSTEM_PROMPT,
    load_and_profile_datasets,
    load_pinned_tokenizer,
)
from .workflow_status import (
    development_workflow_status,
    training_workflow_status,
)

_HARD_CLUSTER_METHOD = (
    "transitive union of math-aware and narrowly source-format-insensitive exact "
    "fingerprints; number-masked templates remain soft audit candidates"
)

_LOCKED_CODEX_TEACHER_CONFIG: dict[str, object] = {
    "schema_version": "gate-b-codex-teacher-config-v1",
    "label": "codex-gpt-5.6-sol-teacher",
    "version": "v1",
    "provider": "chatgpt_codex_cli",
    "model_id": "gpt-5.6-sol",
    "model_revision": "gpt-5.6-sol",
    "initial_reasoning_effort": "high",
    "repair_reasoning_effort": "xhigh",
    "seed": 20_260_731,
    "initial_chunk_size": 64,
    "repair_chunk_size": 16,
    "max_attempts": 3,
    "pilot_size": 128,
    "max_concurrent_workers": 2,
    "reference_answer_in_prompt": False,
    "allow_tool_use": False,
    "network_scope": "training_only",
}

# This profile is intentionally separate from the historic v1 teacher ledger
# and from the later ``gate-b-teacher-v2-*`` development-CV expansion commands.
# It creates a fresh 128-row pilot with the same safety limits but a versioned
# question-only prompt and smaller, immutable initial chunks.
_LOCKED_CODEX_TEACHER_PILOT_V2_CONFIG: dict[str, object] = {
    "schema_version": "gate-b-codex-teacher-pilot-config-v2",
    "label": "codex-gpt-5.6-sol-teacher-pilot-v2",
    "version": "pilot-v2",
    "provider": "chatgpt_codex_cli",
    "model_id": "gpt-5.6-sol",
    "model_revision": "gpt-5.6-sol",
    "initial_reasoning_effort": "high",
    "repair_reasoning_effort": "xhigh",
    "seed": 20_260_731,
    "initial_chunk_size": 32,
    "repair_chunk_size": 16,
    "max_attempts": 3,
    "pilot_size": 128,
    "max_concurrent_workers": 2,
    "reference_answer_in_prompt": False,
    "allow_tool_use": False,
    "network_scope": "training_only",
    "prompt_version": "gate-b-codex-teacher-prompt-v2",
    "prompt_template_sha256": "743fb09547055475a8d73856859e9f068d6332cdb2a2bcd9802052c3d5b917b0",
}

# V3 changes one quality variable only: the teacher must derive and then
# independently verify each signed integer before emitting the unchanged JSON
# contract.  Scheduling, isolation, retry, and no-tool boundaries remain v2-identical.
_LOCKED_CODEX_TEACHER_PILOT_V3_CONFIG: dict[str, object] = {
    "schema_version": "gate-b-codex-teacher-pilot-config-v3",
    "label": "codex-gpt-5.6-sol-teacher-pilot-v3",
    "version": "pilot-v3",
    "provider": "chatgpt_codex_cli",
    "model_id": "gpt-5.6-sol",
    "model_revision": "gpt-5.6-sol",
    "initial_reasoning_effort": "high",
    "repair_reasoning_effort": "xhigh",
    "seed": 20_260_731,
    "initial_chunk_size": 32,
    "repair_chunk_size": 16,
    "max_attempts": 3,
    "pilot_size": 128,
    "max_concurrent_workers": 2,
    "reference_answer_in_prompt": False,
    "allow_tool_use": False,
    "network_scope": "training_only",
    "prompt_version": "gate-b-codex-teacher-prompt-v3",
    "prompt_template_sha256": "cf56fc2c021410337f8be8f5f519912eabf6390aa8892ecd92cac1ced6175c72",
}

# V4 changes only the v3 instruction preflight: before returning the already
# locked JSON schema, the teacher must check cardinality, ID uniqueness, and
# canonical INPUT_JSON order.  Its organizer-data plan is additionally gated
# by an independently reverified synthetic replay/live authorization.
_LOCKED_CODEX_TEACHER_PILOT_V4_CONFIG: dict[str, object] = {
    "schema_version": "gate-b-codex-teacher-pilot-config-v4",
    "label": "codex-gpt-5.6-sol-teacher-pilot-v4",
    "version": "pilot-v4",
    "provider": "chatgpt_codex_cli",
    "model_id": "gpt-5.6-sol",
    "model_revision": "gpt-5.6-sol",
    "initial_reasoning_effort": "high",
    "repair_reasoning_effort": "xhigh",
    "seed": 20_260_731,
    "initial_chunk_size": 32,
    "repair_chunk_size": 16,
    "max_attempts": 3,
    "pilot_size": 128,
    "max_concurrent_workers": 2,
    "reference_answer_in_prompt": False,
    "allow_tool_use": False,
    "network_scope": "training_only",
    "prompt_version": "gate-b-codex-teacher-prompt-v4",
    "prompt_template_sha256": "3029e9297bdda504e0f48e1ce4d57e363e5d3a5342edf18253b11c4f75ecd8a7",
}

_LOCKED_CODEX_TEACHER_CONFIGS = {
    str(_LOCKED_CODEX_TEACHER_CONFIG["schema_version"]): _LOCKED_CODEX_TEACHER_CONFIG,
    str(_LOCKED_CODEX_TEACHER_PILOT_V2_CONFIG["schema_version"]): (
        _LOCKED_CODEX_TEACHER_PILOT_V2_CONFIG
    ),
    str(_LOCKED_CODEX_TEACHER_PILOT_V3_CONFIG["schema_version"]): (
        _LOCKED_CODEX_TEACHER_PILOT_V3_CONFIG
    ),
    str(_LOCKED_CODEX_TEACHER_PILOT_V4_CONFIG["schema_version"]): (
        _LOCKED_CODEX_TEACHER_PILOT_V4_CONFIG
    ),
}

# The synthetic harness has its own fixed config rather than reusing a
# production teacher profile.  It is limited to two answer-free arithmetic
# chunks and cannot create a bank, repair, or retry ledger.
_LOCKED_CODEX_TEACHER_HARNESS_CONFIG: dict[str, object] = {
    "schema_version": "gate-b-codex-teacher-harness-config-v1",
    "label": "codex-gpt-5.6-sol-teacher-harness-v1",
    "version": "harness-v1",
    "provider": "chatgpt_codex_cli",
    "model_id": "gpt-5.6-sol",
    "model_revision": "gpt-5.6-sol",
    "seed": 20_260_731,
    "initial_chunk_size": 32,
    "chunk_count": 2,
    "max_workers": 1,
    "max_invocations": 2,
    "max_attempts": 1,
    "retry_count": 0,
    "repair_count": 0,
    "bank_output_count": 0,
    "initial_reasoning_effort": "high",
    "reference_answer_in_prompt": False,
    "allow_tool_use": False,
    "network_scope": "synthetic_canary_only",
    "fixture_version": "gate-b-codex-teacher-harness-fixture-v1",
    "fixture_sha256": "bc314a24ec872edf26bae13e296fb8fd500f80bcf6c00023518844d1b407e3b7",
}

_LOCKED_CODEX_TEACHER_HARNESS_V2_CONFIG: dict[str, object] = {
    "schema_version": "gate-b-codex-teacher-harness-config-v2",
    "label": "codex-gpt-5.6-sol-teacher-harness-v2",
    "version": "harness-v2",
    "provider": "chatgpt_codex_cli",
    "model_id": "gpt-5.6-sol",
    "model_revision": "gpt-5.6-sol",
    "seed": 20_260_731,
    "initial_chunk_size": 16,
    "chunk_count": 8,
    "max_workers": 1,
    "max_invocations": 8,
    "max_attempts": 1,
    "retry_count": 0,
    "repair_count": 0,
    "bank_output_count": 0,
    "initial_reasoning_effort": "high",
    "reference_answer_in_prompt": False,
    "allow_tool_use": False,
    "network_scope": "synthetic_canary_only",
    "fixture_version": "gate-b-codex-teacher-harness-fixture-v2",
    "fixture_sha256": "dea9b4cc3c3262de831abba2c7ce36bf6ac2612ee215cbccd34c5d4b3d1a3388",
}

_LOCKED_CODEX_TEACHER_HARNESS_CONFIGS = {
    HARNESS_CONFIG_SCHEMA: _LOCKED_CODEX_TEACHER_HARNESS_CONFIG,
    HARNESS_CONFIG_V2_SCHEMA: _LOCKED_CODEX_TEACHER_HARNESS_V2_CONFIG,
}

# A live harness must exercise a policy-bound candidate profile.  Historic v1
# and v2 failure evidence remains readable/diagnosable but cannot be selected
# for a fresh synthetic call.
_HARNESS_ALLOWED_TEACHER_CONFIG_SCHEMAS = frozenset(
    {
        "gate-b-codex-teacher-pilot-config-v3",
        "gate-b-codex-teacher-pilot-config-v4",
    }
)
_V4_TEACHER_CONFIG_SCHEMA = "gate-b-codex-teacher-pilot-config-v4"
_V4_INITIAL_THRESHOLD_FAILURE_SCHEMA = "gate-b-codex-teacher-pilot-v4-initial-threshold-v1"
_V4_INITIAL_THRESHOLD_FAILURE_FILENAME = "initial-threshold-failure-v1.json"
_V4_INITIAL_MIN_ACCEPTED = 103

# This is a separate immutable profile from the rationale-generation profile
# above.  It deliberately has no editable file: the audit must always sample
# exactly 64 accepted teacher rows and require at least 60 internally
# consistent judgments before the pilot can advance.  The model/provider
# contract remains bound to ``_LOCKED_CODEX_TEACHER_CONFIG``.
_LOCKED_CODEX_LOGICAL_AUDIT_PROFILE: dict[str, object] = {
    "schema_version": "gate-b-codex-teacher-logical-audit-cli-v1",
    "label": "codex-gpt-5.6-sol-logical-audit",
    "version": "v1",
    "sample_size": 64,
    "min_consistent": 60,
    "reference_answer_read": False,
    "allow_tool_use": False,
    "network_scope": "training_only",
}

# v2 is a distinct, post-probe expansion scope.  It is intentionally not an
# edit to the v1 teacher profile or plan: v1 covers fold-0 training IDs, while
# v2 can cover only the remaining eligible development-CV IDs after a verified
# candidate-probe decision has authorized completing OOF work.
_CODEX_TEACHER_V2_PLAN_LABEL = "codex-gpt-5.6-sol-teacher-development-v2"
_CODEX_TEACHER_V2_PLAN_VERSION = "v2"
_CODEX_TEACHER_V2_SCOPE = "remaining_development_cv_after_fold0_training"
_CODEX_TEACHER_V2_AUTHORIZATION_SCHEMA = (
    "gate-b-codex-teacher-v2-authorization-v1"
)
_CODEX_TEACHER_V2_AUTHORIZATION_FILENAME = "v2-authorization.json"
_CANDIDATE_PROBE_DECISION_SCHEMA = "gate-b-candidate-probe-decision-v1"

_CODEX_TEACHER_SAFE_ENV_NAMES = frozenset(
    {
        "HOME",
        "LANG",
        "LC_ALL",
        "LOGNAME",
        "PATH",
        "TMPDIR",
        "USER",
    }
)


def _add_dataset_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train", required=True, type=Path, help="official train CSV")
    parser.add_argument(
        "--leaderboard", required=True, type=Path, help="official leaderboard CSV"
    )


def _add_train_exclusion_path(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--train-exclusions",
        type=Path,
        help=(
            "organizer train_filtered_ids CSV; only its ID column is used and the "
            "source train file is never rewritten"
        ),
    )


def _add_split_scope(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--split-artifact",
        type=Path,
        help="validated existing split artifact used to keep the locked holdout sealed",
    )
    parser.add_argument(
        "--train-scope",
        choices=("all", "cv-only", "fold-training", "fold-validation"),
        default=None,
    )
    parser.add_argument("--fold", type=int)
    parser.add_argument(
        "--allow-all-train-scope",
        action="store_true",
        help="explicit pre-lock acknowledgement; never use for post-lock model selection",
    )


def _add_required_gate_b_data_contract(
    parser: argparse.ArgumentParser,
    *,
    require_development_shard: bool = True,
    require_development_shard_sha256: bool = True,
) -> None:
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--train-exclusions", required=True, type=Path)
    parser.add_argument("--split-artifact", required=True, type=Path)
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--expected-train-sha256", required=True)
    parser.add_argument("--expected-exclusions-sha256", required=True)
    parser.add_argument("--expected-exclusion-count", required=True, type=int)
    parser.add_argument("--expected-split-sha256", required=True)
    if require_development_shard_sha256:
        parser.add_argument("--expected-development-shard-sha256", required=True)
    if require_development_shard:
        parser.add_argument("--development-shard", required=True, type=Path)


def _add_teacher_v2_authorization_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the non-optional probe-decision binding for v2 teacher work."""

    parser.add_argument("--candidate-probe-decision", required=True, type=Path)
    parser.add_argument("--candidate-label", required=True)


def _add_teacher_pilot_authorization_arguments(
    parser: argparse.ArgumentParser, *, require_receipt: bool
) -> None:
    """Add explicit private evidence paths for the v1 full-bank promotion gate.

    These paths are intentionally supplied again when a full plan is created:
    the immutable receipt alone cannot detect a source-bank or logical-audit
    artifact that was tampered with after receipt publication.
    """

    if require_receipt:
        parser.add_argument("--pilot-authorization", type=Path)
    parser.add_argument("--pilot-plan-dir", type=Path)
    parser.add_argument("--pilot-source-jsonl", type=Path)
    parser.add_argument("--pilot-source-manifest", type=Path)
    parser.add_argument("--pilot-logical-audit-dir", type=Path)


def _add_optional_teacher_harness_authorization_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    """Accept v4-only synthetic evidence without changing historic invocations.

    V1--v3 commands retain their existing argument contract.  The v4 profile
    treats this complete bundle as mandatory and re-verifies it on every
    organizer-data operation instead of trusting a plan-local hash alone.
    """

    parser.add_argument("--harness-config", type=Path)
    parser.add_argument("--harness-replay-report", type=Path)
    parser.add_argument("--harness-live-report", type=Path)
    parser.add_argument("--harness-live-plan-dir", type=Path)
    parser.add_argument("--harness-source-root", type=Path)
    parser.add_argument("--harness-source-manifest", type=Path)


def _add_gpu_runtime_gate(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--preflight-report", required=True, type=Path)
    parser.add_argument("--gpu-smoke-report", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--acknowledge-gpu-use",
        action="store_true",
        help="required explicit acknowledgement; this command creates a CUDA workload",
    )


def _add_current_source_provenance(parser: argparse.ArgumentParser) -> None:
    """Require a source snapshot that still matches the local source tree."""

    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)


def _add_optional_rationale_training_inputs(parser: argparse.ArgumentParser) -> None:
    """Add the all-or-none private corpus inputs for the rationale candidate."""

    parser.add_argument("--rationale-corpus", type=Path)
    parser.add_argument("--rationale-manifest", type=Path)
    parser.add_argument("--rationale-audit", type=Path)
    parser.add_argument("--rationale-config", type=Path)


def build_parser() -> argparse.ArgumentParser:
    """Build the public CLI parser."""

    parser = argparse.ArgumentParser(
        prog="deep-challenge",
        description="Offline-first audit and submission toolkit",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit-data", help="audit train and leaderboard CSVs")
    _add_dataset_paths(audit)
    _add_train_exclusion_path(audit)
    _add_split_scope(audit)
    audit.add_argument("--output", required=True, type=Path)
    audit.add_argument(
        "--source-root",
        type=Path,
        default=Path.cwd(),
        help="root included in the source-tree provenance manifest",
    )
    audit.set_defaults(handler=_command_audit)

    split = subparsers.add_parser("build-splits", help="build leakage-resistant splits")
    split.add_argument("--train", required=True, type=Path)
    split.add_argument("--output", required=True, type=Path)
    split.add_argument("--folds", type=int, default=5)
    split.add_argument("--holdout-fraction", type=float, default=0.10)
    split.add_argument("--seed", type=int, default=20260731)
    split.add_argument("--version", default="v4")
    split.set_defaults(handler=_command_splits)

    development_shard = subparsers.add_parser(
        "build-development-shard",
        help="atomically seal a CV-only organizer-train shard after split creation",
    )
    development_shard.add_argument("--train", required=True, type=Path)
    development_shard.add_argument("--split-artifact", required=True, type=Path)
    development_shard.add_argument("--expected-train-sha256", required=True)
    development_shard.add_argument("--expected-split-sha256", required=True)
    development_shard.add_argument("--output-dir", required=True, type=Path)
    development_shard.set_defaults(handler=_command_build_development_shard)

    eligibility = subparsers.add_parser(
        "build-eligibility-overlay",
        help="bind organizer train exclusions to an existing locked split artifact",
    )
    eligibility.add_argument("--train", required=True, type=Path)
    eligibility.add_argument("--train-exclusions", required=True, type=Path)
    eligibility.add_argument("--split-artifact", required=True, type=Path)
    eligibility.add_argument("--output", required=True, type=Path)
    eligibility.add_argument("--expected-train-sha256")
    eligibility.add_argument("--expected-exclusions-sha256")
    eligibility.add_argument("--expected-exclusion-count", type=int)
    eligibility.add_argument("--expected-split-sha256")
    eligibility.set_defaults(handler=_command_eligibility_overlay)

    profile = subparsers.add_parser(
        "profile-tokenizer", help="profile exact Qwen tokenizer input lengths"
    )
    _add_dataset_paths(profile)
    _add_train_exclusion_path(profile)
    _add_split_scope(profile)
    profile.add_argument("--output", required=True, type=Path)
    profile.add_argument("--model-id", default="Qwen/Qwen2.5-3B-Instruct")
    profile.add_argument(
        "--revision", required=True, help="immutable 40-hex Hugging Face commit"
    )
    profile.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    profile.add_argument(
        "--allow-download",
        action="store_true",
        help="allow Hugging Face access; default requires an existing local cache",
    )
    profile.set_defaults(handler=_command_profile)

    model_preflight = subparsers.add_parser(
        "model-preflight", help="inspect cached weights and accelerator readiness"
    )
    model_preflight.add_argument("--output", required=True, type=Path)
    model_preflight.add_argument("--model-id", default="Qwen/Qwen2.5-3B-Instruct")
    model_preflight.add_argument(
        "--revision", required=True, help="immutable 40-hex Hugging Face commit"
    )
    model_preflight.set_defaults(handler=_command_model_preflight)

    gpu_smoke = subparsers.add_parser(
        "gpu-smoke",
        help="run the explicit final NF4 model-load/backward/generation GPU gate",
    )
    gpu_smoke.add_argument("--preflight-report", required=True, type=Path)
    gpu_smoke.add_argument("--output", required=True, type=Path)
    gpu_smoke.add_argument(
        "--acknowledge-gpu-use",
        action="store_true",
        help="required explicit acknowledgement; this command creates a CUDA workload",
    )
    gpu_smoke.set_defaults(handler=_command_gpu_smoke)

    sft_preflight = subparsers.add_parser(
        "gate-b-sft-preflight",
        help="CPU-only real-tokenizer response-only encoding proof",
    )
    _add_required_gate_b_data_contract(sft_preflight)
    sft_preflight.add_argument("--revision", required=True)
    sft_preflight.add_argument("--config", required=True, type=Path)
    _add_optional_rationale_training_inputs(sft_preflight)
    sft_preflight.add_argument("--output", required=True, type=Path)
    sft_preflight.set_defaults(handler=_command_gate_b_sft_preflight)

    rationale_build = subparsers.add_parser(
        "build-rationale-corpus",
        help="CPU-only canonicalization of a private fold-training teacher JSONL",
    )
    _add_required_gate_b_data_contract(rationale_build)
    rationale_build.add_argument("--source-jsonl", required=True, type=Path)
    rationale_build.add_argument("--rationale-config", required=True, type=Path)
    rationale_build.add_argument("--output-jsonl", required=True, type=Path)
    rationale_build.add_argument("--output-manifest", required=True, type=Path)
    rationale_build.set_defaults(handler=_command_build_rationale_corpus)

    teacher_materialize = subparsers.add_parser(
        "gate-b-materialize-teacher-bank",
        help=(
            "CPU-only exact fold-training selection from one or more finalized "
            "private Codex teacher banks"
        ),
    )
    _add_required_gate_b_data_contract(teacher_materialize)
    teacher_materialize.add_argument(
        "--teacher-bank",
        action="append",
        nargs=3,
        type=Path,
        metavar=("PLAN_DIR", "SOURCE_JSONL", "SOURCE_MANIFEST"),
        required=True,
        help=(
            "repeat for each finalized private bank; each value is its plan "
            "directory, source JSONL, and source manifest"
        ),
    )
    teacher_materialize.add_argument("--output-jsonl", required=True, type=Path)
    teacher_materialize.add_argument("--output-manifest", required=True, type=Path)
    teacher_materialize.set_defaults(handler=_command_gate_b_materialize_teacher_bank)

    rationale_audit = subparsers.add_parser(
        "audit-rationale-corpus",
        help="CPU-only raw-free audit of a canonical concise-rationale corpus",
    )
    _add_required_gate_b_data_contract(rationale_audit)
    rationale_audit.add_argument("--rationale-corpus", required=True, type=Path)
    rationale_audit.add_argument("--rationale-manifest", required=True, type=Path)
    rationale_audit.add_argument("--rationale-config", required=True, type=Path)
    rationale_audit.add_argument("--output", required=True, type=Path)
    rationale_audit.set_defaults(handler=_command_audit_rationale_corpus)

    teacher_plan = subparsers.add_parser(
        "gate-b-teacher-plan",
        help=(
            "create a question-only Codex teacher plan from fold-0 training IDs "
            "without opening the locked holdout"
        ),
    )
    _add_required_gate_b_data_contract(teacher_plan)
    teacher_plan.add_argument("--teacher-config", required=True, type=Path)
    teacher_plan.add_argument("--output-dir", required=True, type=Path)
    teacher_plan.add_argument(
        "--pilot-size",
        type=int,
        help=(
            "use the locked deterministic 128-problem stratified pilot; omit for "
            "the complete fold-0 training bank, which requires a passed pilot receipt"
        ),
    )
    _add_teacher_pilot_authorization_arguments(teacher_plan, require_receipt=True)
    _add_optional_teacher_harness_authorization_arguments(teacher_plan)
    teacher_plan.set_defaults(handler=_command_gate_b_teacher_plan)

    teacher_run = subparsers.add_parser(
        "gate-b-teacher-run",
        help=(
            "execute pending question-only Codex teacher chunks with a ChatGPT "
            "login; raw events remain private"
        ),
    )
    teacher_run.add_argument("--plan-dir", required=True, type=Path)
    teacher_run.add_argument("--teacher-config", required=True, type=Path)
    teacher_run.add_argument(
        "--acknowledge-codex-teacher",
        action="store_true",
        help="required acknowledgement before organizer training questions are sent to Codex",
    )
    teacher_run.add_argument(
        "--max-invocations",
        type=int,
        help="optional positive cap on this invocation's Codex calls for a bounded pilot",
    )
    teacher_run.add_argument(
        "--max-workers",
        type=int,
        choices=(1, 2),
        default=1,
        help="bounded concurrent Codex calls; start pilots with one worker",
    )
    teacher_run.add_argument(
        "--timeout-seconds",
        type=int,
        default=1_800,
        help="per-Codex-call timeout; a timeout is recorded as a failed private attempt",
    )
    teacher_run.add_argument(
        "--recover-stale-lock",
        action="store_true",
        help="explicitly recover a verified stale teacher-plan lock before resuming",
    )
    _add_optional_teacher_harness_authorization_arguments(teacher_run)
    teacher_run.set_defaults(handler=_command_gate_b_teacher_run)

    teacher_status_parser = subparsers.add_parser(
        "gate-b-teacher-status",
        help="show a raw-free Codex teacher ledger status",
    )
    teacher_status_parser.add_argument("--plan-dir", required=True, type=Path)
    teacher_status_parser.add_argument("--teacher-config", required=True, type=Path)
    teacher_status_parser.add_argument(
        "--output",
        type=Path,
        help="optional atomic raw-free status snapshot; safe to refresh in place",
    )
    _add_optional_teacher_harness_authorization_arguments(teacher_status_parser)
    teacher_status_parser.set_defaults(handler=_command_gate_b_teacher_status)

    teacher_diagnose = subparsers.add_parser(
        "gate-b-teacher-diagnose",
        help=(
            "classify an existing private teacher ledger into immutable raw-free "
            "failure counts without modifying it"
        ),
    )
    teacher_diagnose.add_argument("--plan-dir", required=True, type=Path)
    teacher_diagnose.add_argument("--teacher-config", required=True, type=Path)
    teacher_diagnose.add_argument("--output", required=True, type=Path)
    teacher_diagnose.set_defaults(handler=_command_gate_b_teacher_diagnose)

    harness_replay = subparsers.add_parser(
        "gate-b-teacher-harness-replay",
        help=("run the fixed offline Codex teacher fault matrix without a process or network call"),
    )
    harness_replay.add_argument("--harness-config", required=True, type=Path)
    harness_replay.add_argument("--teacher-config", required=True, type=Path)
    harness_replay.add_argument("--output", required=True, type=Path)
    harness_replay.set_defaults(handler=_command_gate_b_teacher_harness_replay)

    harness_live = subparsers.add_parser(
        "gate-b-teacher-harness-live",
        help=(
            "run the exact versioned answer-free synthetic Codex canary chunks after a "
            "frozen source-manifest check"
        ),
    )
    harness_live.add_argument("--harness-config", required=True, type=Path)
    harness_live.add_argument("--teacher-config", required=True, type=Path)
    harness_live.add_argument("--source-manifest", required=True, type=Path)
    harness_live.add_argument("--source-root", type=Path, default=Path.cwd())
    harness_live.add_argument("--plan-dir", required=True, type=Path)
    harness_live.add_argument("--report", required=True, type=Path)
    harness_live.add_argument(
        "--acknowledge-synthetic-codex-canary",
        action="store_true",
        help=(
            "required acknowledgement before the fixed synthetic questions "
            "are sent to the ChatGPT-login Codex CLI"
        ),
    )
    harness_live.set_defaults(handler=_command_gate_b_teacher_harness_live)

    teacher_finalize = subparsers.add_parser(
        "gate-b-teacher-finalize",
        help=(
            "locally exact-match Codex answers and publish a private rationale "
            "source bank only when every planned row is accepted"
        ),
    )
    _add_required_gate_b_data_contract(teacher_finalize)
    teacher_finalize.add_argument("--teacher-config", required=True, type=Path)
    teacher_finalize.add_argument("--plan-dir", required=True, type=Path)
    teacher_finalize.add_argument("--output-jsonl", required=True, type=Path)
    teacher_finalize.add_argument("--output-manifest", required=True, type=Path)
    teacher_finalize.add_argument(
        "--pilot-size",
        type=int,
        help="must repeat the locked pilot selection used when the plan was created",
    )
    teacher_finalize.add_argument(
        "--recover-stale-lock",
        action="store_true",
        help="explicitly recover a verified stale teacher-plan lock before finalizing",
    )
    _add_optional_teacher_harness_authorization_arguments(teacher_finalize)
    teacher_finalize.set_defaults(handler=_command_gate_b_teacher_finalize)

    teacher_pilot_authorize = subparsers.add_parser(
        "gate-b-teacher-pilot-authorize",
        help=(
            "verify the complete 128-row Codex teacher pilot and publish one "
            "raw-free immutable authorization receipt before full v1 planning"
        ),
    )
    _add_required_gate_b_data_contract(teacher_pilot_authorize)
    teacher_pilot_authorize.add_argument("--teacher-config", required=True, type=Path)
    _add_teacher_pilot_authorization_arguments(
        teacher_pilot_authorize,
        require_receipt=False,
    )
    _add_optional_teacher_harness_authorization_arguments(teacher_pilot_authorize)
    teacher_pilot_authorize.add_argument("--output", required=True, type=Path)
    teacher_pilot_authorize.set_defaults(
        handler=_command_gate_b_teacher_pilot_authorize
    )

    teacher_v2_plan = subparsers.add_parser(
        "gate-b-teacher-v2-plan",
        help=(
            "create a post-probe question-only Codex teacher plan from only the "
            "remaining eligible development-CV IDs"
        ),
    )
    _add_required_gate_b_data_contract(teacher_v2_plan)
    _add_teacher_v2_authorization_arguments(teacher_v2_plan)
    teacher_v2_plan.add_argument("--teacher-config", required=True, type=Path)
    teacher_v2_plan.add_argument("--output-dir", required=True, type=Path)
    teacher_v2_plan.set_defaults(handler=_command_gate_b_teacher_v2_plan)

    teacher_v2_run = subparsers.add_parser(
        "gate-b-teacher-v2-run",
        help=(
            "execute the authorized remaining-development-CV Codex teacher plan; "
            "raw events remain private"
        ),
    )
    _add_required_gate_b_data_contract(
        teacher_v2_run,
        require_development_shard=False,
        require_development_shard_sha256=False,
    )
    _add_teacher_v2_authorization_arguments(teacher_v2_run)
    teacher_v2_run.add_argument("--teacher-config", required=True, type=Path)
    teacher_v2_run.add_argument("--plan-dir", required=True, type=Path)
    teacher_v2_run.add_argument(
        "--acknowledge-codex-teacher",
        action="store_true",
        help="required acknowledgement before organizer training questions are sent to Codex",
    )
    teacher_v2_run.add_argument(
        "--max-invocations",
        type=int,
        help="optional positive cap on this invocation's Codex calls",
    )
    teacher_v2_run.add_argument(
        "--max-workers",
        type=int,
        choices=(1, 2),
        default=1,
        help="bounded concurrent Codex calls; do not exceed the locked profile cap",
    )
    teacher_v2_run.add_argument(
        "--timeout-seconds",
        type=int,
        default=1_800,
        help="per-Codex-call timeout; a timeout is recorded as a failed private attempt",
    )
    teacher_v2_run.add_argument(
        "--recover-stale-lock",
        action="store_true",
        help="explicitly recover a verified stale teacher-plan lock before resuming",
    )
    teacher_v2_run.set_defaults(handler=_command_gate_b_teacher_v2_run)

    teacher_v2_status = subparsers.add_parser(
        "gate-b-teacher-v2-status",
        help="show raw-free status for an authorized v2 teacher ledger",
    )
    _add_required_gate_b_data_contract(
        teacher_v2_status,
        require_development_shard=False,
        require_development_shard_sha256=False,
    )
    _add_teacher_v2_authorization_arguments(teacher_v2_status)
    teacher_v2_status.add_argument("--teacher-config", required=True, type=Path)
    teacher_v2_status.add_argument("--plan-dir", required=True, type=Path)
    teacher_v2_status.add_argument(
        "--output",
        type=Path,
        help="optional atomic raw-free status snapshot; safe to refresh in place",
    )
    teacher_v2_status.set_defaults(handler=_command_gate_b_teacher_v2_status)

    teacher_v2_finalize = subparsers.add_parser(
        "gate-b-teacher-v2-finalize",
        help=(
            "locally exact-match an authorized v2 ledger and publish its private "
            "rationale source bank only when every row is accepted"
        ),
    )
    _add_required_gate_b_data_contract(teacher_v2_finalize)
    _add_teacher_v2_authorization_arguments(teacher_v2_finalize)
    teacher_v2_finalize.add_argument("--teacher-config", required=True, type=Path)
    teacher_v2_finalize.add_argument("--plan-dir", required=True, type=Path)
    teacher_v2_finalize.add_argument("--output-jsonl", required=True, type=Path)
    teacher_v2_finalize.add_argument("--output-manifest", required=True, type=Path)
    teacher_v2_finalize.add_argument(
        "--recover-stale-lock",
        action="store_true",
        help="explicitly recover a verified stale teacher-plan lock before finalizing",
    )
    teacher_v2_finalize.set_defaults(handler=_command_gate_b_teacher_v2_finalize)

    teacher_audit_plan = subparsers.add_parser(
        "gate-b-teacher-logical-audit-plan",
        help=(
            "create the fixed 64-row, candidate-only Codex logical-audit plan "
            "from a finalized private teacher bank"
        ),
    )
    teacher_audit_plan.add_argument("--teacher-config", required=True, type=Path)
    teacher_audit_plan.add_argument("--teacher-plan-dir", required=True, type=Path)
    teacher_audit_plan.add_argument("--source-jsonl", required=True, type=Path)
    teacher_audit_plan.add_argument("--source-manifest", required=True, type=Path)
    teacher_audit_plan.add_argument("--output-dir", required=True, type=Path)
    _add_optional_teacher_harness_authorization_arguments(teacher_audit_plan)
    teacher_audit_plan.set_defaults(handler=_command_gate_b_teacher_logical_audit_plan)

    teacher_audit_run = subparsers.add_parser(
        "gate-b-teacher-logical-audit-run",
        help=(
            "run one restartable, candidate-only Codex logical audit with a "
            "ChatGPT login; raw events remain private"
        ),
    )
    teacher_audit_run.add_argument("--teacher-config", required=True, type=Path)
    teacher_audit_run.add_argument("--teacher-plan-dir", required=True, type=Path)
    teacher_audit_run.add_argument("--audit-dir", required=True, type=Path)
    teacher_audit_run.add_argument(
        "--acknowledge-codex-teacher",
        action="store_true",
        help=(
            "required acknowledgement before Codex receives private training "
            "questions and candidate rationales"
        ),
    )
    teacher_audit_run.add_argument(
        "--timeout-seconds",
        type=int,
        default=1_800,
        help="per-Codex-call timeout; a timeout is recorded as a failed private attempt",
    )
    teacher_audit_run.add_argument(
        "--recover-stale-lock",
        action="store_true",
        help="explicitly recover a verified stale logical-audit lock before resuming",
    )
    _add_optional_teacher_harness_authorization_arguments(teacher_audit_run)
    teacher_audit_run.set_defaults(handler=_command_gate_b_teacher_logical_audit_run)

    teacher_audit_status_parser = subparsers.add_parser(
        "gate-b-teacher-logical-audit-status",
        help="show a raw-free Codex teacher logical-audit ledger status",
    )
    teacher_audit_status_parser.add_argument(
        "--teacher-config", required=True, type=Path
    )
    teacher_audit_status_parser.add_argument(
        "--teacher-plan-dir", required=True, type=Path
    )
    teacher_audit_status_parser.add_argument("--audit-dir", required=True, type=Path)
    teacher_audit_status_parser.add_argument(
        "--output",
        type=Path,
        help="optional atomic raw-free status snapshot; safe to refresh in place",
    )
    _add_optional_teacher_harness_authorization_arguments(teacher_audit_status_parser)
    teacher_audit_status_parser.set_defaults(
        handler=_command_gate_b_teacher_logical_audit_status
    )

    teacher_audit_finalize = subparsers.add_parser(
        "gate-b-teacher-logical-audit-finalize",
        help=(
            "publish or inspect the immutable fixed-64 logical-audit verdict "
            "without reading organizer answers"
        ),
    )
    teacher_audit_finalize.add_argument("--teacher-config", required=True, type=Path)
    teacher_audit_finalize.add_argument(
        "--teacher-plan-dir", required=True, type=Path
    )
    teacher_audit_finalize.add_argument("--audit-dir", required=True, type=Path)
    teacher_audit_finalize.add_argument(
        "--recover-stale-lock",
        action="store_true",
        help="explicitly recover a verified stale logical-audit lock before finalizing",
    )
    _add_optional_teacher_harness_authorization_arguments(teacher_audit_finalize)
    teacher_audit_finalize.set_defaults(
        handler=_command_gate_b_teacher_logical_audit_finalize
    )

    development = subparsers.add_parser(
        "gate-b-development",
        help="run one fixed base or verified-adapter development fold",
    )
    _add_required_gate_b_data_contract(development)
    _add_gpu_runtime_gate(development)
    _add_current_source_provenance(development)
    development.add_argument("--output-jsonl", required=True, type=Path)
    development.add_argument("--output-manifest", required=True, type=Path)
    development.add_argument(
        "--resume-dir",
        type=Path,
        help=(
            "private persistent chunk ledger for interruption-safe development "
            "generation; final output paths must still be new"
        ),
    )
    development.add_argument(
        "--resume-chunk-size",
        type=int,
        default=25,
        help="validated development problems per private resume chunk",
    )
    development.add_argument("--adapter", type=Path)
    development.add_argument(
        "--base-baseline-manifest",
        type=Path,
        help="required successful fixed-base run manifest before adapter evaluation",
    )
    development.set_defaults(handler=_command_gate_b_development)

    development_status = subparsers.add_parser(
        "gate-b-development-status",
        help="read one raw-free resumable development-generation status",
    )
    development_status.add_argument("--resume-dir", required=True, type=Path)
    development_status.add_argument("--output", type=Path)
    development_status.set_defaults(handler=_command_gate_b_development_status)

    parser_golden = subparsers.add_parser(
        "audit-parser-golden",
        help="validate one development bundle and emit redacted parser-golden evidence",
    )
    parser_golden.add_argument("--records", required=True, type=Path)
    parser_golden.add_argument("--manifest", required=True, type=Path)
    parser_golden.add_argument("--output", required=True, type=Path)
    parser_golden.set_defaults(handler=_command_audit_parser_golden)

    parser_rescore = subparsers.add_parser(
        "audit-parser-rescore",
        help="compare stored/current parser outcomes without selection promotion",
    )
    parser_rescore.add_argument("--records", required=True, type=Path)
    parser_rescore.add_argument("--manifest", required=True, type=Path)
    parser_rescore.add_argument("--output", required=True, type=Path)
    parser_rescore.set_defaults(handler=_command_audit_parser_rescore)

    train_fold = subparsers.add_parser(
        "gate-b-train-fold",
        help="train one locked direct-answer QLoRA fold after the base baseline",
    )
    _add_required_gate_b_data_contract(train_fold)
    _add_gpu_runtime_gate(train_fold)
    _add_current_source_provenance(train_fold)
    _add_optional_rationale_training_inputs(train_fold)
    train_fold.add_argument("--base-baseline-manifest", required=True, type=Path)
    train_fold.add_argument("--output-dir", required=True, type=Path)
    train_fold.add_argument(
        "--resume-dir",
        type=Path,
        help=(
            "private persistent QLoRA checkpoint ledger; resume is allowed only "
            "when its immutable contract exactly matches this invocation"
        ),
    )
    train_fold.set_defaults(handler=_command_gate_b_train_fold)

    training_status = subparsers.add_parser(
        "gate-b-training-status",
        help="read one raw-free persistent QLoRA training status",
    )
    training_status.add_argument("--resume-dir", required=True, type=Path)
    training_status.add_argument("--output", type=Path)
    training_status.set_defaults(handler=_command_gate_b_training_status)

    compare = subparsers.add_parser(
        "compare-development",
        help="validate development JSONL and run paired cluster statistics",
    )
    compare.add_argument("--train", required=True, type=Path)
    compare.add_argument("--train-exclusions", required=True, type=Path)
    compare.add_argument("--split-artifact", required=True, type=Path)
    compare.add_argument("--fold", required=True, type=int)
    compare.add_argument(
        "--reference",
        required=True,
        nargs=3,
        metavar=("LABEL", "RECORDS_JSONL", "RUN_MANIFEST"),
    )
    compare.add_argument(
        "--candidate",
        required=True,
        action="append",
        nargs=3,
        metavar=("LABEL", "RECORDS_JSONL", "RUN_MANIFEST"),
    )
    compare.add_argument("--bootstrap-samples", type=int, default=10_000)
    compare.add_argument("--bootstrap-seed", type=int, default=20_260_804)
    compare.add_argument("--confidence", type=float, default=0.95)
    compare.add_argument("--alpha", type=float, default=0.05)
    compare.add_argument("--expected-train-sha256", required=True)
    compare.add_argument("--expected-exclusions-sha256", required=True)
    compare.add_argument("--expected-exclusion-count", required=True, type=int)
    compare.add_argument("--expected-split-sha256", required=True)
    compare.add_argument("--development-shard", required=True, type=Path)
    compare.add_argument("--expected-development-shard-sha256", required=True)
    compare.add_argument("--output", required=True, type=Path)
    compare.set_defaults(handler=_command_compare_development)

    probe_decision = subparsers.add_parser(
        "decide-candidate-probe",
        help="apply the fixed single-fold harm screen before more GPU folds",
    )
    probe_decision.add_argument("--comparison-artifact", required=True, type=Path)
    probe_decision.add_argument("--candidate-label", required=True)
    probe_decision.add_argument("--output", required=True, type=Path)
    probe_decision.set_defaults(handler=_command_decide_candidate_probe)

    compare_oof = subparsers.add_parser(
        "compare-development-oof",
        help="pool complete out-of-fold runs and run paired cluster statistics",
    )
    compare_oof.add_argument("--train", required=True, type=Path)
    compare_oof.add_argument("--train-exclusions", required=True, type=Path)
    compare_oof.add_argument("--split-artifact", required=True, type=Path)
    compare_oof.add_argument("--deployment-fold", required=True, type=int)
    compare_oof.add_argument("--reference-label", required=True)
    compare_oof.add_argument(
        "--candidate-label", required=True, action="append"
    )
    compare_oof.add_argument(
        "--base-run",
        required=True,
        action="append",
        nargs=3,
        metavar=("FOLD", "RECORDS_JSONL", "RUN_MANIFEST"),
        help="one fixed-base validation run; repeat exactly once per fold",
    )
    compare_oof.add_argument(
        "--adapter-run",
        required=True,
        action="append",
        nargs=5,
        metavar=("FOLD", "LABEL", "RECORDS_JSONL", "RUN_MANIFEST", "ADAPTER_DIR"),
        help="one adapter validation run with its exact bundle; repeat per label/fold",
    )
    compare_oof.add_argument("--bootstrap-samples", type=int, default=10_000)
    compare_oof.add_argument("--bootstrap-seed", type=int, default=20_260_804)
    compare_oof.add_argument("--confidence", type=float, default=0.95)
    compare_oof.add_argument("--alpha", type=float, default=0.05)
    compare_oof.add_argument("--expected-train-sha256", required=True)
    compare_oof.add_argument("--expected-exclusions-sha256", required=True)
    compare_oof.add_argument("--expected-exclusion-count", required=True, type=int)
    compare_oof.add_argument("--expected-split-sha256", required=True)
    compare_oof.add_argument("--development-shard", required=True, type=Path)
    compare_oof.add_argument("--expected-development-shard-sha256", required=True)
    compare_oof.add_argument("--output", required=True, type=Path)
    compare_oof.set_defaults(handler=_command_compare_development_oof)

    verify_base_oof = subparsers.add_parser(
        "verify-base-development-oof",
        help="qualify complete fixed-base OOF evidence without a candidate",
    )
    verify_base_oof.add_argument("--train", required=True, type=Path)
    verify_base_oof.add_argument("--train-exclusions", required=True, type=Path)
    verify_base_oof.add_argument("--split-artifact", required=True, type=Path)
    verify_base_oof.add_argument("--deployment-fold", required=True, type=int)
    verify_base_oof.add_argument("--base-label", required=True)
    verify_base_oof.add_argument(
        "--base-run",
        required=True,
        action="append",
        nargs=3,
        metavar=("FOLD", "RECORDS_JSONL", "RUN_MANIFEST"),
        help="one fixed-base validation run; repeat exactly once per fold",
    )
    verify_base_oof.add_argument("--expected-train-sha256", required=True)
    verify_base_oof.add_argument("--expected-exclusions-sha256", required=True)
    verify_base_oof.add_argument("--expected-exclusion-count", required=True, type=int)
    verify_base_oof.add_argument("--expected-split-sha256", required=True)
    verify_base_oof.add_argument("--development-shard", required=True, type=Path)
    verify_base_oof.add_argument("--expected-development-shard-sha256", required=True)
    verify_base_oof.add_argument("--output", required=True, type=Path)
    verify_base_oof.set_defaults(handler=_command_verify_base_development_oof)

    freeze = subparsers.add_parser(
        "freeze-development-selection",
        help="freeze primary/fallback methods from development evidence",
    )
    freeze.add_argument("--comparison-artifact", required=True, type=Path)
    freeze.add_argument("--primary-label", required=True)
    freeze.add_argument("--fallback-label")
    freeze.add_argument("--decision-note", required=True)
    freeze.add_argument("--source-manifest", required=True, type=Path)
    freeze.add_argument("--lockfile", required=True, type=Path)
    freeze.add_argument("--output", required=True, type=Path)
    freeze.add_argument("--confirm-no-leaderboard-selection", action="store_true")
    freeze.set_defaults(handler=_command_freeze_development_selection)

    freeze_base = subparsers.add_parser(
        "freeze-development-base",
        help="freeze one fixed-base method from qualified complete OOF evidence",
    )
    freeze_base.add_argument("--base-oof-artifact", required=True, type=Path)
    freeze_base.add_argument("--primary-label", required=True)
    freeze_base.add_argument("--decision-note", required=True)
    freeze_base.add_argument("--source-manifest", required=True, type=Path)
    freeze_base.add_argument("--lockfile", required=True, type=Path)
    freeze_base.add_argument("--output", required=True, type=Path)
    freeze_base.add_argument("--confirm-no-leaderboard-selection", action="store_true")
    freeze_base.set_defaults(handler=_command_freeze_development_base)

    holdout = subparsers.add_parser(
        "gate-b-locked-holdout-evaluate",
        help="claim and immediately evaluate frozen primary/fallback methods exactly once",
    )
    _add_required_gate_b_data_contract(holdout, require_development_shard=False)
    _add_gpu_runtime_gate(holdout)
    holdout.add_argument("--freeze-artifact", required=True, type=Path)
    holdout.add_argument("--primary-kind", required=True, choices=("base", "adapter"))
    holdout.add_argument("--primary-adapter", type=Path)
    holdout.add_argument(
        "--fallback-kind", required=True, choices=("none", "base", "adapter")
    )
    holdout.add_argument("--fallback-adapter", type=Path)
    holdout.add_argument("--output", required=True, type=Path)
    holdout.add_argument(
        "--acknowledge-one-time-locked-holdout",
        action="store_true",
        help="required acknowledgement that this irreversibly consumes the sole holdout claim",
    )
    holdout.set_defaults(handler=_command_gate_b_locked_holdout_evaluate)

    predict = subparsers.add_parser(
        "gate-b-predict-evaluation",
        help="run frozen offline leaderboard/test inference and strict parsing",
    )
    _add_required_gate_b_data_contract(predict, require_development_shard=False)
    _add_gpu_runtime_gate(predict)
    predict.add_argument("--evaluation", required=True, type=Path)
    predict.add_argument("--dataset-role", required=True, choices=("leaderboard", "test"))
    predict.add_argument("--expected-evaluation-sha256", required=True)
    predict.add_argument("--freeze-artifact", required=True, type=Path)
    predict.add_argument("--primary-kind", required=True, choices=("base", "adapter"))
    predict.add_argument("--primary-adapter", type=Path)
    predict.add_argument(
        "--fallback-kind", required=True, choices=("none", "base", "adapter")
    )
    predict.add_argument("--fallback-adapter", type=Path)
    predict.add_argument("--output-artifact", required=True, type=Path)
    predict.add_argument("--output-predictions", required=True, type=Path)
    predict.set_defaults(handler=_command_gate_b_predict_evaluation)

    parse = subparsers.add_parser("parse-answer", help="parse an assistant completion")
    parse_input = parse.add_mutually_exclusive_group(required=True)
    parse_input.add_argument("--text")
    parse_input.add_argument("--file", type=Path)
    parse.set_defaults(handler=_command_parse)

    validate = subparsers.add_parser(
        "validate-submission", help="validate a submission against expected IDs"
    )
    validate.add_argument("--submission", required=True, type=Path)
    validate.add_argument("--expected", required=True, type=Path)
    validate.add_argument("--id-column", default="ID")
    validate.add_argument("--answer-column", default="answer")
    validate.set_defaults(handler=_command_validate_submission)

    independent_validate = subparsers.add_parser(
        "verify-submission-independent",
        help="cross-check final CSV with an independent minimal parser",
    )
    independent_validate.add_argument("--submission", required=True, type=Path)
    independent_validate.add_argument("--expected", required=True, type=Path)
    independent_validate.add_argument("--expected-sha256", required=True)
    independent_validate.set_defaults(handler=_command_verify_submission_independent)

    write_submission = subparsers.add_parser(
        "write-submission",
        help="atomically build a submission from a JSON ID-to-integer mapping",
    )
    write_submission.add_argument("--predictions", required=True, type=Path)
    write_submission.add_argument("--expected", required=True, type=Path)
    write_submission.add_argument("--output", required=True, type=Path)
    write_submission.add_argument("--id-column", default="ID")
    write_submission.add_argument("--answer-column", default="answer")
    write_submission.add_argument("--overwrite", action="store_true")
    write_submission.set_defaults(handler=_command_write_submission)

    source = subparsers.add_parser(
        "source-manifest", help="hash the current source/config/document tree"
    )
    source.add_argument("--root", type=Path, default=Path.cwd())
    source.add_argument("--output", required=True, type=Path)
    source.set_defaults(handler=_command_source_manifest)
    return parser


def _command_audit(args: argparse.Namespace) -> int:
    raw_train, train, exclusions = _load_train_with_optional_exclusions(
        args.train, args.train_exclusions
    )
    train, scope_metadata = _apply_train_scope(
        raw_train=raw_train,
        filtered_train=train,
        exclusions=exclusions,
        split_artifact=args.split_artifact,
        train_scope=args.train_scope,
        fold=args.fold,
        allow_all_train_scope=args.allow_all_train_scope,
    )
    leaderboard = load_leaderboard_csv(args.leaderboard)
    report = build_data_audit_report_from_datasets(
        train,
        leaderboard,
        audit_version=(
            FILTERED_AUDIT_VERSION if exclusions is not None else AUDIT_VERSION
        ),
        eligibility=(
            _audit_eligibility_metadata(train, exclusions, scope_metadata)
            if exclusions is not None
            else None
        ),
        source_tree_root=args.source_root,
        source_tree_excluded_paths=(args.output,),
    )
    write_json_atomic(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "train_rows": report["train"]["row_count"],
                "leaderboard_rows": report["leaderboard"]["row_count"],
                "train_sha256": report["train"]["manifest"]["sha256"],
                "leaderboard_sha256": report["leaderboard"]["manifest"]["sha256"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def _command_eligibility_overlay(args: argparse.Namespace) -> int:
    train = load_train_csv(args.train)
    exclusions = load_train_exclusions_csv(
        args.train_exclusions,
        train_ids=(record.id for record in train),
    )
    if not exclusions:
        raise ValueError("train exclusion overlay must contain at least one organizer ID")
    manifest, _ = _load_verified_split(train, args.split_artifact)
    _require_expected_contract(
        actual_train_sha256=train.manifest.sha256,
        actual_exclusions_sha256=exclusions.manifest.sha256,
        actual_exclusion_count=len(exclusions),
        actual_split_sha256=manifest.sha256,
        expected_train_sha256=args.expected_train_sha256,
        expected_exclusions_sha256=args.expected_exclusions_sha256,
        expected_exclusion_count=args.expected_exclusion_count,
        expected_split_sha256=args.expected_split_sha256,
    )

    expanded_ids = expand_hard_group_exclusions(manifest, exclusions.ids)
    fold_summaries: dict[str, Any] = {}
    for fold in range(manifest.n_folds):
        training = eligible_training_ids(manifest, fold, exclusions.ids)
        validation = eligible_validation_ids(manifest, fold, exclusions.ids)
        fold_summaries[str(fold)] = {
            "training_count": len(training),
            "training_ids_sha256": _ids_sha256(training),
            "validation_count": len(validation),
            "validation_ids_sha256": _ids_sha256(validation),
        }
    payload: dict[str, Any] = {
        "schema_version": "train-eligibility-v1",
        "policy": {
            "organizer_exclusion_source": "train_filtered_ids.csv ID column only",
            "hard_group_expansion": True,
            "hard_group_source": "existing split manifest group_id only",
            "number_masked_templates_used_as_hard_groups": False,
            "leaderboard_or_test_used": False,
            "source_train_rewritten": False,
            "expected_contract_checked": any(
                value is not None
                for value in (
                    args.expected_train_sha256,
                    args.expected_exclusions_sha256,
                    args.expected_exclusion_count,
                    args.expected_split_sha256,
                )
            ),
        },
        "expected_contract": {
            "train_sha256": args.expected_train_sha256,
            "exclusions_sha256": args.expected_exclusions_sha256,
            "exclusion_count": args.expected_exclusion_count,
            "split_sha256": args.expected_split_sha256,
        },
        "train_manifest": asdict(train.manifest),
        "exclusion_manifest": asdict(exclusions.manifest),
        "exclusion_raw_header": exclusions.raw_header,
        "direct_excluded_count": len(exclusions),
        "direct_excluded_ids_sha256": exclusions.ids_sha256,
        "expanded_excluded_count": len(expanded_ids),
        "expanded_excluded_ids_sha256": _ids_sha256(expanded_ids),
        "extra_hard_group_member_count": len(expanded_ids) - len(exclusions),
        "eligible_global_count": len(train) - len(expanded_ids),
        "split_artifact": {
            "path": str(args.split_artifact.resolve()),
            "sha256": sha256_file(args.split_artifact),
            "split_sha256": manifest.sha256,
            "source_groups_sha256": manifest.source_groups_sha256,
            "version": manifest.version,
            "n_folds": manifest.n_folds,
        },
        "folds": fold_summaries,
        "locked_holdout": {
            "accessed_for_model_selection": False,
            "ids_or_answers_emitted": False,
            "policy": (
                "remain sealed until primary/fallback methods are frozen from "
                "development evidence"
            ),
        },
    }
    write_json_atomic(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "direct_excluded_count": len(exclusions),
                "expanded_excluded_count": len(expanded_ids),
                "eligible_global_count": payload["eligible_global_count"],
                "split_sha256": manifest.sha256,
            },
            sort_keys=True,
        )
    )
    return 0


def _command_splits(args: argparse.Namespace) -> int:
    train = load_train_csv(args.train)
    labels = {
        record.id: (
            f"math:{math_aware_fingerprint(record.question_raw)}",
            f"source:{source_format_fingerprint(record.question_raw)}",
        )
        for record in train
    }
    group_by_id = build_group_clusters(
        (record.id for record in train), group_labels=labels
    )
    manifest = make_grouped_split_manifest(
        (record.id for record in train),
        group_by_id,
        n_folds=args.folds,
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
        version=args.version,
    )
    payload: dict[str, Any] = {
        "dataset_manifest": asdict(train.manifest),
        "cluster_method": _HARD_CLUSTER_METHOD,
        "cluster_count": len(set(group_by_id.values())),
        "split": manifest.to_dict(),
        "strata_summary": _split_strata_summary(train.records, manifest),
    }
    write_json_atomic(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "split_sha256": manifest.sha256,
                "cluster_count": payload["cluster_count"],
                "holdout_rows": manifest.actual_counts()["partition_records"]
                ["final_locked_holdout"],
                "fold_rows": [len(manifest.fold_ids(index)) for index in range(args.folds)],
            },
            sort_keys=True,
        )
    )
    return 0


def _command_build_development_shard(args: argparse.Namespace) -> int:
    train = load_train_csv(args.train)
    if train.manifest.sha256 != args.expected_train_sha256:
        raise ValueError("organizer train SHA-256 does not match the shard-build contract")
    manifest, _ = _load_verified_split(train, args.split_artifact)
    if manifest.sha256 != args.expected_split_sha256:
        raise ValueError("logical split SHA-256 does not match the shard-build contract")
    result = build_development_cv_shard(
        train,
        split_manifest=manifest,
        split_artifact_sha256=sha256_file(args.split_artifact),
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "output": result.path,
                "bundle_sha256": result.bundle_sha256,
                "csv_sha256": result.csv_sha256,
                "row_count": result.row_count,
                "locked_holdout_rows_emitted": False,
            },
            sort_keys=True,
        )
    )
    return 0


def _split_strata_summary(
    records: Sequence[MathRecord], manifest: SplitManifest
) -> dict[str, Any]:
    """Summarize answer/quality strata without changing split identity."""

    record_by_id = {record.id: record for record in records}

    def summarize(ids: Sequence[str]) -> dict[str, Any]:
        sign_counts: Counter[str] = Counter()
        magnitude_counts: Counter[str] = Counter()
        quality_counts: Counter[str] = Counter()
        answers: Counter[int] = Counter()
        for record_id in ids:
            record = record_by_id[record_id]
            if record.answer is None:
                raise ValueError("split strata require train answers")
            answer = record.answer
            answers[answer] += 1
            sign_counts["negative" if answer < 0 else "positive" if answer > 0 else "zero"] += 1
            absolute = abs(answer)
            if absolute == 0:
                bucket = "0"
            elif absolute <= 10:
                bucket = "1_to_10"
            elif absolute <= 100:
                bucket = "11_to_100"
            elif absolute <= 1_000:
                bucket = "101_to_1000"
            elif absolute < 1_000_000:
                bucket = "1001_to_999999"
            else:
                bucket = "ge_1e6"
            magnitude_counts[bucket] += 1
            quality_counts.update(
                flag.value for flag in assess_question(record.question_raw).flags
            )
        return {
            "row_count": len(ids),
            "answer_sign_counts": dict(sorted(sign_counts.items())),
            "absolute_answer_bucket_counts": dict(sorted(magnitude_counts.items())),
            "quality_flag_counts": dict(sorted(quality_counts.items())),
            "top_answers": [
                {"answer": answer, "count": count}
                for answer, count in sorted(
                    answers.items(), key=lambda item: (-item[1], item[0])
                )[:10]
            ],
        }

    cv_ids = tuple(
        record_id
        for fold in range(manifest.n_folds)
        for record_id in manifest.fold_ids(fold)
    )
    holdout_row_count = manifest.actual_counts()["partition_records"][
        "final_locked_holdout"
    ]
    return {
        "overall": {
            "row_count": len(records),
            "sealed_metrics": True,
            "note": "only row accounting is emitted after the holdout is locked",
        },
        "development_cross_validation": summarize(cv_ids),
        "final_locked_holdout": {
            "row_count": holdout_row_count,
            "sealed": True,
            "answer_or_question_statistics_emitted": False,
        },
        "folds": {
            str(fold): summarize(manifest.fold_ids(fold))
            for fold in range(manifest.n_folds)
        },
        "contract": (
            "development-only diagnostics; locked holdout answers/questions are not "
            "summarized and derived strata do not alter split SHA-256"
        ),
    }


def _command_profile(args: argparse.Namespace) -> int:
    raw_train, train, exclusions = _load_train_with_optional_exclusions(
        args.train, args.train_exclusions
    )
    train, scope_metadata = _apply_train_scope(
        raw_train=raw_train,
        filtered_train=train,
        exclusions=exclusions,
        split_artifact=args.split_artifact,
        train_scope=args.train_scope,
        fold=args.fold,
        allow_all_train_scope=args.allow_all_train_scope,
    )
    leaderboard = load_leaderboard_csv(args.leaderboard)
    common = {
        "model_id": args.model_id,
        "revision": args.revision,
        "system_prompt": args.system_prompt,
        "local_files_only": not args.allow_download,
    }
    report = load_and_profile_datasets(
        {"train": train.records, "leaderboard": leaderboard.records}, **common
    )
    if report["train"]["provenance"] != report["leaderboard"]["provenance"]:
        raise RuntimeError("train and leaderboard tokenizer provenance differ")
    report["_eligibility"] = (
        None
        if exclusions is None
        else _audit_eligibility_metadata(train, exclusions, scope_metadata)
    )
    write_json_atomic(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "resolved_commit": report["train"]["provenance"]["resolved_commit"],
                "train_max_chat_tokens": report["train"]["chat_input_tokens"]["max"],
                "leaderboard_max_chat_tokens": report["leaderboard"]["chat_input_tokens"][
                    "max"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


def _load_train_with_optional_exclusions(
    train_path: Path,
    exclusion_path: Path | None,
) -> tuple[CsvDataset, CsvDataset, TrainExclusionSet | None]:
    train = load_train_csv(train_path)
    if exclusion_path is None:
        return train, train, None
    exclusions = load_train_exclusions_csv(
        exclusion_path,
        train_ids=(record.id for record in train),
    )
    excluded_ids = set(exclusions.ids)
    filtered = CsvDataset(
        kind=train.kind,
        records=tuple(record for record in train if record.id not in excluded_ids),
        manifest=train.manifest,
        raw_header=train.raw_header,
    )
    if len(filtered) + len(exclusions) != len(train):  # pragma: no cover - invariant guard
        raise RuntimeError("train eligibility filtering did not preserve row accounting")
    return train, filtered, exclusions


def _apply_train_scope(
    *,
    raw_train: CsvDataset,
    filtered_train: CsvDataset,
    exclusions: TrainExclusionSet | None,
    split_artifact: Path | None,
    train_scope: str | None,
    fold: int | None,
    allow_all_train_scope: bool,
) -> tuple[CsvDataset, dict[str, Any] | None]:
    if train_scope is None:
        train_scope = "cv-only" if exclusions is not None else "all"
    if train_scope == "all":
        if split_artifact is not None or fold is not None:
            raise ValueError(
                "--split-artifact/--fold require an explicit non-all --train-scope"
            )
        if exclusions is not None and not allow_all_train_scope:
            raise ValueError(
                "all-train metrics after organizer exclusions require explicit "
                "--allow-all-train-scope; use --train-scope cv-only after split lock"
            )
        return filtered_train, None
    if allow_all_train_scope:
        raise ValueError("--allow-all-train-scope is valid only with --train-scope all")
    if exclusions is None:
        raise ValueError("a non-all train scope requires --train-exclusions")
    if split_artifact is None:
        raise ValueError("a non-all train scope requires --split-artifact")
    manifest, _ = _load_verified_split(raw_train, split_artifact)
    if train_scope == "cv-only":
        if fold is not None:
            raise ValueError("--fold is not valid with --train-scope cv-only")
        selected_ids = tuple(
            record_id
            for fold_index in range(manifest.n_folds)
            for record_id in eligible_validation_ids(
                manifest, fold_index, exclusions.ids
            )
        )
    else:
        if fold is None:
            raise ValueError(f"--train-scope {train_scope} requires --fold")
        if train_scope == "fold-training":
            selected_ids = eligible_training_ids(manifest, fold, exclusions.ids)
        elif train_scope == "fold-validation":
            selected_ids = eligible_validation_ids(manifest, fold, exclusions.ids)
        else:  # pragma: no cover - argparse constrains this value
            raise ValueError(f"unsupported train scope {train_scope!r}")
    if len(set(selected_ids)) != len(selected_ids):  # pragma: no cover - manifest invariant
        raise RuntimeError("selected split scope contains duplicate IDs")
    record_by_id = {record.id: record for record in filtered_train}
    missing = sorted(set(selected_ids) - set(record_by_id))
    if missing:
        raise RuntimeError(f"split scope references filtered-out or missing IDs: {missing[:5]!r}")
    selected = CsvDataset(
        kind=filtered_train.kind,
        records=tuple(record_by_id[record_id] for record_id in sorted(selected_ids)),
        manifest=filtered_train.manifest,
        raw_header=filtered_train.raw_header,
    )
    expanded_excluded_ids = expand_hard_group_exclusions(manifest, exclusions.ids)
    metadata = {
        "train_scope": train_scope,
        "fold": fold,
        "selected_count": len(selected),
        "selected_ids_sha256": _ids_sha256(tuple(sorted(selected_ids))),
        "split_artifact": {
            "path": str(split_artifact.resolve()),
            "sha256": sha256_file(split_artifact),
            "split_sha256": manifest.sha256,
            "source_groups_sha256": manifest.source_groups_sha256,
        },
        "locked_holdout_used_for_metrics_or_examples": False,
        "hard_group_expansion_applied": True,
        "expanded_excluded_count": len(expanded_excluded_ids),
        "expanded_excluded_ids_sha256": _ids_sha256(expanded_excluded_ids),
    }
    return selected, metadata


def _audit_eligibility_metadata(
    filtered_train: CsvDataset,
    exclusions: TrainExclusionSet,
    scope_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "policy": "organizer train_filtered_ids ID column only; source train not rewritten",
        "exclusion_manifest": asdict(exclusions.manifest),
        "exclusion_raw_header": exclusions.raw_header,
        "excluded_count": len(exclusions),
        "excluded_ids_sha256": exclusions.ids_sha256,
        "eligible_count": len(filtered_train),
        "hard_group_expansion_applied": bool(
            scope_metadata and scope_metadata.get("hard_group_expansion_applied")
        ),
        "expanded_excluded_count": (
            scope_metadata.get("expanded_excluded_count")
            if scope_metadata is not None
            else None
        ),
        "expanded_excluded_ids_sha256": (
            scope_metadata.get("expanded_excluded_ids_sha256")
            if scope_metadata is not None
            else None
        ),
        "hard_group_note": (
            "audit describes organizer row eligibility; use build-eligibility-overlay "
            "for group-safe fold training"
        ),
        "scope": scope_metadata or {"train_scope": "all-organizer-eligible"},
    }


def _ids_sha256(ids: Sequence[str]) -> str:
    return hashlib.sha256(canonical_json_bytes(list(ids))).hexdigest()


def _source_groups_sha256(
    ids: Sequence[str], group_by_id: dict[str, str]
) -> str:
    payload = [[record_id, group_by_id[record_id]] for record_id in sorted(ids)]
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _load_verified_split(
    train: CsvDataset,
    split_artifact: Path,
) -> tuple[SplitManifest, dict[str, Any]]:
    wrapper = _load_json_object(split_artifact)
    dataset_manifest = wrapper.get("dataset_manifest")
    if not isinstance(dataset_manifest, dict):
        raise ValueError("split artifact is missing dataset_manifest")
    split_dataset_sha = dataset_manifest.get("sha256")
    if split_dataset_sha != train.manifest.sha256:
        raise ValueError(
            "split artifact train SHA-256 does not match the supplied train CSV: "
            f"split={split_dataset_sha!r}, train={train.manifest.sha256!r}"
        )
    manifest = SplitManifest.from_dict(_required_mapping(wrapper, "split"))
    train_ids = tuple(record.id for record in train)
    assignment_ids = tuple(assignment.record_id for assignment in manifest.assignments)
    if assignment_ids != tuple(sorted(train_ids)):
        missing = sorted(set(train_ids) - set(assignment_ids))
        extra = sorted(set(assignment_ids) - set(train_ids))
        raise ValueError(
            "split assignments do not exactly match the supplied train IDs; "
            f"missing={missing[:5]!r}, extra={extra[:5]!r}"
        )
    labels = {
        record.id: (
            f"math:{math_aware_fingerprint(record.question_raw)}",
            f"source:{source_format_fingerprint(record.question_raw)}",
        )
        for record in train
    }
    recomputed_group_by_id = build_group_clusters(train_ids, group_labels=labels)
    recomputed_cluster_count = len(set(recomputed_group_by_id.values()))
    if wrapper.get("cluster_method") != _HARD_CLUSTER_METHOD:
        raise ValueError("split artifact cluster_method is not the canonical hard-group policy")
    if wrapper.get("cluster_count") != recomputed_cluster_count:
        raise ValueError(
            "split artifact cluster_count does not match recomputed hard groups: "
            f"stored={wrapper.get('cluster_count')!r}, "
            f"computed={recomputed_cluster_count}"
        )
    recomputed_source_groups_sha256 = _source_groups_sha256(
        train_ids, recomputed_group_by_id
    )
    if manifest.source_groups_sha256 != recomputed_source_groups_sha256:
        raise ValueError(
            "split source_groups_sha256 does not match hard groups recomputed from train"
        )
    mismatched_groups = [
        assignment.record_id
        for assignment in manifest.assignments
        if assignment.group_id != recomputed_group_by_id[assignment.record_id]
    ]
    if mismatched_groups:  # pragma: no cover - source hash already makes this infeasible
        raise ValueError(
            "split assignment group IDs differ from recomputed hard groups: "
            f"{mismatched_groups[:5]!r}"
        )
    return manifest, wrapper


def _require_expected_contract(
    *,
    actual_train_sha256: str,
    actual_exclusions_sha256: str,
    actual_exclusion_count: int,
    actual_split_sha256: str,
    expected_train_sha256: str | None,
    expected_exclusions_sha256: str | None,
    expected_exclusion_count: int | None,
    expected_split_sha256: str | None,
) -> None:
    comparisons = (
        ("train SHA-256", actual_train_sha256, expected_train_sha256),
        (
            "train exclusions SHA-256",
            actual_exclusions_sha256,
            expected_exclusions_sha256,
        ),
        ("train exclusion count", actual_exclusion_count, expected_exclusion_count),
        ("split SHA-256", actual_split_sha256, expected_split_sha256),
    )
    for label, actual, expected in comparisons:
        if expected is not None and actual != expected:
            raise ValueError(
                f"expected contract mismatch for {label}: expected={expected!r}, "
                f"actual={actual!r}"
            )


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8", errors="strict"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} is not allowed")


def _required_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact field {key!r} must be an object")
    return value


def _command_parse(args: argparse.Namespace) -> int:
    completion = args.text
    if args.file is not None:
        completion = args.file.read_text(encoding="utf-8")
    result = parse_answer(completion)
    print(json.dumps(asdict(result), ensure_ascii=False, sort_keys=True))
    return 0 if result.ok else 1


def _command_model_preflight(args: argparse.Namespace) -> int:
    report = run_model_preflight(model_id=args.model_id, revision=args.revision)
    write_json_atomic(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "tokenizer_ready": report["tokenizer_ready"],
                "model_runtime_ready": report["model_runtime_ready"],
                "training_ready": report["training_ready"],
                "blockers": report["blockers"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["training_ready"] else 1


def _command_gpu_smoke(args: argparse.Namespace) -> int:
    result = run_final_gpu_smoke(
        args.output,
        preflight_report_path=args.preflight_report,
        acknowledge_gpu_execution=args.acknowledge_gpu_use,
    )
    print(
        json.dumps(
            {
                "output": str(result.path),
                "size_bytes": result.size_bytes,
                "sha256": result.sha256,
                "payload_sha256": result.payload_sha256,
                "status": "green",
            },
            sort_keys=True,
        )
    )
    return 0


def _command_gate_b_sft_preflight(args: argparse.Namespace) -> int:
    config = _load_locked_gate_b_config(args.config)
    if args.revision != config.revision:
        raise ValueError("--revision does not match the locked Gate B config")
    train, exclusions, manifest = _load_gate_b_data_contract(args)
    training_ids = eligible_training_ids(manifest, args.fold, exclusions.ids)
    validation_ids = eligible_validation_ids(manifest, args.fold, exclusions.ids)
    required_ids = set(training_ids) | set(validation_ids)
    development_records = tuple(record for record in train if record.id in required_ids)
    if len(development_records) != len(required_ids):  # pragma: no cover - split guard
        raise RuntimeError("SFT preflight could not resolve every development-CV row")
    training_records = _records_for_ids(train.records, training_ids)
    rationale_corpus, rationale_config = _load_optional_rationale_training_inputs(
        args,
        training_records=training_records,
        split_manifest=manifest,
        exclusions=exclusions,
    )
    tokenizer, tokenizer_provenance = load_pinned_tokenizer(
        revision=args.revision,
        local_files_only=True,
    )
    result = run_sft_encoding_preflight(
        development_records,
        split_manifest=manifest,
        excluded_ids=exclusions.ids,
        tokenizer=tokenizer,
        tokenizer_provenance=tokenizer_provenance,
        train_file_sha256=sha256_file(args.train),
        exclusions_file_sha256=exclusions.manifest.sha256,
        split_artifact_sha256=sha256_file(args.split_artifact),
        development_shard_sha256=args.expected_development_shard_sha256,
        output_path=args.output,
        folds=(args.fold,),
        rationale_corpus=rationale_corpus,
        rationale_config=rationale_config,
        config=config,
    )
    print(
        json.dumps(
            {
                "output": result.path,
                "sha256": result.sha256,
                "payload_sha256": result.payload_sha256,
                "fold": args.fold,
                "development_cv_count": len(required_ids),
                "torch_or_cuda_used": False,
                "locked_holdout_accessed": False,
                "training_target_kind": (
                    "direct_answer"
                    if rationale_config is None
                    else rationale_config.training_target_kind
                ),
            },
            sort_keys=True,
        )
    )
    return 0


def _command_build_rationale_corpus(args: argparse.Namespace) -> int:
    train, exclusions, manifest = _load_gate_b_data_contract(args)
    training_records = _records_for_ids(
        train.records,
        eligible_training_ids(manifest, args.fold, exclusions.ids),
    )
    config, config_file_sha256 = load_concise_rationale_config(
        args.rationale_config
    )
    result = build_rationale_corpus(
        args.source_jsonl,
        training_records,
        split_manifest=manifest,
        fold=args.fold,
        excluded_ids=exclusions.ids,
        candidate_config_file_sha256=config_file_sha256,
        output_jsonl=args.output_jsonl,
        output_manifest=args.output_manifest,
        config=config,
    )
    print(
        json.dumps(
            {
                **asdict(result),
                "training_target_kind": config.training_target_kind,
                "leaderboard_or_test_used": False,
                "locked_holdout_accessed": False,
                "torch_or_cuda_used": False,
            },
            sort_keys=True,
        )
    )
    return 0


def _command_gate_b_materialize_teacher_bank(args: argparse.Namespace) -> int:
    """Select an exact fold-training source from private finalized teacher banks."""

    train, exclusions, manifest = _load_gate_b_data_contract(args)
    expected_ids = eligible_training_ids(manifest, args.fold, exclusions.ids)
    bank_arguments = args.teacher_bank
    if not isinstance(bank_arguments, list) or not bank_arguments:
        raise ValueError("--teacher-bank is required at least once")
    banks: list[FinalizedTeacherBank] = []
    for index, raw_bank in enumerate(bank_arguments):
        if not isinstance(raw_bank, list) or len(raw_bank) != 3 or not all(
            isinstance(value, Path) for value in raw_bank
        ):
            raise ValueError(
                f"--teacher-bank occurrence {index + 1} must contain three paths"
            )
        banks.append(
            FinalizedTeacherBank(
                plan_dir=raw_bank[0],
                source_jsonl=raw_bank[1],
                source_manifest=raw_bank[2],
            )
        )
    result = materialize_teacher_bank_source(
        banks,
        _records_for_ids(train.records, expected_ids),
        split_manifest=manifest,
        fold=args.fold,
        excluded_ids=exclusions.ids,
        output_jsonl=args.output_jsonl,
        output_manifest=args.output_manifest,
    )
    print(
        json.dumps(
            {
                "event": "gate_b_teacher_bank_materialized",
                **result.as_dict(),
                "raw_rationale_serialized": False,
                "problem_id_serialized": False,
                "reference_answer_serialized": False,
                "leaderboard_or_test_used": False,
                "locked_holdout_accessed": False,
                "torch_or_cuda_used": False,
            },
            sort_keys=True,
        )
    )
    return 0


def _command_audit_rationale_corpus(args: argparse.Namespace) -> int:
    train, exclusions, manifest = _load_gate_b_data_contract(args)
    training_records = _records_for_ids(
        train.records,
        eligible_training_ids(manifest, args.fold, exclusions.ids),
    )
    config, config_file_sha256 = load_concise_rationale_config(
        args.rationale_config
    )
    result = audit_rationale_corpus(
        args.rationale_corpus,
        args.rationale_manifest,
        training_records,
        split_manifest=manifest,
        fold=args.fold,
        excluded_ids=exclusions.ids,
        candidate_config_file_sha256=config_file_sha256,
        output_path=args.output,
        config=config,
    )
    print(
        json.dumps(
            {
                **asdict(result),
                "raw_rationale_serialized": False,
                "leaderboard_or_test_used": False,
                "locked_holdout_accessed": False,
                "torch_or_cuda_used": False,
            },
            sort_keys=True,
        )
    )
    return 0


def _command_gate_b_teacher_plan(args: argparse.Namespace) -> int:
    """Create a question-only, fold-0-only Codex teacher plan."""

    teacher_config, config_file_sha256 = _load_locked_codex_teacher_config(
        args.teacher_config
    )
    _require_teacher_fold_zero(args.fold)
    train, exclusions, manifest = _load_gate_b_data_contract(args)
    fold0_training_ids = tuple(
        sorted(eligible_training_ids(manifest, args.fold, exclusions.ids))
    )
    allowed_ids = _teacher_plan_ids(
        train.records,
        fold0_training_ids,
        pilot_size=args.pilot_size,
        configured_pilot_size=_teacher_config_int(teacher_config, "pilot_size"),
    )
    pilot_receipt = None
    if args.pilot_size is None:
        pilot_receipt = _require_teacher_full_v1_pilot_authorization(
            args,
            teacher_config=teacher_config,
            config_file_sha256=config_file_sha256,
            train=train,
            exclusions=exclusions,
            manifest=manifest,
            fold0_training_ids=fold0_training_ids,
        )
    elif _teacher_pilot_authorization_argument_values(args):
        raise ValueError(
            "pilot authorization evidence is only valid when --pilot-size is omitted "
            "for the complete fold-0 v1 bank"
        )
    harness_evidence = _require_v4_teacher_harness_evidence(
        args,
        teacher_config=teacher_config,
        teacher_config_file_sha256=config_file_sha256,
    )
    codex_binary, codex_cli_version = _probe_codex_chatgpt_cli()
    execution = _teacher_execution_from_config(
        teacher_config,
        codex_binary=codex_binary,
        codex_cli_version=codex_cli_version,
    )
    if harness_evidence is not None:
        require_harness_live_execution_matches(
            harness_evidence.live_report,
            execution=execution,
        )
    plan = create_teacher_plan(
        _records_for_ids(train.records, allowed_ids),
        allowed_ids,
        args.output_dir,
        chunk_size=_teacher_config_int(teacher_config, "initial_chunk_size"),
        label=_teacher_config_string(teacher_config, "label"),
        version=_teacher_config_string(teacher_config, "version"),
        execution=execution,
        prompt_policy=_teacher_prompt_policy_from_config(teacher_config),
    )
    full_v1_authorization_payload_sha256 = None
    if pilot_receipt is not None:
        # The full v1 plan is an immutable promotion artifact.  Bind it to the
        # already re-verified pilot receipt so later private materialization
        # cannot accept an unqualified plan/source tuple.
        full_v1_authorization_payload_sha256 = write_teacher_full_v1_bank_authorization(
            plan.plan_dir,
            pilot_receipt,
        )
    harness_authorization_payload_sha256 = _write_v4_teacher_harness_authorization(
        plan,
        teacher_config=teacher_config,
        teacher_config_file_sha256=config_file_sha256,
        evidence=harness_evidence,
    )
    print(
        json.dumps(
            {
                "event": "gate_b_teacher_plan_created",
                "plan_dir": str(plan.plan_dir),
                "plan_sha256": plan.plan_sha256,
                "allowed_problem_count": len(plan.problem_ids),
                "chunk_count": len(plan.chunks),
                "allowed_ids_sha256": plan.allowed_ids_sha256,
                "teacher_config_sha256": _teacher_config_sha256(teacher_config),
                "teacher_config_file_sha256": config_file_sha256,
                "codex_cli_version": codex_cli_version,
                "pilot_authorization_file_sha256": (
                    pilot_receipt.file_sha256 if pilot_receipt is not None else None
                ),
                "pilot_authorization_payload_sha256": (
                    pilot_receipt.payload_sha256 if pilot_receipt is not None else None
                ),
                "full_v1_authorization_payload_sha256": (
                    full_v1_authorization_payload_sha256
                ),
                "harness_authorization_payload_sha256": harness_authorization_payload_sha256,
                "reference_answer_in_prompt": False,
                "locked_holdout_accessed": False,
                "leaderboard_or_test_used": False,
            },
            sort_keys=True,
        )
    )
    return 0


def _command_gate_b_teacher_pilot_authorize(args: argparse.Namespace) -> int:
    """Publish one immutable raw-free receipt only after every pilot gate passes."""

    teacher_config, config_file_sha256 = _load_locked_codex_teacher_config(
        args.teacher_config
    )
    _require_teacher_fold_zero(args.fold)
    train, exclusions, manifest = _load_gate_b_data_contract(args)
    fold0_training_ids = tuple(
        sorted(eligible_training_ids(manifest, args.fold, exclusions.ids))
    )
    contract = _teacher_pilot_authorization_contract(
        args,
        teacher_config=teacher_config,
        config_file_sha256=config_file_sha256,
        train=train,
        exclusions=exclusions,
        manifest=manifest,
        fold0_training_ids=fold0_training_ids,
    )
    pilot_plan_dir, source_jsonl, source_manifest, audit_dir = (
        _required_teacher_pilot_evidence_paths(args, require_receipt=False)
    )
    pilot_plan = load_teacher_plan(pilot_plan_dir)
    _require_teacher_plan_matches_config(pilot_plan, teacher_config)
    _require_v4_teacher_harness_evidence(
        args,
        teacher_config=teacher_config,
        teacher_config_file_sha256=config_file_sha256,
        plan_dir=pilot_plan.plan_dir,
    )
    audit_plan = load_teacher_logical_audit_plan(audit_dir)
    _require_teacher_logical_audit_plan_matches_contract(
        audit_plan,
        pilot_plan,
        teacher_config,
    )
    receipt = create_teacher_pilot_authorization(
        args.output,
        contract=contract,
        pilot_plan_dir=pilot_plan_dir,
        source_jsonl=source_jsonl,
        source_manifest=source_manifest,
        logical_audit_dir=audit_dir,
    )
    print(
        json.dumps(
            {
                "event": "gate_b_teacher_pilot_authorized",
                "teacher_config_sha256": _teacher_config_sha256(teacher_config),
                "teacher_config_file_sha256": config_file_sha256,
                "receipt": receipt.as_dict(),
                "pilot_problem_count": _teacher_config_int(teacher_config, "pilot_size"),
                "initial_exact_match_threshold_percent": (
                    PILOT_AUTHORIZATION_INITIAL_EXACT_MATCH_PERCENT
                ),
                "logical_audit_sample_size": PILOT_AUTHORIZATION_LOGICAL_AUDIT_SAMPLE_SIZE,
                "logical_audit_min_consistent": (
                    PILOT_AUTHORIZATION_LOGICAL_AUDIT_MIN_CONSISTENT
                ),
                "raw_generation_serialized": False,
                "reference_answer_in_prompt": False,
                "locked_holdout_accessed": False,
                "leaderboard_or_test_used": False,
                "torch_or_cuda_used": False,
            },
            sort_keys=True,
        )
    )
    return 0


def _command_gate_b_teacher_run(args: argparse.Namespace) -> int:
    """Execute a bounded, restartable Codex teacher round with no shell."""

    if args.acknowledge_codex_teacher is not True:
        raise ValueError(
            "--acknowledge-codex-teacher is required before Codex receives "
            "organizer training questions"
        )
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be a positive integer")
    teacher_config, config_file_sha256 = _load_locked_codex_teacher_config(
        args.teacher_config
    )
    plan = load_teacher_plan(args.plan_dir)
    _require_teacher_plan_matches_config(plan, teacher_config)
    _require_v4_teacher_harness_evidence(
        args,
        teacher_config=teacher_config,
        teacher_config_file_sha256=config_file_sha256,
        plan_dir=plan.plan_dir,
    )
    _require_v4_initial_threshold_not_failed(
        plan,
        teacher_config=teacher_config,
        teacher_config_file_sha256=config_file_sha256,
    )
    worker_cap = _teacher_config_int(teacher_config, "max_concurrent_workers")
    if args.max_workers > worker_cap:
        raise ValueError(
            f"--max-workers exceeds locked teacher cap: {args.max_workers} > {worker_cap}"
        )
    if (
        len(plan.problem_ids) == _teacher_config_int(teacher_config, "pilot_size")
        and args.max_workers != 1
    ):
        raise ValueError("the locked Codex teacher pilot requires --max-workers 1")
    _require_v4_teacher_run_wave(
        plan,
        teacher_config=teacher_config,
        max_invocations=args.max_invocations,
    )
    _require_current_codex_cli_execution(plan.execution)
    with tempfile.TemporaryDirectory(
        prefix="deep-challenge-codex-teacher-workspace-"
    ) as working_dir, tempfile.TemporaryDirectory(
        prefix="deep-challenge-codex-teacher-auth-"
    ) as auth_root:

        def run_one_command(command: tuple[str, ...]) -> CodexCommandResult:
            # The model's `-C` workspace remains empty.  Its auth-only
            # CODEX_HOME is a separately managed sibling temp tree, never
            # reachable from that workspace.
            with tempfile.TemporaryDirectory(
                prefix="codex-home-", dir=auth_root
            ) as home_parent:
                return _run_trusted_codex_teacher_command(
                    command,
                    execution=plan.execution,
                    timeout_seconds=args.timeout_seconds,
                    isolated_codex_home=_prepare_isolated_codex_home(
                        Path(home_parent)
                    ),
                )

        result = run_teacher_plan(
            plan.plan_dir,
            run_one_command,
            max_attempts=_teacher_config_int(teacher_config, "max_attempts"),
            repair_chunk_size=_teacher_config_int(
                teacher_config, "repair_chunk_size"
            ),
            max_chunks=args.max_invocations,
            max_workers=args.max_workers,
            working_directory=working_dir,
            allow_stale_lock_recovery=args.recover_stale_lock,
        )
    status = teacher_status(
        plan.plan_dir,
        max_attempts=_teacher_config_int(teacher_config, "max_attempts"),
    )
    print(
        json.dumps(
            {
                "event": "gate_b_teacher_run_finished",
                "teacher_config_sha256": _teacher_config_sha256(teacher_config),
                "teacher_config_file_sha256": config_file_sha256,
                "reasoning_effort_policy": {
                    "initial": _teacher_config_string(
                        teacher_config, "initial_reasoning_effort"
                    ),
                    "repair": _teacher_config_string(
                        teacher_config, "repair_reasoning_effort"
                    ),
                },
                "max_workers": args.max_workers,
                "run": result.as_dict(),
                "status": status.as_dict(),
                "raw_generation_serialized": False,
                "reference_answer_in_prompt": False,
                "locked_holdout_accessed": False,
                "leaderboard_or_test_used": False,
            },
            sort_keys=True,
        )
    )
    return 0


def _command_gate_b_teacher_status(args: argparse.Namespace) -> int:
    """Emit a monitor-safe teacher status without raw problems or answers."""

    teacher_config, config_file_sha256 = _load_locked_codex_teacher_config(
        args.teacher_config
    )
    plan = load_teacher_plan(args.plan_dir)
    _require_teacher_plan_matches_config(plan, teacher_config)
    _require_v4_teacher_harness_evidence(
        args,
        teacher_config=teacher_config,
        teacher_config_file_sha256=config_file_sha256,
        plan_dir=plan.plan_dir,
    )
    initial_threshold_failure = _load_v4_initial_threshold_failure(
        plan,
        teacher_config=teacher_config,
        teacher_config_file_sha256=config_file_sha256,
    )
    status = teacher_status(
        plan.plan_dir,
        max_attempts=_teacher_config_int(teacher_config, "max_attempts"),
    )
    payload = {
        "event": "gate_b_teacher_status",
        "teacher_config_sha256": _teacher_config_sha256(teacher_config),
        "teacher_config_file_sha256": config_file_sha256,
        "status": status.as_dict(),
        "initial_threshold_failure": initial_threshold_failure,
    }
    if args.output is not None:
        write_json_atomic(args.output, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


def _command_gate_b_teacher_diagnose(args: argparse.Namespace) -> int:
    """Classify private teacher failures without a resume or raw output leak."""

    teacher_config, config_file_sha256 = _load_locked_codex_teacher_config(args.teacher_config)
    plan = load_teacher_plan(args.plan_dir)
    _require_teacher_plan_matches_config(plan, teacher_config)
    result = diagnose_teacher_ledger(
        plan.plan_dir,
        args.output,
        teacher_config_sha256=_teacher_config_sha256(teacher_config),
        teacher_config_file_sha256=config_file_sha256,
        prompt_policy=_teacher_prompt_policy_from_config(teacher_config),
    )
    print(
        json.dumps(
            {
                "event": "gate_b_teacher_diagnostic_finished",
                "qualified": result.qualified,
                "report_sha256": result.report_sha256,
                "raw_generation_serialized": False,
                "ledger_modified": False,
            },
            sort_keys=True,
        )
    )
    return 0 if result.qualified else 1


def _command_gate_b_teacher_harness_replay(args: argparse.Namespace) -> int:
    """Execute the versioned synthetic fault matrix without Codex/network use."""

    harness_config, harness_file_sha256, profile = _load_locked_codex_teacher_harness_config(
        args.harness_config
    )
    teacher_config, teacher_file_sha256 = _load_locked_codex_teacher_config(args.teacher_config)
    _require_harness_allowed_teacher_config(teacher_config)
    result = run_harness_replay(
        args.output,
        harness_config_sha256=_teacher_config_sha256(harness_config),
        harness_config_file_sha256=harness_file_sha256,
        teacher_config_sha256=_teacher_config_sha256(teacher_config),
        teacher_config_file_sha256=teacher_file_sha256,
        prompt_policy=_teacher_prompt_policy_from_config(teacher_config),
        profile=profile,
    )
    print(
        json.dumps(
            {
                "event": "gate_b_teacher_harness_replay_finished",
                "qualified": result.qualified,
                "report_sha256": result.report_sha256,
                "codex_or_network_called": False,
            },
            sort_keys=True,
        )
    )
    return 0 if result.qualified else 1


def _command_gate_b_teacher_harness_live(args: argparse.Namespace) -> int:
    """Run the fixed versioned canary only after all pre-call contracts match."""

    if args.acknowledge_synthetic_codex_canary is not True:
        raise ValueError("--acknowledge-synthetic-codex-canary is required before the live canary")
    harness_config, harness_file_sha256, profile = _load_locked_codex_teacher_harness_config(
        args.harness_config
    )
    teacher_config, teacher_file_sha256 = _load_locked_codex_teacher_config(args.teacher_config)
    _require_harness_allowed_teacher_config(teacher_config)
    _require_new_harness_runtime_target(args.plan_dir, "synthetic harness plan")
    _require_new_harness_runtime_target(args.report, "synthetic harness report")
    if args.plan_dir.resolve(strict=False) == args.report.resolve(strict=False):
        raise ValueError("synthetic harness plan and report paths must differ")
    _require_harness_runtime_target_is_manifest_excluded(args.plan_dir, args.source_root)
    _require_harness_runtime_target_is_manifest_excluded(args.report, args.source_root)
    source_manifest = validate_source_tree_manifest_artifact(
        args.source_manifest,
        root=args.source_root,
    )
    codex_binary, codex_cli_version = _probe_codex_chatgpt_cli()
    execution = _teacher_execution_from_config(
        teacher_config,
        codex_binary=codex_binary,
        codex_cli_version=codex_cli_version,
    )
    with (
        tempfile.TemporaryDirectory(
            prefix="deep-challenge-codex-harness-workspace-"
        ) as working_dir,
        tempfile.TemporaryDirectory(prefix="deep-challenge-codex-harness-auth-") as auth_root,
    ):

        def run_one_command(command: tuple[str, ...]) -> CodexCommandResult:
            with tempfile.TemporaryDirectory(prefix="codex-home-", dir=auth_root) as home_parent:
                return _run_trusted_codex_teacher_command(
                    command,
                    execution=execution,
                    timeout_seconds=1_800,
                    isolated_codex_home=_prepare_isolated_codex_home(Path(home_parent)),
                )

        result = run_harness_live(
            args.plan_dir,
            args.report,
            harness_config_sha256=_teacher_config_sha256(harness_config),
            harness_config_file_sha256=harness_file_sha256,
            teacher_config_sha256=_teacher_config_sha256(teacher_config),
            teacher_config_file_sha256=teacher_file_sha256,
            prompt_policy=_teacher_prompt_policy_from_config(teacher_config),
            execution=execution,
            profile=profile,
            source_manifest=source_manifest,
            command_runner=run_one_command,
            working_directory=working_dir,
        )
    print(
        json.dumps(
            {
                "event": "gate_b_teacher_harness_live_finished",
                "qualified": result.qualified,
                "report_sha256": result.report_sha256,
                "plan_sha256": result.plan_sha256,
                "invocations": profile.chunk_count,
                "max_workers": profile.max_workers,
                "retry_count": profile.retry_count,
                "repair_count": profile.repair_count,
                "bank_output_count": profile.bank_output_count,
                "raw_generation_serialized": False,
            },
            sort_keys=True,
        )
    )
    return 0 if result.qualified else 1


def _require_new_harness_runtime_target(path: Path, label: str) -> None:
    """Reject overwritten/symlinked runtime targets before any Codex call starts."""

    if path.is_symlink() or path.parent.is_symlink() or not path.parent.is_dir():
        raise ValueError(f"{label} target is invalid")
    if path.exists():
        raise ValueError(f"{label} target already exists")


def _require_harness_runtime_target_is_manifest_excluded(path: Path, source_root: Path) -> None:
    """Keep runtime evidence out of the already-frozen source tree manifest."""

    root = source_root.resolve(strict=True)
    target = path.resolve(strict=False)
    if target.is_relative_to(root):
        relative = target.relative_to(root)
        if not relative.parts or relative.parts[0] != "artifacts":
            raise ValueError("synthetic harness runtime targets must be under excluded artifacts")


def _command_gate_b_teacher_finalize(args: argparse.Namespace) -> int:
    """Locally assess the private ledger against fold-0 training answers only."""

    teacher_config, config_file_sha256 = _load_locked_codex_teacher_config(
        args.teacher_config
    )
    _require_teacher_fold_zero(args.fold)
    train, exclusions, manifest = _load_gate_b_data_contract(args)
    expected_ids = _teacher_plan_ids(
        train.records,
        eligible_training_ids(manifest, args.fold, exclusions.ids),
        pilot_size=args.pilot_size,
        configured_pilot_size=_teacher_config_int(teacher_config, "pilot_size"),
    )
    plan = load_teacher_plan(args.plan_dir)
    _require_teacher_plan_matches_config(plan, teacher_config)
    _require_v4_teacher_harness_evidence(
        args,
        teacher_config=teacher_config,
        teacher_config_file_sha256=config_file_sha256,
        plan_dir=plan.plan_dir,
    )
    if plan.problem_ids != expected_ids:
        raise ValueError(
            "teacher plan IDs do not exactly match the derived fold-0 training scope"
        )
    result = finalize_teacher_bank(
        plan.plan_dir,
        _records_for_ids(train.records, expected_ids),
        output_jsonl=args.output_jsonl,
        output_manifest=args.output_manifest,
        max_attempts=_teacher_config_int(teacher_config, "max_attempts"),
        allow_stale_lock_recovery=args.recover_stale_lock,
    )
    initial_threshold_failure = _write_v4_initial_threshold_failure(
        plan,
        teacher_config=teacher_config,
        teacher_config_file_sha256=config_file_sha256,
        result=result,
    )
    print(
        json.dumps(
            {
                "event": "gate_b_teacher_finalize",
                "teacher_config_sha256": _teacher_config_sha256(teacher_config),
                "teacher_config_file_sha256": config_file_sha256,
                "result": result.as_dict(),
                "initial_threshold_failure": initial_threshold_failure,
                "reference_answer_used_locally": True,
                "reference_answer_in_prompt": False,
                "locked_holdout_accessed": False,
                "leaderboard_or_test_used": False,
            },
            sort_keys=True,
        )
    )
    return 1 if initial_threshold_failure is not None else 0


def _command_gate_b_teacher_logical_audit_plan(args: argparse.Namespace) -> int:
    """Create a fixed 64-row audit from an already verified private bank.

    This handler intentionally has no train/leaderboard arguments.  The
    lower-level planner re-derives the source-bank provenance from the original
    teacher ledger without opening organizer reference answers.
    """

    teacher_config, config_file_sha256 = _load_locked_codex_teacher_config(
        args.teacher_config
    )
    teacher_plan = load_teacher_plan(args.teacher_plan_dir)
    _require_teacher_plan_matches_config(teacher_plan, teacher_config)
    _require_v4_teacher_harness_evidence(
        args,
        teacher_config=teacher_config,
        teacher_config_file_sha256=config_file_sha256,
        plan_dir=teacher_plan.plan_dir,
    )
    codex_binary, codex_cli_version = _probe_codex_chatgpt_cli()
    execution = _teacher_execution_from_config(
        teacher_config,
        codex_binary=codex_binary,
        codex_cli_version=codex_cli_version,
    )
    profile = _LOCKED_CODEX_LOGICAL_AUDIT_PROFILE
    audit_plan = create_teacher_logical_audit_plan(
        teacher_plan.plan_dir,
        args.source_jsonl,
        args.source_manifest,
        args.output_dir,
        sample_size=profile["sample_size"],
        min_consistent=profile["min_consistent"],
        label=profile["label"],
        version=profile["version"],
        execution=execution,
    )
    print(
        json.dumps(
            {
                "event": "gate_b_teacher_logical_audit_plan_created",
                "audit_plan_dir": str(audit_plan.audit_dir),
                "audit_plan_sha256": audit_plan.plan_sha256,
                "teacher_plan_sha256": audit_plan.teacher_plan_sha256,
                "sample_size": audit_plan.sample_size,
                "min_consistent": audit_plan.min_consistent,
                "selected_ids_sha256": audit_plan.selected_ids_sha256,
                "teacher_config_sha256": _teacher_config_sha256(teacher_config),
                "teacher_config_file_sha256": config_file_sha256,
                "logical_audit_profile_sha256": _logical_audit_profile_sha256(),
                "codex_cli_version": codex_cli_version,
                "source_bank_provenance_reverified": True,
                "reference_answer_read": False,
                "reference_answer_in_prompt": False,
                "candidate_rationale_in_prompt": True,
                "tool_use_allowed": False,
                "locked_holdout_accessed": False,
                "leaderboard_or_test_used": False,
            },
            sort_keys=True,
        )
    )
    return 0


def _command_gate_b_teacher_logical_audit_run(args: argparse.Namespace) -> int:
    """Run at most one restartable candidate-only Codex logical audit."""

    if args.acknowledge_codex_teacher is not True:
        raise ValueError(
            "--acknowledge-codex-teacher is required before Codex receives private "
            "training questions and candidate rationales"
        )
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be a positive integer")
    (
        teacher_config,
        config_file_sha256,
        teacher_plan,
        audit_plan,
    ) = _load_verified_teacher_logical_audit_cli_contract(
        teacher_config_path=args.teacher_config,
        teacher_plan_dir=args.teacher_plan_dir,
        audit_dir=args.audit_dir,
    )
    _require_v4_teacher_harness_evidence(
        args,
        teacher_config=teacher_config,
        teacher_config_file_sha256=config_file_sha256,
        plan_dir=teacher_plan.plan_dir,
    )
    max_attempts = _teacher_config_int(teacher_config, "max_attempts")
    status_before = teacher_logical_audit_status(
        audit_plan.audit_dir,
        max_attempts=max_attempts,
    )
    reasoning_effort = _logical_audit_reasoning_effort(
        total_attempts=status_before.total_attempts,
        parsed_attempts=status_before.parsed_attempts,
        exhausted=status_before.exhausted,
        config=teacher_config,
    )
    _require_current_codex_cli_execution(audit_plan.execution)
    with tempfile.TemporaryDirectory(
        prefix="deep-challenge-codex-logical-audit-workspace-"
    ) as working_dir, tempfile.TemporaryDirectory(
        prefix="deep-challenge-codex-logical-audit-auth-"
    ) as auth_root:

        def run_one_command(command: tuple[str, ...]) -> CodexCommandResult:
            # Keep the auth-only home outside the model workspace, even for
            # an unexpected read-only tool call.
            with tempfile.TemporaryDirectory(
                prefix="codex-home-", dir=auth_root
            ) as home_parent:
                return _run_trusted_codex_teacher_command(
                    command,
                    execution=audit_plan.execution,
                    timeout_seconds=args.timeout_seconds,
                    isolated_codex_home=_prepare_isolated_codex_home(
                        Path(home_parent)
                    ),
                )

        result = run_teacher_logical_audit(
            audit_plan.audit_dir,
            run_one_command,
            max_attempts=max_attempts,
            working_directory=working_dir,
            allow_stale_lock_recovery=args.recover_stale_lock,
        )
    status = teacher_logical_audit_status(
        audit_plan.audit_dir,
        max_attempts=max_attempts,
    )
    print(
        json.dumps(
            {
                "event": "gate_b_teacher_logical_audit_run_finished",
                "teacher_plan_sha256": teacher_plan.plan_sha256,
                "teacher_config_sha256": _teacher_config_sha256(teacher_config),
                "teacher_config_file_sha256": config_file_sha256,
                "logical_audit_profile_sha256": _logical_audit_profile_sha256(),
                "reasoning_effort": reasoning_effort,
                "run": result.as_dict(),
                "status": status.as_dict(),
                "raw_generation_serialized": False,
                "reference_answer_read": False,
                "reference_answer_in_prompt": False,
                "candidate_rationale_in_prompt": True,
                "tool_use_allowed": False,
                "locked_holdout_accessed": False,
                "leaderboard_or_test_used": False,
            },
            sort_keys=True,
        )
    )
    return 0


def _command_gate_b_teacher_logical_audit_status(args: argparse.Namespace) -> int:
    """Emit a monitor-safe logical-audit status without questions or text."""

    teacher_config, config_file_sha256, teacher_plan, audit_plan = (
        _load_verified_teacher_logical_audit_cli_contract(
            teacher_config_path=args.teacher_config,
            teacher_plan_dir=args.teacher_plan_dir,
            audit_dir=args.audit_dir,
        )
    )
    _require_v4_teacher_harness_evidence(
        args,
        teacher_config=teacher_config,
        teacher_config_file_sha256=config_file_sha256,
        plan_dir=teacher_plan.plan_dir,
    )
    status = teacher_logical_audit_status(
        audit_plan.audit_dir,
        max_attempts=_teacher_config_int(teacher_config, "max_attempts"),
    )
    payload = {
        "event": "gate_b_teacher_logical_audit_status",
        "teacher_plan_sha256": teacher_plan.plan_sha256,
        "teacher_config_sha256": _teacher_config_sha256(teacher_config),
        "teacher_config_file_sha256": config_file_sha256,
        "logical_audit_profile_sha256": _logical_audit_profile_sha256(),
        "status": status.as_dict(),
        "raw_generation_serialized": False,
        "reference_answer_read": False,
        "locked_holdout_accessed": False,
        "leaderboard_or_test_used": False,
    }
    if args.output is not None:
        write_json_atomic(args.output, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


def _command_gate_b_teacher_logical_audit_finalize(args: argparse.Namespace) -> int:
    """Publish only the raw-free 64/60 audit verdict; no answers are loaded."""

    teacher_config, config_file_sha256, teacher_plan, audit_plan = (
        _load_verified_teacher_logical_audit_cli_contract(
            teacher_config_path=args.teacher_config,
            teacher_plan_dir=args.teacher_plan_dir,
            audit_dir=args.audit_dir,
        )
    )
    _require_v4_teacher_harness_evidence(
        args,
        teacher_config=teacher_config,
        teacher_config_file_sha256=config_file_sha256,
        plan_dir=teacher_plan.plan_dir,
    )
    result = finalize_teacher_logical_audit(
        audit_plan.audit_dir,
        max_attempts=_teacher_config_int(teacher_config, "max_attempts"),
        allow_stale_lock_recovery=args.recover_stale_lock,
    )
    print(
        json.dumps(
            {
                "event": "gate_b_teacher_logical_audit_finalize",
                "teacher_plan_sha256": teacher_plan.plan_sha256,
                "teacher_config_sha256": _teacher_config_sha256(teacher_config),
                "teacher_config_file_sha256": config_file_sha256,
                "logical_audit_profile_sha256": _logical_audit_profile_sha256(),
                "result": result.as_dict(),
                "reference_answer_read": False,
                "reference_answer_in_prompt": False,
                "candidate_rationale_in_prompt": True,
                "tool_use_allowed": False,
                "locked_holdout_accessed": False,
                "leaderboard_or_test_used": False,
            },
            sort_keys=True,
        )
    )
    return 0


def _command_gate_b_teacher_v2_plan(args: argparse.Namespace) -> int:
    """Create the separately authorized remaining-development-CV teacher plan."""

    teacher_config, config_file_sha256 = _load_locked_codex_teacher_config(
        args.teacher_config
    )
    _require_teacher_fold_zero(args.fold)
    train, exclusions, manifest = _load_gate_b_data_contract(args)
    scope = _derive_teacher_v2_scope(manifest, exclusions.ids, fold=args.fold)
    decision = _load_verified_candidate_probe_decision(
        args.candidate_probe_decision,
        candidate_label=args.candidate_label,
        manifest=manifest,
        fold=args.fold,
    )
    codex_binary, codex_cli_version = _probe_codex_chatgpt_cli()
    execution = _teacher_execution_from_config(
        teacher_config,
        codex_binary=codex_binary,
        codex_cli_version=codex_cli_version,
    )
    plan = create_teacher_plan(
        _records_for_ids(train.records, scope["remaining_ids"]),
        scope["remaining_ids"],
        args.output_dir,
        chunk_size=_teacher_config_int(teacher_config, "initial_chunk_size"),
        label=_CODEX_TEACHER_V2_PLAN_LABEL,
        version=_CODEX_TEACHER_V2_PLAN_VERSION,
        execution=execution,
        prompt_policy=_teacher_prompt_policy_from_config(teacher_config),
    )
    authorization_sha256 = _write_teacher_v2_authorization(
        plan,
        decision=decision,
        scope=scope,
    )
    print(
        json.dumps(
            {
                "event": "gate_b_teacher_v2_plan_created",
                "plan_dir": str(plan.plan_dir),
                "plan_sha256": plan.plan_sha256,
                "plan_label": plan.label,
                "plan_version": plan.version,
                "scope": _CODEX_TEACHER_V2_SCOPE,
                "allowed_problem_count": len(plan.problem_ids),
                "chunk_count": len(plan.chunks),
                "allowed_ids_sha256": plan.allowed_ids_sha256,
                "authorization_sha256": authorization_sha256,
                "candidate_probe_decision_sha256": decision["file_sha256"],
                "candidate_probe_decision_payload_sha256": decision["payload_sha256"],
                "teacher_config_sha256": _teacher_config_sha256(teacher_config),
                "teacher_config_file_sha256": config_file_sha256,
                "codex_cli_version": codex_cli_version,
                "reference_answer_in_prompt": False,
                "locked_holdout_accessed": False,
                "leaderboard_or_test_used": False,
            },
            sort_keys=True,
        )
    )
    return 0


def _command_gate_b_teacher_v2_run(args: argparse.Namespace) -> int:
    """Run an already authorized v2 teacher ledger without opening train Q/A."""

    if args.acknowledge_codex_teacher is not True:
        raise ValueError(
            "--acknowledge-codex-teacher is required before Codex receives "
            "organizer training questions"
        )
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be a positive integer")
    teacher_config, config_file_sha256 = _load_locked_codex_teacher_config(
        args.teacher_config
    )
    _require_teacher_fold_zero(args.fold)
    exclusions, manifest = _load_gate_b_preclaim_contract(args)
    scope = _derive_teacher_v2_scope(manifest, exclusions.ids, fold=args.fold)
    decision = _load_verified_candidate_probe_decision(
        args.candidate_probe_decision,
        candidate_label=args.candidate_label,
        manifest=manifest,
        fold=args.fold,
    )
    plan = load_teacher_plan(args.plan_dir)
    _require_teacher_v2_plan_matches_config(plan, teacher_config)
    _require_teacher_v2_authorization(plan, decision=decision, scope=scope)
    worker_cap = _teacher_config_int(teacher_config, "max_concurrent_workers")
    if args.max_workers > worker_cap:
        raise ValueError(
            f"--max-workers exceeds locked teacher cap: {args.max_workers} > {worker_cap}"
        )
    _require_current_codex_cli_execution(plan.execution)
    with tempfile.TemporaryDirectory(
        prefix="deep-challenge-codex-teacher-workspace-"
    ) as working_dir, tempfile.TemporaryDirectory(
        prefix="deep-challenge-codex-teacher-auth-"
    ) as auth_root:

        def run_one_command(command: tuple[str, ...]) -> CodexCommandResult:
            with tempfile.TemporaryDirectory(
                prefix="codex-home-", dir=auth_root
            ) as home_parent:
                return _run_trusted_codex_teacher_command(
                    command,
                    execution=plan.execution,
                    timeout_seconds=args.timeout_seconds,
                    isolated_codex_home=_prepare_isolated_codex_home(
                        Path(home_parent)
                    ),
                )

        result = run_teacher_plan(
            plan.plan_dir,
            run_one_command,
            max_attempts=_teacher_config_int(teacher_config, "max_attempts"),
            repair_chunk_size=_teacher_config_int(
                teacher_config, "repair_chunk_size"
            ),
            max_chunks=args.max_invocations,
            max_workers=args.max_workers,
            working_directory=working_dir,
            allow_stale_lock_recovery=args.recover_stale_lock,
        )
    status = teacher_status(
        plan.plan_dir,
        max_attempts=_teacher_config_int(teacher_config, "max_attempts"),
    )
    print(
        json.dumps(
            {
                "event": "gate_b_teacher_v2_run_finished",
                "scope": _CODEX_TEACHER_V2_SCOPE,
                "teacher_config_sha256": _teacher_config_sha256(teacher_config),
                "teacher_config_file_sha256": config_file_sha256,
                "candidate_probe_decision_sha256": decision["file_sha256"],
                "reasoning_effort_policy": {
                    "initial": _teacher_config_string(
                        teacher_config, "initial_reasoning_effort"
                    ),
                    "repair": _teacher_config_string(
                        teacher_config, "repair_reasoning_effort"
                    ),
                },
                "max_workers": args.max_workers,
                "run": result.as_dict(),
                "status": status.as_dict(),
                "raw_generation_serialized": False,
                "reference_answer_in_prompt": False,
                "locked_holdout_accessed": False,
                "leaderboard_or_test_used": False,
            },
            sort_keys=True,
        )
    )
    return 0


def _command_gate_b_teacher_v2_status(args: argparse.Namespace) -> int:
    """Emit raw-free status only after re-verifying v2 authorization binding."""

    teacher_config, config_file_sha256 = _load_locked_codex_teacher_config(
        args.teacher_config
    )
    _require_teacher_fold_zero(args.fold)
    exclusions, manifest = _load_gate_b_preclaim_contract(args)
    scope = _derive_teacher_v2_scope(manifest, exclusions.ids, fold=args.fold)
    decision = _load_verified_candidate_probe_decision(
        args.candidate_probe_decision,
        candidate_label=args.candidate_label,
        manifest=manifest,
        fold=args.fold,
    )
    plan = load_teacher_plan(args.plan_dir)
    _require_teacher_v2_plan_matches_config(plan, teacher_config)
    _require_teacher_v2_authorization(plan, decision=decision, scope=scope)
    status = teacher_status(
        plan.plan_dir,
        max_attempts=_teacher_config_int(teacher_config, "max_attempts"),
    )
    payload = {
        "event": "gate_b_teacher_v2_status",
        "scope": _CODEX_TEACHER_V2_SCOPE,
        "teacher_config_sha256": _teacher_config_sha256(teacher_config),
        "teacher_config_file_sha256": config_file_sha256,
        "candidate_probe_decision_sha256": decision["file_sha256"],
        "status": status.as_dict(),
    }
    if args.output is not None:
        write_json_atomic(args.output, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


def _command_gate_b_teacher_v2_finalize(args: argparse.Namespace) -> int:
    """Assess only the sealed v2 scope against local organizer-train answers."""

    teacher_config, config_file_sha256 = _load_locked_codex_teacher_config(
        args.teacher_config
    )
    _require_teacher_fold_zero(args.fold)
    train, exclusions, manifest = _load_gate_b_data_contract(args)
    scope = _derive_teacher_v2_scope(manifest, exclusions.ids, fold=args.fold)
    decision = _load_verified_candidate_probe_decision(
        args.candidate_probe_decision,
        candidate_label=args.candidate_label,
        manifest=manifest,
        fold=args.fold,
    )
    plan = load_teacher_plan(args.plan_dir)
    _require_teacher_v2_plan_matches_config(plan, teacher_config)
    _require_teacher_v2_authorization(plan, decision=decision, scope=scope)
    if plan.problem_ids != scope["remaining_ids"]:
        raise ValueError(
            "v2 teacher plan IDs do not exactly match the derived remaining development-CV scope"
        )
    result = finalize_teacher_bank(
        plan.plan_dir,
        _records_for_ids(train.records, scope["remaining_ids"]),
        output_jsonl=args.output_jsonl,
        output_manifest=args.output_manifest,
        max_attempts=_teacher_config_int(teacher_config, "max_attempts"),
        allow_stale_lock_recovery=args.recover_stale_lock,
    )
    print(
        json.dumps(
            {
                "event": "gate_b_teacher_v2_finalize",
                "scope": _CODEX_TEACHER_V2_SCOPE,
                "teacher_config_sha256": _teacher_config_sha256(teacher_config),
                "teacher_config_file_sha256": config_file_sha256,
                "candidate_probe_decision_sha256": decision["file_sha256"],
                "result": result.as_dict(),
                "reference_answer_used_locally": True,
                "reference_answer_in_prompt": False,
                "locked_holdout_accessed": False,
                "leaderboard_or_test_used": False,
            },
            sort_keys=True,
        )
    )
    return 0


def _command_gate_b_development(args: argparse.Namespace) -> int:
    _require_gpu_cli_acknowledgement(args.acknowledge_gpu_use)
    config = _load_locked_gate_b_config(args.config)
    source_manifest = validate_source_tree_manifest_artifact(
        args.source_manifest,
        root=args.source_root,
    )
    _require_new_development_targets(args.output_jsonl, args.output_manifest)
    train, exclusions, manifest = _load_gate_b_data_contract(args)
    validation_records = _records_for_ids(
        train.records,
        eligible_validation_ids(manifest, args.fold, exclusions.ids),
    )
    if args.adapter is None:
        if args.base_baseline_manifest is not None:
            raise ValueError(
                "--base-baseline-manifest is valid only when --adapter is supplied"
            )
        backend = create_base_development_backend(
            preflight_artifact=args.preflight_report,
            gpu_smoke_artifact=args.gpu_smoke_report,
            gpu_acknowledgement=GPU_EXECUTION_ACKNOWLEDGEMENT,
            config=config,
        )
        run_kind = "fixed_base"
    else:
        if args.base_baseline_manifest is None:
            raise ValueError(
                "adapter development requires --base-baseline-manifest from the same fold"
            )
        require_base_development_artifact(
            args.base_baseline_manifest,
            validation_records,
            split_manifest=manifest,
            fold=args.fold,
            excluded_ids=exclusions.ids,
            expected_checkpoint_sha256=BASE_MODEL_CHECKPOINT_SHA256,
        )
        backend = create_adapted_development_backend(
            preflight_artifact=args.preflight_report,
            gpu_smoke_artifact=args.gpu_smoke_report,
            adapter_path=args.adapter,
            split_manifest=manifest,
            fold=args.fold,
            excluded_ids=exclusions.ids,
            train_file_sha256=sha256_file(args.train),
            exclusions_file_sha256=exclusions.manifest.sha256,
            split_artifact_sha256=sha256_file(args.split_artifact),
            development_shard_sha256=args.expected_development_shard_sha256,
            gpu_acknowledgement=GPU_EXECUTION_ACKNOWLEDGEMENT,
            config=config,
        )
        run_kind = "verified_adapter"
    execution_evidence = _capture_development_execution_evidence(
        backend,
        source_manifest=source_manifest,
        config_path=args.config,
        config=config,
        preflight_report=args.preflight_report,
        gpu_smoke_report=args.gpu_smoke_report,
    )
    try:
        def report_progress(completed: int, total: int) -> None:
            if completed == 1 or completed == total or completed % 25 == 0:
                print(
                    json.dumps(
                        {
                            "event": "gate_b_development_progress",
                            "completed_generations": completed,
                            "total_generations": total,
                            "fold": args.fold,
                            "run_kind": run_kind,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        records = run_development_baseline(
            validation_records,
            split_manifest=manifest,
            fold=args.fold,
            excluded_ids=exclusions.ids,
            backend=backend,
            checkpoint_sha256=backend.checkpoint_sha256,
            config=config,
            samples_per_problem=1,
            progress_callback=report_progress,
            resume_dir=args.resume_dir,
            chunk_size=args.resume_chunk_size,
        )
        result = write_development_artifacts(
            records,
            jsonl_path=args.output_jsonl,
            manifest_path=args.output_manifest,
            execution_evidence=execution_evidence,
            resume_dir=args.resume_dir,
        )
    finally:
        backend.close()
    print(
        json.dumps(
            {
                "run_kind": run_kind,
                "records": result.records_path,
                "manifest": result.manifest_path,
                "record_count": result.record_count,
                "records_sha256": result.records_sha256,
                "manifest_sha256": result.manifest_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


def _command_gate_b_development_status(args: argparse.Namespace) -> int:
    status = development_workflow_status(
        read_development_resume_status(args.resume_dir)
    ).as_dict()
    if args.output is not None:
        write_json_atomic(args.output, status)
    print(json.dumps(status, sort_keys=True))
    return 0


def _command_audit_parser_golden(args: argparse.Namespace) -> int:
    result = audit_development_parser_golden(
        args.records,
        args.manifest,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "output": result.path,
                "size_bytes": result.size_bytes,
                "sha256": result.sha256,
                "payload_sha256": result.payload_sha256,
                "case_count": result.case_count,
                "raw_completion_serialized": False,
                "locked_holdout_accessed": False,
                "leaderboard_or_test_used": False,
            },
            sort_keys=True,
        )
    )
    return 0


def _command_audit_parser_rescore(args: argparse.Namespace) -> int:
    result = audit_development_parser_rescore(
        args.records,
        args.manifest,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "output": result.path,
                "size_bytes": result.size_bytes,
                "sha256": result.sha256,
                "payload_sha256": result.payload_sha256,
                "changed_parser_result_count": result.case_count,
                "selection_eligible": False,
                "raw_completion_serialized": False,
                "locked_holdout_accessed": False,
                "leaderboard_or_test_used": False,
            },
            sort_keys=True,
        )
    )
    return 0


def _command_gate_b_train_fold(args: argparse.Namespace) -> int:
    _require_gpu_cli_acknowledgement(args.acknowledge_gpu_use)
    config = _load_locked_gate_b_config(args.config)
    source_manifest = validate_source_tree_manifest_artifact(
        args.source_manifest,
        root=args.source_root,
    )
    train, exclusions, manifest = _load_gate_b_data_contract(args)
    training_records = _records_for_ids(
        train.records,
        eligible_training_ids(manifest, args.fold, exclusions.ids),
    )
    validation_records = _records_for_ids(
        train.records,
        eligible_validation_ids(manifest, args.fold, exclusions.ids),
    )
    rationale_corpus, rationale_config = _load_optional_rationale_training_inputs(
        args,
        training_records=training_records,
        split_manifest=manifest,
        exclusions=exclusions,
    )
    require_base_development_artifact(
        args.base_baseline_manifest,
        validation_records,
        split_manifest=manifest,
        fold=args.fold,
        excluded_ids=exclusions.ids,
        expected_checkpoint_sha256=BASE_MODEL_CHECKPOINT_SHA256,
    )
    result = train_qlora_fold(
        training_records,
        validation_records,
        split_manifest=manifest,
        fold=args.fold,
        excluded_ids=exclusions.ids,
        train_file_sha256=sha256_file(args.train),
        exclusions_file_sha256=exclusions.manifest.sha256,
        split_artifact_sha256=sha256_file(args.split_artifact),
        development_shard_sha256=args.expected_development_shard_sha256,
        source_manifest=source_manifest,
        preflight_artifact=args.preflight_report,
        gpu_smoke_artifact=args.gpu_smoke_report,
        output_dir=args.output_dir,
        gpu_acknowledgement=GPU_EXECUTION_ACKNOWLEDGEMENT,
        resume_dir=args.resume_dir,
        rationale_corpus=rationale_corpus,
        rationale_config=rationale_config,
        config=config,
    )
    print(json.dumps(asdict(result), sort_keys=True))
    return 0


def _command_gate_b_training_status(args: argparse.Namespace) -> int:
    status = training_workflow_status(
        read_training_resume_status(args.resume_dir)
    ).as_dict()
    if args.output is not None:
        write_json_atomic(args.output, status)
    print(json.dumps(status, sort_keys=True))
    return 0


def _command_gate_b_locked_holdout_evaluate(args: argparse.Namespace) -> int:
    _require_gpu_cli_acknowledgement(args.acknowledge_gpu_use)
    if args.acknowledge_one_time_locked_holdout is not True:
        raise ValueError(
            "--acknowledge-one-time-locked-holdout is required before consuming the claim"
        )
    config = _load_locked_gate_b_config(args.config)
    exclusions, manifest = _load_gate_b_preclaim_contract(args)
    train_file_sha256 = sha256_file(args.train)
    exclusions_file_sha256 = exclusions.manifest.sha256
    split_artifact_sha256 = sha256_file(args.split_artifact)

    def load_records_after_claim() -> Sequence[MathRecord]:
        train = load_train_csv(args.train)
        verified_manifest, _ = _load_verified_split(train, args.split_artifact)
        if verified_manifest.sha256 != manifest.sha256:  # pragma: no cover - byte binding
            raise RuntimeError("split manifest changed after the holdout claim")
        verified_exclusions = load_train_exclusions_csv(
            args.train_exclusions,
            train_ids=(record.id for record in train),
        )
        if (
            verified_exclusions.manifest.sha256 != exclusions_file_sha256
            or verified_exclusions.ids != exclusions.ids
        ):
            raise RuntimeError("train exclusions changed after the holdout claim")
        if train.manifest.sha256 != train_file_sha256:
            raise RuntimeError("organizer train changed after the holdout claim")
        return train.records

    primary = None
    fallback = None
    try:
        primary = _create_gate_b_backend(
            kind=args.primary_kind,
            adapter_path=args.primary_adapter,
            role="primary",
            manifest=manifest,
            fold=args.fold,
            excluded_ids=exclusions.ids,
            train_file_sha256=train_file_sha256,
            exclusions_file_sha256=exclusions_file_sha256,
            split_artifact_sha256=split_artifact_sha256,
            development_shard_sha256=args.expected_development_shard_sha256,
            preflight_report=args.preflight_report,
            gpu_smoke_report=args.gpu_smoke_report,
            config=config,
        )
        if args.fallback_kind == "none":
            if args.fallback_adapter is not None:
                raise ValueError("--fallback-adapter is invalid with --fallback-kind none")
        else:
            fallback = _create_gate_b_backend(
                kind=args.fallback_kind,
                adapter_path=args.fallback_adapter,
                role="fallback",
                manifest=manifest,
                fold=args.fold,
                excluded_ids=exclusions.ids,
                train_file_sha256=train_file_sha256,
                exclusions_file_sha256=exclusions_file_sha256,
                split_artifact_sha256=split_artifact_sha256,
                development_shard_sha256=args.expected_development_shard_sha256,
                preflight_report=args.preflight_report,
                gpu_smoke_report=args.gpu_smoke_report,
                config=config,
            )
        result = evaluate_locked_holdout_once(
            load_records_after_claim,
            split_manifest=manifest,
            excluded_ids=exclusions.ids,
            train_file_sha256=train_file_sha256,
            exclusions_file_sha256=exclusions_file_sha256,
            excluded_ids_sha256=exclusions.ids_sha256,
            split_artifact_sha256=split_artifact_sha256,
            development_shard_sha256=args.expected_development_shard_sha256,
            fold=args.fold,
            freeze_artifact=args.freeze_artifact,
            primary_backend=primary,
            fallback_backend=fallback,
            output_path=args.output,
            holdout_acknowledgement=HOLDOUT_ACCESS_ACKNOWLEDGEMENT,
            config=config,
        )
    finally:
        if fallback is not None:
            fallback.close()
        if primary is not None:
            primary.close()
    print(
        json.dumps(
            {
                "output": result.path,
                "sha256": result.sha256,
                "payload_sha256": result.payload_sha256,
                "holdout_claim_consumed": True,
                "selection_after_evaluation_forbidden": True,
            },
            sort_keys=True,
        )
    )
    return 0


def _command_gate_b_predict_evaluation(args: argparse.Namespace) -> int:
    _require_gpu_cli_acknowledgement(args.acknowledge_gpu_use)
    config = _load_locked_gate_b_config(args.config)
    exclusions, manifest = _load_gate_b_preclaim_contract(args)
    train_file_sha256 = sha256_file(args.train)
    exclusions_file_sha256 = exclusions.manifest.sha256
    split_artifact_sha256 = sha256_file(args.split_artifact)
    primary = None
    fallback = None
    try:
        primary = _create_gate_b_backend(
            kind=args.primary_kind,
            adapter_path=args.primary_adapter,
            role="primary",
            manifest=manifest,
            fold=args.fold,
            excluded_ids=exclusions.ids,
            train_file_sha256=train_file_sha256,
            exclusions_file_sha256=exclusions_file_sha256,
            split_artifact_sha256=split_artifact_sha256,
            development_shard_sha256=args.expected_development_shard_sha256,
            preflight_report=args.preflight_report,
            gpu_smoke_report=args.gpu_smoke_report,
            config=config,
        )
        if args.fallback_kind == "none":
            if args.fallback_adapter is not None:
                raise ValueError("--fallback-adapter is invalid with --fallback-kind none")
        else:
            fallback = _create_gate_b_backend(
                kind=args.fallback_kind,
                adapter_path=args.fallback_adapter,
                role="fallback",
                manifest=manifest,
                fold=args.fold,
                excluded_ids=exclusions.ids,
                train_file_sha256=train_file_sha256,
                exclusions_file_sha256=exclusions_file_sha256,
                split_artifact_sha256=split_artifact_sha256,
                development_shard_sha256=args.expected_development_shard_sha256,
                preflight_report=args.preflight_report,
                gpu_smoke_report=args.gpu_smoke_report,
                config=config,
            )
        result = run_frozen_evaluation_inference(
            dataset_role=args.dataset_role,
            evaluation_file_path=args.evaluation,
            expected_evaluation_sha256=args.expected_evaluation_sha256,
            split_manifest=manifest,
            excluded_ids=exclusions.ids,
            train_file_sha256=train_file_sha256,
            exclusions_file_sha256=exclusions_file_sha256,
            excluded_ids_sha256=exclusions.ids_sha256,
            split_artifact_sha256=split_artifact_sha256,
            development_shard_sha256=args.expected_development_shard_sha256,
            fold=args.fold,
            freeze_artifact=args.freeze_artifact,
            primary_backend=primary,
            fallback_backend=fallback,
            artifact_path=args.output_artifact,
            predictions_path=args.output_predictions,
            config=config,
        )
    finally:
        if fallback is not None:
            fallback.close()
        if primary is not None:
            primary.close()
    print(json.dumps(asdict(result), sort_keys=True))
    return 0 if result.invalid_count == 0 else 1


def _create_gate_b_backend(
    *,
    kind: str,
    adapter_path: Path | None,
    role: str,
    manifest: SplitManifest,
    fold: int,
    excluded_ids: Sequence[str],
    train_file_sha256: str,
    exclusions_file_sha256: str,
    split_artifact_sha256: str,
    development_shard_sha256: str,
    preflight_report: Path,
    gpu_smoke_report: Path,
    config: GateBConfig,
):
    if kind == "base":
        if adapter_path is not None:
            raise ValueError(f"--{role}-adapter is invalid when --{role}-kind is base")
        return create_base_development_backend(
            preflight_artifact=preflight_report,
            gpu_smoke_artifact=gpu_smoke_report,
            gpu_acknowledgement=GPU_EXECUTION_ACKNOWLEDGEMENT,
            config=config,
        )
    if kind != "adapter":  # pragma: no cover - argparse constrains the value
        raise ValueError(f"unsupported {role} backend kind: {kind!r}")
    if adapter_path is None:
        raise ValueError(f"--{role}-adapter is required when --{role}-kind is adapter")
    return create_adapted_development_backend(
        preflight_artifact=preflight_report,
        gpu_smoke_artifact=gpu_smoke_report,
        adapter_path=adapter_path,
        split_manifest=manifest,
        fold=fold,
        excluded_ids=excluded_ids,
        train_file_sha256=train_file_sha256,
        exclusions_file_sha256=exclusions_file_sha256,
        split_artifact_sha256=split_artifact_sha256,
        development_shard_sha256=development_shard_sha256,
        gpu_acknowledgement=GPU_EXECUTION_ACKNOWLEDGEMENT,
        config=config,
    )


def _require_gpu_cli_acknowledgement(acknowledged: bool) -> None:
    if acknowledged is not True:
        raise ValueError(
            "--acknowledge-gpu-use is required before any Gate B CUDA command"
        )


def _capture_development_execution_evidence(
    backend: object,
    *,
    source_manifest: object,
    config_path: Path,
    config: GateBConfig,
    preflight_report: Path,
    gpu_smoke_report: Path,
):
    """Bind a development manifest to the exact B0 gate used by its backend."""

    runtime_gate = getattr(backend, "runtime_gate_evidence", None)
    if not isinstance(runtime_gate, RuntimeGateEvidence):
        raise RuntimeError(
            "development backend must expose validated RuntimeGateEvidence for publication"
        )
    return create_development_execution_evidence(
        source_manifest=source_manifest,
        config_path=config_path,
        config_sha256=config.sha256,
        preflight_report_path=preflight_report,
        preflight_report_sha256=runtime_gate.preflight_sha256,
        gpu_smoke_report_path=gpu_smoke_report,
        gpu_smoke_report_sha256=runtime_gate.smoke_sha256,
        gpu_device_name=runtime_gate.device_name,
    )


def _load_locked_codex_teacher_config(path: Path) -> tuple[dict[str, object], str]:
    """Load the public no-API teacher profile without permitting drift."""

    payload = _load_json_object(path)
    stored_sha256 = payload.pop("config_sha256", None)
    if not isinstance(stored_sha256, str):
        raise ValueError("Codex teacher config is missing config_sha256")
    semantic_sha256 = _teacher_config_sha256(payload)
    schema_version = payload.get("schema_version")
    expected = (
        _LOCKED_CODEX_TEACHER_CONFIGS.get(schema_version)
        if isinstance(schema_version, str)
        else None
    )
    if payload != expected or stored_sha256 != semantic_sha256:
        raise ValueError("Codex teacher config differs from the locked no-API profile")
    return payload, sha256_file(path)


def _load_locked_codex_teacher_harness_config(
    path: Path,
) -> tuple[dict[str, object], str, HarnessProfile]:
    """Load an immutable versioned synthetic canary profile without drift."""

    payload = _load_json_object(path)
    stored_sha256 = payload.pop("config_sha256", None)
    if not isinstance(stored_sha256, str):
        raise ValueError("Codex teacher harness config is missing config_sha256")
    semantic_sha256 = _teacher_config_sha256(payload)
    expected = _LOCKED_CODEX_TEACHER_HARNESS_CONFIGS.get(payload.get("schema_version"))
    if payload != expected or stored_sha256 != semantic_sha256:
        raise ValueError("Codex teacher harness config differs from the locked profile")
    return payload, sha256_file(path), profile_from_config(payload)


@dataclass(frozen=True, slots=True)
class _TeacherHarnessEvidence:
    """Verified private paths and portable hashes for one v4 authorization."""

    harness_config_sha256: str
    harness_config_file_sha256: str
    replay_report: Path
    live_report: Path
    live_plan_dir: Path
    source_manifest: SourceTreeArtifactEvidence


def _teacher_harness_argument_values(args: argparse.Namespace) -> tuple[object, ...]:
    """Return the complete v4 evidence bundle without serializing any path."""

    return (
        args.harness_config,
        args.harness_replay_report,
        args.harness_live_report,
        args.harness_live_plan_dir,
        args.harness_source_root,
        args.harness_source_manifest,
    )


def _require_v4_teacher_harness_evidence(
    args: argparse.Namespace,
    *,
    teacher_config: dict[str, object],
    teacher_config_file_sha256: str,
    plan_dir: Path | None = None,
) -> _TeacherHarnessEvidence | None:
    """Fail closed unless v4 replay/live evidence matches the current source.

    Historic profiles neither require nor accept these new arguments, preserving
    their established CLI behavior and serialized ledgers.  V4 validates the
    same qualified evidence before plan creation and revalidates the plan-local
    immutable sidecar before every later operation.
    """

    values = _teacher_harness_argument_values(args)
    is_v4 = teacher_config.get("schema_version") == _V4_TEACHER_CONFIG_SCHEMA
    if not is_v4:
        if any(value is not None for value in values):
            raise ValueError("synthetic harness evidence is only valid for teacher-pilot-v4")
        return None
    if any(value is None for value in values):
        raise ValueError(
            "teacher-pilot-v4 requires harness config, qualified replay/live evidence, "
            "and a current harness source manifest"
        )

    harness_config_path = args.harness_config
    replay_report = args.harness_replay_report
    live_report = args.harness_live_report
    live_plan_dir = args.harness_live_plan_dir
    source_root = args.harness_source_root
    source_manifest_path = args.harness_source_manifest
    assert isinstance(harness_config_path, Path)
    assert isinstance(replay_report, Path)
    assert isinstance(live_report, Path)
    assert isinstance(live_plan_dir, Path)
    assert isinstance(source_root, Path)
    assert isinstance(source_manifest_path, Path)

    harness_config, harness_config_file_sha256, _profile = (
        _load_locked_codex_teacher_harness_config(harness_config_path)
    )
    _require_harness_allowed_teacher_config(teacher_config)
    source_manifest = validate_source_tree_manifest_artifact(
        source_manifest_path,
        root=source_root,
    )
    evidence = _TeacherHarnessEvidence(
        harness_config_sha256=_teacher_config_sha256(harness_config),
        harness_config_file_sha256=harness_config_file_sha256,
        replay_report=replay_report,
        live_report=live_report,
        live_plan_dir=live_plan_dir,
        source_manifest=source_manifest,
    )
    authorization_kwargs = {
        "replay_report": evidence.replay_report,
        "live_report": evidence.live_report,
        "live_plan_dir": evidence.live_plan_dir,
        "harness_config_sha256": evidence.harness_config_sha256,
        "harness_config_file_sha256": evidence.harness_config_file_sha256,
        "teacher_config_sha256": _teacher_config_sha256(teacher_config),
        "teacher_config_file_sha256": teacher_config_file_sha256,
        "prompt_policy": _teacher_prompt_policy_from_config(teacher_config),
        "source_manifest": evidence.source_manifest,
    }
    if plan_dir is None:
        validate_harness_evidence(**authorization_kwargs)
    else:
        verify_harness_authorization(
            plan_dir / HARNESS_AUTHORIZATION_FILENAME,
            **authorization_kwargs,
        )
        require_harness_live_execution_matches(
            evidence.live_report,
            execution=load_teacher_plan(plan_dir).execution,
        )
    return evidence


def _write_v4_teacher_harness_authorization(
    plan: TeacherPlan,
    *,
    teacher_config: dict[str, object],
    teacher_config_file_sha256: str,
    evidence: _TeacherHarnessEvidence | None,
) -> str | None:
    """Publish the no-overwrite v4 sidecar after a pre-validated plan exists."""

    if evidence is None:
        return None
    require_harness_live_execution_matches(
        evidence.live_report,
        execution=plan.execution,
    )
    return create_harness_authorization(
        plan.plan_dir / HARNESS_AUTHORIZATION_FILENAME,
        replay_report=evidence.replay_report,
        live_report=evidence.live_report,
        live_plan_dir=evidence.live_plan_dir,
        harness_config_sha256=evidence.harness_config_sha256,
        harness_config_file_sha256=evidence.harness_config_file_sha256,
        teacher_config_sha256=_teacher_config_sha256(teacher_config),
        teacher_config_file_sha256=teacher_config_file_sha256,
        prompt_policy=_teacher_prompt_policy_from_config(teacher_config),
        source_manifest=evidence.source_manifest,
    )


def _load_v4_initial_threshold_failure(
    plan: TeacherPlan,
    *,
    teacher_config: dict[str, object],
    teacher_config_file_sha256: str,
) -> dict[str, object] | None:
    """Load a raw-free v4 fail-fast marker without weakening historic plans."""

    if teacher_config.get("schema_version") != _V4_TEACHER_CONFIG_SCHEMA:
        return None
    path = plan.plan_dir / _V4_INITIAL_THRESHOLD_FAILURE_FILENAME
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise ValueError("teacher-pilot-v4 initial threshold marker is invalid")
    payload = _load_json_object(path)
    expected_keys = {
        "schema_version",
        "plan_sha256",
        "teacher_config_sha256",
        "teacher_config_file_sha256",
        "total_problem_count",
        "accepted_problem_count",
        "retryable_problem_count",
        "exhausted_problem_count",
        "unassessed_problem_count",
        "total_attempts",
        "minimum_initial_accepted_count",
        "failure_category",
        "payload_sha256",
    }
    if set(payload) != expected_keys:
        raise ValueError("teacher-pilot-v4 initial threshold marker keys are invalid")
    payload_sha256 = payload.get("payload_sha256")
    if (
        not isinstance(payload_sha256, str)
        or len(payload_sha256) != 64
        or any(character not in "0123456789abcdef" for character in payload_sha256)
    ):
        raise ValueError("teacher-pilot-v4 initial threshold marker SHA is invalid")
    unhashed = dict(payload)
    unhashed.pop("payload_sha256")
    if hashlib.sha256(canonical_json_bytes(unhashed)).hexdigest() != payload_sha256:
        raise ValueError("teacher-pilot-v4 initial threshold marker SHA does not match")
    integer_fields = (
        "total_problem_count",
        "accepted_problem_count",
        "retryable_problem_count",
        "exhausted_problem_count",
        "unassessed_problem_count",
        "total_attempts",
        "minimum_initial_accepted_count",
    )
    if any(
        isinstance(payload[field], bool)
        or not isinstance(payload[field], int)
        or payload[field] < 0
        for field in integer_fields
    ):
        raise ValueError("teacher-pilot-v4 initial threshold marker counts are invalid")
    if (
        payload["schema_version"] != _V4_INITIAL_THRESHOLD_FAILURE_SCHEMA
        or payload["plan_sha256"] != plan.plan_sha256
        or payload["teacher_config_sha256"] != _teacher_config_sha256(teacher_config)
        or payload["teacher_config_file_sha256"] != teacher_config_file_sha256
        or payload["total_problem_count"] != _teacher_config_int(teacher_config, "pilot_size")
        or payload["accepted_problem_count"] >= _V4_INITIAL_MIN_ACCEPTED
        or payload["minimum_initial_accepted_count"] != _V4_INITIAL_MIN_ACCEPTED
        or payload["failure_category"] != "initial_exact_match_below_threshold"
    ):
        raise ValueError("teacher-pilot-v4 initial threshold marker does not match the plan")
    return payload


def _write_v4_initial_threshold_failure(
    plan: TeacherPlan,
    *,
    teacher_config: dict[str, object],
    teacher_config_file_sha256: str,
    result: TeacherBankFinalizeResult,
) -> dict[str, object] | None:
    """Atomically mark an unrecoverable v4 first-wave threshold failure.

    A marker is written only after every initial chunk has one finalized attempt.
    It contains counts and hashes only, and blocks every later teacher execution
    for this plan while preserving status inspection and forensic evidence.
    """

    if teacher_config.get("schema_version") != _V4_TEACHER_CONFIG_SCHEMA:
        return None
    existing = _load_v4_initial_threshold_failure(
        plan,
        teacher_config=teacher_config,
        teacher_config_file_sha256=teacher_config_file_sha256,
    )
    if existing is not None:
        return existing
    if (
        result.complete
        or result.total_problem_count != _teacher_config_int(teacher_config, "pilot_size")
        or result.accepted_problem_count >= _V4_INITIAL_MIN_ACCEPTED
    ):
        return None
    status = teacher_status(
        plan.plan_dir,
        max_attempts=_teacher_config_int(teacher_config, "max_attempts"),
    )
    if (
        status.total_problem_count != result.total_problem_count
        or status.accepted_problem_count != result.accepted_problem_count
        or status.total_attempts != len(plan.chunks)
    ):
        return None
    payload_without_hash: dict[str, object] = {
        "schema_version": _V4_INITIAL_THRESHOLD_FAILURE_SCHEMA,
        "plan_sha256": plan.plan_sha256,
        "teacher_config_sha256": _teacher_config_sha256(teacher_config),
        "teacher_config_file_sha256": teacher_config_file_sha256,
        "total_problem_count": status.total_problem_count,
        "accepted_problem_count": status.accepted_problem_count,
        "retryable_problem_count": status.retryable_problem_count,
        "exhausted_problem_count": status.exhausted_problem_count,
        "unassessed_problem_count": status.unassessed_problem_count,
        "total_attempts": status.total_attempts,
        "minimum_initial_accepted_count": _V4_INITIAL_MIN_ACCEPTED,
        "failure_category": "initial_exact_match_below_threshold",
    }
    payload = {
        **payload_without_hash,
        "payload_sha256": hashlib.sha256(
            canonical_json_bytes(payload_without_hash)
        ).hexdigest(),
    }
    target = plan.plan_dir / _V4_INITIAL_THRESHOLD_FAILURE_FILENAME
    if target.is_symlink() or target.exists():
        raise ValueError("teacher-pilot-v4 initial threshold marker already exists")
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
            raise ValueError("teacher-pilot-v4 initial threshold marker already exists") from exc
        directory_descriptor = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()
    return payload


def _require_v4_initial_threshold_not_failed(
    plan: TeacherPlan,
    *,
    teacher_config: dict[str, object],
    teacher_config_file_sha256: str,
) -> None:
    """Refuse any post-threshold teacher call while keeping status readable."""

    if _load_v4_initial_threshold_failure(
        plan,
        teacher_config=teacher_config,
        teacher_config_file_sha256=teacher_config_file_sha256,
    ) is not None:
        raise ValueError(
            "teacher-pilot-v4 initial threshold failed; refusing further teacher execution"
        )


def _require_v4_teacher_run_wave(
    plan: TeacherPlan,
    *,
    teacher_config: dict[str, object],
    max_invocations: int | None,
) -> None:
    """Fix v4's first and repair waves before any Codex command is probed.

    A four-chunk initial wave is a single, bounded comparison.  It cannot be
    split into selective resumes.  After a local finalizer has assessed it,
    repair remains bounded to two canonical chunks per wave.  This does not
    change historic profiles or their existing command behavior.
    """

    if teacher_config.get("schema_version") != _V4_TEACHER_CONFIG_SCHEMA:
        return
    if len(plan.problem_ids) != _teacher_config_int(teacher_config, "pilot_size"):
        return
    status = teacher_status(
        plan.plan_dir,
        max_attempts=_teacher_config_int(teacher_config, "max_attempts"),
    )
    if status.total_attempts == 0:
        if max_invocations != len(plan.chunks):
            raise ValueError(
                "teacher-pilot-v4 initial wave requires exactly --max-invocations 4"
            )
        return
    if status.total_attempts < len(plan.chunks):
        raise ValueError(
            "teacher-pilot-v4 initial wave must complete all four chunks before a resume"
        )
    if status.unassessed_problem_count:
        raise ValueError(
            "teacher-pilot-v4 requires local finalization before any repair wave"
        )
    if max_invocations not in (1, 2):
        raise ValueError(
            "teacher-pilot-v4 repair waves require --max-invocations 1 or 2"
        )


def _require_harness_allowed_teacher_config(config: dict[str, object]) -> None:
    """Disallow historic ledger profiles from starting a fresh harness run."""

    schema_version = config.get("schema_version")
    if schema_version not in _HARNESS_ALLOWED_TEACHER_CONFIG_SCHEMAS:
        raise ValueError("teacher config is not allowlisted for the live synthetic harness")


def _teacher_config_sha256(config: dict[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(config)).hexdigest()


def _logical_audit_profile_sha256() -> str:
    """Return the semantic hash for the fixed candidate-only audit contract."""

    return _teacher_config_sha256(_LOCKED_CODEX_LOGICAL_AUDIT_PROFILE)


def _teacher_config_string(config: dict[str, object], field: str) -> str:
    value = config.get(field)
    if not isinstance(value, str):  # pragma: no cover - locked config check above
        raise RuntimeError(f"locked Codex teacher config field is not text: {field}")
    return value


def _teacher_config_int(config: dict[str, object], field: str) -> int:
    value = config.get(field)
    if isinstance(value, bool) or not isinstance(value, int):  # pragma: no cover
        raise RuntimeError(f"locked Codex teacher config field is not an integer: {field}")
    return value


def _teacher_prompt_policy_from_config(config: dict[str, object]) -> TeacherPromptPolicy:
    """Return the only prompt policy approved by one immutable config profile."""

    prompt_version = config.get("prompt_version", "gate-b-codex-teacher-prompt-v1")
    if not isinstance(prompt_version, str):  # pragma: no cover - locked config check above
        raise RuntimeError("locked Codex teacher prompt version is not text")
    template_sha256 = config.get("prompt_template_sha256")
    if template_sha256 is not None and not isinstance(template_sha256, str):  # pragma: no cover
        raise RuntimeError("locked Codex teacher prompt template SHA is not text")
    return TeacherPromptPolicy(
        prompt_version=prompt_version,
        prompt_template_sha256=template_sha256,
    )


def _teacher_execution_from_config(
    config: dict[str, object],
    *,
    codex_binary: str,
    codex_cli_version: str,
) -> TeacherExecutionConfig:
    return TeacherExecutionConfig(
        provider=_teacher_config_string(config, "provider"),
        model_id=_teacher_config_string(config, "model_id"),
        model_revision=_teacher_config_string(config, "model_revision"),
        codex_cli_version=codex_cli_version,
        reasoning_effort=_teacher_config_string(config, "initial_reasoning_effort"),
        codex_binary=codex_binary,
        seed=_teacher_config_int(config, "seed"),
    )


def _require_current_codex_cli_execution(
    execution: TeacherExecutionConfig,
) -> tuple[str, str]:
    """Fail closed if a resume would use a different Codex executable/version.

    The private plan records the resolved executable and CLI version at plan
    creation.  Re-probing immediately before every privileged teacher run
    makes that provenance live rather than merely historical, and prevents a
    recomputed plan from selecting an arbitrary executable.
    """

    codex_binary, codex_cli_version = _probe_codex_chatgpt_cli()
    if (
        execution.codex_binary != codex_binary
        or execution.codex_cli_version != codex_cli_version
    ):
        raise ValueError(
            "fresh ChatGPT Codex CLI binary/version does not match the immutable "
            "teacher execution plan"
        )
    return codex_binary, codex_cli_version


def _teacher_pilot_authorization_argument_values(
    args: argparse.Namespace,
) -> tuple[Path, ...]:
    """Return supplied private v1-pilot evidence paths, omitting absent flags."""

    values = (
        getattr(args, "pilot_authorization", None),
        getattr(args, "pilot_plan_dir", None),
        getattr(args, "pilot_source_jsonl", None),
        getattr(args, "pilot_source_manifest", None),
        getattr(args, "pilot_logical_audit_dir", None),
    )
    if any(value is not None and not isinstance(value, Path) for value in values):
        raise RuntimeError("teacher pilot evidence argument is not a path")
    return tuple(value for value in values if isinstance(value, Path))


def _required_teacher_pilot_evidence_paths(
    args: argparse.Namespace,
    *,
    require_receipt: bool,
) -> tuple[Path, ...]:
    """Require every live private artifact needed to re-verify promotion."""

    names = (
        "pilot_authorization",
        "pilot_plan_dir",
        "pilot_source_jsonl",
        "pilot_source_manifest",
        "pilot_logical_audit_dir",
    )
    if not require_receipt:
        names = names[1:]
    values = tuple(getattr(args, name, None) for name in names)
    missing = [name for name, value in zip(names, values, strict=True) if value is None]
    if missing:
        raise ValueError(
            "complete fold-0 v1 teacher planning requires a passed pilot authorization "
            f"and all live private pilot evidence; missing={missing!r}"
        )
    if any(not isinstance(value, Path) for value in values):
        raise RuntimeError("teacher pilot evidence argument is not a path")
    return tuple(value for value in values if isinstance(value, Path))


def _teacher_pilot_authorization_contract(
    args: argparse.Namespace,
    *,
    teacher_config: dict[str, object],
    config_file_sha256: str,
    train: CsvDataset,
    exclusions: TrainExclusionSet,
    manifest: SplitManifest,
    fold0_training_ids: tuple[str, ...],
) -> TeacherPilotAuthorizationContract:
    """Re-derive the exact 128-row pilot from the current sealed split contract."""

    pilot_ids = _teacher_plan_ids(
        train.records,
        fold0_training_ids,
        pilot_size=_teacher_config_int(teacher_config, "pilot_size"),
        configured_pilot_size=_teacher_config_int(teacher_config, "pilot_size"),
    )
    return TeacherPilotAuthorizationContract(
        teacher_config_sha256=_teacher_config_sha256(teacher_config),
        teacher_config_file_sha256=config_file_sha256,
        teacher_prompt_policy_sha256=_teacher_prompt_policy_from_config(
            teacher_config
        ).sha256,
        train_sha256=train.manifest.sha256,
        exclusions_sha256=exclusions.manifest.sha256,
        exclusion_count=len(exclusions),
        split_artifact_sha256=sha256_file(args.split_artifact),
        development_shard_sha256=sha256_file(
            args.development_shard / "CHECKSUMS.sha256"
        ),
        split_version=manifest.version,
        split_sha256=manifest.sha256,
        source_groups_sha256=manifest.source_groups_sha256,
        fold=args.fold,
        fold0_training_ids=fold0_training_ids,
        pilot_ids=pilot_ids,
        teacher_plan_label=_teacher_config_string(teacher_config, "label"),
        teacher_plan_version=_teacher_config_string(teacher_config, "version"),
        logical_audit_label=str(_LOCKED_CODEX_LOGICAL_AUDIT_PROFILE["label"]),
        logical_audit_version=str(_LOCKED_CODEX_LOGICAL_AUDIT_PROFILE["version"]),
    )


def _require_teacher_full_v1_pilot_authorization(
    args: argparse.Namespace,
    *,
    teacher_config: dict[str, object],
    config_file_sha256: str,
    train: CsvDataset,
    exclusions: TrainExclusionSet,
    manifest: SplitManifest,
    fold0_training_ids: tuple[str, ...],
):
    """Fail closed unless an immutable receipt and live evidence still agree."""

    (
        authorization_path,
        pilot_plan_dir,
        source_jsonl,
        source_manifest,
        audit_dir,
    ) = _required_teacher_pilot_evidence_paths(args, require_receipt=True)
    contract = _teacher_pilot_authorization_contract(
        args,
        teacher_config=teacher_config,
        config_file_sha256=config_file_sha256,
        train=train,
        exclusions=exclusions,
        manifest=manifest,
        fold0_training_ids=fold0_training_ids,
    )
    pilot_plan = load_teacher_plan(pilot_plan_dir)
    _require_teacher_plan_matches_config(pilot_plan, teacher_config)
    audit_plan = load_teacher_logical_audit_plan(audit_dir)
    _require_teacher_logical_audit_plan_matches_contract(
        audit_plan,
        pilot_plan,
        teacher_config,
    )
    return verify_teacher_pilot_authorization(
        authorization_path,
        contract=contract,
        pilot_plan_dir=pilot_plan_dir,
        source_jsonl=source_jsonl,
        source_manifest=source_manifest,
        logical_audit_dir=audit_dir,
    )


def _require_teacher_fold_zero(fold: int) -> None:
    if fold != 0:
        raise ValueError("Codex teacher plans are locked to fold 0 development training IDs")


def _require_teacher_plan_matches_config(
    plan: TeacherPlan, config: dict[str, object]
) -> None:
    """Ensure an immutable private plan still corresponds to the public profile."""

    if (
        plan.label != _teacher_config_string(config, "label")
        or plan.version != _teacher_config_string(config, "version")
        or plan.execution.provider != _teacher_config_string(config, "provider")
        or plan.execution.model_id != _teacher_config_string(config, "model_id")
        or plan.execution.model_revision != _teacher_config_string(config, "model_revision")
        or plan.execution.reasoning_effort
        != _teacher_config_string(config, "initial_reasoning_effort")
        or plan.execution.seed != _teacher_config_int(config, "seed")
        or plan.execution.codex_cli_version == "unknown"
        or plan.prompt_policy != _teacher_prompt_policy_from_config(config)
    ):
        raise ValueError("teacher plan does not match the locked Codex teacher profile")


def _require_teacher_logical_audit_plan_matches_contract(
    audit_plan: TeacherLogicalAuditPlan,
    teacher_plan: TeacherPlan,
    config: dict[str, object],
) -> None:
    """Bind an audit to its immutable teacher plan and the fixed 64/60 gate."""

    profile = _LOCKED_CODEX_LOGICAL_AUDIT_PROFILE
    execution = audit_plan.execution
    if (
        audit_plan.label != profile["label"]
        or audit_plan.version != profile["version"]
        or audit_plan.sample_size != profile["sample_size"]
        or audit_plan.min_consistent != profile["min_consistent"]
        or audit_plan.teacher_plan_sha256 != teacher_plan.plan_sha256
        or execution.provider != _teacher_config_string(config, "provider")
        or execution.model_id != _teacher_config_string(config, "model_id")
        or execution.model_revision != _teacher_config_string(config, "model_revision")
        or execution.reasoning_effort
        != _teacher_config_string(config, "initial_reasoning_effort")
        or execution.seed != _teacher_config_int(config, "seed")
        or execution.codex_cli_version == "unknown"
    ):
        raise ValueError(
            "teacher logical-audit plan does not match the locked 64/60 Codex contract"
        )


def _load_verified_teacher_logical_audit_cli_contract(
    *,
    teacher_config_path: Path,
    teacher_plan_dir: Path,
    audit_dir: Path,
) -> tuple[dict[str, object], str, TeacherPlan, TeacherLogicalAuditPlan]:
    """Load only provenance-bound private plans; never organizer train answers."""

    teacher_config, config_file_sha256 = _load_locked_codex_teacher_config(
        teacher_config_path
    )
    teacher_plan = load_teacher_plan(teacher_plan_dir)
    _require_teacher_plan_matches_config(teacher_plan, teacher_config)
    audit_plan = load_teacher_logical_audit_plan(audit_dir)
    _require_teacher_logical_audit_plan_matches_contract(
        audit_plan,
        teacher_plan,
        teacher_config,
    )
    return teacher_config, config_file_sha256, teacher_plan, audit_plan


def _logical_audit_reasoning_effort(
    *,
    total_attempts: int,
    parsed_attempts: int,
    exhausted: bool,
    config: dict[str, object],
) -> str:
    """Select high initially and xhigh only for an automatic audit retry."""

    if parsed_attempts > 0 or exhausted:
        # ``run_teacher_logical_audit`` will publish no new attempt in either
        # terminal state.  Keep the returned value deterministic and private.
        return _teacher_config_string(config, "initial_reasoning_effort")
    return _teacher_config_string(
        config,
        "initial_reasoning_effort" if total_attempts == 0 else "repair_reasoning_effort",
    )


def _require_teacher_v2_plan_matches_config(
    plan: TeacherPlan, config: dict[str, object]
) -> None:
    """Verify the immutable v2 scope label plus the unchanged teacher profile."""

    if (
        plan.label != _CODEX_TEACHER_V2_PLAN_LABEL
        or plan.version != _CODEX_TEACHER_V2_PLAN_VERSION
        or plan.execution.provider != _teacher_config_string(config, "provider")
        or plan.execution.model_id != _teacher_config_string(config, "model_id")
        or plan.execution.model_revision != _teacher_config_string(config, "model_revision")
        or plan.execution.reasoning_effort
        != _teacher_config_string(config, "initial_reasoning_effort")
        or plan.execution.seed != _teacher_config_int(config, "seed")
        or plan.execution.codex_cli_version == "unknown"
        or plan.prompt_policy != _teacher_prompt_policy_from_config(config)
    ):
        raise ValueError("v2 teacher plan does not match its locked scope/profile")


def _derive_teacher_v2_scope(
    manifest: SplitManifest,
    excluded_ids: Sequence[str],
    *,
    fold: int,
) -> dict[str, Any]:
    """Derive the only permissible v2 IDs without loading organizer answers.

    The v1 bank contains ``training_ids(0)``.  The only remaining eligible
    development-CV rows are therefore fold 0 validation rows.  Re-deriving the
    result from the complete CV union makes that implication explicit and
    fails closed if future split behavior would violate it.
    """

    _require_teacher_fold_zero(fold)
    fold0_training_ids = tuple(
        sorted(eligible_training_ids(manifest, fold, excluded_ids))
    )
    development_cv_ids = tuple(
        sorted(
            problem_id
            for fold_index in range(manifest.n_folds)
            for problem_id in eligible_validation_ids(
                manifest, fold_index, excluded_ids
            )
        )
    )
    if len(development_cv_ids) != len(set(development_cv_ids)):
        raise RuntimeError("eligible development-CV scope contains duplicate IDs")
    remaining_ids = tuple(
        sorted(set(development_cv_ids) - set(fold0_training_ids))
    )
    fold0_validation_ids = tuple(
        sorted(eligible_validation_ids(manifest, fold, excluded_ids))
    )
    if remaining_ids != fold0_validation_ids:
        raise RuntimeError(
            "remaining development-CV scope must equal fold-0 validation IDs"
        )
    if not remaining_ids:
        raise ValueError("v2 teacher scope has no eligible remaining development-CV IDs")
    holdout_ids = set(manifest.final_holdout_ids())
    if (
        set(remaining_ids) & set(fold0_training_ids)
        or set(remaining_ids) & holdout_ids
        or set(fold0_training_ids) & holdout_ids
    ):
        raise RuntimeError("v2 teacher scope crosses a sealed split boundary")
    return {
        "fold": fold,
        "development_cv_ids": development_cv_ids,
        "fold0_training_ids": fold0_training_ids,
        "remaining_ids": remaining_ids,
    }


def _load_verified_candidate_probe_decision(
    decision_path: Path,
    *,
    candidate_label: str,
    manifest: SplitManifest,
    fold: int,
) -> dict[str, Any]:
    """Recompute and bind a positive single-fold probe decision fail-closed.

    A self-hash alone does not establish that a decision was generated from its
    comparison evidence.  This intentionally re-runs the pure CPU decision
    function into a temporary no-publish path and requires byte-equivalent
    payload semantics before the teacher expansion can start.
    """

    raw = Path(decision_path)
    if raw.is_symlink():
        raise ValueError("candidate probe decision refuses symbolic links")
    source = raw.resolve(strict=True)
    if not source.is_file() or source.is_symlink():
        raise ValueError("candidate probe decision must be a regular file")
    payload = _load_json_object(source)
    expected_keys = {
        "schema_version",
        "decision_scope",
        "policy",
        "comparison_artifact",
        "model_id",
        "revision",
        "split_version",
        "split_sha256",
        "source_groups_sha256",
        "fold",
        "reference_label",
        "candidate_label",
        "evidence",
        "significant_regression",
        "candidate_action",
        "candidate_full_oof_authorized",
        "final_selection_eligible",
        "complete_oof_required_before_freeze",
        "selection_frozen",
        "locked_holdout_accessed",
        "leaderboard_or_test_used",
        "payload_sha256",
    }
    if set(payload) != expected_keys:
        raise ValueError("candidate probe decision schema differs from the locked v1 schema")
    stored_payload_sha256 = payload.get("payload_sha256")
    if (
        not isinstance(stored_payload_sha256, str)
        or len(stored_payload_sha256) != 64
        or any(character not in "0123456789abcdef" for character in stored_payload_sha256)
    ):
        raise ValueError("candidate probe decision payload SHA-256 is invalid")
    payload_without_hash = dict(payload)
    payload_without_hash.pop("payload_sha256")
    if (
        hashlib.sha256(canonical_json_bytes(payload_without_hash)).hexdigest()
        != stored_payload_sha256
    ):
        raise ValueError("candidate probe decision payload SHA-256 does not match content")
    if payload.get("schema_version") != _CANDIDATE_PROBE_DECISION_SCHEMA:
        raise ValueError("candidate probe decision schema version is unsupported")
    if payload.get("candidate_label") != candidate_label:
        raise ValueError("candidate probe decision does not authorize the requested candidate")
    if (
        payload.get("candidate_full_oof_authorized") is not True
        or payload.get("candidate_action") != "continue_to_complete_oof"
        or payload.get("significant_regression") is not False
        or payload.get("final_selection_eligible") is not False
        or payload.get("complete_oof_required_before_freeze") is not True
        or payload.get("selection_frozen") is not False
        or payload.get("locked_holdout_accessed") is not False
        or payload.get("leaderboard_or_test_used") is not False
    ):
        raise ValueError("candidate probe decision does not authorize v2 teacher expansion")
    comparison = payload.get("comparison_artifact")
    if not isinstance(comparison, dict):
        raise ValueError("candidate probe decision comparison evidence is invalid")
    comparison_path = comparison.get("path")
    if not isinstance(comparison_path, str) or not comparison_path:
        raise ValueError("candidate probe decision comparison path is invalid")
    comparison_source = Path(comparison_path)
    if comparison_source.is_symlink() or not comparison_source.is_file():
        raise ValueError("candidate probe comparison artifact is unavailable")
    try:
        with tempfile.TemporaryDirectory(
            prefix="deep-challenge-probe-decision-verify-"
        ) as temporary_directory:
            recomputed_path = Path(temporary_directory) / "decision.json"
            decide_candidate_probe_promotion(
                comparison_source,
                candidate_label=candidate_label,
                output_path=recomputed_path,
            )
            recomputed = _load_json_object(recomputed_path)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(
            "candidate probe decision cannot be independently verified from its comparison"
        ) from exc
    if payload != recomputed:
        raise ValueError(
            "candidate probe decision does not match its independently recomputed evidence"
        )
    if (
        payload.get("model_id") != OFFICIAL_MODEL_ID
        or payload.get("revision") != OFFICIAL_REVISION
        or payload.get("decision_scope") != "single_fold_gpu_cost_control_only"
        or payload.get("split_version") != manifest.version
        or payload.get("split_sha256") != manifest.sha256
        or payload.get("source_groups_sha256") != manifest.source_groups_sha256
        or payload.get("fold") != fold
    ):
        raise ValueError("candidate probe decision does not match the current split/model contract")
    return {
        "candidate_label": candidate_label,
        "candidate_full_oof_authorized": payload["candidate_full_oof_authorized"],
        "candidate_action": payload["candidate_action"],
        "file_sha256": sha256_file(source),
        "payload_sha256": stored_payload_sha256,
        "split_sha256": manifest.sha256,
        "source_groups_sha256": manifest.source_groups_sha256,
        "fold": fold,
    }


def _teacher_v2_authorization_payload(
    plan: TeacherPlan,
    *,
    decision: dict[str, Any],
    scope: dict[str, Any],
) -> dict[str, Any]:
    """Return the raw-free immutable v2-plan authorization payload."""

    return {
        "schema_version": _CODEX_TEACHER_V2_AUTHORIZATION_SCHEMA,
        "plan_sha256": plan.plan_sha256,
        "plan_label": plan.label,
        "plan_version": plan.version,
        "scope": _CODEX_TEACHER_V2_SCOPE,
        "fold": scope["fold"],
        "candidate_label": decision["candidate_label"],
        "candidate_full_oof_authorized": decision["candidate_full_oof_authorized"],
        "candidate_action": decision["candidate_action"],
        "candidate_probe_decision_sha256": decision["file_sha256"],
        "candidate_probe_decision_payload_sha256": decision["payload_sha256"],
        "split_sha256": decision["split_sha256"],
        "source_groups_sha256": decision["source_groups_sha256"],
        "eligible_development_cv_ids_sha256": _ids_sha256(
            scope["development_cv_ids"]
        ),
        "fold0_training_ids_sha256": _ids_sha256(scope["fold0_training_ids"]),
        "allowed_ids_sha256": plan.allowed_ids_sha256,
        "allowed_problem_count": len(plan.problem_ids),
    }


def _write_teacher_v2_authorization(
    plan: TeacherPlan,
    *,
    decision: dict[str, Any],
    scope: dict[str, Any],
) -> str:
    """Publish the v2 decision binding once, atomically and without overwrite."""

    if plan.problem_ids != scope["remaining_ids"]:
        raise RuntimeError("v2 authorization cannot bind a plan outside its remaining scope")
    payload_without_hash = _teacher_v2_authorization_payload(
        plan,
        decision=decision,
        scope=scope,
    )
    payload = {
        **payload_without_hash,
        "payload_sha256": hashlib.sha256(
            canonical_json_bytes(payload_without_hash)
        ).hexdigest(),
    }
    serialized = (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")
    _write_bytes_noreplace(
        plan.plan_dir / _CODEX_TEACHER_V2_AUTHORIZATION_FILENAME,
        serialized,
        label="v2 teacher authorization",
    )
    return hashlib.sha256(serialized).hexdigest()


def _require_teacher_v2_authorization(
    plan: TeacherPlan,
    *,
    decision: dict[str, Any],
    scope: dict[str, Any],
) -> None:
    """Ensure run/status/finalize receive exactly the approval bound at planning."""

    if plan.problem_ids != scope["remaining_ids"]:
        raise ValueError("v2 teacher plan IDs are outside the sealed remaining scope")
    path = plan.plan_dir / _CODEX_TEACHER_V2_AUTHORIZATION_FILENAME
    if path.is_symlink() or not path.is_file():
        raise ValueError("v2 teacher plan is missing its immutable authorization binding")
    payload = _load_json_object(path)
    expected_without_hash = _teacher_v2_authorization_payload(
        plan,
        decision=decision,
        scope=scope,
    )
    expected = {
        **expected_without_hash,
        "payload_sha256": hashlib.sha256(
            canonical_json_bytes(expected_without_hash)
        ).hexdigest(),
    }
    if payload != expected:
        raise ValueError("v2 teacher authorization does not match its plan/probe/scope")


def _write_bytes_noreplace(path: Path, payload: bytes, *, label: str) -> None:
    """Atomically publish one private sidecar without replacing an existing file."""

    raw = Path(path)
    if raw.is_symlink() or raw.parent.is_symlink():
        raise ValueError(f"{label} refuses symbolic links")
    target = raw.resolve(strict=False)
    if not target.parent.is_dir():
        raise ValueError(f"{label} parent must be an existing regular directory")
    if target.exists():
        raise ValueError(f"refusing to overwrite {label}")
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
            raise ValueError(f"refusing to overwrite {label}") from exc
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


def _teacher_plan_ids(
    records: Sequence[MathRecord],
    training_ids: Sequence[str],
    *,
    pilot_size: int | None,
    configured_pilot_size: int,
) -> tuple[str, ...]:
    """Return all fold-0 training IDs or its locked deterministic pilot subset."""

    canonical_ids = tuple(sorted(training_ids))
    if len(canonical_ids) != len(set(canonical_ids)):
        raise RuntimeError("fold-0 training IDs must be unique")
    if pilot_size is None:
        return canonical_ids
    if pilot_size != configured_pilot_size:
        raise ValueError(
            f"--pilot-size is locked to {configured_pilot_size} when it is supplied"
        )
    if pilot_size > len(canonical_ids):
        raise ValueError("locked teacher pilot is larger than the fold-0 training scope")
    by_id = {record.id: record for record in records}
    if set(canonical_ids) - set(by_id):  # pragma: no cover - data contract precedes this
        raise RuntimeError("teacher pilot IDs are missing from the development shard")
    strata: dict[str, list[str]] = {}
    for problem_id in canonical_ids:
        answer = by_id[problem_id].answer
        if answer is None:  # pragma: no cover - train loader contract precedes this
            raise RuntimeError("teacher pilot may use only organizer-train integer answers")
        sign = "negative" if answer < 0 else "zero" if answer == 0 else "positive"
        magnitude = abs(answer)
        if magnitude <= 9:
            magnitude_bucket = "single_digit"
        elif magnitude <= 99:
            magnitude_bucket = "double_digit"
        elif magnitude <= 999:
            magnitude_bucket = "triple_digit"
        else:
            magnitude_bucket = "four_plus_digit"
        strata.setdefault(f"{sign}:{magnitude_bucket}", []).append(problem_id)

    total = len(canonical_ids)
    quotas = {
        key: (pilot_size * len(problem_ids)) // total
        for key, problem_ids in strata.items()
    }
    remaining = pilot_size - sum(quotas.values())
    priority = sorted(
        strata,
        key=lambda key: (
            -((pilot_size * len(strata[key])) % total),
            key,
        ),
    )
    for key in priority:
        if remaining == 0:
            break
        if quotas[key] < len(strata[key]):
            quotas[key] += 1
            remaining -= 1
    if remaining != 0:  # pragma: no cover - quota arithmetic above is exhaustive
        raise RuntimeError("teacher pilot stratification could not allocate its exact size")

    selected: list[str] = []
    for key in sorted(strata):
        ranked = sorted(
            strata[key],
            key=lambda problem_id: hashlib.sha256(
                f"gate-b-teacher-pilot-v1:{problem_id}".encode()
            ).hexdigest(),
        )
        selected.extend(ranked[: quotas[key]])
    if len(selected) != pilot_size or len(set(selected)) != pilot_size:
        raise RuntimeError("teacher pilot selection has invalid exact coverage")
    return tuple(sorted(selected))


def _codex_teacher_environment(
    *, isolated_codex_home: Path | None = None
) -> dict[str, str]:
    """Keep ChatGPT login discovery while excluding supplied API credentials."""

    environment = {
        key: value
        for key, value in os.environ.items()
        if key in _CODEX_TEACHER_SAFE_ENV_NAMES and value
    }
    if "PATH" not in environment:
        environment["PATH"] = os.defpath
    if "HOME" not in environment:
        raise RuntimeError("Codex teacher requires HOME to read the existing ChatGPT login")
    if isolated_codex_home is not None:
        if isolated_codex_home.is_symlink() or not isolated_codex_home.is_dir():
            raise RuntimeError("isolated Codex teacher home must be a regular directory")
        environment["CODEX_HOME"] = str(isolated_codex_home.resolve(strict=True))
    return environment


def _prepare_isolated_codex_home(working_directory: Path) -> Path:
    """Copy only the existing ChatGPT auth state into an empty private Codex home.

    ``CODEX_HOME`` includes skills and configuration as well as authentication.
    A short-lived home prevents global skills from entering a question-only teacher
    turn while retaining the user's already authenticated ChatGPT session.
    """

    if working_directory.is_symlink() or not working_directory.is_dir():
        raise RuntimeError("teacher working directory must be a regular directory")
    source_home = Path(_codex_teacher_environment()["HOME"]) / ".codex"
    source_auth = source_home / "auth.json"
    if source_auth.is_symlink() or not source_auth.is_file():
        raise RuntimeError("existing ChatGPT Codex auth state is unavailable")
    isolated_home = working_directory / "codex-home"
    isolated_home.mkdir(mode=0o700)
    target_auth = isolated_home / "auth.json"
    try:
        shutil.copyfile(source_auth, target_auth)
        os.chmod(target_auth, 0o600)
    except OSError as exc:
        raise RuntimeError("could not prepare isolated Codex teacher authentication") from exc
    return isolated_home


def _probe_codex_chatgpt_cli() -> tuple[str, str]:
    """Verify the locally installed CLI and ChatGPT login without exposing either output."""

    environment = _codex_teacher_environment()
    binary = shutil.which("codex", path=environment["PATH"])
    if binary is None:
        raise RuntimeError("Codex CLI is unavailable on PATH")
    try:
        version = subprocess.run(
            (binary, "--version"),
            check=False,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            env=environment,
        )
        login = subprocess.run(
            (binary, "login", "status"),
            check=False,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("Codex CLI probe did not complete") from exc
    version_text = (version.stdout or "").strip().splitlines()
    if version.returncode != 0 or not version_text:
        raise RuntimeError("Codex CLI version probe failed")
    codex_cli_version = version_text[0]
    if "\x00" in codex_cli_version or len(codex_cli_version) > 200:
        raise RuntimeError("Codex CLI version probe returned unsafe text")
    login_text = f"{login.stdout or ''}\n{login.stderr or ''}".lower()
    if login.returncode != 0 or "logged in using chatgpt" not in login_text:
        raise RuntimeError("Codex CLI is not logged in through ChatGPT")
    return str(Path(binary).resolve()), codex_cli_version


def _run_trusted_codex_teacher_command(
    command: tuple[str, ...],
    *,
    execution: TeacherExecutionConfig,
    timeout_seconds: int,
    isolated_codex_home: Path,
) -> CodexCommandResult:
    """Execute only the executable freshly bound to the immutable plan.

    ``run_teacher_plan`` reloads its private plan when a resumable run starts.
    This final argv[0] check therefore also closes the gap between the live
    pre-run probe and command construction without exposing the auth-only
    home to an unexpected executable.
    """

    if not command or command[0] != execution.codex_binary:
        raise RuntimeError(
            "teacher command executable differs from the freshly verified Codex plan"
        )
    return _run_codex_teacher_command(
        command,
        timeout_seconds=timeout_seconds,
        isolated_codex_home=isolated_codex_home,
    )


def _run_codex_teacher_command(
    command: tuple[str, ...],
    *,
    timeout_seconds: int,
    isolated_codex_home: Path | None = None,
) -> CodexCommandResult:
    """Run a prevalidated no-shell Codex command and keep raw output private."""

    started = time.monotonic_ns()
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            env=_codex_teacher_environment(isolated_codex_home=isolated_codex_home),
        )
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        returncode = completed.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = _subprocess_text(exc.stdout)
        stderr = "codex_timeout"
        returncode = 124
    except OSError:
        stdout = ""
        stderr = "codex_subprocess_os_error"
        returncode = 127
    latency_ms = (time.monotonic_ns() - started) // 1_000_000
    return CodexCommandResult(
        stdout=stdout,
        stderr=stderr,
        returncode=returncode,
        latency_ms=latency_ms,
    )


def _subprocess_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _load_locked_gate_b_config(path: Path) -> GateBConfig:
    payload = _load_json_object(path)
    stored_sha256 = payload.pop("config_sha256", None)
    if not isinstance(stored_sha256, str):
        raise ValueError("Gate B config is missing config_sha256")
    try:
        config = GateBConfig(**payload)
    except TypeError as exc:
        raise ValueError(f"Gate B config schema is invalid: {exc}") from exc
    if config != DEFAULT_GATE_B_CONFIG or config.sha256 != stored_sha256:
        raise ValueError("Gate B config differs from the locked RTX 4070 SUPER profile")
    return config


def _load_gate_b_data_contract(
    args: argparse.Namespace,
) -> tuple[CsvDataset, TrainExclusionSet, SplitManifest]:
    exclusions, manifest = _load_gate_b_preclaim_contract(args)
    evidence = load_development_cv_shard(
        args.development_shard,
        source_train_sha256=sha256_file(args.train),
        split_manifest=manifest,
        split_artifact_sha256=sha256_file(args.split_artifact),
        expected_bundle_sha256=args.expected_development_shard_sha256,
    )
    return evidence.dataset, exclusions, manifest


def _load_optional_rationale_training_inputs(
    args: argparse.Namespace,
    *,
    training_records: Sequence[MathRecord],
    split_manifest: SplitManifest,
    exclusions: TrainExclusionSet,
) -> tuple[RationaleCorpusEvidence | None, ConciseRationaleConfig | None]:
    names = (
        "rationale_corpus",
        "rationale_manifest",
        "rationale_audit",
        "rationale_config",
    )
    values = tuple(getattr(args, name, None) for name in names)
    if all(value is None for value in values):
        return None, None
    if any(value is None for value in values):
        missing = [name for name, value in zip(names, values, strict=True) if value is None]
        raise ValueError(
            "concise-rationale training inputs are all-or-none; "
            f"missing={missing!r}"
        )
    corpus_path, manifest_path, audit_path, config_path = values
    assert isinstance(corpus_path, Path)
    assert isinstance(manifest_path, Path)
    assert isinstance(audit_path, Path)
    assert isinstance(config_path, Path)
    config, config_file_sha256 = load_concise_rationale_config(config_path)
    evidence = load_verified_rationale_corpus(
        corpus_path,
        manifest_path,
        training_records,
        split_manifest=split_manifest,
        fold=args.fold,
        excluded_ids=exclusions.ids,
        candidate_config_file_sha256=config_file_sha256,
        audit_path=audit_path,
        config=config,
    )
    return evidence, config


def _load_gate_b_preclaim_contract(
    args: argparse.Namespace,
) -> tuple[TrainExclusionSet, SplitManifest]:
    """Validate bytes, IDs, and split structure without opening train Q/A rows."""

    train_sha256 = sha256_file(args.train)
    wrapper = _load_json_object(args.split_artifact)
    dataset_manifest = _required_mapping(wrapper, "dataset_manifest")
    if dataset_manifest.get("sha256") != train_sha256:
        raise ValueError("split artifact train SHA-256 does not match the train bytes")
    manifest = SplitManifest.from_dict(_required_mapping(wrapper, "split"))
    assignment_ids = tuple(assignment.record_id for assignment in manifest.assignments)
    if len(set(assignment_ids)) != len(assignment_ids):  # pragma: no cover - manifest validates
        raise ValueError("split artifact contains duplicate train IDs")
    if wrapper.get("cluster_method") != _HARD_CLUSTER_METHOD:
        raise ValueError("split artifact cluster_method is not canonical")
    if wrapper.get("cluster_count") != len(
        {assignment.group_id for assignment in manifest.assignments}
    ):
        raise ValueError("split artifact cluster_count does not match its assignments")
    exclusions = load_train_exclusions_csv(
        args.train_exclusions,
        train_ids=assignment_ids,
    )
    _require_expected_contract(
        actual_train_sha256=train_sha256,
        actual_exclusions_sha256=exclusions.manifest.sha256,
        actual_exclusion_count=len(exclusions),
        actual_split_sha256=manifest.sha256,
        expected_train_sha256=args.expected_train_sha256,
        expected_exclusions_sha256=args.expected_exclusions_sha256,
        expected_exclusion_count=args.expected_exclusion_count,
        expected_split_sha256=args.expected_split_sha256,
    )
    return exclusions, manifest


def _records_for_ids(
    records: Sequence[MathRecord], identifiers: Sequence[str]
) -> tuple[MathRecord, ...]:
    by_id = {record.id: record for record in records}
    missing = sorted(set(identifiers) - set(by_id))
    if missing:  # pragma: no cover - split/train binding catches this first
        raise RuntimeError(f"split records are missing from organizer train: {missing[:5]!r}")
    return tuple(by_id[problem_id] for problem_id in identifiers)


def _require_new_development_targets(jsonl_path: Path, manifest_path: Path) -> None:
    raw_paths = (jsonl_path, manifest_path)
    if jsonl_path.resolve(strict=False) == manifest_path.resolve(strict=False):
        raise ValueError("development JSONL and manifest paths must differ")
    resolved_parents: list[Path] = []
    for path in raw_paths:
        if path.is_symlink() or path.exists():
            raise ValueError(f"refusing to overwrite development artifact: {path}")
        if path.parent.is_symlink():
            raise ValueError("development artifact parent must not be a symlink")
        parent = path.parent.resolve(strict=True)
        if not parent.is_dir():
            raise ValueError("development artifact parent must be a directory")
        resolved_parents.append(parent)
    if resolved_parents[0] != resolved_parents[1]:
        raise ValueError("development JSONL and manifest must share one directory")


def _command_compare_development(args: argparse.Namespace) -> int:
    train, exclusions, manifest = _load_gate_b_data_contract(args)
    validation_ids = eligible_validation_ids(manifest, args.fold, exclusions.ids)
    record_by_id = {record.id: record for record in train}
    validation_records = tuple(record_by_id[record_id] for record_id in validation_ids)
    reference = NamedDevelopmentRun(
        args.reference[0], Path(args.reference[1]), Path(args.reference[2])
    )
    candidates = tuple(
        NamedDevelopmentRun(values[0], Path(values[1]), Path(values[2]))
        for values in args.candidate
    )
    result = compare_development_runs(
        reference,
        candidates,
        validation_records,
        split_manifest=manifest,
        fold=args.fold,
        excluded_ids=exclusions.ids,
        train_file_sha256=sha256_file(args.train),
        exclusions_file_sha256=exclusions.manifest.sha256,
        split_artifact_sha256=sha256_file(args.split_artifact),
        development_shard_sha256=args.expected_development_shard_sha256,
        output_path=args.output,
        bootstrap_samples=args.bootstrap_samples,
        confidence=args.confidence,
        alpha=args.alpha,
        seed=args.bootstrap_seed,
    )
    print(
        json.dumps(
            {
                "output": result.path,
                "sha256": result.sha256,
                "payload_sha256": result.payload_sha256,
                "reference_label": reference.label,
                "candidate_count": len(candidates),
                "bootstrap_unit": "duplicate_cluster",
                "holdout_accessed": False,
            },
            sort_keys=True,
        )
    )
    return 0


def _command_compare_development_oof(args: argparse.Namespace) -> int:
    exclusions, manifest = _load_gate_b_preclaim_contract(args)
    development = load_development_cv_shard(
        args.development_shard,
        source_train_sha256=sha256_file(args.train),
        split_manifest=manifest,
        split_artifact_sha256=sha256_file(args.split_artifact),
        expected_bundle_sha256=args.expected_development_shard_sha256,
    )
    eligible_ids = tuple(
        sorted(
            problem_id
            for fold in range(manifest.n_folds)
            for problem_id in eligible_validation_ids(
                manifest, fold, exclusions.ids
            )
        )
    )
    development_records = _records_for_ids(development.dataset.records, eligible_ids)
    fold_runs: list[FoldDevelopmentRun] = []
    for values in args.base_run:
        try:
            fold = int(values[0])
        except ValueError as exc:
            raise ValueError("--base-run FOLD must be an integer") from exc
        fold_runs.append(
            FoldDevelopmentRun(
                fold=fold,
                label=args.reference_label,
                records_path=Path(values[1]),
                manifest_path=Path(values[2]),
                method_kind=FIXED_BASE_METHOD_KIND,
            )
        )
    for values in args.adapter_run:
        try:
            fold = int(values[0])
        except ValueError as exc:
            raise ValueError("--adapter-run FOLD must be an integer") from exc
        fold_runs.append(
            FoldDevelopmentRun(
                fold=fold,
                label=values[1],
                records_path=Path(values[2]),
                manifest_path=Path(values[3]),
                method_kind=ADAPTER_METHOD_KIND,
                adapter_path=Path(values[4]),
            )
        )
    result = compare_cross_fold_development_runs(
        args.reference_label,
        tuple(args.candidate_label),
        tuple(fold_runs),
        development_records,
        split_manifest=manifest,
        deployment_fold=args.deployment_fold,
        excluded_ids=exclusions.ids,
        train_file_sha256=sha256_file(args.train),
        exclusions_file_sha256=exclusions.manifest.sha256,
        split_artifact_sha256=sha256_file(args.split_artifact),
        development_shard_sha256=args.expected_development_shard_sha256,
        output_path=args.output,
        bootstrap_samples=args.bootstrap_samples,
        confidence=args.confidence,
        alpha=args.alpha,
        seed=args.bootstrap_seed,
    )
    print(
        json.dumps(
            {
                "output": result.path,
                "sha256": result.sha256,
                "payload_sha256": result.payload_sha256,
                "reference_label": args.reference_label,
                "candidate_count": len(args.candidate_label),
                "fold_count": manifest.n_folds,
                "deployment_fold": args.deployment_fold,
                "bootstrap_unit": "duplicate_cluster",
                "holdout_accessed": False,
            },
            sort_keys=True,
        )
    )
    return 0


def _command_verify_base_development_oof(args: argparse.Namespace) -> int:
    exclusions, manifest = _load_gate_b_preclaim_contract(args)
    development = load_development_cv_shard(
        args.development_shard,
        source_train_sha256=sha256_file(args.train),
        split_manifest=manifest,
        split_artifact_sha256=sha256_file(args.split_artifact),
        expected_bundle_sha256=args.expected_development_shard_sha256,
    )
    eligible_ids = tuple(
        sorted(
            problem_id
            for fold in range(manifest.n_folds)
            for problem_id in eligible_validation_ids(
                manifest, fold, exclusions.ids
            )
        )
    )
    development_records = _records_for_ids(development.dataset.records, eligible_ids)
    fold_runs: list[FoldDevelopmentRun] = []
    for values in args.base_run:
        try:
            fold = int(values[0])
        except ValueError as exc:
            raise ValueError("--base-run FOLD must be an integer") from exc
        fold_runs.append(
            FoldDevelopmentRun(
                fold=fold,
                label=args.base_label,
                records_path=Path(values[1]),
                manifest_path=Path(values[2]),
                method_kind=FIXED_BASE_METHOD_KIND,
            )
        )
    result = verify_base_development_oof(
        args.base_label,
        tuple(fold_runs),
        development_records,
        split_manifest=manifest,
        deployment_fold=args.deployment_fold,
        excluded_ids=exclusions.ids,
        train_file_sha256=sha256_file(args.train),
        exclusions_file_sha256=exclusions.manifest.sha256,
        split_artifact_sha256=sha256_file(args.split_artifact),
        development_shard_sha256=args.expected_development_shard_sha256,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "output": result.path,
                "sha256": result.sha256,
                "payload_sha256": result.payload_sha256,
                "base_label": args.base_label,
                "fold_count": manifest.n_folds,
                "deployment_fold": args.deployment_fold,
                "qualified": True,
                "holdout_accessed": False,
            },
            sort_keys=True,
        )
    )
    return 0


def _command_decide_candidate_probe(args: argparse.Namespace) -> int:
    result = decide_candidate_probe_promotion(
        args.comparison_artifact,
        candidate_label=args.candidate_label,
        output_path=args.output,
    )
    payload = json.loads(Path(result.path).read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "output": result.path,
                "sha256": result.sha256,
                "payload_sha256": result.payload_sha256,
                "candidate_label": args.candidate_label,
                "candidate_action": payload["candidate_action"],
                "candidate_full_oof_authorized": payload[
                    "candidate_full_oof_authorized"
                ],
                "selection_frozen": False,
                "holdout_accessed": False,
            },
            sort_keys=True,
        )
    )
    return 0


def _command_freeze_development_selection(args: argparse.Namespace) -> int:
    if not args.confirm_no_leaderboard_selection:
        raise ValueError(
            "--confirm-no-leaderboard-selection is required before freezing methods"
        )
    result = freeze_development_selection(
        args.comparison_artifact,
        primary_label=args.primary_label,
        fallback_label=args.fallback_label,
        decision_note=args.decision_note,
        source_manifest_path=args.source_manifest,
        lockfile_path=args.lockfile,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "output": result.path,
                "sha256": result.sha256,
                "payload_sha256": result.payload_sha256,
                "primary_label": args.primary_label,
                "fallback_label": args.fallback_label,
                "selection_frozen": True,
                "holdout_accessed": False,
            },
            sort_keys=True,
        )
    )
    return 0


def _command_freeze_development_base(args: argparse.Namespace) -> int:
    if not args.confirm_no_leaderboard_selection:
        raise ValueError(
            "--confirm-no-leaderboard-selection is required before freezing methods"
        )
    result = freeze_base_development_selection(
        args.base_oof_artifact,
        primary_label=args.primary_label,
        decision_note=args.decision_note,
        source_manifest_path=args.source_manifest,
        lockfile_path=args.lockfile,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "output": result.path,
                "sha256": result.sha256,
                "payload_sha256": result.payload_sha256,
                "primary_label": args.primary_label,
                "fallback_label": None,
                "routing_policy": "primary_only",
                "selection_frozen": True,
                "holdout_accessed": False,
            },
            sort_keys=True,
        )
    )
    return 0


def _command_validate_submission(args: argparse.Namespace) -> int:
    expected = load_leaderboard_csv(args.expected)
    schema = SubmissionSchema(args.id_column, args.answer_column)
    report = validate_submission_csv(
        args.submission,
        [record.id for record in expected],
        schema=schema,
    )
    print(
        json.dumps(
            {
                "valid": report.valid,
                "row_count": report.row_count,
                "expected_count": report.expected_count,
                "issues": [asdict(issue) for issue in report.issues],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if report.valid else 1


def _command_verify_submission_independent(args: argparse.Namespace) -> int:
    report = verify_submission_independently(
        args.submission,
        args.expected,
        expected_file_sha256=args.expected_sha256,
    )
    print(json.dumps(asdict(report), ensure_ascii=False, sort_keys=True))
    return 0 if report.valid else 1


def _command_write_submission(args: argparse.Namespace) -> int:
    expected = load_leaderboard_csv(args.expected)
    try:
        predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"predictions JSON is invalid: {exc}") from exc
    if not isinstance(predictions, dict):
        raise ValueError("predictions JSON must be an object mapping IDs to integers")
    schema = SubmissionSchema(args.id_column, args.answer_column)
    result = write_submission_csv(
        args.output,
        predictions,
        [record.id for record in expected],
        schema=schema,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "output": str(result.path),
                "row_count": result.report.row_count,
                "size_bytes": result.size_bytes,
                "sha256": result.sha256,
                "fallback_count": result.fallback_count,
            },
            sort_keys=True,
        )
    )
    return 0


def _command_source_manifest(args: argparse.Namespace) -> int:
    manifest = build_source_tree_manifest(args.root, excluded_paths=(args.output,))
    write_json_atomic(args.output, manifest.as_dict())
    print(
        json.dumps(
            {
                "output": str(args.output),
                "tree_sha256": manifest.tree_sha256,
                "file_count": len(manifest.files),
            },
            sort_keys=True,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and convert expected validation errors into exit status 2."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

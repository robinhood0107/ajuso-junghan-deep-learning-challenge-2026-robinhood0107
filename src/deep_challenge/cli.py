"""Command-line entry points for the offline competition toolkit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict
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
    run_development_baseline,
    write_development_artifacts,
)
from .gate_b_holdout import evaluate_locked_holdout_once
from .gate_b_prediction import run_frozen_evaluation_inference
from .gate_b_runtime import (
    BASE_MODEL_CHECKPOINT_SHA256,
    GPU_EXECUTION_ACKNOWLEDGEMENT,
    create_adapted_development_backend,
    create_base_development_backend,
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
    freeze_development_selection,
    require_base_development_artifact,
)
from .gate_b_sft_preflight import run_sft_encoding_preflight
from .gpu_smoke import run_final_gpu_smoke
from .independent_submission import verify_submission_independently
from .model_preflight import run_model_preflight
from .provenance import (
    build_source_tree_manifest,
    canonical_json_bytes,
    sha256_file,
    write_json_atomic,
)
from .quality import (
    assess_question,
    math_aware_fingerprint,
    source_format_fingerprint,
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
from .tokenizer_profile import (
    DEFAULT_SYSTEM_PROMPT,
    load_and_profile_datasets,
    load_pinned_tokenizer,
)

_HARD_CLUSTER_METHOD = (
    "transitive union of math-aware and narrowly source-format-insensitive exact "
    "fingerprints; number-masked templates remain soft audit candidates"
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
    parser: argparse.ArgumentParser, *, require_development_shard: bool = True
) -> None:
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--train-exclusions", required=True, type=Path)
    parser.add_argument("--split-artifact", required=True, type=Path)
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--expected-train-sha256", required=True)
    parser.add_argument("--expected-exclusions-sha256", required=True)
    parser.add_argument("--expected-exclusion-count", required=True, type=int)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--expected-development-shard-sha256", required=True)
    if require_development_shard:
        parser.add_argument("--development-shard", required=True, type=Path)


def _add_gpu_runtime_gate(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--preflight-report", required=True, type=Path)
    parser.add_argument("--gpu-smoke-report", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--acknowledge-gpu-use",
        action="store_true",
        help="required explicit acknowledgement; this command creates a CUDA workload",
    )


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
    sft_preflight.add_argument("--output", required=True, type=Path)
    sft_preflight.set_defaults(handler=_command_gate_b_sft_preflight)

    development = subparsers.add_parser(
        "gate-b-development",
        help="run one fixed base or verified-adapter development fold",
    )
    _add_required_gate_b_data_contract(development)
    _add_gpu_runtime_gate(development)
    development.add_argument("--output-jsonl", required=True, type=Path)
    development.add_argument("--output-manifest", required=True, type=Path)
    development.add_argument("--adapter", type=Path)
    development.add_argument(
        "--base-baseline-manifest",
        type=Path,
        help="required successful fixed-base run manifest before adapter evaluation",
    )
    development.set_defaults(handler=_command_gate_b_development)

    train_fold = subparsers.add_parser(
        "gate-b-train-fold",
        help="train one locked direct-answer QLoRA fold after the base baseline",
    )
    _add_required_gate_b_data_contract(train_fold)
    _add_gpu_runtime_gate(train_fold)
    train_fold.add_argument("--base-baseline-manifest", required=True, type=Path)
    train_fold.add_argument("--output-dir", required=True, type=Path)
    train_fold.set_defaults(handler=_command_gate_b_train_fold)

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
            },
            sort_keys=True,
        )
    )
    return 0


def _command_gate_b_development(args: argparse.Namespace) -> int:
    _require_gpu_cli_acknowledgement(args.acknowledge_gpu_use)
    config = _load_locked_gate_b_config(args.config)
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
    try:
        records = run_development_baseline(
            validation_records,
            split_manifest=manifest,
            fold=args.fold,
            excluded_ids=exclusions.ids,
            backend=backend,
            checkpoint_sha256=backend.checkpoint_sha256,
            config=config,
            samples_per_problem=1,
        )
        result = write_development_artifacts(
            records,
            jsonl_path=args.output_jsonl,
            manifest_path=args.output_manifest,
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


def _command_gate_b_train_fold(args: argparse.Namespace) -> int:
    _require_gpu_cli_acknowledgement(args.acknowledge_gpu_use)
    config = _load_locked_gate_b_config(args.config)
    train, exclusions, manifest = _load_gate_b_data_contract(args)
    training_records = _records_for_ids(
        train.records,
        eligible_training_ids(manifest, args.fold, exclusions.ids),
    )
    validation_records = _records_for_ids(
        train.records,
        eligible_validation_ids(manifest, args.fold, exclusions.ids),
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
        preflight_artifact=args.preflight_report,
        gpu_smoke_artifact=args.gpu_smoke_report,
        output_dir=args.output_dir,
        gpu_acknowledgement=GPU_EXECUTION_ACKNOWLEDGEMENT,
        config=config,
    )
    print(json.dumps(asdict(result), sort_keys=True))
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

"""Executable, fail-closed Gate B QLoRA and NF4 generation runtime.

The module itself is CPU-safe: importing it never imports PyTorch, Transformers,
PEFT, Accelerate, or bitsandbytes and never initializes CUDA.  GPU work only
starts after a validated preflight report and a separately recorded final GPU
smoke artifact have both been supplied.
"""

from __future__ import annotations

import ctypes
import errno
import gc
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager, nullcontext, suppress
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from .answers import parse_answer
from .data import MathRecord
from .gate_b import (
    DEFAULT_GATE_B_CONFIG,
    PINNED_MODEL_REVISION,
    ChatMessage,
    ChatTokenizer,
    DirectAnswerSFTExample,
    EncodedSFTExample,
    GateBArtifactExistsError,
    GateBConfig,
    GateBPreflightRequiredError,
    GateBValidationError,
    GenerationRequest,
    GenerationResult,
    build_concise_rationale_sft_examples,
    build_direct_answer_sft_examples,
    encode_response_only_example,
)
from .gpu_smoke import (
    DEFAULT_GPU_SMOKE_CONFIG,
    SYNTHETIC_SMOKE_EXPECTED_ANSWER,
    SYNTHETIC_SMOKE_USER_PROMPT,
)
from .model_preflight import OFFICIAL_MODEL_ID, PINNED_WEIGHT_ARTIFACTS
from .provenance import SourceTreeArtifactEvidence, canonical_json_bytes, sha256_file
from .rationale_corpus import (
    DEFAULT_CONCISE_RATIONALE_CONFIG,
    ConciseRationaleConfig,
    RationaleCorpusEvidence,
)
from .splits import (
    SplitManifest,
    SplitPartition,
    SplitValidationError,
    eligible_training_ids,
    eligible_validation_ids,
)

_TRAIN_ID_RE = re.compile(r"train-\d{6}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_ADAPTER_MANIFEST_SCHEMA = "gate-b-qlora-adapter-v3"
_RATIONALE_ADAPTER_MANIFEST_SCHEMA = "gate-b-qlora-adapter-v4"
_CHECKSUM_FILENAME = "CHECKSUMS.sha256"
_MANIFEST_FILENAME = "manifest.json"
_TRAINING_RESUME_CONTRACT_SCHEMA = "gate-b-qlora-training-resume-v1"
_TRAINING_RESUME_CONTRACT_FILENAME = "resume-contract.json"
_TRAINING_RESUME_STARTED_FILENAME = "training-started.json"
_TRAINING_RESUME_LOCK_FILENAME = ".training-resume.lock"
_TRAINING_RESUME_CHECKPOINT_FILENAME = "resume-checkpoint.json"
_TRAINING_RESUME_FORENSIC_DIRECTORY = "forensics"
_TRAINING_RESUME_FORENSIC_SCHEMA = "gate-b-qlora-training-forensic-v1"
_TRAINER_CHECKPOINT_RE = re.compile(r"checkpoint-([1-9][0-9]*)\Z")
_REQUIRED_TRAINER_CHECKPOINT_FILES = frozenset(
    {
        "adapter_config.json",
        "optimizer.pt",
        _TRAINING_RESUME_CHECKPOINT_FILENAME,
        "rng_state.pth",
        "scheduler.pt",
        "trainer_state.json",
        "training_args.bin",
    }
)
GPU_EXECUTION_ACKNOWLEDGEMENT = "USE_GPU_AFTER_FINAL_SMOKE"
BASE_MODEL_CHECKPOINT_SHA256 = hashlib.sha256(
    canonical_json_bytes(
        {
            "kind": "huggingface_pinned_weight_contract",
            "model_id": OFFICIAL_MODEL_ID,
            "revision": PINNED_MODEL_REVISION,
            "files": PINNED_WEIGHT_ARTIFACTS,
        }
    )
).hexdigest()
PINNED_QWEN_ALL_LINEAR_TARGET_MODULES = tuple(
    sorted(
        (
            "down_proj",
            "gate_proj",
            "k_proj",
            "o_proj",
            "q_proj",
            "up_proj",
            "v_proj",
        )
    )
)
_REQUIRED_TOKENIZER_FILES = frozenset({"tokenizer.json", "tokenizer_config.json"})
_PINNED_TOKENIZER_JSON_SHA256 = (
    "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
)
_PINNED_TOKENIZER_CONFIG_JSON_SHA256 = (
    "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583"
)
_PINNED_TOKENIZER_CHAT_TEMPLATE_SHA256 = (
    "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f"
)
_PINNED_QWEN_LAYER_COUNT = 36
_PINNED_QWEN_PROJECTION_DIMS = {
    "q_proj": (2048, 2048),
    "k_proj": (256, 2048),
    "v_proj": (256, 2048),
    "o_proj": (2048, 2048),
    "gate_proj": (11008, 2048),
    "up_proj": (11008, 2048),
    "down_proj": (2048, 11008),
}
_ALLOWED_LORA_DTYPES = frozenset({"BF16", "F16", "F32"})
_OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
}


@dataclass(frozen=True, slots=True)
class PreflightEvidence:
    """Validated CPU/pre-GPU-smoke preflight evidence."""

    path: str
    sha256: str
    model_id: str
    revision: str
    config_sha256: str


@dataclass(frozen=True, slots=True)
class RuntimeGateEvidence:
    """Two-artifact authorization required before any real CUDA operation."""

    preflight_path: str
    preflight_sha256: str
    smoke_path: str
    smoke_sha256: str
    model_id: str
    revision: str
    config_sha256: str
    device_name: str


@dataclass(frozen=True, slots=True)
class FoldSFTPlan:
    """Exact split-derived direct-answer training and validation plan."""

    fold: int
    split_version: str
    split_sha256: str
    source_groups_sha256: str
    excluded_ids_sha256: str
    training_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    training_examples: tuple[DirectAnswerSFTExample, ...]
    validation_examples: tuple[DirectAnswerSFTExample, ...]
    config_sha256: str
    training_target_kind: str = "direct_answer"
    rationale_candidate_config_sha256: str | None = None
    rationale_candidate_config_file_sha256: str | None = None
    rationale_corpus_records_sha256: str | None = None
    rationale_corpus_manifest_sha256: str | None = None
    rationale_corpus_audit_sha256: str | None = None

    @property
    def training_ids_sha256(self) -> str:
        return _ids_sha256(self.training_ids)

    @property
    def validation_ids_sha256(self) -> str:
        return _ids_sha256(self.validation_ids)

    @property
    def training_examples_sha256(self) -> str:
        return _sft_examples_sha256(self.training_examples)

    @property
    def validation_examples_sha256(self) -> str:
        return _sft_examples_sha256(self.validation_examples)


@dataclass(frozen=True, slots=True)
class RuntimeTrainingResult:
    """Small, JSON-safe summary returned by an injected training runtime."""

    global_step: int
    metrics: Mapping[str, str | int | float | bool | None]
    package_versions: Mapping[str, str]

    def __post_init__(self) -> None:
        if isinstance(self.global_step, bool) or not isinstance(self.global_step, int):
            raise GateBValidationError("global_step must be an integer")
        if self.global_step <= 0:
            raise GateBValidationError("global_step must be positive before publication")
        _validated_flat_json_mapping(self.metrics, "training metrics")
        _validated_string_mapping(self.package_versions, "package_versions")


@dataclass(frozen=True, slots=True)
class TrainingArtifact:
    """Content identity of one atomically published adapter bundle."""

    path: str
    artifact_sha256: str
    manifest_sha256: str
    checksums_sha256: str
    file_count: int
    training_count: int
    validation_count: int
    training_target_kind: str = "direct_answer"


@dataclass(frozen=True, slots=True)
class AdapterArtifactEvidence:
    """Validated completeness and content identity of an adapter bundle."""

    path: str
    artifact_sha256: str
    manifest_sha256: str
    checksums_sha256: str
    file_count: int
    config_sha256: str
    split_version: str
    split_sha256: str
    source_groups_sha256: str
    fold: int
    excluded_ids_sha256: str
    training_count: int
    training_ids_sha256: str
    validation_count: int
    validation_ids_sha256: str
    training_examples_sha256: str
    validation_examples_sha256: str
    train_file_sha256: str
    exclusions_file_sha256: str
    split_artifact_sha256: str
    development_shard_sha256: str
    preflight_sha256: str
    gpu_smoke_sha256: str
    source_manifest_sha256: str
    source_tree_sha256: str
    source_file_count: int
    training_target_kind: str = "direct_answer"
    rationale_candidate_config_sha256: str | None = None
    rationale_candidate_config_file_sha256: str | None = None
    rationale_corpus_records_sha256: str | None = None
    rationale_corpus_manifest_sha256: str | None = None
    rationale_corpus_audit_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class _TrainingResumeContext:
    """Validated persistent Trainer workspace for one exact training contract."""

    root: Path
    contract_sha256: str


@dataclass(frozen=True, slots=True)
class _RecoverableTrainerCheckpoint:
    """One incomplete/corrupt checkpoint preserved outside the active attempt."""

    path: Path
    global_step: int
    reason: str


class FoldTrainingRuntime(Protocol):
    """Injectable runtime boundary used by CPU-only orchestration tests."""

    @property
    def tokenizer(self) -> ChatTokenizer: ...

    def train(
        self,
        *,
        training_examples: Sequence[EncodedSFTExample],
        validation_examples: Sequence[EncodedSFTExample],
        work_dir: Path,
        export_dir: Path,
        plan: FoldSFTPlan,
        config: GateBConfig,
        resume_checkpoint: Path | None = None,
        retain_checkpoints: bool = False,
    ) -> RuntimeTrainingResult: ...

    def close(self) -> None: ...


def build_fold_sft_plan(
    training_records: Iterable[MathRecord],
    validation_records: Iterable[MathRecord],
    *,
    split_manifest: SplitManifest,
    fold: int,
    excluded_ids: Iterable[str],
    rationale_corpus: RationaleCorpusEvidence | None = None,
    rationale_config: ConciseRationaleConfig | None = None,
    config: GateBConfig = DEFAULT_GATE_B_CONFIG,
) -> FoldSFTPlan:
    """Derive and validate one fold without exposing holdout/full-train paths."""

    _require_default_config(config)
    if not isinstance(split_manifest, SplitManifest):
        raise TypeError("split_manifest must be a SplitManifest")
    try:
        split_manifest.validate()
    except SplitValidationError as exc:
        raise GateBValidationError(f"invalid split manifest: {exc}") from exc

    exclusions = _validated_train_ids(excluded_ids, "excluded_ids", allow_empty=True)
    try:
        training_ids = eligible_training_ids(split_manifest, fold, exclusions)
        validation_ids = eligible_validation_ids(split_manifest, fold, exclusions)
    except SplitValidationError as exc:
        raise GateBValidationError(f"invalid fold/exclusion boundary: {exc}") from exc
    training_ids = _validated_train_ids(training_ids, "derived training IDs")
    validation_ids = _validated_train_ids(validation_ids, "derived validation IDs")
    if set(training_ids) & set(validation_ids):  # pragma: no cover - split invariant
        raise GateBValidationError("training and validation IDs overlap")

    training_materialized = _validated_exact_records(
        training_records, training_ids, "training_records"
    )
    validation_materialized = _validated_exact_records(
        validation_records, validation_ids, "validation_records"
    )
    if (rationale_corpus is None) != (rationale_config is None):
        raise GateBValidationError(
            "rationale_corpus and rationale_config must be supplied together"
        )
    if rationale_corpus is None:
        training_examples = build_direct_answer_sft_examples(
            training_materialized,
            split_manifest=split_manifest,
            fold=fold,
            excluded_ids=exclusions,
            config=config,
        )
        target_binding: dict[str, str | None] = {
            "training_target_kind": "direct_answer",
            "rationale_candidate_config_sha256": None,
            "rationale_candidate_config_file_sha256": None,
            "rationale_corpus_records_sha256": None,
            "rationale_corpus_manifest_sha256": None,
            "rationale_corpus_audit_sha256": None,
        }
    else:
        assert rationale_config is not None
        training_examples = build_concise_rationale_sft_examples(
            training_materialized,
            split_manifest=split_manifest,
            fold=fold,
            excluded_ids=exclusions,
            rationale_corpus=rationale_corpus,
            rationale_config=rationale_config,
            config=config,
        )
        target_binding = {
            "training_target_kind": rationale_config.training_target_kind,
            "rationale_candidate_config_sha256": rationale_config.sha256,
            "rationale_candidate_config_file_sha256": (
                rationale_corpus.candidate_config_file_sha256
            ),
            "rationale_corpus_records_sha256": rationale_corpus.records_sha256,
            "rationale_corpus_manifest_sha256": rationale_corpus.manifest_sha256,
            "rationale_corpus_audit_sha256": rationale_corpus.audit_sha256,
        }
    validation_examples = _build_validation_examples(
        validation_materialized,
        validation_ids=validation_ids,
        split_manifest=split_manifest,
        fold=fold,
        config=config,
    )
    return FoldSFTPlan(
        fold=fold,
        split_version=split_manifest.version,
        split_sha256=split_manifest.sha256,
        source_groups_sha256=split_manifest.source_groups_sha256,
        excluded_ids_sha256=_ids_sha256(exclusions),
        training_ids=training_ids,
        validation_ids=validation_ids,
        training_examples=training_examples,
        validation_examples=validation_examples,
        config_sha256=config.sha256,
        **target_binding,
    )


def validate_preflight_artifact(
    path: str | Path,
    *,
    config: GateBConfig = DEFAULT_GATE_B_CONFIG,
) -> PreflightEvidence:
    """Require a green pinned-model pre-smoke report without initializing CUDA."""

    _require_default_config(config)
    source, payload, digest = _load_json_artifact(path, "model preflight")
    required_true = (
        "snapshot_consistent",
        "tokenizer_ready",
        "weights_ready",
        "model_runtime_ready",
        "host_runtime_ready",
        "qlora_dependencies_ready",
        "training_ready",
    )
    if payload.get("model_id") != OFFICIAL_MODEL_ID:
        raise GateBPreflightRequiredError("preflight does not bind the official fixed model")
    if payload.get("requested_revision") != PINNED_MODEL_REVISION:
        raise GateBPreflightRequiredError("preflight does not bind the pinned revision")
    if payload.get("resolved_commit") != PINNED_MODEL_REVISION:
        raise GateBPreflightRequiredError("preflight snapshot did not resolve to the pinned commit")
    failed = [field for field in required_true if payload.get(field) is not True]
    if failed:
        raise GateBPreflightRequiredError(
            f"preflight prerequisites are not green: {failed!r}"
        )
    if payload.get("training_ready_scope") != "pre_gpu_smoke_prerequisites":
        raise GateBPreflightRequiredError("unknown preflight readiness scope")
    if payload.get("training_profile") != "nf4_qlora_bf16":
        raise GateBPreflightRequiredError("preflight training profile is not NF4 QLoRA BF16")
    if payload.get("blockers") not in ([], ()):
        raise GateBPreflightRequiredError("preflight still reports blockers")
    if payload.get("runtime_blockers") not in ([], ()):
        raise GateBPreflightRequiredError("preflight still reports runtime blockers")
    return PreflightEvidence(
        path=str(source),
        sha256=digest,
        model_id=OFFICIAL_MODEL_ID,
        revision=PINNED_MODEL_REVISION,
        config_sha256=config.sha256,
    )


def validate_runtime_gate(
    *,
    preflight_artifact: str | Path,
    gpu_smoke_artifact: str | Path,
    config: GateBConfig = DEFAULT_GATE_B_CONFIG,
) -> RuntimeGateEvidence:
    """Bind a successful final smoke to one exact green preflight report."""

    preflight = validate_preflight_artifact(preflight_artifact, config=config)
    smoke_path, payload, smoke_digest = _load_json_artifact(
        gpu_smoke_artifact, "final GPU smoke"
    )
    expected = {
        "schema_version": DEFAULT_GPU_SMOKE_CONFIG.schema_version,
        "status": "green",
        "model_id": OFFICIAL_MODEL_ID,
        "revision": PINNED_MODEL_REVISION,
        "config": DEFAULT_GPU_SMOKE_CONFIG.as_dict(),
        "config_sha256": DEFAULT_GPU_SMOKE_CONFIG.sha256,
        "expected_answer": SYNTHETIC_SMOKE_EXPECTED_ANSWER,
        "exact_match": True,
    }
    mismatched = [key for key, value in expected.items() if payload.get(key) != value]
    if mismatched:
        raise GateBPreflightRequiredError(
            f"final GPU smoke is missing or mismatches required evidence: {mismatched!r}"
        )
    _validate_smoke_payload_hash(payload)
    _validate_smoke_preflight_binding(payload, preflight)
    _validate_smoke_input_and_parser(payload)
    _validate_smoke_runtime_evidence(payload)
    return RuntimeGateEvidence(
        preflight_path=preflight.path,
        preflight_sha256=preflight.sha256,
        smoke_path=str(smoke_path),
        smoke_sha256=smoke_digest,
        model_id=OFFICIAL_MODEL_ID,
        revision=PINNED_MODEL_REVISION,
        config_sha256=config.sha256,
        device_name=str(payload["runtime"]["device_name"]),  # type: ignore[index]
    )


def train_qlora_fold(
    training_records: Iterable[MathRecord],
    validation_records: Iterable[MathRecord],
    *,
    split_manifest: SplitManifest,
    fold: int,
    excluded_ids: Iterable[str],
    train_file_sha256: str,
    exclusions_file_sha256: str,
    split_artifact_sha256: str,
    development_shard_sha256: str,
    source_manifest: SourceTreeArtifactEvidence,
    preflight_artifact: str | Path,
    gpu_smoke_artifact: str | Path,
    output_dir: str | Path,
    gpu_acknowledgement: str,
    resume_dir: str | Path | None = None,
    rationale_corpus: RationaleCorpusEvidence | None = None,
    rationale_config: ConciseRationaleConfig | None = None,
    runtime_factory: Callable[[RuntimeGateEvidence], FoldTrainingRuntime] | None = None,
    config: GateBConfig = DEFAULT_GATE_B_CONFIG,
) -> TrainingArtifact:
    """Train and atomically publish one exact organizer-train CV adapter.

    The caller cannot pass a free-form ID list.  Both partitions are derived
    internally from ``training_ids(fold)``/``fold_ids(fold)`` after hard-group
    exclusions, and each supplied record collection must match exactly.
    """

    _require_gpu_acknowledgement(gpu_acknowledgement)
    source_evidence = _validated_source_manifest_evidence(source_manifest)
    data_provenance = {
        "train_file_sha256": _required_sha256(
            train_file_sha256, "train_file_sha256"
        ),
        "exclusions_file_sha256": _required_sha256(
            exclusions_file_sha256, "exclusions_file_sha256"
        ),
        "split_artifact_sha256": _required_sha256(
            split_artifact_sha256, "split_artifact_sha256"
        ),
        "development_shard_sha256": _required_sha256(
            development_shard_sha256, "development_shard_sha256"
        ),
    }
    gate = validate_runtime_gate(
        preflight_artifact=preflight_artifact,
        gpu_smoke_artifact=gpu_smoke_artifact,
        config=config,
    )
    plan = build_fold_sft_plan(
        training_records,
        validation_records,
        split_manifest=split_manifest,
        fold=fold,
        excluded_ids=excluded_ids,
        rationale_corpus=rationale_corpus,
        rationale_config=rationale_config,
        config=config,
    )
    target = _validated_new_directory_target(output_dir)
    resume = _prepare_training_resume_context(
        resume_dir,
        output_dir=target,
        plan=plan,
        gate=gate,
        data_provenance=data_provenance,
        source_manifest=source_evidence,
        config=config,
    )
    factory = runtime_factory or (
        lambda evidence: TransformersQLoRATrainingRuntime(
            evidence=evidence,
            config=config,
        )
    )
    build_root = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.training-", dir=target.parent)
    )
    work_dir = build_root / "work" if resume is None else resume.root
    export_dir = build_root / "publish"
    if resume is None:
        work_dir.mkdir(mode=0o700)
    export_dir.mkdir(mode=0o700)
    runtime: FoldTrainingRuntime | None = None
    try:
        lock = (
            _locked_training_resume_checkpoint(resume)
            if resume is not None
            else nullcontext(None)
        )
        with lock as resume_checkpoint:
            runtime = factory(gate)
            if resume is not None:
                _require_resume_capable_runtime(runtime)
            tokenizer = runtime.tokenizer
            encoded_training = tuple(
                encode_response_only_example(example, tokenizer, config=config)
                for example in plan.training_examples
            )
            encoded_validation = tuple(
                encode_response_only_example(example, tokenizer, config=config)
                for example in plan.validation_examples
            )
            _validate_encoded_partition(encoded_training, plan.training_ids, "training")
            _validate_encoded_partition(
                encoded_validation, plan.validation_ids, "validation"
            )
            if resume is not None:
                _mark_training_resume_started(resume)
            result = _run_training_runtime(
                runtime,
                training_examples=encoded_training,
                validation_examples=encoded_validation,
                work_dir=work_dir,
                export_dir=export_dir,
                plan=plan,
                config=config,
                resume_checkpoint=resume_checkpoint,
                retain_checkpoints=resume is not None,
            )
            if not isinstance(result, RuntimeTrainingResult):
                raise TypeError("FoldTrainingRuntime.train() must return RuntimeTrainingResult")
            _prepare_adapter_bundle(
                export_dir,
                plan=plan,
                gate=gate,
                result=result,
                data_provenance=data_provenance,
                source_manifest=source_evidence,
                config=config,
            )
            staged = validate_adapter_artifact(export_dir, config=config)
            _validate_adapter_target_binding(staged, plan)
            _publish_directory_noreplace(export_dir, target)
            published = validate_adapter_artifact(target, config=config)
            _validate_adapter_target_binding(published, plan)
            if published.artifact_sha256 != staged.artifact_sha256:
                raise GateBValidationError(
                    "published adapter digest changed after atomic rename"
                )
            return TrainingArtifact(
                path=published.path,
                artifact_sha256=published.artifact_sha256,
                manifest_sha256=published.manifest_sha256,
                checksums_sha256=published.checksums_sha256,
                file_count=published.file_count,
                training_count=len(plan.training_ids),
                validation_count=len(plan.validation_ids),
                training_target_kind=plan.training_target_kind,
            )
    finally:
        if runtime is not None:
            with suppress(Exception):
                runtime.close()
        if build_root.exists():
            shutil.rmtree(build_root)


def validate_adapter_artifact(
    path: str | Path,
    *,
    config: GateBConfig = DEFAULT_GATE_B_CONFIG,
) -> AdapterArtifactEvidence:
    """Verify adapter/tokenizer files, shards, manifest, and checksum closure."""

    _require_default_config(config)
    supplied = Path(path)
    if supplied.is_symlink():
        raise GateBValidationError("adapter artifact must not be a symbolic link")
    root = supplied.resolve(strict=True)
    if not root.is_dir():
        raise GateBValidationError("adapter artifact must be a real directory")
    files = _regular_artifact_files(root)
    relative_names = {item.relative_to(root).as_posix() for item in files}
    required = {_MANIFEST_FILENAME, _CHECKSUM_FILENAME, *_REQUIRED_TOKENIZER_FILES}
    missing = sorted(required - relative_names)
    if missing:
        raise GateBValidationError(f"adapter bundle is incomplete; missing={missing!r}")
    _validate_saved_tokenizer(root)
    _validate_adapter_config(root)
    _validate_adapter_weight_files(root, relative_names)

    manifest_path = root / _MANIFEST_FILENAME
    _, manifest, manifest_digest = _load_json_artifact(manifest_path, "adapter manifest")
    schema_version = manifest.get("schema_version")
    if schema_version not in {
        _ADAPTER_MANIFEST_SCHEMA,
        _RATIONALE_ADAPTER_MANIFEST_SCHEMA,
    }:
        raise GateBValidationError("adapter manifest schema_version is unsupported")
    expected_manifest = {
        "model_id": OFFICIAL_MODEL_ID,
        "revision": PINNED_MODEL_REVISION,
        "config": config.as_dict(),
        "config_sha256": config.sha256,
        "partition": "fold_training",
        "response_only_labels": True,
        "truncation": False,
    }
    mismatched = [key for key, value in expected_manifest.items() if manifest.get(key) != value]
    if mismatched:
        raise GateBValidationError(f"adapter manifest binding mismatch: {mismatched!r}")
    training_target_kind = "direct_answer"
    rationale_candidate_config_sha256: str | None = None
    rationale_candidate_config_file_sha256: str | None = None
    rationale_corpus_records_sha256: str | None = None
    rationale_corpus_manifest_sha256: str | None = None
    rationale_corpus_audit_sha256: str | None = None
    training_target = manifest.get("training_target")
    if schema_version == _ADAPTER_MANIFEST_SCHEMA:
        if training_target is not None:
            raise GateBValidationError(
                "direct-answer adapter manifest must not contain training_target"
            )
    else:
        if not isinstance(training_target, Mapping):
            raise GateBValidationError(
                "concise-rationale adapter manifest requires training_target"
            )
        expected_target_keys = {
            "kind",
            "candidate_config_sha256",
            "candidate_config_file_sha256",
            "corpus_records_sha256",
            "corpus_manifest_sha256",
            "corpus_audit_sha256",
        }
        if set(training_target) != expected_target_keys:
            raise GateBValidationError(
                "concise-rationale adapter training_target keys are invalid"
            )
        training_target_kind = str(training_target.get("kind"))
        if training_target_kind != "verified_concise_rationale":
            raise GateBValidationError(
                "concise-rationale adapter training target kind is invalid"
            )
        rationale_candidate_config_sha256 = _required_sha256(
            training_target.get("candidate_config_sha256"),
            "adapter rationale candidate_config_sha256",
        )
        if (
            rationale_candidate_config_sha256
            != DEFAULT_CONCISE_RATIONALE_CONFIG.sha256
        ):
            raise GateBValidationError(
                "adapter rationale candidate config is not the locked v1 policy"
            )
        rationale_candidate_config_file_sha256 = _required_sha256(
            training_target.get("candidate_config_file_sha256"),
            "adapter rationale candidate_config_file_sha256",
        )
        rationale_corpus_records_sha256 = _required_sha256(
            training_target.get("corpus_records_sha256"),
            "adapter rationale corpus_records_sha256",
        )
        rationale_corpus_manifest_sha256 = _required_sha256(
            training_target.get("corpus_manifest_sha256"),
            "adapter rationale corpus_manifest_sha256",
        )
        rationale_corpus_audit_sha256 = _required_sha256(
            training_target.get("corpus_audit_sha256"),
            "adapter rationale corpus_audit_sha256",
        )
    split_version = manifest.get("split_version")
    if not isinstance(split_version, str) or not split_version:
        raise GateBValidationError("adapter manifest split_version must be non-empty")
    split_sha256 = _required_sha256(manifest.get("split_sha256"), "adapter split_sha256")
    source_groups_sha256 = _required_sha256(
        manifest.get("source_groups_sha256"), "adapter source_groups_sha256"
    )
    excluded_ids_sha256 = _required_sha256(
        manifest.get("excluded_ids_sha256"), "adapter excluded_ids_sha256"
    )
    training_ids_sha256 = _required_sha256(
        manifest.get("training_ids_sha256"), "adapter training_ids_sha256"
    )
    validation_ids_sha256 = _required_sha256(
        manifest.get("validation_ids_sha256"), "adapter validation_ids_sha256"
    )
    training_examples_sha256 = _required_sha256(
        manifest.get("training_examples_sha256"),
        "adapter training_examples_sha256",
    )
    validation_examples_sha256 = _required_sha256(
        manifest.get("validation_examples_sha256"),
        "adapter validation_examples_sha256",
    )
    train_file_sha256 = _required_sha256(
        manifest.get("train_file_sha256"), "adapter train_file_sha256"
    )
    exclusions_file_sha256 = _required_sha256(
        manifest.get("exclusions_file_sha256"), "adapter exclusions_file_sha256"
    )
    split_artifact_sha256 = _required_sha256(
        manifest.get("split_artifact_sha256"), "adapter split_artifact_sha256"
    )
    development_shard_sha256 = _required_sha256(
        manifest.get("development_shard_sha256"),
        "adapter development_shard_sha256",
    )
    preflight_sha256 = _required_sha256(
        manifest.get("preflight_sha256"), "adapter preflight_sha256"
    )
    gpu_smoke_sha256 = _required_sha256(
        manifest.get("gpu_smoke_sha256"), "adapter gpu_smoke_sha256"
    )
    source_manifest_sha256 = _required_sha256(
        manifest.get("source_manifest_sha256"), "adapter source_manifest_sha256"
    )
    source_tree_sha256 = _required_sha256(
        manifest.get("source_tree_sha256"), "adapter source_tree_sha256"
    )
    source_file_count = manifest.get("source_file_count")
    if (
        isinstance(source_file_count, bool)
        or not isinstance(source_file_count, int)
        or source_file_count < 1
    ):
        raise GateBValidationError("adapter source_file_count must be positive")
    integer_fields: dict[str, int] = {}
    for field_name in ("fold", "training_count", "validation_count"):
        value = manifest.get(field_name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise GateBValidationError(f"adapter manifest {field_name} is invalid")
        integer_fields[field_name] = value
    if integer_fields["training_count"] < 1 or integer_fields["validation_count"] < 1:
        raise GateBValidationError("adapter training and validation counts must be positive")
    manifest_files = manifest.get("files")
    if not isinstance(manifest_files, list):
        raise GateBValidationError("adapter manifest files must be an array")
    computed_payload = _file_inventory(
        root,
        excluded={_MANIFEST_FILENAME, _CHECKSUM_FILENAME},
    )
    if manifest_files != computed_payload:
        raise GateBValidationError("adapter manifest file inventory does not match bytes on disk")

    expected_checksum_entries = [
        *computed_payload,
        {
            "path": _MANIFEST_FILENAME,
            "size_bytes": manifest_path.stat().st_size,
            "sha256": manifest_digest,
        },
    ]
    expected_checksum_entries.sort(key=lambda item: str(item["path"]))
    checksum_path = root / _CHECKSUM_FILENAME
    actual_checksums = _parse_checksum_file(checksum_path)
    expected_checksums = {
        str(entry["path"]): str(entry["sha256"]) for entry in expected_checksum_entries
    }
    if actual_checksums != expected_checksums:
        raise GateBValidationError("CHECKSUMS.sha256 does not close over adapter payload+manifest")
    checksums_digest = sha256_file(checksum_path)
    artifact_inventory = [
        *expected_checksum_entries,
        {
            "path": _CHECKSUM_FILENAME,
            "size_bytes": checksum_path.stat().st_size,
            "sha256": checksums_digest,
        },
    ]
    artifact_inventory.sort(key=lambda item: str(item["path"]))
    artifact_digest = hashlib.sha256(canonical_json_bytes(artifact_inventory)).hexdigest()
    return AdapterArtifactEvidence(
        path=str(root),
        artifact_sha256=artifact_digest,
        manifest_sha256=manifest_digest,
        checksums_sha256=checksums_digest,
        file_count=len(artifact_inventory),
        config_sha256=config.sha256,
        split_version=split_version,
        split_sha256=split_sha256,
        source_groups_sha256=source_groups_sha256,
        fold=integer_fields["fold"],
        excluded_ids_sha256=excluded_ids_sha256,
        training_count=integer_fields["training_count"],
        training_ids_sha256=training_ids_sha256,
        validation_count=integer_fields["validation_count"],
        validation_ids_sha256=validation_ids_sha256,
        training_examples_sha256=training_examples_sha256,
        validation_examples_sha256=validation_examples_sha256,
        train_file_sha256=train_file_sha256,
        exclusions_file_sha256=exclusions_file_sha256,
        split_artifact_sha256=split_artifact_sha256,
        development_shard_sha256=development_shard_sha256,
        preflight_sha256=preflight_sha256,
        gpu_smoke_sha256=gpu_smoke_sha256,
        source_manifest_sha256=source_manifest_sha256,
        source_tree_sha256=source_tree_sha256,
        source_file_count=source_file_count,
        training_target_kind=training_target_kind,
        rationale_candidate_config_sha256=rationale_candidate_config_sha256,
        rationale_candidate_config_file_sha256=(
            rationale_candidate_config_file_sha256
        ),
        rationale_corpus_records_sha256=rationale_corpus_records_sha256,
        rationale_corpus_manifest_sha256=rationale_corpus_manifest_sha256,
        rationale_corpus_audit_sha256=rationale_corpus_audit_sha256,
    )


def validate_adapter_for_fold(
    path: str | Path,
    *,
    split_manifest: SplitManifest,
    fold: int,
    excluded_ids: Iterable[str],
    train_file_sha256: str,
    exclusions_file_sha256: str,
    split_artifact_sha256: str,
    development_shard_sha256: str,
    preflight_sha256: str,
    gpu_smoke_sha256: str,
    config: GateBConfig = DEFAULT_GATE_B_CONFIG,
) -> AdapterArtifactEvidence:
    """Bind an adapter to the exact current split, fold, and exclusion overlay."""

    evidence = validate_adapter_artifact(path, config=config)
    exclusions = _validated_train_ids(excluded_ids, "excluded_ids", allow_empty=True)
    try:
        training_ids = eligible_training_ids(split_manifest, fold, exclusions)
        validation_ids = eligible_validation_ids(split_manifest, fold, exclusions)
    except SplitValidationError as exc:
        raise GateBValidationError(f"invalid adapter fold boundary: {exc}") from exc
    expected = {
        "split_version": split_manifest.version,
        "split_sha256": split_manifest.sha256,
        "source_groups_sha256": split_manifest.source_groups_sha256,
        "fold": fold,
        "excluded_ids_sha256": _ids_sha256(exclusions),
        "training_count": len(training_ids),
        "training_ids_sha256": _ids_sha256(training_ids),
        "validation_count": len(validation_ids),
        "validation_ids_sha256": _ids_sha256(validation_ids),
        "train_file_sha256": _required_sha256(
            train_file_sha256, "train_file_sha256"
        ),
        "exclusions_file_sha256": _required_sha256(
            exclusions_file_sha256, "exclusions_file_sha256"
        ),
        "split_artifact_sha256": _required_sha256(
            split_artifact_sha256, "split_artifact_sha256"
        ),
        "development_shard_sha256": _required_sha256(
            development_shard_sha256, "development_shard_sha256"
        ),
        "preflight_sha256": _required_sha256(preflight_sha256, "preflight_sha256"),
        "gpu_smoke_sha256": _required_sha256(gpu_smoke_sha256, "gpu_smoke_sha256"),
    }
    mismatched = [key for key, value in expected.items() if getattr(evidence, key) != value]
    if mismatched:
        raise GateBValidationError(
            f"adapter does not belong to the requested split/fold: {mismatched!r}"
        )
    return evidence


def create_base_development_backend(
    *,
    preflight_artifact: str | Path,
    gpu_smoke_artifact: str | Path,
    gpu_acknowledgement: str,
    config: GateBConfig = DEFAULT_GATE_B_CONFIG,
) -> TransformersNF4GenerationBackend:
    """Create the pinned base-model backend for ``run_development_baseline``."""

    return TransformersNF4GenerationBackend(
        preflight_artifact=preflight_artifact,
        gpu_smoke_artifact=gpu_smoke_artifact,
        gpu_acknowledgement=gpu_acknowledgement,
        adapter_path=None,
        config=config,
    )


def create_adapted_development_backend(
    *,
    preflight_artifact: str | Path,
    gpu_smoke_artifact: str | Path,
    adapter_path: str | Path,
    split_manifest: SplitManifest,
    fold: int,
    excluded_ids: Iterable[str],
    train_file_sha256: str,
    exclusions_file_sha256: str,
    split_artifact_sha256: str,
    development_shard_sha256: str,
    gpu_acknowledgement: str,
    config: GateBConfig = DEFAULT_GATE_B_CONFIG,
) -> TransformersNF4GenerationBackend:
    """Create a verified QLoRA-adapter backend for development generation."""

    return TransformersNF4GenerationBackend(
        preflight_artifact=preflight_artifact,
        gpu_smoke_artifact=gpu_smoke_artifact,
        gpu_acknowledgement=gpu_acknowledgement,
        adapter_path=adapter_path,
        adapter_split_manifest=split_manifest,
        adapter_fold=fold,
        adapter_excluded_ids=excluded_ids,
        adapter_train_file_sha256=train_file_sha256,
        adapter_exclusions_file_sha256=exclusions_file_sha256,
        adapter_split_artifact_sha256=split_artifact_sha256,
        adapter_development_shard_sha256=development_shard_sha256,
        config=config,
    )


class TransformersQLoRATrainingRuntime:  # pragma: no cover - requires the final GPU gate
    """Lazy offline Transformers Trainer implementation for the locked profile."""

    def __init__(
        self,
        *,
        evidence: RuntimeGateEvidence,
        config: GateBConfig = DEFAULT_GATE_B_CONFIG,
    ) -> None:
        _require_runtime_evidence(evidence, config)
        self._evidence = evidence
        self._config = config
        self._modules: Mapping[str, Any] | None = None
        self._tokenizer: Any = None
        self._model: Any = None
        self._closed = False

    @property
    def tokenizer(self) -> ChatTokenizer:
        if self._closed:
            raise GateBValidationError("training runtime is closed")
        if self._tokenizer is None:
            with _offline_environment():
                modules = self._load_modules()
                transformers = modules["transformers"]
                self._tokenizer = transformers.AutoTokenizer.from_pretrained(
                    OFFICIAL_MODEL_ID,
                    revision=PINNED_MODEL_REVISION,
                    local_files_only=True,
                    trust_remote_code=False,
                    use_fast=True,
                )
                _ensure_padding_token(self._tokenizer)
        return self._tokenizer

    def train(
        self,
        *,
        training_examples: Sequence[EncodedSFTExample],
        validation_examples: Sequence[EncodedSFTExample],
        work_dir: Path,
        export_dir: Path,
        plan: FoldSFTPlan,
        config: GateBConfig,
        resume_checkpoint: Path | None = None,
        retain_checkpoints: bool = False,
    ) -> RuntimeTrainingResult:
        _require_default_config(config)
        _validate_plan_binding(plan, config)
        _validate_encoded_partition(training_examples, plan.training_ids, "training")
        _validate_encoded_partition(validation_examples, plan.validation_ids, "validation")
        if self._closed:
            raise GateBValidationError("training runtime is closed")
        if any(export_dir.iterdir()):
            raise GateBValidationError("training export directory must start empty")
        trainer_dir = work_dir / "trainer"
        resume_contract_sha256: str | None = None
        if retain_checkpoints:
            resume_contract_sha256 = _training_resume_contract_from_root(work_dir)
        if resume_checkpoint is not None:
            if not retain_checkpoints:
                raise GateBValidationError(
                    "resuming requires retained Trainer checkpoints"
                )
            assert resume_contract_sha256 is not None
            _validate_runtime_resume_checkpoint_path(
                resume_checkpoint,
                trainer_dir,
                contract_sha256=resume_contract_sha256,
            )

        with _offline_environment():
            modules = self._load_modules()
            torch = modules["torch"]
            transformers = modules["transformers"]
            peft = modules["peft"]
            _require_smoke_bound_cuda_device(torch, self._evidence)
            tokenizer = self.tokenizer
            _, model = _load_pinned_nf4_base(modules, config, tokenizer=tokenizer)
            model = peft.prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=config.gradient_checkpointing,
                gradient_checkpointing_kwargs={
                    "use_reentrant": config.gradient_checkpointing_use_reentrant
                },
            )
            lora_config = peft.LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=config.lora_target_modules,
                bias=config.lora_bias,
                task_type="CAUSAL_LM",
            )
            model = peft.get_peft_model(model, lora_config)
            model.config.use_cache = config.use_cache
            self._model = model

            arguments = _training_arguments(
                transformers.TrainingArguments,
                output_dir=trainer_dir,
                config=config,
                retain_checkpoints=retain_checkpoints,
            )
            collator = transformers.DataCollatorForSeq2Seq(
                tokenizer=tokenizer,
                padding=True,
                label_pad_token_id=-100,
                pad_to_multiple_of=8,
                return_tensors="pt",
            )
            trainer_kwargs: dict[str, Any] = {
                "model": model,
                "args": arguments,
                "train_dataset": _EncodedDataset(training_examples),
                "eval_dataset": _EncodedDataset(validation_examples),
                "data_collator": collator,
            }
            trainer_signature = __import__("inspect").signature(
                transformers.Trainer.__init__
            ).parameters
            if "processing_class" in trainer_signature:
                trainer_kwargs["processing_class"] = tokenizer
            else:  # transformers 4.45 compatibility
                trainer_kwargs["tokenizer"] = tokenizer
            if resume_contract_sha256 is not None:
                trainer_kwargs["callbacks"] = [
                    _resume_checkpoint_callback(
                        transformers.TrainerCallback,
                        contract_sha256=resume_contract_sha256,
                    )
                ]
            trainer = transformers.Trainer(**trainer_kwargs)
            train_output = (
                trainer.train()
                if resume_checkpoint is None
                else trainer.train(resume_from_checkpoint=str(resume_checkpoint))
            )
            model.save_pretrained(export_dir, safe_serialization=True)
            _save_pinned_tokenizer_snapshot(
                export_dir,
                cached_file=transformers.utils.cached_file,
            )
            metrics = _coerce_flat_metrics(getattr(train_output, "metrics", {}))
            global_step = int(getattr(trainer.state, "global_step", 0))
            return RuntimeTrainingResult(
                global_step=global_step,
                metrics=metrics,
                package_versions=_package_versions(
                    ("torch", "transformers", "peft", "accelerate", "bitsandbytes")
                ),
            )

    def _load_modules(self) -> Mapping[str, Any]:
        if self._modules is None:
            self._modules = _lazy_gpu_modules()
        return self._modules

    def close(self) -> None:
        if self._closed:
            return
        self._model = None
        self._tokenizer = None
        gc.collect()
        if self._modules is not None:
            torch = self._modules.get("torch")
            if torch is not None and torch.cuda.is_available():
                with suppress(Exception):
                    torch.cuda.empty_cache()
        self._modules = None
        self._closed = True


class TransformersNF4GenerationBackend:
    """Reusable lazy offline NF4 backend implementing ``GenerationBackend``."""

    def __init__(
        self,
        *,
        preflight_artifact: str | Path,
        gpu_smoke_artifact: str | Path,
        gpu_acknowledgement: str,
        adapter_path: str | Path | None = None,
        adapter_split_manifest: SplitManifest | None = None,
        adapter_fold: int | None = None,
        adapter_excluded_ids: Iterable[str] | None = None,
        adapter_train_file_sha256: str | None = None,
        adapter_exclusions_file_sha256: str | None = None,
        adapter_split_artifact_sha256: str | None = None,
        adapter_development_shard_sha256: str | None = None,
        config: GateBConfig = DEFAULT_GATE_B_CONFIG,
    ) -> None:
        _require_gpu_acknowledgement(gpu_acknowledgement)
        self._gate = validate_runtime_gate(
            preflight_artifact=preflight_artifact,
            gpu_smoke_artifact=gpu_smoke_artifact,
            config=config,
        )
        _require_default_config(config)
        self._config = config
        if adapter_path is None:
            if (
                adapter_split_manifest is not None
                or adapter_fold is not None
                or adapter_excluded_ids is not None
                or adapter_train_file_sha256 is not None
                or adapter_exclusions_file_sha256 is not None
                or adapter_split_artifact_sha256 is not None
                or adapter_development_shard_sha256 is not None
            ):
                raise GateBValidationError("base backend must not receive adapter scope")
            self._adapter = None
        else:
            if (
                adapter_split_manifest is None
                or adapter_fold is None
                or adapter_excluded_ids is None
                or adapter_train_file_sha256 is None
                or adapter_exclusions_file_sha256 is None
                or adapter_split_artifact_sha256 is None
                or adapter_development_shard_sha256 is None
            ):
                raise GateBValidationError(
                    "adapter backend requires exact split, fold, and exclusion scope"
                )
            self._adapter = validate_adapter_for_fold(
                adapter_path,
                split_manifest=adapter_split_manifest,
                fold=adapter_fold,
                excluded_ids=adapter_excluded_ids,
                train_file_sha256=adapter_train_file_sha256,
                exclusions_file_sha256=adapter_exclusions_file_sha256,
                split_artifact_sha256=adapter_split_artifact_sha256,
                development_shard_sha256=adapter_development_shard_sha256,
                preflight_sha256=self._gate.preflight_sha256,
                gpu_smoke_sha256=self._gate.smoke_sha256,
                config=config,
            )
        self._modules: Mapping[str, Any] | None = None
        self._model: Any = None
        self._tokenizer: Any = None
        self._closed = False

    @property
    def checkpoint_sha256(self) -> str:
        """Return the exact digest to pass to ``run_development_baseline``."""

        return (
            BASE_MODEL_CHECKPOINT_SHA256
            if self._adapter is None
            else self._adapter.artifact_sha256
        )

    @property
    def runtime_gate_evidence(self) -> RuntimeGateEvidence:
        """Expose the exact validated B0 pair for private run-manifest binding."""

        return self._gate

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self._validate_request(request)
        if self._closed:
            raise GateBValidationError("generation backend is closed")
        with _offline_environment():
            self._ensure_loaded()
            assert self._modules is not None  # internal lazy-load invariant
            torch = self._modules["torch"]
            messages = [message.as_dict() for message in request.messages]
            input_ids = self._tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            input_count = int(input_ids.shape[-1])
            if input_count > self._config.max_sequence_length:
                raise GateBValidationError(
                    f"generation prompt has {input_count} tokens; truncation is forbidden"
                )
            input_ids = input_ids.to(self._model.device)
            # Qwen's pad token is its EOS token.  Even though this backend currently
            # sends one unpadded prompt at a time, Transformers cannot infer a safe
            # mask from those IDs and may otherwise change generation behavior.
            attention_mask = torch.ones_like(input_ids)
            torch.manual_seed(request.seed)
            torch.cuda.manual_seed_all(request.seed)
            torch.cuda.reset_peak_memory_stats(self._model.device)
            generation_kwargs: dict[str, Any] = {
                "do_sample": request.decoding_policy.do_sample,
                "num_beams": request.decoding_policy.num_beams,
                "max_new_tokens": request.decoding_policy.max_new_tokens,
                "repetition_penalty": request.decoding_policy.repetition_penalty,
                "pad_token_id": self._tokenizer.eos_token_id,
                "eos_token_id": self._tokenizer.eos_token_id,
                # Gradient-checkpointed QLoRA training keeps use_cache=False,
                # but this is an eval-only, one-prompt generation path.  The
                # KV cache is therefore both safe and required for practical
                # serial development inference.  The implementation is bound
                # by the source-tree manifest before model selection.
                "use_cache": True,
            }
            if request.decoding_policy.temperature is not None:
                generation_kwargs["temperature"] = request.decoding_policy.temperature
            if request.decoding_policy.top_p is not None:
                generation_kwargs["top_p"] = request.decoding_policy.top_p
            if request.decoding_policy.top_k is not None:
                generation_kwargs["top_k"] = request.decoding_policy.top_k
            with torch.inference_mode():
                sequences = self._model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **generation_kwargs,
                )
            torch.cuda.synchronize(self._model.device)
            generated = sequences[0, input_count:]
            output_count = int(generated.shape[-1])
            text = self._tokenizer.decode(generated, skip_special_tokens=True)
            finish_reason = (
                "length"
                if output_count >= request.decoding_policy.max_new_tokens
                else "stop"
            )
            return GenerationResult(
                text=text,
                finish_reason=finish_reason,
                input_token_count=input_count,
                output_token_count=output_count,
                peak_vram_allocated_bytes=int(
                    torch.cuda.max_memory_allocated(self._model.device)
                ),
            )

    def _validate_request(self, request: GenerationRequest) -> None:
        if not isinstance(request, GenerationRequest):
            raise TypeError("request must be a GenerationRequest")
        expected = {
            "model_id": OFFICIAL_MODEL_ID,
            "revision": PINNED_MODEL_REVISION,
            "route": self._config.route,
            "config_sha256": self._config.sha256,
        }
        mismatched = [key for key, value in expected.items() if getattr(request, key) != value]
        if mismatched:
            raise GateBValidationError(f"generation request binding mismatch: {mismatched!r}")
        if request.decoding_policy != self._config.decoding_policy:
            raise GateBValidationError("generation request changes the locked decoding policy")
        if isinstance(request.seed, bool) or not isinstance(request.seed, int) or request.seed < 0:
            raise GateBValidationError("generation seed must be a non-negative integer")
        if not request.messages or any(
            not isinstance(item, ChatMessage) for item in request.messages
        ):
            raise GateBValidationError("generation messages must be non-empty ChatMessage values")
        prompt_digest = hashlib.sha256(
            canonical_json_bytes(
                {
                    "messages": [message.as_dict() for message in request.messages],
                    "add_generation_prompt": True,
                }
            )
        ).hexdigest()
        if request.prompt_sha256 != prompt_digest:
            raise GateBValidationError("generation prompt_sha256 does not match messages")

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        modules = _lazy_gpu_modules()
        torch = modules["torch"]
        _require_smoke_bound_cuda_device(torch, self._gate)
        tokenizer, model = _load_pinned_nf4_base(modules, self._config)
        if self._adapter is not None:
            peft = modules["peft"]
            model = peft.PeftModel.from_pretrained(
                model,
                self._adapter.path,
                is_trainable=False,
                local_files_only=True,
            )
        model.eval()
        self._modules = modules
        self._tokenizer = tokenizer
        self._model = model

    def close(self) -> None:
        if self._closed:
            return
        self._model = None
        self._tokenizer = None
        gc.collect()
        if self._modules is not None:
            torch = self._modules.get("torch")
            if torch is not None and torch.cuda.is_available():
                with suppress(Exception):
                    torch.cuda.empty_cache()
        self._modules = None
        self._closed = True

    def __enter__(self) -> TransformersNF4GenerationBackend:
        if self._closed:
            raise GateBValidationError("generation backend is closed")
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class _EncodedDataset:
    def __init__(self, values: Sequence[EncodedSFTExample]) -> None:
        self._values = tuple(values)

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        value = self._values[index]
        return {
            "input_ids": list(value.input_ids),
            "attention_mask": list(value.attention_mask),
            "labels": list(value.labels),
        }


def _build_validation_examples(
    records: Sequence[MathRecord],
    *,
    validation_ids: Sequence[str],
    split_manifest: SplitManifest,
    fold: int,
    config: GateBConfig,
) -> tuple[DirectAnswerSFTExample, ...]:
    by_id = {record.id: record for record in records}
    assignments = split_manifest.assignment_by_id()
    eligibility_digest = _ids_sha256(tuple(validation_ids))
    output: list[DirectAnswerSFTExample] = []
    for problem_id in validation_ids:
        record = by_id[problem_id]
        answer = _required_answer(record)
        assignment = assignments[problem_id]
        if (
            assignment.partition is not SplitPartition.CROSS_VALIDATION
            or assignment.fold != fold
        ):
            raise GateBValidationError(f"{problem_id}: validation record is outside fold {fold}")
        output.append(
            DirectAnswerSFTExample(
                problem_id=problem_id,
                question_sha256=hashlib.sha256(record.question_raw.encode("utf-8")).hexdigest(),
                prompt_messages=(
                    ChatMessage("system", config.system_prompt),
                    ChatMessage("user", record.question_raw),
                ),
                target_text=f"Final answer: {answer}",
                split_version=split_manifest.version,
                split_sha256=split_manifest.sha256,
                source_groups_sha256=split_manifest.source_groups_sha256,
                fold=fold,
                partition="fold_validation",
                split_partition=SplitPartition.CROSS_VALIDATION.value,
                group_id=assignment.group_id,
                eligibility_ids_sha256=eligibility_digest,
            )
        )
    return tuple(output)


def _validated_exact_records(
    records: Iterable[MathRecord],
    expected_ids: Sequence[str],
    field_name: str,
) -> tuple[MathRecord, ...]:
    materialized = tuple(records)
    if not materialized:
        raise GateBValidationError(f"{field_name} must not be empty")
    if any(not isinstance(record, MathRecord) for record in materialized):
        raise TypeError(f"{field_name} must contain MathRecord values")
    actual_ids = tuple(record.id for record in materialized)
    _validated_train_ids(actual_ids, field_name)
    if len(set(actual_ids)) != len(actual_ids):
        raise GateBValidationError(f"{field_name} contains duplicate IDs")
    missing = sorted(set(expected_ids) - set(actual_ids))
    extra = sorted(set(actual_ids) - set(expected_ids))
    if missing or extra:
        raise GateBValidationError(
            f"{field_name} must exactly match its split-derived partition; "
            f"missing={missing[:5]!r}, extra={extra[:5]!r}"
        )
    by_id = {record.id: record for record in materialized}
    ordered = tuple(by_id[record_id] for record_id in expected_ids)
    for record in ordered:
        _required_answer(record)
    return ordered


def _required_answer(record: MathRecord) -> int:
    if not record.question_raw or not record.question_raw.strip():
        raise GateBValidationError(f"{record.id}: organizer train question is empty")
    if isinstance(record.answer, bool) or not isinstance(record.answer, int):
        raise GateBValidationError(f"{record.id}: organizer train answer is missing or invalid")
    return record.answer


def _validated_train_ids(
    values: Iterable[str],
    field_name: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    materialized = tuple(values)
    if not materialized and not allow_empty:
        raise GateBValidationError(f"{field_name} must not be empty")
    invalid = [
        value
        for value in materialized
        if not isinstance(value, str) or not _TRAIN_ID_RE.fullmatch(value)
    ]
    if invalid:
        raise GateBValidationError(
            f"{field_name} accepts organizer train-XXXXXX IDs only; invalid={invalid[:5]!r}"
        )
    if len(set(materialized)) != len(materialized):
        raise GateBValidationError(f"{field_name} contains duplicate IDs")
    return tuple(materialized)


def _validate_encoded_partition(
    values: Sequence[EncodedSFTExample],
    expected_ids: Sequence[str],
    role: str,
) -> None:
    if tuple(value.problem_id for value in values) != tuple(expected_ids):
        raise GateBValidationError(f"encoded {role} examples changed split-derived ID order")
    for value in values:
        if value.config_sha256 != DEFAULT_GATE_B_CONFIG.sha256:
            raise GateBValidationError(f"encoded {role} example has wrong config binding")
        if value.sequence_token_count > DEFAULT_GATE_B_CONFIG.max_sequence_length:
            raise GateBValidationError(f"encoded {role} example exceeds sequence limit")
        if value.labels[: value.prompt_token_count] != (-100,) * value.prompt_token_count:
            raise GateBValidationError(f"encoded {role} example exposes prompt labels")
        if not any(label != -100 for label in value.labels):
            raise GateBValidationError(f"encoded {role} example has no response labels")


def _require_default_config(config: GateBConfig) -> None:
    if not isinstance(config, GateBConfig):
        raise TypeError("config must be a GateBConfig")
    if config != DEFAULT_GATE_B_CONFIG or config.sha256 != DEFAULT_GATE_B_CONFIG.sha256:
        raise GateBValidationError("runtime is locked to DEFAULT_GATE_B_CONFIG exactly")
    if config.model_id != OFFICIAL_MODEL_ID or config.revision != PINNED_MODEL_REVISION:
        raise GateBValidationError("runtime model identity is not the fixed official revision")


def _require_gpu_acknowledgement(value: str) -> None:
    if value != GPU_EXECUTION_ACKNOWLEDGEMENT:
        raise GateBPreflightRequiredError(
            "explicit GPU acknowledgement is required; pass "
            f"{GPU_EXECUTION_ACKNOWLEDGEMENT!r} only at the deferred GPU gate"
        )


def _require_runtime_evidence(evidence: RuntimeGateEvidence, config: GateBConfig) -> None:
    _require_default_config(config)
    if not isinstance(evidence, RuntimeGateEvidence):
        raise TypeError("evidence must be RuntimeGateEvidence")
    expected = (OFFICIAL_MODEL_ID, PINNED_MODEL_REVISION, config.sha256)
    if (evidence.model_id, evidence.revision, evidence.config_sha256) != expected:
        raise GateBPreflightRequiredError("runtime evidence is not bound to this locked config")
    _required_sha256(evidence.preflight_sha256, "preflight_sha256")
    _required_sha256(evidence.smoke_sha256, "smoke_sha256")
    if not isinstance(evidence.device_name, str) or not evidence.device_name.strip():
        raise GateBPreflightRequiredError("runtime evidence has no bound GPU device name")


def _validated_source_manifest_evidence(
    value: SourceTreeArtifactEvidence,
) -> SourceTreeArtifactEvidence:
    """Require the source snapshot bytes that a training manifest will cite."""

    if not isinstance(value, SourceTreeArtifactEvidence):
        raise TypeError("source_manifest must be SourceTreeArtifactEvidence")
    supplied = Path(value.path)
    if supplied.is_symlink():
        raise GateBValidationError("source manifest must not be a symbolic link")
    source = supplied.resolve(strict=True)
    if not source.is_file():
        raise GateBValidationError("source manifest must be a regular file")
    digest = sha256_file(source)
    if digest != _required_sha256(value.sha256, "source manifest sha256"):
        raise GateBValidationError("source manifest bytes changed after validation")
    tree_sha256 = _required_sha256(value.tree_sha256, "source tree_sha256")
    if (
        isinstance(value.file_count, bool)
        or not isinstance(value.file_count, int)
        or value.file_count < 1
    ):
        raise GateBValidationError("source manifest file_count must be positive")
    return SourceTreeArtifactEvidence(
        path=str(source),
        sha256=digest,
        tree_sha256=tree_sha256,
        file_count=value.file_count,
    )


def _prepare_training_resume_context(
    resume_dir: str | Path | None,
    *,
    output_dir: Path,
    plan: FoldSFTPlan,
    gate: RuntimeGateEvidence,
    data_provenance: Mapping[str, str],
    source_manifest: SourceTreeArtifactEvidence,
    config: GateBConfig,
) -> _TrainingResumeContext | None:
    """Create or verify the durable, exact-contract Trainer workspace.

    The final adapter is still staged outside this directory so a failed export
    cannot alter reusable Trainer checkpoints.  A pre-existing directory must
    already contain an exact contract; accepting an unbound checkpoint would
    make a resume indistinguishable from a different split, corpus, or B0 run.
    """

    if resume_dir is None:
        return None
    if not isinstance(resume_dir, (str, Path)):
        raise TypeError("resume_dir must be a path string or Path")
    supplied = Path(resume_dir)
    if supplied.is_symlink():
        raise GateBValidationError("resume_dir must not be a symbolic link")
    root = supplied.resolve(strict=False)
    if not root.name or root.name in {".", ".."}:
        raise GateBValidationError("resume_dir must name a child directory")
    if not root.parent.is_dir() or root.parent.is_symlink():
        raise GateBValidationError(
            "resume_dir parent must be an existing real directory"
        )
    if _paths_overlap(root, output_dir):
        raise GateBValidationError(
            "resume_dir and output_dir must be disjoint to protect checkpoints"
        )

    expected = _training_resume_contract_payload(
        plan=plan,
        gate=gate,
        data_provenance=data_provenance,
        source_manifest=source_manifest,
        config=config,
    )
    if root.exists():
        if root.is_symlink() or not root.is_dir():
            raise GateBValidationError("resume_dir must be a real directory")
        contract_path = root / _TRAINING_RESUME_CONTRACT_FILENAME
        if contract_path.exists():
            _validate_training_resume_contract(root, expected)
        elif any(root.iterdir()):
            raise GateBValidationError("persistent training resume contract is missing")
        else:
            _write_training_resume_contract(root, expected)
    else:
        try:
            root.mkdir(mode=0o700)
        except FileExistsError as exc:
            # A concurrent creator may have completed the contract after the
            # existence check.  Treat it exactly like an ordinary resume.
            if root.is_symlink() or not root.is_dir():
                raise GateBValidationError("resume_dir must be a real directory") from exc
            _validate_training_resume_contract(root, expected)
        else:
            _write_training_resume_contract(root, expected)
    return _TrainingResumeContext(
        root=root,
        contract_sha256=_required_sha256(
            expected.get("contract_sha256"), "training resume contract sha256"
        ),
    )


def _write_training_resume_contract(root: Path, payload: Mapping[str, object]) -> None:
    serialized = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    try:
        _atomic_write_new_file(root / _TRAINING_RESUME_CONTRACT_FILENAME, serialized)
    except GateBArtifactExistsError:
        _validate_training_resume_contract(root, payload)
    _fsync_directory(root)


def _training_resume_contract_payload(
    *,
    plan: FoldSFTPlan,
    gate: RuntimeGateEvidence,
    data_provenance: Mapping[str, str],
    source_manifest: SourceTreeArtifactEvidence,
    config: GateBConfig,
) -> dict[str, object]:
    """Return the immutable evidence contract for retained Trainer state."""

    _validate_plan_binding(plan, config)
    payload: dict[str, object] = {
        "schema_version": _TRAINING_RESUME_CONTRACT_SCHEMA,
        "model_id": OFFICIAL_MODEL_ID,
        "revision": PINNED_MODEL_REVISION,
        "base_model_checkpoint_sha256": BASE_MODEL_CHECKPOINT_SHA256,
        "config": config.as_dict(),
        "config_sha256": config.sha256,
        "runtime_gate": {
            "preflight_sha256": gate.preflight_sha256,
            "gpu_smoke_sha256": gate.smoke_sha256,
            "model_id": gate.model_id,
            "revision": gate.revision,
            "config_sha256": gate.config_sha256,
            "device_name": gate.device_name,
        },
        "source_manifest": {
            "sha256": source_manifest.sha256,
            "tree_sha256": source_manifest.tree_sha256,
            "file_count": source_manifest.file_count,
        },
        "data_provenance": dict(sorted(data_provenance.items())),
        "split": {
            "version": plan.split_version,
            "sha256": plan.split_sha256,
            "source_groups_sha256": plan.source_groups_sha256,
            "fold": plan.fold,
            "excluded_ids_sha256": plan.excluded_ids_sha256,
            "training_ids_sha256": plan.training_ids_sha256,
            "validation_ids_sha256": plan.validation_ids_sha256,
            "training_examples_sha256": plan.training_examples_sha256,
            "validation_examples_sha256": plan.validation_examples_sha256,
        },
        "training_target": {
            "kind": plan.training_target_kind,
            "candidate_config_sha256": plan.rationale_candidate_config_sha256,
            "candidate_config_file_sha256": (
                plan.rationale_candidate_config_file_sha256
            ),
            "corpus_records_sha256": plan.rationale_corpus_records_sha256,
            "corpus_manifest_sha256": plan.rationale_corpus_manifest_sha256,
            "corpus_audit_sha256": plan.rationale_corpus_audit_sha256,
        },
        "tokenizer_evidence": {
            "tokenizer_json_sha256": _PINNED_TOKENIZER_JSON_SHA256,
            "tokenizer_config_json_sha256": _PINNED_TOKENIZER_CONFIG_JSON_SHA256,
            "chat_template_sha256": _PINNED_TOKENIZER_CHAT_TEMPLATE_SHA256,
            "required_files": sorted(_REQUIRED_TOKENIZER_FILES),
        },
        "checkpoint_retention": "retain_all",
    }
    payload["contract_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return payload


def _validate_training_resume_contract(
    root: Path, expected: Mapping[str, object]
) -> None:
    contract_path = root / _TRAINING_RESUME_CONTRACT_FILENAME
    if not contract_path.exists():
        raise GateBValidationError("persistent training resume contract is missing")
    _, actual, _ = _load_json_artifact(contract_path, "training resume contract")
    stored_digest = _required_sha256(
        actual.get("contract_sha256"), "training resume contract sha256"
    )
    unhashed = dict(actual)
    unhashed.pop("contract_sha256", None)
    computed_digest = hashlib.sha256(canonical_json_bytes(unhashed)).hexdigest()
    if stored_digest != computed_digest:
        raise GateBValidationError("training resume contract hash is invalid")
    if dict(actual) != dict(expected):
        mismatched = sorted(
            key
            for key in set(actual) | set(expected)
            if actual.get(key) != expected.get(key)
        )
        raise GateBValidationError(
            "training resume contract mismatches requested evidence: "
            f"{mismatched!r}"
        )


@contextmanager
def _locked_training_resume_checkpoint(
    context: _TrainingResumeContext,
) -> Iterable[Path | None]:
    """Hold an exclusive fail-closed lock while selecting a checkpoint."""

    if context.root.is_symlink() or not context.root.is_dir():
        raise GateBValidationError("resume_dir disappeared or is no longer a real directory")
    lock_path = context.root / _TRAINING_RESUME_LOCK_FILENAME
    payload = canonical_json_bytes(
        {
            "schema_version": _TRAINING_RESUME_CONTRACT_SCHEMA,
            "contract_sha256": context.contract_sha256,
            "pid": os.getpid(),
        }
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except FileExistsError as exc:
        raise GateBValidationError(
            "training resume lock exists; refusing concurrent or stale resume"
        ) from exc
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(context.root)
        expected = _training_resume_contract_from_root(context.root)
        if expected != context.contract_sha256:
            raise GateBValidationError("training resume contract changed while acquiring lock")
        yield _select_resume_checkpoint(context)
    finally:
        with suppress(FileNotFoundError):
            lock_path.unlink()
        _fsync_directory(context.root)


def _training_resume_contract_from_root(root: Path) -> str:
    """Validate the stored self-hash without accepting an unbound contract."""

    _, payload, _ = _load_json_artifact(
        root / _TRAINING_RESUME_CONTRACT_FILENAME, "training resume contract"
    )
    stored_digest = _required_sha256(
        payload.get("contract_sha256"), "training resume contract sha256"
    )
    unhashed = dict(payload)
    unhashed.pop("contract_sha256", None)
    if hashlib.sha256(canonical_json_bytes(unhashed)).hexdigest() != stored_digest:
        raise GateBValidationError("training resume contract hash is invalid")
    return stored_digest


def _select_resume_checkpoint(context: _TrainingResumeContext) -> Path | None:
    """Choose the newest complete checkpoint without destroying failed attempts.

    A Trainer creates ``checkpoint-N`` before all of its files are durable.  An
    interruption may therefore leave a partial directory after an earlier valid
    checkpoint.  Such a directory is not reusable, but it also must not make a
    verified earlier checkpoint unusable.  We move only contract-unbound or
    structurally-corrupt candidates to a private forensic attempt.  A present
    but mismatched checkpoint contract remains a hard failure: that is evidence
    of a different run or tampering, not an interrupted write.
    """

    trainer_root = context.root / "trainer"
    marker_path = context.root / _TRAINING_RESUME_STARTED_FILENAME
    _validate_training_resume_started(marker_path, context.contract_sha256)
    if not trainer_root.exists():
        if marker_path.exists():
            raise GateBValidationError(
                "persistent training resume has no checkpoint after training started"
            )
        return None
    if trainer_root.is_symlink() or not trainer_root.is_dir():
        raise GateBValidationError("Trainer checkpoint root is missing or unsafe")

    checkpoints: list[tuple[int, Path]] = []
    recoverable: list[_RecoverableTrainerCheckpoint] = []
    for candidate in sorted(trainer_root.iterdir(), key=lambda item: item.name):
        if candidate.is_symlink():
            raise GateBValidationError("Trainer checkpoint root contains a symbolic link")
        match = _TRAINER_CHECKPOINT_RE.fullmatch(candidate.name)
        if match is None:
            continue
        step = int(match.group(1))
        reason = _recoverable_trainer_checkpoint_reason(
            candidate,
            expected_global_step=step,
            contract_sha256=context.contract_sha256,
        )
        if reason is not None:
            recoverable.append(
                _RecoverableTrainerCheckpoint(
                    path=candidate,
                    global_step=step,
                    reason=reason,
                )
            )
            continue
        checkpoints.append((step, candidate))
    if recoverable:
        _archive_recoverable_trainer_checkpoints(context, trainer_root, recoverable)
    if not checkpoints:
        if marker_path.exists() and not recoverable:
            raise GateBValidationError(
                "persistent training resume has no complete checkpoint after training started"
            )
        return None
    return max(checkpoints, key=lambda item: item[0])[1]


def _recoverable_trainer_checkpoint_reason(
    checkpoint: Path,
    *,
    expected_global_step: int,
    contract_sha256: str,
) -> str | None:
    """Return a forensic reason, or fail closed for unsafe/foreign state.

    The sidecar is written only after Trainer's normal checkpoint save.  Its
    absence therefore identifies an interrupted/partial save.  Once present,
    it must exactly bind this resume contract; a mismatch is never downgraded to
    a recoverable corruption.
    """

    if not checkpoint.is_dir():
        return "checkpoint_not_directory"
    try:
        descendants = tuple(checkpoint.rglob("*"))
    except OSError as exc:
        raise GateBValidationError(
            f"cannot inspect Trainer checkpoint {checkpoint.name}: {exc}"
        ) from exc
    if any(item.is_symlink() for item in descendants):
        raise GateBValidationError("Trainer checkpoint contains a symbolic link")

    sidecar = checkpoint / _TRAINING_RESUME_CHECKPOINT_FILENAME
    if sidecar.is_symlink():
        raise GateBValidationError("Trainer checkpoint resume contract is a symbolic link")
    if not sidecar.exists():
        return "missing_resume_contract"
    if not sidecar.is_file():
        raise GateBValidationError("Trainer checkpoint resume contract is not a regular file")

    # Do this before the structural validator.  A checkpoint with both a
    # missing optimizer and another run's sidecar must remain fail-closed.
    _validate_training_resume_checkpoint_contract(
        checkpoint,
        expected_global_step=expected_global_step,
        contract_sha256=contract_sha256,
    )
    try:
        _validate_trainer_checkpoint(
            checkpoint,
            expected_global_step=expected_global_step,
            contract_sha256=contract_sha256,
        )
    except GateBValidationError:
        return "corrupt_after_contract_binding"
    return None


def _archive_recoverable_trainer_checkpoints(
    context: _TrainingResumeContext,
    trainer_root: Path,
    checkpoints: Sequence[_RecoverableTrainerCheckpoint],
) -> None:
    """Move invalid checkpoint entries to an immutable, private forensic attempt.

    The active ``trainer`` directory is then safe for either resuming an older
    complete checkpoint or beginning the next exact-contract attempt.  Entries
    are renamed rather than deleted or overwritten, so their original bytes
    remain available for diagnosis.
    """

    if not checkpoints:
        return
    forensic_root = context.root / _TRAINING_RESUME_FORENSIC_DIRECTORY
    if forensic_root.exists():
        if forensic_root.is_symlink() or not forensic_root.is_dir():
            raise GateBValidationError("training resume forensic root is unsafe")
    else:
        forensic_root.mkdir(mode=0o700)
    attempt_root = _new_training_resume_forensic_attempt(forensic_root)
    entries = []
    try:
        for checkpoint in checkpoints:
            source = checkpoint.path
            destination = attempt_root / source.name
            if destination.exists() or destination.is_symlink():  # pragma: no cover - new attempt
                raise GateBArtifactExistsError(
                    "training resume forensic destination already exists: "
                    f"{destination}"
                )
            os.rename(source, destination)
            entries.append(
                {
                    "checkpoint": source.name,
                    "global_step": checkpoint.global_step,
                    "reason": checkpoint.reason,
                }
            )
        payload = {
            "schema_version": _TRAINING_RESUME_FORENSIC_SCHEMA,
            "contract_sha256": context.contract_sha256,
            "entries": entries,
        }
        _atomic_write_new_file(
            attempt_root / "manifest.json",
            (
                json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
            ).encode("utf-8"),
        )
        _fsync_directory(attempt_root)
        _fsync_directory(trainer_root)
        _fsync_directory(forensic_root)
        _fsync_directory(context.root)
    except BaseException:
        # Renamed entries deliberately stay in the forensic attempt.  Rolling
        # them back could overwrite a concurrently recreated Trainer checkpoint.
        _fsync_directory(attempt_root)
        _fsync_directory(trainer_root)
        raise


def _new_training_resume_forensic_attempt(forensic_root: Path) -> Path:
    """Reserve a no-overwrite forensic attempt directory under the held lock."""

    for index in range(1, 1_000_000):
        attempt = forensic_root / f"attempt-{index:06d}"
        try:
            attempt.mkdir(mode=0o700)
        except FileExistsError as exc:
            if attempt.is_symlink() or not attempt.is_dir():
                raise GateBValidationError(
                    "training resume forensic attempt is unsafe"
                ) from exc
            continue
        return attempt
    raise GateBArtifactExistsError("training resume forensic attempt namespace is exhausted")


def _mark_training_resume_started(context: _TrainingResumeContext) -> None:
    marker_path = context.root / _TRAINING_RESUME_STARTED_FILENAME
    if marker_path.exists():
        _validate_training_resume_started(marker_path, context.contract_sha256)
        return
    payload = (
        json.dumps(
            {
                "schema_version": _TRAINING_RESUME_CONTRACT_SCHEMA,
                "contract_sha256": context.contract_sha256,
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")
    _atomic_write_new_file(marker_path, payload)
    _fsync_directory(context.root)


def _validate_training_resume_started(path: Path, contract_sha256: str) -> None:
    if not path.exists():
        return
    _, payload, _ = _load_json_artifact(path, "training resume start marker")
    expected = {
        "schema_version": _TRAINING_RESUME_CONTRACT_SCHEMA,
        "contract_sha256": contract_sha256,
    }
    if dict(payload) != expected:
        raise GateBValidationError("training resume start marker does not match contract")


def _validate_trainer_checkpoint(
    checkpoint: Path,
    *,
    expected_global_step: int,
    contract_sha256: str,
) -> None:
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise GateBValidationError("Trainer checkpoint must be a real directory")
    files = _regular_artifact_files(checkpoint)
    names = {path.relative_to(checkpoint).as_posix() for path in files}
    missing = sorted(_REQUIRED_TRAINER_CHECKPOINT_FILES - names)
    if missing:
        raise GateBValidationError(
            f"Trainer checkpoint is incomplete; missing={missing!r}"
        )
    _validate_adapter_config(checkpoint)
    _validate_adapter_weight_files(checkpoint, names)
    _validate_training_resume_checkpoint_contract(
        checkpoint,
        expected_global_step=expected_global_step,
        contract_sha256=contract_sha256,
    )
    _, state, _ = _load_json_artifact(
        checkpoint / "trainer_state.json", "Trainer checkpoint state"
    )
    global_step = state.get("global_step")
    if (
        isinstance(global_step, bool)
        or not isinstance(global_step, int)
        or global_step != expected_global_step
    ):
        raise GateBValidationError(
            "Trainer checkpoint global_step does not match its directory name"
        )


def _validate_runtime_resume_checkpoint_path(
    resume_checkpoint: Path,
    trainer_root: Path,
    *,
    contract_sha256: str,
) -> None:
    if resume_checkpoint.is_symlink():
        raise GateBValidationError("resume checkpoint must not be a symbolic link")
    try:
        resolved_checkpoint = resume_checkpoint.resolve(strict=True)
        resolved_trainer_root = trainer_root.resolve(strict=True)
    except FileNotFoundError as exc:
        raise GateBValidationError("resume checkpoint or Trainer root is missing") from exc
    if resolved_checkpoint.parent != resolved_trainer_root:
        raise GateBValidationError("resume checkpoint is outside the persistent Trainer root")
    match = _TRAINER_CHECKPOINT_RE.fullmatch(resolved_checkpoint.name)
    if match is None:
        raise GateBValidationError("resume checkpoint directory name is invalid")
    _validate_trainer_checkpoint(
        resolved_checkpoint,
        expected_global_step=int(match.group(1)),
        contract_sha256=contract_sha256,
    )


def _validate_training_resume_checkpoint_contract(
    checkpoint: Path,
    *,
    expected_global_step: int,
    contract_sha256: str,
) -> None:
    _, payload, _ = _load_json_artifact(
        checkpoint / _TRAINING_RESUME_CHECKPOINT_FILENAME,
        "Trainer checkpoint resume contract",
    )
    expected = {
        "schema_version": _TRAINING_RESUME_CONTRACT_SCHEMA,
        "contract_sha256": contract_sha256,
        "global_step": expected_global_step,
    }
    if dict(payload) != expected:
        raise GateBValidationError("Trainer checkpoint resume contract does not match")


def _resume_checkpoint_callback(
    callback_base: type[Any], *, contract_sha256: str
) -> Any:
    """Bind each Trainer-created checkpoint to the immutable resume contract."""

    class ResumeCheckpointContractCallback(callback_base):
        def on_save(
            self,
            arguments: Any,
            state: Any,
            control: Any,
            **_kwargs: Any,
        ) -> Any:
            global_step = getattr(state, "global_step", None)
            if (
                isinstance(global_step, bool)
                or not isinstance(global_step, int)
                or global_step <= 0
            ):
                raise GateBValidationError(
                    "Trainer saved a checkpoint with an invalid global_step"
                )
            checkpoint = Path(str(arguments.output_dir)) / f"checkpoint-{global_step}"
            _write_training_resume_checkpoint_contract(
                checkpoint,
                global_step=global_step,
                contract_sha256=contract_sha256,
            )
            return control

    return ResumeCheckpointContractCallback()


def _write_training_resume_checkpoint_contract(
    checkpoint: Path,
    *,
    global_step: int,
    contract_sha256: str,
) -> None:
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise GateBValidationError("Trainer checkpoint is missing when save callback runs")
    payload = (
        json.dumps(
            {
                "schema_version": _TRAINING_RESUME_CONTRACT_SCHEMA,
                "contract_sha256": contract_sha256,
                "global_step": global_step,
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")
    target = checkpoint / _TRAINING_RESUME_CHECKPOINT_FILENAME
    try:
        _atomic_write_new_file(target, payload)
    except GateBArtifactExistsError:
        _validate_training_resume_checkpoint_contract(
            checkpoint,
            expected_global_step=global_step,
            contract_sha256=contract_sha256,
        )
    _fsync_directory(checkpoint)


def _require_resume_capable_runtime(runtime: FoldTrainingRuntime) -> None:
    """Keep legacy injected runtimes working for non-resume calls only."""

    try:
        parameters = inspect.signature(runtime.train).parameters.values()
    except (TypeError, ValueError) as exc:
        raise GateBValidationError(
            "persistent resume requires an inspectable runtime.train() signature"
        ) from exc
    names = {parameter.name for parameter in parameters}
    supports_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters
    )
    required = {"resume_checkpoint", "retain_checkpoints"}
    if not supports_kwargs and not required.issubset(names):
        raise GateBValidationError(
            "persistent resume requires runtime.train() to accept "
            "resume_checkpoint and retain_checkpoints"
        )


def _run_training_runtime(
    runtime: FoldTrainingRuntime,
    *,
    training_examples: Sequence[EncodedSFTExample],
    validation_examples: Sequence[EncodedSFTExample],
    work_dir: Path,
    export_dir: Path,
    plan: FoldSFTPlan,
    config: GateBConfig,
    resume_checkpoint: Path | None,
    retain_checkpoints: bool,
) -> RuntimeTrainingResult:
    kwargs: dict[str, object] = {
        "training_examples": training_examples,
        "validation_examples": validation_examples,
        "work_dir": work_dir,
        "export_dir": export_dir,
        "plan": plan,
        "config": config,
    }
    if retain_checkpoints:
        kwargs["resume_checkpoint"] = resume_checkpoint
        kwargs["retain_checkpoints"] = True
    return runtime.train(**kwargs)  # type: ignore[arg-type]


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _validate_plan_binding(plan: FoldSFTPlan, config: GateBConfig) -> None:
    if not isinstance(plan, FoldSFTPlan):
        raise TypeError("plan must be FoldSFTPlan")
    if plan.config_sha256 != config.sha256:
        raise GateBValidationError("fold plan is not bound to the locked config")
    _validated_train_ids(plan.training_ids, "plan training IDs")
    _validated_train_ids(plan.validation_ids, "plan validation IDs")
    rationale_values = (
        plan.rationale_candidate_config_sha256,
        plan.rationale_candidate_config_file_sha256,
        plan.rationale_corpus_records_sha256,
        plan.rationale_corpus_manifest_sha256,
        plan.rationale_corpus_audit_sha256,
    )
    if plan.training_target_kind == "direct_answer":
        if any(value is not None for value in rationale_values):
            raise GateBValidationError(
                "direct-answer fold plan contains rationale provenance"
            )
        return
    if plan.training_target_kind != "verified_concise_rationale":
        raise GateBValidationError("fold plan training target kind is unsupported")
    for value, label in zip(
        rationale_values,
        (
            "rationale candidate config sha256",
            "rationale candidate config file sha256",
            "rationale corpus records sha256",
            "rationale corpus manifest sha256",
            "rationale corpus audit sha256",
        ),
        strict=True,
    ):
        _required_sha256(value, label)
    if (
        plan.rationale_candidate_config_sha256
        != DEFAULT_CONCISE_RATIONALE_CONFIG.sha256
    ):
        raise GateBValidationError(
            "fold plan rationale policy is not the locked concise-rationale v1 config"
        )


def _validate_adapter_target_binding(
    adapter: AdapterArtifactEvidence, plan: FoldSFTPlan
) -> None:
    expected = {
        "training_target_kind": plan.training_target_kind,
        "rationale_candidate_config_sha256": (
            plan.rationale_candidate_config_sha256
        ),
        "rationale_candidate_config_file_sha256": (
            plan.rationale_candidate_config_file_sha256
        ),
        "rationale_corpus_records_sha256": plan.rationale_corpus_records_sha256,
        "rationale_corpus_manifest_sha256": plan.rationale_corpus_manifest_sha256,
        "rationale_corpus_audit_sha256": plan.rationale_corpus_audit_sha256,
    }
    mismatched = [
        field_name
        for field_name, expected_value in expected.items()
        if getattr(adapter, field_name) != expected_value
    ]
    if mismatched:
        raise GateBValidationError(
            f"adapter training-target provenance mismatch: {mismatched!r}"
        )


def _load_json_artifact(
    path: str | Path, label: str
) -> tuple[Path, Mapping[str, object], str]:
    supplied = Path(path)
    if supplied.is_symlink():
        raise GateBValidationError(f"{label} must not be a symbolic link")
    source = supplied.resolve(strict=True)
    if not source.is_file():
        raise GateBValidationError(f"{label} must be a regular, non-symlink file")
    try:
        text = source.read_text(encoding="utf-8", errors="strict")
        payload = json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GateBValidationError(f"cannot load {label}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise GateBValidationError(f"{label} must contain one JSON object")
    return source, payload, sha256_file(source)


def _validate_smoke_payload_hash(payload: Mapping[str, object]) -> None:
    stored = _required_sha256(payload.get("payload_sha256"), "GPU smoke payload_sha256")
    unhashed = dict(payload)
    unhashed.pop("payload_sha256", None)
    computed = hashlib.sha256(canonical_json_bytes(unhashed)).hexdigest()
    if stored != computed:
        raise GateBPreflightRequiredError("final GPU smoke payload_sha256 is invalid")


def _validate_smoke_preflight_binding(
    payload: Mapping[str, object], preflight: PreflightEvidence
) -> None:
    binding = payload.get("preflight")
    if not isinstance(binding, Mapping):
        raise GateBPreflightRequiredError("final GPU smoke preflight binding is missing")
    if binding.get("sha256") != preflight.sha256:
        raise GateBPreflightRequiredError("final GPU smoke binds a different preflight file")
    if binding.get("training_ready") is not True:
        raise GateBPreflightRequiredError("final GPU smoke lacks green preflight readiness")
    size = binding.get("size_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise GateBPreflightRequiredError("final GPU smoke preflight size is invalid")
    if size != Path(preflight.path).stat().st_size:
        raise GateBPreflightRequiredError("final GPU smoke preflight size binding changed")
    selected = binding.get("selected_physical_gpu")
    if not isinstance(selected, Mapping) or selected.get("index") != 0:
        raise GateBPreflightRequiredError("final GPU smoke selected GPU evidence is invalid")
    if not isinstance(selected.get("name"), str) or not str(selected["name"]).strip():
        raise GateBPreflightRequiredError("final GPU smoke selected GPU name is missing")


def _validate_smoke_input_and_parser(payload: Mapping[str, object]) -> None:
    provenance = payload.get("input_provenance")
    if not isinstance(provenance, Mapping):
        raise GateBPreflightRequiredError("final GPU smoke input provenance is missing")
    expected_prompt_sha = hashlib.sha256(
        SYNTHETIC_SMOKE_USER_PROMPT.encode("utf-8")
    ).hexdigest()
    expected_provenance = {
        "kind": "locally_authored_synthetic_arithmetic",
        "competition_data_used": False,
        "caller_supplied_prompt_accepted": False,
        "prompt_sha256": expected_prompt_sha,
    }
    if any(provenance.get(key) != value for key, value in expected_provenance.items()):
        raise GateBPreflightRequiredError(
            "final GPU smoke may not use organizer or caller-supplied problem data"
        )
    runtime = payload.get("runtime")
    if not isinstance(runtime, Mapping):
        raise GateBPreflightRequiredError("final GPU smoke runtime evidence is missing")
    raw_generation = runtime.get("raw_generation")
    if not isinstance(raw_generation, str) or not raw_generation:
        raise GateBPreflightRequiredError("final GPU smoke raw generation is missing")
    parsed = parse_answer(raw_generation)
    stored_parser = payload.get("parser")
    expected_parser = {
        "status": parsed.status,
        "value": parsed.value,
        "source": parsed.source,
        "reason": parsed.reason,
    }
    if stored_parser != expected_parser:
        raise GateBPreflightRequiredError("final GPU smoke parser evidence is inconsistent")
    if not parsed.ok or parsed.value != SYNTHETIC_SMOKE_EXPECTED_ANSWER:
        raise GateBPreflightRequiredError("final GPU smoke did not exactly solve synthetic 2+3")


def _validate_smoke_runtime_evidence(payload: Mapping[str, object]) -> None:
    runtime = payload.get("runtime")
    if not isinstance(runtime, Mapping):  # pragma: no cover - checked by parser validator
        raise GateBPreflightRequiredError("final GPU smoke runtime evidence is missing")
    packages = runtime.get("package_versions")
    required_packages = (
        "torch",
        "transformers",
        "accelerate",
        "peft",
        "bitsandbytes",
        "triton",
    )
    if not isinstance(packages, Mapping) or any(
        not isinstance(packages.get(name), str) or not packages[name]
        for name in required_packages
    ):
        raise GateBPreflightRequiredError("final GPU smoke package evidence is incomplete")
    if runtime.get("cuda_device_count") != 1 or runtime.get("device_index") != 0:
        raise GateBPreflightRequiredError("final GPU smoke did not use exactly CUDA device 0")
    if not isinstance(runtime.get("device_name"), str) or not str(
        runtime["device_name"]
    ).strip():
        raise GateBPreflightRequiredError("final GPU smoke runtime device name is missing")
    if not isinstance(runtime.get("pre_context_physical_device_name"), str) or not str(
        runtime["pre_context_physical_device_name"]
    ).strip():
        raise GateBPreflightRequiredError(
            "final GPU smoke pre-context physical device name is missing"
        )
    integer_fields = (
        "pre_context_physical_total_bytes",
        "pre_context_physical_used_bytes",
        "pre_context_physical_free_bytes",
        "physical_total_bytes",
        "physical_free_bytes_before",
        "physical_free_bytes_after_cleanup",
        "allocated_bytes_before",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
    )
    integers: dict[str, int] = {}
    for field_name in integer_fields:
        value = runtime.get(field_name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise GateBPreflightRequiredError(
                f"final GPU smoke runtime {field_name} is invalid"
            )
        integers[field_name] = value
    if integers["physical_total_bytes"] <= 0:
        raise GateBPreflightRequiredError("final GPU smoke physical VRAM total is invalid")
    if integers["pre_context_physical_total_bytes"] <= 0:
        raise GateBPreflightRequiredError(
            "final GPU smoke pre-context physical VRAM total is invalid"
        )
    if (
        integers["pre_context_physical_used_bytes"]
        > integers["pre_context_physical_total_bytes"]
        or integers["pre_context_physical_free_bytes"]
        > integers["pre_context_physical_total_bytes"]
        or (
            integers["pre_context_physical_used_bytes"]
            + integers["pre_context_physical_free_bytes"]
            > integers["pre_context_physical_total_bytes"]
        )
    ):
        raise GateBPreflightRequiredError(
            "final GPU smoke pre-context physical VRAM is inconsistent"
        )
    if (
        integers["pre_context_physical_free_bytes"]
        < DEFAULT_GPU_SMOKE_CONFIG.minimum_free_vram_mib * 1024 * 1024
    ):
        raise GateBPreflightRequiredError(
            "final GPU smoke pre-context physical free VRAM is insufficient"
        )
    if (
        integers["pre_context_physical_used_bytes"]
        > DEFAULT_GPU_SMOKE_CONFIG.maximum_preexisting_used_vram_mib * 1024 * 1024
    ):
        raise GateBPreflightRequiredError(
            "final GPU smoke pre-context physical GPU was occupied"
        )
    if integers["physical_free_bytes_before"] > integers["physical_total_bytes"]:
        raise GateBPreflightRequiredError("final GPU smoke free VRAM exceeds total")
    if integers["physical_free_bytes_after_cleanup"] > integers["physical_total_bytes"]:
        raise GateBPreflightRequiredError("final GPU smoke cleanup VRAM exceeds total")
    if integers["peak_reserved_bytes"] < integers["peak_allocated_bytes"]:
        raise GateBPreflightRequiredError("final GPU smoke peak VRAM evidence is inconsistent")
    for field_name in (
        "training_latency_ms",
        "generation_latency_ms",
        "total_latency_ms",
        "training_loss",
    ):
        value = runtime.get(field_name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise GateBPreflightRequiredError(
                f"final GPU smoke runtime {field_name} is invalid"
            )


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise GateBValidationError(f"duplicate JSON key {key!r}")
        output[key] = value
    return output


def _reject_json_constant(value: str) -> None:
    raise GateBValidationError(f"non-standard JSON numeric constant {value!r}")


def _required_sha256(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise GateBValidationError(f"{field_name} must be a lowercase SHA-256")
    return value


def _ids_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256(canonical_json_bytes(list(values))).hexdigest()


def _sft_examples_sha256(values: Sequence[DirectAnswerSFTExample]) -> str:
    return hashlib.sha256(
        canonical_json_bytes([asdict(example) for example in values])
    ).hexdigest()


def _validated_new_file_target(path: str | Path, label: str) -> Path:
    target = Path(path).resolve(strict=False)
    if not target.parent.is_dir() or target.parent.is_symlink():
        raise GateBValidationError(f"{label} parent must be an existing real directory")
    if target.exists() or target.is_symlink():
        raise GateBArtifactExistsError(f"refusing to overwrite existing {label}: {target}")
    return target


def _validated_new_directory_target(path: str | Path) -> Path:
    target = Path(path).resolve(strict=False)
    if not target.name or target.name in {".", ".."}:
        raise GateBValidationError("output_dir must name a child directory")
    if not target.parent.is_dir() or target.parent.is_symlink():
        raise GateBValidationError("output_dir parent must be an existing real directory")
    if target.exists() or target.is_symlink():
        raise GateBArtifactExistsError(f"refusing to overwrite adapter directory: {target}")
    return target


def _atomic_write_new_file(target: Path, payload: bytes) -> None:
    descriptor, raw_temp = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(raw_temp)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            raise GateBArtifactExistsError(f"refusing to overwrite artifact: {target}") from exc
        _fsync_directory(target.parent)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


def _prepare_adapter_bundle(
    root: Path,
    *,
    plan: FoldSFTPlan,
    gate: RuntimeGateEvidence,
    result: RuntimeTrainingResult,
    data_provenance: Mapping[str, str],
    source_manifest: SourceTreeArtifactEvidence,
    config: GateBConfig,
) -> None:
    if not any(root.iterdir()):
        raise GateBValidationError("training runtime produced an empty adapter export")
    if (root / _MANIFEST_FILENAME).exists() or (root / _CHECKSUM_FILENAME).exists():
        raise GateBValidationError("training runtime must not pre-create provenance files")
    _canonicalize_adapter_config(root)
    files = _regular_artifact_files(root)
    names = {path.relative_to(root).as_posix() for path in files}
    missing_tokenizer = sorted(_REQUIRED_TOKENIZER_FILES - names)
    if missing_tokenizer:
        raise GateBValidationError(
            f"training runtime did not save a complete tokenizer: {missing_tokenizer!r}"
        )
    _validate_saved_tokenizer(root)
    _validate_adapter_config(root)
    _validate_adapter_weight_files(root, names)
    inventory = _file_inventory(root, excluded=set())
    manifest = {
        "schema_version": (
            _ADAPTER_MANIFEST_SCHEMA
            if plan.training_target_kind == "direct_answer"
            else _RATIONALE_ADAPTER_MANIFEST_SCHEMA
        ),
        "model_id": OFFICIAL_MODEL_ID,
        "revision": PINNED_MODEL_REVISION,
        "config": config.as_dict(),
        "config_sha256": config.sha256,
        "preflight_sha256": gate.preflight_sha256,
        "gpu_smoke_sha256": gate.smoke_sha256,
        "source_manifest_sha256": source_manifest.sha256,
        "source_tree_sha256": source_manifest.tree_sha256,
        "source_file_count": source_manifest.file_count,
        "train_file_sha256": data_provenance["train_file_sha256"],
        "exclusions_file_sha256": data_provenance["exclusions_file_sha256"],
        "split_artifact_sha256": data_provenance["split_artifact_sha256"],
        "development_shard_sha256": data_provenance[
            "development_shard_sha256"
        ],
        "split_version": plan.split_version,
        "split_sha256": plan.split_sha256,
        "source_groups_sha256": plan.source_groups_sha256,
        "fold": plan.fold,
        "partition": "fold_training",
        "excluded_ids_sha256": plan.excluded_ids_sha256,
        "training_count": len(plan.training_ids),
        "training_ids_sha256": plan.training_ids_sha256,
        "training_examples_sha256": plan.training_examples_sha256,
        "validation_count": len(plan.validation_ids),
        "validation_ids_sha256": plan.validation_ids_sha256,
        "validation_examples_sha256": plan.validation_examples_sha256,
        "response_only_labels": True,
        "truncation": False,
        "global_step": result.global_step,
        "metrics": dict(sorted(result.metrics.items())),
        "package_versions": dict(sorted(result.package_versions.items())),
        "files": inventory,
    }
    if plan.training_target_kind != "direct_answer":
        if plan.training_target_kind != "verified_concise_rationale":
            raise GateBValidationError(
                f"unsupported training target kind: {plan.training_target_kind!r}"
            )
        manifest["training_target"] = {
            "kind": plan.training_target_kind,
            "candidate_config_sha256": _required_sha256(
                plan.rationale_candidate_config_sha256,
                "rationale candidate config sha256",
            ),
            "candidate_config_file_sha256": _required_sha256(
                plan.rationale_candidate_config_file_sha256,
                "rationale candidate config file sha256",
            ),
            "corpus_records_sha256": _required_sha256(
                plan.rationale_corpus_records_sha256,
                "rationale corpus records sha256",
            ),
            "corpus_manifest_sha256": _required_sha256(
                plan.rationale_corpus_manifest_sha256,
                "rationale corpus manifest sha256",
            ),
            "corpus_audit_sha256": _required_sha256(
                plan.rationale_corpus_audit_sha256,
                "rationale corpus audit sha256",
            ),
        }
    manifest_path = root / _MANIFEST_FILENAME
    _write_fsynced_file(
        manifest_path,
        (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
            "utf-8"
        ),
    )
    checksum_entries = [
        *inventory,
        {
            "path": _MANIFEST_FILENAME,
            "size_bytes": manifest_path.stat().st_size,
            "sha256": sha256_file(manifest_path),
        },
    ]
    checksum_entries.sort(key=lambda item: str(item["path"]))
    checksum_text = "".join(
        f"{entry['sha256']}  {entry['path']}\n" for entry in checksum_entries
    )
    _write_fsynced_file(root / _CHECKSUM_FILENAME, checksum_text.encode("utf-8"))
    _fsync_directory(root)


def _regular_artifact_files(root: Path) -> tuple[Path, ...]:
    output: list[Path] = []
    for candidate in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if candidate.is_symlink():
            raise GateBValidationError(
                f"adapter bundle refuses symbolic links: {candidate.relative_to(root)}"
            )
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise GateBValidationError(
                f"adapter bundle contains a non-regular entry: {candidate.relative_to(root)}"
            )
        relative = candidate.relative_to(root).as_posix()
        _validated_relative_artifact_path(relative)
        if candidate.stat().st_size <= 0:
            raise GateBValidationError(f"adapter artifact file is empty: {relative}")
        output.append(candidate)
    return tuple(output)


def _file_inventory(root: Path, *, excluded: set[str]) -> list[dict[str, str | int]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in _regular_artifact_files(root)
        if path.relative_to(root).as_posix() not in excluded
    ]


def _validate_adapter_weight_files(root: Path, names: set[str]) -> None:
    direct = "adapter_model.safetensors" in names
    index_name = "adapter_model.safetensors.index.json"
    indexed = index_name in names
    if direct == indexed:
        raise GateBValidationError(
            "adapter bundle must contain exactly one direct safetensors file or one shard index"
        )
    shard_names = {
        name
        for name in names
        if name.startswith("adapter_model-") and name.endswith(".safetensors")
    }
    if direct:
        if shard_names:
            raise GateBValidationError("unindexed adapter shards are forbidden")
        weight_map: Mapping[str, object] | None = None
        tensor_files = ("adapter_model.safetensors",)
    else:
        _, index, _ = _load_json_artifact(root / index_name, "adapter shard index")
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, Mapping) or not weight_map:
            raise GateBValidationError("adapter shard index weight_map must be non-empty")
        referenced: set[str] = set()
        for tensor_name, shard_name in weight_map.items():
            if not isinstance(tensor_name, str) or not tensor_name:
                raise GateBValidationError("adapter shard index has an invalid tensor name")
            if not isinstance(shard_name, str):
                raise GateBValidationError("adapter shard index has a non-string shard name")
            _validated_relative_artifact_path(shard_name)
            if "/" in shard_name or not (
                shard_name.startswith("adapter_model-")
                and shard_name.endswith(".safetensors")
            ):
                raise GateBValidationError(
                    "adapter shard index references an invalid shard path"
                )
            referenced.add(shard_name)
        if referenced != shard_names:
            raise GateBValidationError(
                "adapter shard files are missing, extra, or inconsistent with the index"
            )
        tensor_files = tuple(sorted(shard_names))

    expected_shapes = _expected_lora_tensor_shapes()
    if weight_map is not None and set(weight_map) != set(expected_shapes):
        raise GateBValidationError("adapter shard index tensor inventory is not exact Qwen LoRA")
    try:
        safe_open = importlib.import_module("safetensors").safe_open
    except (ImportError, AttributeError) as exc:
        raise GateBValidationError(
            "safetensors is required to verify adapter tensor structure"
        ) from exc
    actual_keys: set[str] = set()
    for filename in tensor_files:
        try:
            with safe_open(str(root / filename), framework="np") as handle:
                keys = tuple(handle.keys())
                for key in keys:
                    if key in actual_keys:
                        raise GateBValidationError(
                            f"adapter tensor {key!r} is duplicated across shards"
                        )
                    actual_keys.add(key)
                    if key not in expected_shapes:
                        raise GateBValidationError(
                            f"adapter contains unexpected tensor {key!r}"
                        )
                    if weight_map is not None and weight_map.get(key) != filename:
                        raise GateBValidationError(
                            f"adapter shard index maps tensor {key!r} to another shard"
                        )
                    tensor = handle.get_slice(key)
                    if tuple(tensor.get_shape()) != expected_shapes[key]:
                        raise GateBValidationError(
                            f"adapter tensor {key!r} has an unexpected shape"
                        )
                    if tensor.get_dtype() not in _ALLOWED_LORA_DTYPES:
                        raise GateBValidationError(
                            f"adapter tensor {key!r} has an unsafe dtype"
                        )
        except GateBValidationError:
            raise
        except Exception as exc:
            raise GateBValidationError(
                f"adapter safetensors file is invalid: {filename}: {exc}"
            ) from exc
    if actual_keys != set(expected_shapes):
        raise GateBValidationError("adapter tensor inventory is incomplete")


def _expected_lora_tensor_shapes() -> dict[str, tuple[int, int]]:
    rank = DEFAULT_GATE_B_CONFIG.lora_rank
    output: dict[str, tuple[int, int]] = {}
    for layer in range(_PINNED_QWEN_LAYER_COUNT):
        for module, (output_size, input_size) in _PINNED_QWEN_PROJECTION_DIMS.items():
            block = "self_attn" if module in {"q_proj", "k_proj", "v_proj", "o_proj"} else "mlp"
            prefix = f"base_model.model.model.layers.{layer}.{block}.{module}"
            output[f"{prefix}.lora_A.weight"] = (rank, input_size)
            output[f"{prefix}.lora_B.weight"] = (output_size, rank)
    return output


def _validate_saved_tokenizer(root: Path) -> None:
    tokenizer_path = root / "tokenizer.json"
    if sha256_file(tokenizer_path) != _PINNED_TOKENIZER_JSON_SHA256:
        raise GateBValidationError("saved tokenizer.json differs from the pinned snapshot")
    tokenizer_config_path = root / "tokenizer_config.json"
    if sha256_file(tokenizer_config_path) != _PINNED_TOKENIZER_CONFIG_JSON_SHA256:
        raise GateBValidationError(
            "saved tokenizer_config.json differs from the pinned snapshot"
        )
    _, config, _ = _load_json_artifact(
        tokenizer_config_path, "saved tokenizer config"
    )
    chat_template = config.get("chat_template")
    if (
        not isinstance(chat_template, str)
        or hashlib.sha256(chat_template.encode("utf-8")).hexdigest()
        != _PINNED_TOKENIZER_CHAT_TEMPLATE_SHA256
        or config.get("eos_token") != "<|im_end|>"
        or config.get("pad_token") != "<|endoftext|>"
        or config.get("model_max_length") != 131_072
    ):
        raise GateBValidationError(
            "saved tokenizer config differs from the pinned Qwen tokenizer contract"
        )


def _save_pinned_tokenizer_snapshot(
    root: Path,
    *,
    cached_file: Callable[..., str | None],
) -> None:
    """Copy exact pinned tokenizer bytes instead of reserializing them.

    Recent Transformers/tokenizers releases may add default fields to
    ``tokenizer.json`` and externalize ``chat_template`` while running
    ``save_pretrained``.  That output can tokenize equivalently but is no longer
    the fixed revision's byte-identical tokenizer contract.
    """

    expected = {
        "tokenizer.json": _PINNED_TOKENIZER_JSON_SHA256,
        "tokenizer_config.json": _PINNED_TOKENIZER_CONFIG_JSON_SHA256,
    }
    payloads: dict[str, bytes] = {}
    for filename, expected_sha256 in expected.items():
        try:
            cached = cached_file(
                OFFICIAL_MODEL_ID,
                filename,
                revision=PINNED_MODEL_REVISION,
                local_files_only=True,
            )
        except Exception as exc:
            raise GateBValidationError(
                f"pinned tokenizer cache lookup failed for {filename}: {exc}"
            ) from exc
        if not isinstance(cached, str) or not cached:
            raise GateBValidationError(
                f"pinned tokenizer cache is missing {filename}"
            )
        source = Path(cached).resolve(strict=True)
        if not source.is_file():
            raise GateBValidationError(
                f"pinned tokenizer cache entry is not a regular file: {filename}"
            )
        payload = source.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise GateBValidationError(
                f"pinned tokenizer cache checksum mismatch: {filename}"
            )
        payloads[filename] = payload

    for filename, payload in payloads.items():
        _write_fsynced_file(root / filename, payload)


def _validate_adapter_config(root: Path) -> Mapping[str, object]:
    config_path = root / "adapter_config.json"
    if not config_path.is_file() or config_path.is_symlink():
        raise GateBValidationError("adapter_config.json is missing or not a regular file")
    _, payload, _ = _load_json_artifact(config_path, "PEFT adapter config")
    expected = {
        "base_model_name_or_path": OFFICIAL_MODEL_ID,
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": DEFAULT_GATE_B_CONFIG.lora_rank,
        "lora_alpha": DEFAULT_GATE_B_CONFIG.lora_alpha,
        "lora_dropout": DEFAULT_GATE_B_CONFIG.lora_dropout,
        "bias": DEFAULT_GATE_B_CONFIG.lora_bias,
        "modules_to_save": None,
        "use_dora": False,
        "use_rslora": False,
        "trainable_token_indices": None,
        "target_parameters": None,
        "rank_pattern": {},
        "alpha_pattern": {},
        "inference_mode": True,
        "fan_in_fan_out": False,
    }
    mismatched = [key for key, value in expected.items() if payload.get(key) != value]
    target_modules = payload.get("target_modules")
    if (
        not isinstance(target_modules, list)
        or any(not isinstance(item, str) for item in target_modules)
        or len(target_modules) != len(set(target_modules))
        or set(target_modules) != set(PINNED_QWEN_ALL_LINEAR_TARGET_MODULES)
    ):
        mismatched.append("target_modules")
    if mismatched:
        raise GateBValidationError(
            f"PEFT adapter config does not match the fixed QLoRA contract: {mismatched!r}"
        )
    return payload


def _canonicalize_adapter_config(root: Path) -> None:
    """Normalize PEFT's set-derived target list before content hashing."""

    config_path = root / "adapter_config.json"
    payload = dict(_validate_adapter_config(root))
    payload["target_modules"] = list(PINNED_QWEN_ALL_LINEAR_TARGET_MODULES)
    serialized = (
        json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
        + "\n"
    ).encode("utf-8")
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=".adapter_config.json.", suffix=".tmp", dir=root
    )
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, config_path)
        _fsync_directory(root)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


def _validated_relative_artifact_path(value: str) -> None:
    pure = PurePosixPath(value)
    if (
        not value
        or pure.is_absolute()
        or "\\" in value
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise GateBValidationError(f"unsafe adapter artifact path: {value!r}")


def _parse_checksum_file(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    except (OSError, UnicodeError) as exc:
        raise GateBValidationError(f"cannot read checksum file: {exc}") from exc
    output: dict[str, str] = {}
    for line_number, line in enumerate(lines, start=1):
        if "  " not in line:
            raise GateBValidationError(f"checksum line {line_number} has invalid format")
        digest, relative = line.split("  ", 1)
        _required_sha256(digest, f"checksum line {line_number}")
        _validated_relative_artifact_path(relative)
        if relative in output:
            raise GateBValidationError(f"checksum file repeats {relative!r}")
        output[relative] = digest
    return output


def _publish_directory_noreplace(source: Path, target: Path) -> None:
    """Atomically rename a complete directory while refusing replacement."""

    if (
        source.parent != target.parent
        and source.parent.parent != target.parent
        and source.stat().st_dev != target.parent.stat().st_dev
    ):
        # Current orchestrator uses build_root/publish -> target; both live on the
        # same filesystem even though their immediate parents differ.
        raise GateBValidationError("atomic publication requires one filesystem")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is not None:
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            os.fsencode(source),
            -100,
            os.fsencode(target),
            1,  # RENAME_NOREPLACE
        )
        if result == 0:
            _fsync_directory(target.parent)
            return
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise GateBArtifactExistsError(f"adapter destination already exists: {target}")
        if error not in {errno.ENOSYS, errno.EINVAL, errno.ENOTSUP}:
            raise GateBValidationError(f"cannot atomically publish adapter: errno={error}")
    # Portable fallback. A same-tool exclusive reservation protects concurrent
    # writers; supported Linux/WSL hosts take the renameat2 path above.
    lock_path = target.parent / f".{target.name}.publish.lock"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except FileExistsError as exc:
        raise GateBArtifactExistsError(f"adapter publication lock exists: {lock_path}") from exc
    try:
        if target.exists() or target.is_symlink():
            raise GateBArtifactExistsError(f"adapter destination already exists: {target}")
        os.rename(source, target)
        _fsync_directory(target.parent)
    finally:
        os.close(descriptor)
        with suppress(FileNotFoundError):
            lock_path.unlink()


def _write_fsynced_file(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise GateBArtifactExistsError(f"refusing to overwrite artifact file: {path}") from exc
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _offline_environment() -> Iterable[None]:
    previous = {name: os.environ.get(name) for name in _OFFLINE_ENVIRONMENT}
    os.environ.update(_OFFLINE_ENVIRONMENT)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _lazy_gpu_modules() -> Mapping[str, Any]:  # pragma: no cover - real GPU environment
    try:
        return {
            "torch": importlib.import_module("torch"),
            "transformers": importlib.import_module("transformers"),
            "peft": importlib.import_module("peft"),
            "bitsandbytes": importlib.import_module("bitsandbytes"),
        }
    except (ImportError, OSError) as exc:
        raise GateBPreflightRequiredError(
            f"GPU runtime imports failed after preflight: {type(exc).__name__}: {exc}"
        ) from exc


def _require_smoke_bound_cuda_device(torch: Any, evidence: RuntimeGateEvidence) -> None:
    if not bool(torch.cuda.is_available()) or not bool(torch.cuda.is_bf16_supported()):
        raise GateBPreflightRequiredError("CUDA BF16 GPU disappeared after final smoke")
    if int(torch.cuda.device_count()) != 1:
        raise GateBPreflightRequiredError(
            "runtime must expose exactly the one GPU validated by final smoke"
        )
    torch.cuda.set_device(0)
    current_name = str(torch.cuda.get_device_name(0))
    if current_name != evidence.device_name:
        raise GateBPreflightRequiredError(
            "runtime CUDA device identity differs from the final GPU smoke"
        )


def _load_pinned_nf4_base(
    modules: Mapping[str, Any],
    config: GateBConfig,
    *,
    tokenizer: Any = None,
) -> tuple[Any, Any]:  # pragma: no cover - loads the pinned 3B model on CUDA
    torch = modules["torch"]
    transformers = modules["transformers"]
    if tokenizer is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            OFFICIAL_MODEL_ID,
            revision=PINNED_MODEL_REVISION,
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
        )
        _ensure_padding_token(tokenizer)
    quantization = transformers.BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_quant_type=config.quantization,
        bnb_4bit_use_double_quant=config.double_quantization,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        OFFICIAL_MODEL_ID,
        revision=PINNED_MODEL_REVISION,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = config.use_cache
    return tokenizer, model


def _ensure_padding_token(tokenizer: Any) -> None:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise GateBValidationError("tokenizer has neither pad nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


def _training_arguments(
    training_arguments: Any,
    *,
    output_dir: Path,
    config: GateBConfig,
    retain_checkpoints: bool = False,
) -> Any:
    if not isinstance(retain_checkpoints, bool):
        raise TypeError("retain_checkpoints must be a bool")
    parameters = __import__("inspect").signature(training_arguments.__init__).parameters
    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "overwrite_output_dir": False,
        "per_device_train_batch_size": config.micro_batch_size,
        "per_device_eval_batch_size": config.micro_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "optim": config.optimizer,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "lr_scheduler_type": config.lr_scheduler_type,
        "warmup_ratio": config.warmup_ratio,
        "num_train_epochs": config.num_train_epochs,
        "max_grad_norm": config.max_grad_norm,
        "eval_steps": config.eval_steps,
        "save_steps": config.save_steps,
        # The default locked profile keeps two temporary checkpoints.  A
        # persistent resume workspace instead retains every checkpoint so an
        # interruption never deletes the last reusable recovery point.
        "save_total_limit": None if retain_checkpoints else config.save_total_limit,
        "logging_steps": config.logging_steps,
        "bf16": config.bf16_training,
        "fp16": False,
        "gradient_checkpointing": config.gradient_checkpointing,
        "gradient_checkpointing_kwargs": {
            "use_reentrant": config.gradient_checkpointing_use_reentrant
        },
        "seed": config.seed,
        "data_seed": config.seed,
        "report_to": [],
        "remove_unused_columns": False,
        "dataloader_num_workers": 0,
        "save_safetensors": True,
    }
    if "eval_strategy" in parameters:
        kwargs["eval_strategy"] = "steps"
    elif "evaluation_strategy" in parameters:
        kwargs["evaluation_strategy"] = "steps"
    else:  # pragma: no cover - unsupported future API guard
        raise GateBPreflightRequiredError("Transformers TrainingArguments lacks eval strategy")
    if "save_strategy" in parameters:
        kwargs["save_strategy"] = "steps"
    return training_arguments(**kwargs)


def _package_versions(names: Sequence[str]) -> dict[str, str]:
    output: dict[str, str] = {}
    for name in names:
        try:
            output[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise GateBPreflightRequiredError(f"runtime package disappeared: {name}") from exc
    return output


def _coerce_flat_metrics(
    values: Mapping[str, object],
) -> dict[str, str | int | float | bool | None]:
    output: dict[str, str | int | float | bool | None] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            raise GateBValidationError("training metric keys must be non-empty strings")
        if (
            isinstance(value, (str, bool, int))
            or value is None
            or isinstance(value, float)
            and math.isfinite(value)
        ):
            output[key] = value
        else:
            raise GateBValidationError(f"training metric {key!r} is not finite JSON scalar")
    return output


def _validated_flat_json_mapping(values: Mapping[str, object], field_name: str) -> None:
    if not isinstance(values, Mapping):
        raise GateBValidationError(f"{field_name} must be a mapping")
    _coerce_flat_metrics(values)


def _validated_string_mapping(values: Mapping[str, str], field_name: str) -> None:
    if not isinstance(values, Mapping) or any(
        not isinstance(key, str)
        or not key
        or not isinstance(value, str)
        or not value
        for key, value in values.items()
    ):
        raise GateBValidationError(f"{field_name} must map non-empty strings to strings")


__all__ = [
    "AdapterArtifactEvidence",
    "BASE_MODEL_CHECKPOINT_SHA256",
    "FoldSFTPlan",
    "FoldTrainingRuntime",
    "GPU_EXECUTION_ACKNOWLEDGEMENT",
    "PINNED_QWEN_ALL_LINEAR_TARGET_MODULES",
    "PreflightEvidence",
    "RuntimeGateEvidence",
    "RuntimeTrainingResult",
    "TrainingArtifact",
    "TransformersNF4GenerationBackend",
    "TransformersQLoRATrainingRuntime",
    "build_fold_sft_plan",
    "create_adapted_development_backend",
    "create_base_development_backend",
    "train_qlora_fold",
    "validate_adapter_artifact",
    "validate_adapter_for_fold",
    "validate_preflight_artifact",
    "validate_runtime_gate",
]

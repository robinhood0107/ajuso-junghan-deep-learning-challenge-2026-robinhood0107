"""CPU-only safety and provenance primitives for the gated model phase.

This module intentionally imports neither PyTorch nor Transformers and never
loads model weights.  Its public training and development entry points derive
their own eligible IDs from a validated :class:`~deep_challenge.splits.SplitManifest`;
callers cannot substitute an arbitrary ``eligible_ids`` list that accidentally
contains a validation fold or the locked final holdout.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from .answers import AnswerParseResult, parse_answer
from .data import MathRecord
from .inference import deterministic_seed
from .model_preflight import OFFICIAL_MODEL_ID, OFFICIAL_REVISION
from .provenance import (
    SourceTreeArtifactEvidence,
    canonical_json_bytes,
    sha256_file,
)
from .rationale_corpus import (
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
from .tokenizer_profile import DEFAULT_SYSTEM_PROMPT

PINNED_MODEL_REVISION = OFFICIAL_REVISION
_TRAIN_ID_RE = re.compile(r"train-\d{6}\Z")
_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}\Z")
_DEVELOPMENT_RESUME_CONTRACT_SCHEMA = "gate-b1-development-resume-contract-v1"
_DEVELOPMENT_RESUME_CHUNK_SCHEMA = "gate-b1-development-resume-chunk-v1"
_DEVELOPMENT_RESUME_PROGRESS_SCHEMA = "gate-b1-development-resume-progress-v1"
_DEVELOPMENT_RESUME_LOCK_SCHEMA = "gate-b1-development-resume-lock-v1"
_DEVELOPMENT_ARTIFACT_PAIR_LOCK_SCHEMA = "gate-b1-development-artifact-pair-lock-v1"
_DEVELOPMENT_RESUME_DEFAULT_CHUNK_SIZE = 25
_DEVELOPMENT_RESUME_CHUNK_RE = re.compile(
    r"chunk-(?P<chunk_index>\d{6})-attempt-(?P<attempt>\d{6})\.manifest\.json\Z"
)
_DEVELOPMENT_RESUME_CHUNK_RECORD_RE = re.compile(
    r"chunk-(?P<chunk_index>\d{6})-attempt-(?P<attempt>\d{6})\.jsonl\Z"
)


class GateBValidationError(ValueError):
    """Raised when a Gate B artifact crosses a declared safety boundary."""


class GateBArtifactExistsError(FileExistsError):
    """Raised when an atomic writer would overwrite or race another run."""


class GateBPreflightRequiredError(RuntimeError):
    """Raised when a real model adapter is requested before runtime preflight."""


@dataclass(frozen=True, slots=True)
class DecodingPolicy:
    """Explicit greedy decoding policy recorded with every generation."""

    do_sample: bool
    num_beams: int
    max_new_tokens: int
    temperature: float | None
    top_p: float | None
    top_k: int | None
    repetition_penalty: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.as_dict())).hexdigest()


@dataclass(frozen=True, slots=True)
class GateBConfig:
    """Locked conservative QLoRA/direct-answer profile for a 12 GB 4070 SUPER.

    Every training and decoding choice below participates in ``sha256``.  A
    materially different experiment should introduce a separately versioned
    profile instead of silently drifting this baseline.
    """

    profile_version: str = "gate-b-direct-rtx4070-super-12gb-v1"
    hardware_profile: str = "nvidia-rtx-4070-super-12gb"
    model_id: str = OFFICIAL_MODEL_ID
    revision: str = PINNED_MODEL_REVISION
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    route: str = "direct_answer"
    max_sequence_length: int = 2_048
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    quantization: str = "nf4"
    load_in_4bit: bool = True
    double_quantization: bool = True
    compute_dtype: str = "bfloat16"
    bf16_training: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = "all-linear"
    lora_bias: str = "none"
    response_only_loss: bool = True
    packing: bool = False
    optimizer: str = "paged_adamw_8bit"
    learning_rate: float = 0.0001
    weight_decay: float = 0.0
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    num_train_epochs: float = 1.0
    max_grad_norm: float = 1.0
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 2
    logging_steps: int = 10
    gradient_checkpointing: bool = True
    gradient_checkpointing_use_reentrant: bool = False
    # This is the training-mode setting.  Inference explicitly enables the
    # KV cache from the pinned runtime source after model.eval().
    use_cache: bool = False
    generation_do_sample: bool = False
    generation_num_beams: int = 1
    max_new_tokens: int = 512
    generation_temperature: float | None = None
    generation_top_p: float | None = None
    generation_top_k: int | None = None
    generation_repetition_penalty: float = 1.0
    seed: int = 20_260_731

    def __post_init__(self) -> None:
        fixed_values: tuple[tuple[str, object, object], ...] = (
            ("profile_version", self.profile_version, "gate-b-direct-rtx4070-super-12gb-v1"),
            ("hardware_profile", self.hardware_profile, "nvidia-rtx-4070-super-12gb"),
            ("model_id", self.model_id, OFFICIAL_MODEL_ID),
            ("revision", self.revision, PINNED_MODEL_REVISION),
            ("route", self.route, "direct_answer"),
            ("max_sequence_length", self.max_sequence_length, 2_048),
            ("micro_batch_size", self.micro_batch_size, 1),
            ("gradient_accumulation_steps", self.gradient_accumulation_steps, 16),
            ("quantization", self.quantization, "nf4"),
            ("load_in_4bit", self.load_in_4bit, True),
            ("double_quantization", self.double_quantization, True),
            ("compute_dtype", self.compute_dtype, "bfloat16"),
            ("bf16_training", self.bf16_training, True),
            ("lora_rank", self.lora_rank, 16),
            ("lora_alpha", self.lora_alpha, 32),
            ("lora_dropout", self.lora_dropout, 0.05),
            ("lora_target_modules", self.lora_target_modules, "all-linear"),
            ("lora_bias", self.lora_bias, "none"),
            ("response_only_loss", self.response_only_loss, True),
            ("packing", self.packing, False),
            ("optimizer", self.optimizer, "paged_adamw_8bit"),
            ("learning_rate", self.learning_rate, 0.0001),
            ("weight_decay", self.weight_decay, 0.0),
            ("lr_scheduler_type", self.lr_scheduler_type, "cosine"),
            ("warmup_ratio", self.warmup_ratio, 0.03),
            ("num_train_epochs", self.num_train_epochs, 1.0),
            ("max_grad_norm", self.max_grad_norm, 1.0),
            ("eval_steps", self.eval_steps, 100),
            ("save_steps", self.save_steps, 100),
            ("save_total_limit", self.save_total_limit, 2),
            ("logging_steps", self.logging_steps, 10),
            ("gradient_checkpointing", self.gradient_checkpointing, True),
            (
                "gradient_checkpointing_use_reentrant",
                self.gradient_checkpointing_use_reentrant,
                False,
            ),
            ("use_cache", self.use_cache, False),
            ("generation_do_sample", self.generation_do_sample, False),
            ("generation_num_beams", self.generation_num_beams, 1),
            ("max_new_tokens", self.max_new_tokens, 512),
            ("generation_temperature", self.generation_temperature, None),
            ("generation_top_p", self.generation_top_p, None),
            ("generation_top_k", self.generation_top_k, None),
            (
                "generation_repetition_penalty",
                self.generation_repetition_penalty,
                1.0,
            ),
        )
        for field_name, value, expected in fixed_values:
            if value != expected or type(value) is not type(expected):
                raise GateBValidationError(
                    f"{field_name} is locked to {expected!r} for this Gate B profile"
                )
        if not isinstance(self.system_prompt, str) or not self.system_prompt.strip():
            raise GateBValidationError("system_prompt must be a non-empty string")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise GateBValidationError("seed must be a non-negative integer")
        if self.eval_steps != self.save_steps:
            raise GateBValidationError("eval_steps and save_steps must remain aligned")
        if self.use_cache and self.gradient_checkpointing:  # pragma: no cover - locked guard
            raise GateBValidationError("use_cache conflicts with gradient checkpointing")
        if self.generation_do_sample and (
            self.generation_temperature is None or self.generation_top_p is None
        ):  # pragma: no cover - locked guard
            raise GateBValidationError("sampling requires temperature and top_p")

    def as_dict(self) -> dict[str, object]:
        """Return the complete canonical-hash input."""

        return asdict(self)

    @property
    def sha256(self) -> str:
        """Return a stable SHA-256 over every run-defining config field."""

        return hashlib.sha256(canonical_json_bytes(self.as_dict())).hexdigest()

    @property
    def decoding_policy(self) -> DecodingPolicy:
        """Return the exact inference policy represented by this profile."""

        return DecodingPolicy(
            do_sample=self.generation_do_sample,
            num_beams=self.generation_num_beams,
            max_new_tokens=self.max_new_tokens,
            temperature=self.generation_temperature,
            top_p=self.generation_top_p,
            top_k=self.generation_top_k,
            repetition_penalty=self.generation_repetition_penalty,
        )


DEFAULT_GATE_B_CONFIG = GateBConfig()


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One immutable chat message."""

    role: str
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True, slots=True)
class DirectAnswerSFTExample:
    """One organizer-only training target with split-bound provenance."""

    problem_id: str
    question_sha256: str
    prompt_messages: tuple[ChatMessage, ...]
    target_text: str
    split_version: str
    split_sha256: str
    source_groups_sha256: str
    fold: int
    partition: str
    split_partition: str
    group_id: str
    eligibility_ids_sha256: str

    @property
    def full_messages(self) -> tuple[ChatMessage, ...]:
        return (*self.prompt_messages, ChatMessage("assistant", self.target_text))


@dataclass(frozen=True, slots=True)
class EncodedSFTExample:
    """Pure-Python response-only encoding suitable for a later tensor collator."""

    problem_id: str
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    labels: tuple[int, ...]
    prompt_token_count: int
    sequence_token_count: int
    config_sha256: str
    split_version: str
    split_sha256: str
    source_groups_sha256: str
    fold: int
    partition: str
    split_partition: str
    group_id: str
    eligibility_ids_sha256: str


class ChatTokenizer(Protocol):
    """Minimum tokenizer surface needed for response-only SFT encoding."""

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> Sequence[int]: ...


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """One deterministic development-generation request."""

    problem_id: str
    messages: tuple[ChatMessage, ...]
    seed: int
    sample_index: int
    model_id: str
    revision: str
    route: str
    prompt_sha256: str
    config_sha256: str
    decoding_policy: DecodingPolicy


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Structured result returned by a preflight-gated generation backend."""

    text: str
    finish_reason: str
    input_token_count: int
    output_token_count: int
    peak_vram_allocated_bytes: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        if (
            not isinstance(self.finish_reason, str)
            or not self.finish_reason
            or self.finish_reason != self.finish_reason.strip()
        ):
            raise GateBValidationError("finish_reason must be a non-empty trimmed string")
        for field_name in ("input_token_count", "output_token_count"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise GateBValidationError(f"{field_name} must be a non-negative integer")
        peak = self.peak_vram_allocated_bytes
        if peak is not None and (
            isinstance(peak, bool) or not isinstance(peak, int) or peak < 0
        ):
            raise GateBValidationError(
                "peak_vram_allocated_bytes must be a non-negative integer or None"
            )


class GenerationBackend(Protocol):
    """Runtime boundary implemented by a future, preflight-gated model adapter."""

    def generate(self, request: GenerationRequest) -> GenerationResult: ...


@dataclass(frozen=True, slots=True)
class DevelopmentBaselineRecord:
    """JSONL-ready raw generation, parser outcome, and complete provenance."""

    problem_id: str
    question_sha256: str
    route: str
    sample_index: int
    seed: int
    model_id: str
    revision: str
    prompt_sha256: str
    checkpoint_sha256: str
    config_sha256: str
    decoding_policy: DecodingPolicy
    raw_completion: str
    raw_completion_sha256: str
    finish_reason: str
    input_token_count: int
    output_token_count: int
    parse: AnswerParseResult
    reference_answer: int
    exact_match: bool
    latency_ms: float
    peak_vram_allocated_bytes: int | None
    split_version: str
    split_sha256: str
    source_groups_sha256: str
    fold: int
    partition: str
    split_partition: str
    group_id: str
    eligibility_ids_sha256: str

    def as_dict(self) -> dict[str, object]:
        """Return a strict JSON-compatible representation."""

        return {
            "schema_version": "gate-b1-development-baseline-v2",
            "problem_id": self.problem_id,
            "question_sha256": self.question_sha256,
            "route": self.route,
            "sample_index": self.sample_index,
            "seed": self.seed,
            "model_id": self.model_id,
            "revision": self.revision,
            "prompt_sha256": self.prompt_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "config_sha256": self.config_sha256,
            "decoding_policy": self.decoding_policy.as_dict(),
            "decoding_policy_sha256": self.decoding_policy.sha256,
            "raw_completion": self.raw_completion,
            "raw_completion_sha256": self.raw_completion_sha256,
            "finish_reason": self.finish_reason,
            "input_token_count": self.input_token_count,
            "output_token_count": self.output_token_count,
            "parse": asdict(self.parse),
            "reference_answer": self.reference_answer,
            "exact_match": self.exact_match,
            "latency_ms": self.latency_ms,
            "peak_vram_allocated_bytes": self.peak_vram_allocated_bytes,
            "split_version": self.split_version,
            "split_sha256": self.split_sha256,
            "source_groups_sha256": self.source_groups_sha256,
            "fold": self.fold,
            "partition": self.partition,
            "split_partition": self.split_partition,
            "group_id": self.group_id,
            "eligibility_ids_sha256": self.eligibility_ids_sha256,
        }

    def to_json_line(self) -> str:
        """Serialize one deterministic, finite JSONL row."""

        return json.dumps(
            self.as_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )


@dataclass(frozen=True, slots=True)
class DevelopmentArtifactWriteResult:
    """Paths and digests produced by the no-overwrite artifact writer."""

    records_path: str
    manifest_path: str
    record_count: int
    records_sha256: str
    manifest_sha256: str


@dataclass(frozen=True, slots=True)
class DevelopmentResumeStatus:
    """Raw-free progress for one resumable development generation run.

    The on-disk progress file deliberately contains no organizer question,
    answer, model completion, or problem identifier.  The private resume
    contract and chunk ledger contain the identifiers needed to resume safely;
    callers that only need to monitor a long generation can use this object
    without loading those records.
    """

    contract_sha256: str
    state: str
    process_id: int | None
    total_chunks: int
    completed_chunks: int
    total_generations: int
    completed_generations: int
    chunk_attempt_count: int
    invalid_chunk_attempt_count: int
    completed_latency_ms: float

    def as_dict(self) -> dict[str, object]:
        """Return the intentionally data-free persisted status payload."""

        return {
            "schema_version": _DEVELOPMENT_RESUME_PROGRESS_SCHEMA,
            "contract_sha256": self.contract_sha256,
            "state": self.state,
            "process_id": self.process_id,
            "total_chunks": self.total_chunks,
            "completed_chunks": self.completed_chunks,
            "total_generations": self.total_generations,
            "completed_generations": self.completed_generations,
            "chunk_attempt_count": self.chunk_attempt_count,
            "invalid_chunk_attempt_count": self.invalid_chunk_attempt_count,
            "completed_latency_ms": self.completed_latency_ms,
        }


@dataclass(frozen=True, slots=True)
class DevelopmentExecutionEvidence:
    """Immutable runtime/source binding stored beside one private development run.

    The raw generation JSONL remains private, but its manifest must make the
    exact B0 artifacts, source snapshot, configuration bytes, and observed GPU
    identity auditable before the run can authorize QLoRA training.
    """

    source_manifest: SourceTreeArtifactEvidence
    config_path: str
    config_file_sha256: str
    config_sha256: str
    preflight_report_path: str
    preflight_report_sha256: str
    gpu_smoke_report_path: str
    gpu_smoke_report_sha256: str
    gpu_device_name: str

    def as_dict(self) -> dict[str, object]:
        """Return the serialized private-manifest representation."""

        return {
            "schema_version": "gate-b1-execution-evidence-v1",
            "source_manifest": {
                "path": self.source_manifest.path,
                **self.source_manifest.as_dict(),
            },
            "config_file": {
                "path": self.config_path,
                "sha256": self.config_file_sha256,
                "config_sha256": self.config_sha256,
            },
            "runtime_gate": {
                "preflight_report": {
                    "path": self.preflight_report_path,
                    "sha256": self.preflight_report_sha256,
                },
                "gpu_smoke_report": {
                    "path": self.gpu_smoke_report_path,
                    "sha256": self.gpu_smoke_report_sha256,
                },
                "gpu_device_name": self.gpu_device_name,
            },
        }


def create_development_execution_evidence(
    *,
    source_manifest: SourceTreeArtifactEvidence,
    config_path: str | Path,
    config_sha256: str,
    preflight_report_path: str | Path,
    preflight_report_sha256: str,
    gpu_smoke_report_path: str | Path,
    gpu_smoke_report_sha256: str,
    gpu_device_name: str,
) -> DevelopmentExecutionEvidence:
    """Re-hash all runtime inputs that the already-created backend used.

    ``source_manifest`` must have been validated against the current source
    tree before backend creation.  The remaining B0/config files are re-hashed
    here and must agree with the runtime gate, so a file replacement between
    preflight validation and artifact publication fails closed.
    """

    if not isinstance(source_manifest, SourceTreeArtifactEvidence):
        raise TypeError("source_manifest must be SourceTreeArtifactEvidence")
    source_path, source_digest = _regular_file_identity(
        source_manifest.path, "source manifest"
    )
    if source_digest != source_manifest.sha256:
        raise GateBValidationError("source manifest bytes changed after validation")
    _validated_sha256(source_manifest.tree_sha256, "source manifest tree_sha256")
    if (
        isinstance(source_manifest.file_count, bool)
        or not isinstance(source_manifest.file_count, int)
        or source_manifest.file_count < 1
    ):
        raise GateBValidationError("source manifest file_count must be positive")
    config_file, config_digest = _regular_file_identity(config_path, "Gate B config")
    expected_config = _validated_sha256(config_sha256, "config_sha256")
    _validate_config_file_semantic_sha(config_file, expected_config)
    preflight_file, preflight_digest = _regular_file_identity(
        preflight_report_path, "model preflight report"
    )
    expected_preflight = _validated_sha256(
        preflight_report_sha256, "preflight_report_sha256"
    )
    if preflight_digest != expected_preflight:
        raise GateBValidationError("model preflight report bytes changed after runtime gate")
    smoke_file, smoke_digest = _regular_file_identity(
        gpu_smoke_report_path, "GPU smoke report"
    )
    expected_smoke = _validated_sha256(gpu_smoke_report_sha256, "gpu_smoke_report_sha256")
    if smoke_digest != expected_smoke:
        raise GateBValidationError("GPU smoke report bytes changed after runtime gate")
    if (
        not isinstance(gpu_device_name, str)
        or not gpu_device_name.strip()
        or gpu_device_name != gpu_device_name.strip()
    ):
        raise GateBValidationError("gpu_device_name must be a non-empty trimmed string")
    return DevelopmentExecutionEvidence(
        source_manifest=SourceTreeArtifactEvidence(
            path=source_path,
            sha256=source_digest,
            tree_sha256=source_manifest.tree_sha256,
            file_count=source_manifest.file_count,
        ),
        config_path=config_file,
        config_file_sha256=config_digest,
        config_sha256=expected_config,
        preflight_report_path=preflight_file,
        preflight_report_sha256=preflight_digest,
        gpu_smoke_report_path=smoke_file,
        gpu_smoke_report_sha256=smoke_digest,
        gpu_device_name=gpu_device_name,
    )


class TransformersGenerationBackend:
    """Explicit placeholder for the real, target-host Transformers adapter."""

    def generate(self, request: GenerationRequest) -> GenerationResult:
        del request
        raise GateBPreflightRequiredError(
            "real Transformers generation is disabled in the CPU preparation layer; "
            "run model-preflight on the target GPU and require training_ready=true "
            "before installing or invoking a separately verified runtime adapter"
        )


def build_direct_answer_sft_examples(
    records: Iterable[MathRecord],
    *,
    split_manifest: SplitManifest,
    fold: int,
    excluded_ids: Iterable[str],
    config: GateBConfig = DEFAULT_GATE_B_CONFIG,
) -> tuple[DirectAnswerSFTExample, ...]:
    """Build direct-answer targets from exactly one eligible fold-training view.

    The eligible ID set is derived internally using ``training_ids(fold)`` and
    hard-group exclusion expansion.  ``records`` must match that derived view
    exactly; passing the full organizer train table, a validation row, or a
    locked-holdout row fails closed.
    """

    eligible_ids = _derive_eligible_ids(split_manifest, fold, excluded_ids, training=True)
    ordered_ids, records_by_id = _validated_partition_records(records, eligible_ids)
    provenance = _partition_provenance(split_manifest, fold, ordered_ids, "fold_training")

    examples: list[DirectAnswerSFTExample] = []
    for problem_id in ordered_ids:
        record = records_by_id[problem_id]
        answer = _required_reference_answer(record)
        target_text = f"Final answer: {answer}"
        parsed = parse_answer(target_text)
        if not parsed.ok or parsed.value != answer:  # pragma: no cover - invariant guard
            raise GateBValidationError(f"internal direct-answer target failed for {problem_id}")
        examples.append(
            DirectAnswerSFTExample(
                problem_id=problem_id,
                question_sha256=_sha256_text(record.question_raw),
                prompt_messages=_prompt_messages(record.question_raw, config),
                target_text=target_text,
                group_id=provenance.group_ids[problem_id],
                **provenance.common,
            )
        )
    return tuple(examples)


def build_concise_rationale_sft_examples(
    records: Iterable[MathRecord],
    *,
    split_manifest: SplitManifest,
    fold: int,
    excluded_ids: Iterable[str],
    rationale_corpus: RationaleCorpusEvidence,
    rationale_config: ConciseRationaleConfig,
    config: GateBConfig = DEFAULT_GATE_B_CONFIG,
) -> tuple[DirectAnswerSFTExample, ...]:
    """Build response-only targets from one audited private rationale corpus.

    The corpus loader has already recomputed every answer and split invariant;
    this boundary rechecks its fold, ID order, question digests, and mandatory
    audit binding before target text can reach the GPU runtime.
    """

    if not isinstance(rationale_corpus, RationaleCorpusEvidence):
        raise TypeError("rationale_corpus must be RationaleCorpusEvidence")
    if not isinstance(rationale_config, ConciseRationaleConfig):
        raise TypeError("rationale_config must be ConciseRationaleConfig")
    if rationale_corpus.audit_path is None or rationale_corpus.audit_sha256 is None:
        raise GateBValidationError(
            "concise-rationale SFT requires a validated redacted corpus audit"
        )
    if rationale_corpus.fold != fold:
        raise GateBValidationError("rationale corpus fold does not match the SFT fold")
    if rationale_corpus.candidate_config_sha256 != rationale_config.sha256:
        raise GateBValidationError(
            "rationale corpus is not bound to the supplied candidate config"
        )
    eligible_ids = _derive_eligible_ids(split_manifest, fold, excluded_ids, training=True)
    ordered_ids, records_by_id = _validated_partition_records(records, eligible_ids)
    if rationale_corpus.training_ids_sha256 != _ids_sha256(ordered_ids):
        raise GateBValidationError(
            "rationale corpus training ID digest does not match the fold partition"
        )
    if rationale_corpus.record_count != len(ordered_ids):
        raise GateBValidationError("rationale corpus record count does not match the fold")
    if tuple(row.problem_id for row in rationale_corpus.rows) != ordered_ids:
        raise GateBValidationError("rationale corpus changed split-derived ID order")
    provenance = _partition_provenance(split_manifest, fold, ordered_ids, "fold_training")
    examples: list[DirectAnswerSFTExample] = []
    for row in rationale_corpus.rows:
        record = records_by_id[row.problem_id]
        answer = _required_reference_answer(record)
        question_sha256 = _sha256_text(record.question_raw)
        if row.question_sha256 != question_sha256:
            raise GateBValidationError(
                f"{row.problem_id}: rationale question SHA changed after corpus audit"
            )
        parsed = parse_answer(row.target_text)
        if not parsed.ok or parsed.value != answer or parsed.source != "final_answer":
            raise GateBValidationError(
                f"{row.problem_id}: rationale target no longer matches the organizer answer"
            )
        examples.append(
            DirectAnswerSFTExample(
                problem_id=row.problem_id,
                question_sha256=question_sha256,
                prompt_messages=_prompt_messages(record.question_raw, config),
                target_text=row.target_text,
                group_id=provenance.group_ids[row.problem_id],
                **provenance.common,
            )
        )
    return tuple(examples)


def encode_response_only_example(
    example: DirectAnswerSFTExample,
    tokenizer: ChatTokenizer,
    *,
    config: GateBConfig = DEFAULT_GATE_B_CONFIG,
) -> EncodedSFTExample:
    """Encode one example without truncation and mask every prompt label."""

    prompt_ids = _token_ids(
        tokenizer.apply_chat_template(
            [message.as_dict() for message in example.prompt_messages],
            tokenize=True,
            add_generation_prompt=True,
        ),
        field_name="prompt input_ids",
    )
    full_ids = _token_ids(
        tokenizer.apply_chat_template(
            [message.as_dict() for message in example.full_messages],
            tokenize=True,
            add_generation_prompt=False,
        ),
        field_name="full input_ids",
    )
    if len(full_ids) > config.max_sequence_length:
        raise GateBValidationError(
            f"{example.problem_id}: encoded sequence has {len(full_ids)} tokens, "
            f"exceeding locked limit {config.max_sequence_length}; truncation is forbidden"
        )
    if len(prompt_ids) >= len(full_ids):
        raise GateBValidationError(
            f"{example.problem_id}: assistant target produced no unmasked tokens"
        )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise GateBValidationError(
            f"{example.problem_id}: chat template prompt is not a prefix of the full sequence"
        )
    labels = (-100,) * len(prompt_ids) + full_ids[len(prompt_ids) :]
    return EncodedSFTExample(
        problem_id=example.problem_id,
        input_ids=full_ids,
        attention_mask=(1,) * len(full_ids),
        labels=labels,
        prompt_token_count=len(prompt_ids),
        sequence_token_count=len(full_ids),
        config_sha256=config.sha256,
        split_version=example.split_version,
        split_sha256=example.split_sha256,
        source_groups_sha256=example.source_groups_sha256,
        fold=example.fold,
        partition=example.partition,
        split_partition=example.split_partition,
        group_id=example.group_id,
        eligibility_ids_sha256=example.eligibility_ids_sha256,
    )


def run_development_baseline(
    records: Iterable[MathRecord],
    *,
    split_manifest: SplitManifest,
    fold: int,
    excluded_ids: Iterable[str],
    backend: GenerationBackend,
    checkpoint_sha256: str,
    config: GateBConfig = DEFAULT_GATE_B_CONFIG,
    samples_per_problem: int = 1,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
    progress_callback: Callable[[int, int], None] | None = None,
    resume_dir: str | Path | None = None,
    chunk_size: int = _DEVELOPMENT_RESUME_DEFAULT_CHUNK_SIZE,
) -> tuple[DevelopmentBaselineRecord, ...]:
    """Generate from exactly one eligible development-validation fold.

    There is no retry, fallback answer, or parser coercion.  Empty, invalid, and
    conflicting generations are retained verbatim.  The full train table and
    locked holdout are rejected because ``records`` must exactly equal the IDs
    derived by :func:`eligible_validation_ids`.  Supplying ``resume_dir`` adds
    a private immutable ledger; ``chunk_size`` counts problems (all samples for
    a problem stay in one chunk) and must match that ledger on every restart.
    """

    eligible_ids = _derive_eligible_ids(split_manifest, fold, excluded_ids, training=False)
    ordered_ids, records_by_id = _validated_partition_records(records, eligible_ids)
    provenance = _partition_provenance(split_manifest, fold, ordered_ids, "fold_validation")
    checkpoint_digest = _validated_sha256(checkpoint_sha256, "checkpoint_sha256")
    if (
        isinstance(samples_per_problem, bool)
        or not isinstance(samples_per_problem, int)
        or samples_per_problem <= 0
    ):
        raise GateBValidationError("samples_per_problem must be a positive integer")
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable or None")

    validated_chunk_size = _validated_resume_chunk_size(chunk_size)
    plan = _DevelopmentGenerationPlan(
        ordered_ids=ordered_ids,
        records_by_id=records_by_id,
        provenance=provenance,
        checkpoint_sha256=checkpoint_digest,
        config=config,
        samples_per_problem=samples_per_problem,
    )
    if resume_dir is not None:
        return _run_resumable_development_baseline(
            plan,
            backend=backend,
            clock_ns=clock_ns,
            progress_callback=progress_callback,
            resume_dir=resume_dir,
            chunk_size=validated_chunk_size,
        )

    output: list[DevelopmentBaselineRecord] = []
    total_generations = plan.total_generations
    for completed_generations, specification in enumerate(plan.specifications, start=1):
        output.append(
            _generate_development_baseline_record(
                specification,
                backend=backend,
                clock_ns=clock_ns,
            )
        )
        if progress_callback is not None:
            progress_callback(completed_generations, total_generations)
    return tuple(output)


def write_development_artifacts(
    records: Iterable[DevelopmentBaselineRecord],
    *,
    jsonl_path: str | Path,
    manifest_path: str | Path,
    execution_evidence: DevelopmentExecutionEvidence,
    resume_dir: str | Path | None = None,
) -> DevelopmentArtifactWriteResult:
    """Atomically publish no-overwrite JSONL records and a checksum manifest.

    Both files must share an existing directory.  Fully written, fsynced
    temporary files are hard-linked into their final names under an exclusive
    bundle lock.  The manifest is the commit marker: a records-only orphan from
    a SIGKILL between links is not a published artifact and can only be
    completed when its bytes exactly match this invocation.  Hard-link creation
    otherwise fails when a destination already exists, so neither a pre-check
    race nor a concurrent writer can overwrite evidence.

    When ``resume_dir`` is supplied, every final row must additionally match a
    complete set of validated chunk artifacts bound to the immutable resume
    contract.  The published v2 JSONL and manifest deliberately remain
    byte-for-byte schema-compatible with non-resumable runs.
    """

    materialized = _validated_baseline_run(records)
    if resume_dir is not None:
        _validate_final_records_against_resume(materialized, resume_dir)
    validated_execution_evidence = _validated_execution_evidence(
        execution_evidence, materialized
    )
    records_bytes = (
        "".join(f"{record.to_json_line()}\n" for record in materialized).encode("utf-8")
    )
    records_digest = hashlib.sha256(records_bytes).hexdigest()
    manifest_payload = _run_manifest_payload(
        materialized,
        records_filename=Path(jsonl_path).name,
        records_bytes=len(records_bytes),
        records_sha256=records_digest,
        execution_evidence=validated_execution_evidence,
    )
    manifest_bytes = (
        json.dumps(
            manifest_payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()

    records_target, manifest_target = _validated_artifact_targets(
        jsonl_path, manifest_path
    )
    _publish_atomic_pair(
        records_target,
        records_bytes,
        manifest_target,
        manifest_bytes,
    )
    return DevelopmentArtifactWriteResult(
        records_path=str(records_target),
        manifest_path=str(manifest_target),
        record_count=len(materialized),
        records_sha256=records_digest,
        manifest_sha256=manifest_digest,
    )


@dataclass(frozen=True, slots=True)
class _DevelopmentGenerationSpec:
    """One static request binding used by normal and resumable generation."""

    problem_id: str
    record: MathRecord
    reference_answer: int
    messages: tuple[ChatMessage, ...]
    prompt_sha256: str
    sample_index: int
    seed: int
    group_id: str
    provenance: _PartitionProvenance
    checkpoint_sha256: str
    config: GateBConfig


@dataclass(frozen=True, slots=True)
class _DevelopmentGenerationPlan:
    """Validated development run inputs independent of runtime generation."""

    ordered_ids: tuple[str, ...]
    records_by_id: Mapping[str, MathRecord]
    provenance: _PartitionProvenance
    checkpoint_sha256: str
    config: GateBConfig
    samples_per_problem: int

    @property
    def total_generations(self) -> int:
        return len(self.ordered_ids) * self.samples_per_problem

    @property
    def specifications(self) -> tuple[_DevelopmentGenerationSpec, ...]:
        specifications: list[_DevelopmentGenerationSpec] = []
        for problem_id in self.ordered_ids:
            record = self.records_by_id[problem_id]
            reference_answer = _required_reference_answer(record)
            messages = _prompt_messages(record.question_raw, self.config)
            prompt_sha256 = hashlib.sha256(
                canonical_json_bytes(
                    {
                        "messages": [message.as_dict() for message in messages],
                        "add_generation_prompt": True,
                    }
                )
            ).hexdigest()
            for sample_index in range(self.samples_per_problem):
                specifications.append(
                    _DevelopmentGenerationSpec(
                        problem_id=problem_id,
                        record=record,
                        reference_answer=reference_answer,
                        messages=messages,
                        prompt_sha256=prompt_sha256,
                        sample_index=sample_index,
                        seed=deterministic_seed(
                            problem_id,
                            sample_index,
                            salt=f"gate-b1-v2:{self.config.seed}:{self.config.sha256}",
                        ),
                        group_id=self.provenance.group_ids[problem_id],
                        provenance=self.provenance,
                        checkpoint_sha256=self.checkpoint_sha256,
                        config=self.config,
                    )
                )
        return tuple(specifications)


def _generate_development_baseline_record(
    specification: _DevelopmentGenerationSpec,
    *,
    backend: GenerationBackend,
    clock_ns: Callable[[], int],
) -> DevelopmentBaselineRecord:
    """Generate one request and preserve its parser result without coercion."""

    config = specification.config
    request = GenerationRequest(
        problem_id=specification.problem_id,
        messages=specification.messages,
        seed=specification.seed,
        sample_index=specification.sample_index,
        model_id=config.model_id,
        revision=config.revision,
        route=config.route,
        prompt_sha256=specification.prompt_sha256,
        config_sha256=config.sha256,
        decoding_policy=config.decoding_policy,
    )
    started = clock_ns()
    generation = backend.generate(request)
    finished = clock_ns()
    if not isinstance(generation, GenerationResult):
        raise TypeError("GenerationBackend.generate() must return a GenerationResult")
    if finished < started:
        raise GateBValidationError("monotonic clock moved backwards during generation")
    latency_ms = (finished - started) / 1_000_000
    if not math.isfinite(latency_ms):  # pragma: no cover - integer clock invariant
        raise GateBValidationError("generation latency must be finite")
    parsed = parse_answer(generation.text)
    return DevelopmentBaselineRecord(
        problem_id=specification.problem_id,
        question_sha256=_sha256_text(specification.record.question_raw),
        route=config.route,
        sample_index=specification.sample_index,
        seed=specification.seed,
        model_id=config.model_id,
        revision=config.revision,
        prompt_sha256=specification.prompt_sha256,
        checkpoint_sha256=specification.checkpoint_sha256,
        config_sha256=config.sha256,
        decoding_policy=config.decoding_policy,
        raw_completion=generation.text,
        raw_completion_sha256=_sha256_text(generation.text),
        finish_reason=generation.finish_reason,
        input_token_count=generation.input_token_count,
        output_token_count=generation.output_token_count,
        parse=parsed,
        reference_answer=specification.reference_answer,
        exact_match=parsed.ok and parsed.value == specification.reference_answer,
        latency_ms=latency_ms,
        peak_vram_allocated_bytes=generation.peak_vram_allocated_bytes,
        group_id=specification.group_id,
        **specification.provenance.common,
    )


@dataclass(frozen=True, slots=True)
class _PartitionProvenance:
    common: Mapping[str, object]
    group_ids: Mapping[str, str]


def read_development_resume_status(resume_dir: str | Path) -> DevelopmentResumeStatus:
    """Read data-free persisted progress without loading private generations.

    A status can be read after a process interruption.  It is intentionally a
    monitor-only view: ``run_development_baseline`` is the authority that
    validates private chunks against the current split and organizer records
    before it decides whether any work may be skipped.
    """

    root = _development_resume_root(resume_dir, create=False)
    contract = _load_development_resume_contract(root)
    progress_path = root / "progress.json"
    if not progress_path.exists():
        return DevelopmentResumeStatus(
            contract_sha256=_required_resume_sha256(
                contract.get("contract_sha256"), "resume contract_sha256"
            ),
            state="planned",
            process_id=None,
            total_chunks=_required_resume_positive_int(
                contract.get("chunk_count"), "resume chunk_count"
            ),
            completed_chunks=0,
            total_generations=_required_resume_positive_int(
                contract.get("generation_count"), "resume generation_count"
            ),
            completed_generations=0,
            chunk_attempt_count=0,
            invalid_chunk_attempt_count=0,
            completed_latency_ms=0.0,
        )
    progress = _read_strict_json_object(progress_path, "development resume progress")
    status = _development_resume_status_from_payload(progress)
    if status.contract_sha256 != contract["contract_sha256"]:
        raise GateBValidationError("development resume progress has another contract")
    return status


def _run_resumable_development_baseline(
    plan: _DevelopmentGenerationPlan,
    *,
    backend: GenerationBackend,
    clock_ns: Callable[[], int],
    progress_callback: Callable[[int, int], None] | None,
    resume_dir: str | Path,
    chunk_size: int,
) -> tuple[DevelopmentBaselineRecord, ...]:
    """Generate only missing, fully contract-bound development chunks."""

    specifications = plan.specifications
    chunks = _development_specification_chunks(plan, specifications, chunk_size)
    root = _development_resume_root(resume_dir, create=True)
    contract = _build_development_resume_contract(
        plan,
        specifications,
        chunk_size=chunk_size,
        chunk_count=len(chunks),
    )
    _initialize_development_resume_contract(root, contract)
    contract_sha256 = _required_resume_sha256(
        contract["contract_sha256"], "development resume contract_sha256"
    )
    lock_path, lock_descriptor = _acquire_development_resume_lock(root, contract_sha256)
    try:
        chunks_dir = root / "chunks"
        chunks_dir.mkdir(exist_ok=True)
        if not chunks_dir.is_dir() or chunks_dir.is_symlink():
            raise GateBValidationError("development resume chunks path must be a directory")

        recovered, attempts, invalid_attempts = _load_reusable_development_chunks(
            chunks_dir,
            contract_sha256=contract_sha256,
            chunks=chunks,
        )
        _write_development_resume_progress(
            root,
            _development_resume_status(
                contract_sha256=contract_sha256,
                state="running",
                chunks=chunks,
                completed=recovered,
                chunk_attempt_count=attempts,
                invalid_chunk_attempt_count=invalid_attempts,
            ),
        )

        for chunk_index, chunk in enumerate(chunks):
            if chunk_index in recovered:
                continue
            generated: list[DevelopmentBaselineRecord] = []
            completed_before = sum(len(rows) for rows in recovered.values())
            try:
                for specification in chunk:
                    generated.append(
                        _generate_development_baseline_record(
                            specification,
                            backend=backend,
                            clock_ns=clock_ns,
                        )
                    )
                    if progress_callback is not None:
                        progress_callback(
                            completed_before + len(generated),
                            plan.total_generations,
                        )
                generated_rows = tuple(generated)
                _validate_chunk_records_against_specifications(generated_rows, chunk)
                attempt = _next_development_chunk_attempt(chunks_dir, chunk_index)
                _publish_development_resume_chunk(
                    chunks_dir,
                    contract_sha256=contract_sha256,
                    chunk_index=chunk_index,
                    attempt=attempt,
                    records=generated_rows,
                )
            except BaseException:
                _write_development_resume_progress(
                    root,
                    _development_resume_status(
                        contract_sha256=contract_sha256,
                        state="interrupted",
                        chunks=chunks,
                        completed=recovered,
                        chunk_attempt_count=attempts,
                        invalid_chunk_attempt_count=invalid_attempts,
                    ),
                )
                raise
            recovered[chunk_index] = generated_rows
            attempts += 1
            _write_development_resume_progress(
                root,
                _development_resume_status(
                    contract_sha256=contract_sha256,
                    state="running",
                    chunks=chunks,
                    completed=recovered,
                    chunk_attempt_count=attempts,
                    invalid_chunk_attempt_count=invalid_attempts,
                ),
            )

        output = tuple(
            row for chunk_index in range(len(chunks)) for row in recovered[chunk_index]
        )
        _validated_baseline_run(output)
        _write_development_resume_progress(
            root,
            _development_resume_status(
                contract_sha256=contract_sha256,
                state="complete",
                chunks=chunks,
                completed=recovered,
                chunk_attempt_count=attempts,
                invalid_chunk_attempt_count=invalid_attempts,
            ),
        )
        return output
    finally:
        _release_development_resume_lock(lock_path, lock_descriptor)


def _validated_resume_chunk_size(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise GateBValidationError("chunk_size must be a positive integer")
    return value


def _development_specification_chunks(
    plan: _DevelopmentGenerationPlan,
    specifications: Sequence[_DevelopmentGenerationSpec],
    chunk_size: int,
) -> tuple[tuple[_DevelopmentGenerationSpec, ...], ...]:
    """Split only between problems, never between samples of one problem."""

    expected_count = plan.total_generations
    if len(specifications) != expected_count:  # pragma: no cover - plan invariant
        raise GateBValidationError("development generation plan has an invalid record count")
    width = chunk_size * plan.samples_per_problem
    chunks = tuple(
        tuple(specifications[offset : offset + width])
        for offset in range(0, len(specifications), width)
    )
    if not chunks or any(not chunk for chunk in chunks):  # pragma: no cover - plan invariant
        raise GateBValidationError("development generation plan did not produce chunks")
    return chunks


def _build_development_resume_contract(
    plan: _DevelopmentGenerationPlan,
    specifications: Sequence[_DevelopmentGenerationSpec],
    *,
    chunk_size: int,
    chunk_count: int,
) -> dict[str, object]:
    """Build the immutable input/config/split binding for one ledger."""

    common = plan.provenance.common
    input_rows = [
        {
            "problem_id": problem_id,
            "question_sha256": _sha256_text(plan.records_by_id[problem_id].question_raw),
            "reference_answer": _required_reference_answer(plan.records_by_id[problem_id]),
            "group_id": plan.provenance.group_ids[problem_id],
        }
        for problem_id in plan.ordered_ids
    ]
    payload: dict[str, object] = {
        "schema_version": _DEVELOPMENT_RESUME_CONTRACT_SCHEMA,
        "model_id": plan.config.model_id,
        "revision": plan.config.revision,
        "route": plan.config.route,
        "checkpoint_sha256": plan.checkpoint_sha256,
        "config_sha256": plan.config.sha256,
        "decoding_policy": plan.config.decoding_policy.as_dict(),
        "decoding_policy_sha256": plan.config.decoding_policy.sha256,
        "split_version": common["split_version"],
        "split_sha256": common["split_sha256"],
        "source_groups_sha256": common["source_groups_sha256"],
        "fold": common["fold"],
        "partition": common["partition"],
        "split_partition": common["split_partition"],
        "eligible_ids": list(plan.ordered_ids),
        "eligibility_ids_sha256": common["eligibility_ids_sha256"],
        "input_records_sha256": hashlib.sha256(canonical_json_bytes(input_rows)).hexdigest(),
        "samples_per_problem": plan.samples_per_problem,
        "chunk_size": chunk_size,
        "chunk_count": chunk_count,
        "generation_count": len(specifications),
    }
    return _with_development_resume_contract_sha256(payload)


def _with_development_resume_contract_sha256(payload: Mapping[str, object]) -> dict[str, object]:
    if "contract_sha256" in payload:
        raise GateBValidationError("development resume contract payload already has a digest")
    digest = hashlib.sha256(canonical_json_bytes(dict(payload))).hexdigest()
    return {**payload, "contract_sha256": digest}


def _development_resume_root(resume_dir: str | Path, *, create: bool) -> Path:
    raw = Path(resume_dir)
    if raw.is_symlink():
        raise GateBValidationError("development resume directory must not be a symbolic link")
    if create:
        raw.mkdir(parents=True, exist_ok=True)
    if not raw.exists() or not raw.is_dir():
        raise GateBValidationError(f"development resume directory does not exist: {raw}")
    root = raw.resolve(strict=True)
    if not root.is_dir():  # pragma: no cover - resolve/check invariant
        raise GateBValidationError("development resume path must be a directory")
    return root


def _initialize_development_resume_contract(root: Path, expected: Mapping[str, object]) -> None:
    path = root / "contract.json"
    with suppress(FileExistsError):
        _publish_new_file(path, _pretty_json_bytes(expected))
    actual = _load_development_resume_contract(root)
    _validate_development_resume_contract(actual, expected=expected)


def _load_development_resume_contract(root: Path) -> dict[str, object]:
    path = root / "contract.json"
    payload = _read_strict_json_object(path, "development resume contract")
    _validate_development_resume_contract(payload)
    return payload


def _validate_development_resume_contract(
    payload: Mapping[str, object],
    *,
    expected: Mapping[str, object] | None = None,
) -> None:
    required = {
        "schema_version",
        "contract_sha256",
        "model_id",
        "revision",
        "route",
        "checkpoint_sha256",
        "config_sha256",
        "decoding_policy",
        "decoding_policy_sha256",
        "split_version",
        "split_sha256",
        "source_groups_sha256",
        "fold",
        "partition",
        "split_partition",
        "eligible_ids",
        "eligibility_ids_sha256",
        "input_records_sha256",
        "samples_per_problem",
        "chunk_size",
        "chunk_count",
        "generation_count",
    }
    if set(payload) != required:
        raise GateBValidationError("development resume contract has an unexpected schema")
    if payload.get("schema_version") != _DEVELOPMENT_RESUME_CONTRACT_SCHEMA:
        raise GateBValidationError("development resume contract schema is unsupported")
    supplied_digest = _required_resume_sha256(
        payload.get("contract_sha256"), "development resume contract_sha256"
    )
    digest_payload = {key: value for key, value in payload.items() if key != "contract_sha256"}
    actual_digest = hashlib.sha256(canonical_json_bytes(digest_payload)).hexdigest()
    if supplied_digest != actual_digest:
        raise GateBValidationError("development resume contract digest does not match")
    _required_resume_sha256(payload.get("checkpoint_sha256"), "resume checkpoint_sha256")
    _required_resume_sha256(payload.get("config_sha256"), "resume config_sha256")
    _required_resume_sha256(
        payload.get("decoding_policy_sha256"), "resume decoding_policy_sha256"
    )
    _required_resume_sha256(payload.get("split_sha256"), "resume split_sha256")
    _required_resume_sha256(
        payload.get("source_groups_sha256"), "resume source_groups_sha256"
    )
    _required_resume_sha256(
        payload.get("eligibility_ids_sha256"), "resume eligibility_ids_sha256"
    )
    _required_resume_sha256(
        payload.get("input_records_sha256"), "resume input_records_sha256"
    )
    eligible_ids = payload.get("eligible_ids")
    if (
        not isinstance(eligible_ids, list)
        or not eligible_ids
        or any(not isinstance(problem_id, str) for problem_id in eligible_ids)
    ):
        raise GateBValidationError("development resume contract eligible_ids are invalid")
    _reject_non_train_ids(tuple(eligible_ids))
    if len(set(eligible_ids)) != len(eligible_ids):
        raise GateBValidationError("development resume contract eligible_ids are duplicated")
    if _ids_sha256(tuple(eligible_ids)) != payload["eligibility_ids_sha256"]:
        raise GateBValidationError("development resume contract eligibility digest does not match")
    samples = _required_resume_positive_int(
        payload.get("samples_per_problem"), "resume samples_per_problem"
    )
    chunk_size = _required_resume_positive_int(payload.get("chunk_size"), "resume chunk_size")
    chunk_count = _required_resume_positive_int(payload.get("chunk_count"), "resume chunk_count")
    generation_count = _required_resume_positive_int(
        payload.get("generation_count"), "resume generation_count"
    )
    if generation_count != len(eligible_ids) * samples:
        raise GateBValidationError("development resume contract generation count does not match")
    if chunk_count != math.ceil(len(eligible_ids) / chunk_size):
        raise GateBValidationError("development resume contract chunk count does not match")
    if not isinstance(payload.get("decoding_policy"), Mapping):
        raise GateBValidationError("development resume contract decoding policy is invalid")
    policy = _decoding_policy_from_payload(payload["decoding_policy"])
    if policy.sha256 != payload["decoding_policy_sha256"]:
        raise GateBValidationError(
            "development resume contract decoding policy digest does not match"
        )
    for field_name in ("model_id", "revision", "route", "split_version", "partition"):
        value = payload.get(field_name)
        if not isinstance(value, str) or not value or value != value.strip():
            raise GateBValidationError(f"development resume contract {field_name} is invalid")
    fold = payload.get("fold")
    if isinstance(fold, bool) or not isinstance(fold, int) or fold < 0:
        raise GateBValidationError("development resume contract fold is invalid")
    if expected is not None and dict(payload) != dict(expected):
        raise GateBValidationError("development resume contract is incompatible with this run")


def _acquire_development_resume_lock(root: Path, contract_sha256: str) -> tuple[Path, int]:
    lock_path = root / ".gate-b-development-resume.lock"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    for _ in range(2):
        try:
            descriptor = os.open(lock_path, flags, 0o600)
        except FileExistsError as exc:
            lock_payload = _read_strict_json_object(lock_path, "development resume lock")
            if set(lock_payload) != {"schema_version", "contract_sha256", "process_id"}:
                raise GateBValidationError(
                    "development resume lock has an unexpected schema"
                ) from exc
            if lock_payload.get("schema_version") != _DEVELOPMENT_RESUME_LOCK_SCHEMA:
                raise GateBValidationError(
                    "development resume lock schema is unsupported"
                ) from exc
            if lock_payload.get("contract_sha256") != contract_sha256:
                raise GateBValidationError(
                    "development resume lock belongs to another contract"
                ) from exc
            stale_pid = lock_payload.get("process_id")
            if isinstance(stale_pid, bool) or not isinstance(stale_pid, int) or stale_pid <= 0:
                raise GateBValidationError(
                    "development resume lock process ID is invalid"
                ) from exc
            if _process_is_alive(stale_pid):
                raise GateBArtifactExistsError(
                    f"development resume lock is active for process {stale_pid}"
                ) from exc
            try:
                lock_path.unlink()
            except FileNotFoundError:
                continue
            continue
        payload = {
            "schema_version": _DEVELOPMENT_RESUME_LOCK_SCHEMA,
            "contract_sha256": contract_sha256,
            "process_id": os.getpid(),
        }
        try:
            _write_all_and_fsync(descriptor, canonical_json_bytes(payload))
        except BaseException:
            os.close(descriptor)
            with suppress(FileNotFoundError):
                lock_path.unlink()
            raise
        return lock_path, descriptor
    raise GateBArtifactExistsError("development resume lock changed while recovering stale state")


def _release_development_resume_lock(lock_path: Path, descriptor: int) -> None:
    try:
        held = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        current = lock_path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return
    if (held.st_dev, held.st_ino) == (current.st_dev, current.st_ino):
        lock_path.unlink()
        _fsync_directory(lock_path.parent)


def _process_is_alive(process_id: int) -> bool:
    if process_id == os.getpid():
        return True
    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _write_all_and_fsync(descriptor: int, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        count = os.write(descriptor, remaining)
        if count <= 0:  # pragma: no cover - OS write contract
            raise OSError("could not write development resume lock")
        remaining = remaining[count:]
    os.fsync(descriptor)


def _load_reusable_development_chunks(
    chunks_dir: Path,
    *,
    contract_sha256: str,
    chunks: Sequence[Sequence[_DevelopmentGenerationSpec]],
) -> tuple[dict[int, tuple[DevelopmentBaselineRecord, ...]], int, int]:
    reusable: dict[int, tuple[DevelopmentBaselineRecord, ...]] = {}
    total_attempts = 0
    invalid_attempts = 0
    for chunk_index, specifications in enumerate(chunks):
        candidates = _development_chunk_manifest_candidates(chunks_dir, chunk_index)
        total_attempts += len(candidates)
        for _attempt, manifest_path in reversed(candidates):
            try:
                records = _load_development_resume_chunk(
                    chunks_dir,
                    manifest_path=manifest_path,
                    contract_sha256=contract_sha256,
                    chunk_index=chunk_index,
                    expected_count=len(specifications),
                )
                _validate_chunk_records_against_specifications(records, specifications)
            except (GateBValidationError, OSError, TypeError, ValueError):
                invalid_attempts += 1
                continue
            reusable[chunk_index] = records
            break
    return reusable, total_attempts, invalid_attempts


def _development_chunk_manifest_candidates(
    chunks_dir: Path, chunk_index: int
) -> list[tuple[int, Path]]:
    candidates: list[tuple[int, Path]] = []
    for path in chunks_dir.glob(f"chunk-{chunk_index:06d}-attempt-*.manifest.json"):
        match = _DEVELOPMENT_RESUME_CHUNK_RE.fullmatch(path.name)
        if match is None or int(match.group("chunk_index")) != chunk_index:
            continue
        candidates.append((int(match.group("attempt")), path))
    return sorted(candidates)


def _next_development_chunk_attempt(chunks_dir: Path, chunk_index: int) -> int:
    manifest_attempts = [
        attempt
        for attempt, _path in _development_chunk_manifest_candidates(chunks_dir, chunk_index)
    ]
    record_attempts = [
        int(match.group("attempt"))
        for path in chunks_dir.glob(f"chunk-{chunk_index:06d}-attempt-*.jsonl")
        if (match := _DEVELOPMENT_RESUME_CHUNK_RECORD_RE.fullmatch(path.name)) is not None
        and int(match.group("chunk_index")) == chunk_index
    ]
    attempts = [*manifest_attempts, *record_attempts]
    return (max(attempts) if attempts else 0) + 1


def _publish_development_resume_chunk(
    chunks_dir: Path,
    *,
    contract_sha256: str,
    chunk_index: int,
    attempt: int,
    records: Sequence[DevelopmentBaselineRecord],
) -> None:
    records_name = f"chunk-{chunk_index:06d}-attempt-{attempt:06d}.jsonl"
    manifest_name = f"chunk-{chunk_index:06d}-attempt-{attempt:06d}.manifest.json"
    records_bytes = "".join(f"{record.to_json_line()}\n" for record in records).encode("utf-8")
    records_sha256 = hashlib.sha256(records_bytes).hexdigest()
    manifest = {
        "schema_version": _DEVELOPMENT_RESUME_CHUNK_SCHEMA,
        "contract_sha256": contract_sha256,
        "chunk_index": chunk_index,
        "attempt": attempt,
        "records_file": records_name,
        "records_bytes": len(records_bytes),
        "records_sha256": records_sha256,
        "record_count": len(records),
        "identity_sha256": _development_identity_sha256(records),
    }
    _publish_atomic_pair(
        chunks_dir / records_name,
        records_bytes,
        chunks_dir / manifest_name,
        _pretty_json_bytes(manifest),
    )


def _load_development_resume_chunk(
    chunks_dir: Path,
    *,
    manifest_path: Path,
    contract_sha256: str,
    chunk_index: int,
    expected_count: int,
) -> tuple[DevelopmentBaselineRecord, ...]:
    manifest = _read_strict_json_object(manifest_path, "development resume chunk manifest")
    expected_keys = {
        "schema_version",
        "contract_sha256",
        "chunk_index",
        "attempt",
        "records_file",
        "records_bytes",
        "records_sha256",
        "record_count",
        "identity_sha256",
    }
    if set(manifest) != expected_keys:
        raise GateBValidationError("development resume chunk manifest has an unexpected schema")
    if manifest.get("schema_version") != _DEVELOPMENT_RESUME_CHUNK_SCHEMA:
        raise GateBValidationError("development resume chunk manifest schema is unsupported")
    if manifest.get("contract_sha256") != contract_sha256:
        raise GateBValidationError("development resume chunk has another contract")
    if manifest.get("chunk_index") != chunk_index:
        raise GateBValidationError("development resume chunk index does not match its filename")
    match = _DEVELOPMENT_RESUME_CHUNK_RE.fullmatch(manifest_path.name)
    if match is None:
        raise GateBValidationError("development resume chunk manifest filename is invalid")
    attempt = _required_resume_positive_int(manifest.get("attempt"), "resume chunk attempt")
    if int(match.group("attempt")) != attempt:
        raise GateBValidationError("development resume chunk attempt does not match its filename")
    records_name = manifest.get("records_file")
    expected_name = f"chunk-{chunk_index:06d}-attempt-{attempt:06d}.jsonl"
    if not isinstance(records_name, str) or records_name != expected_name:
        raise GateBValidationError("development resume chunk records filename is invalid")
    records_path = chunks_dir / records_name
    if records_path.is_symlink() or not records_path.is_file():
        raise GateBValidationError("development resume chunk records must be a regular file")
    raw = records_path.read_bytes()
    if manifest.get("records_bytes") != len(raw):
        raise GateBValidationError("development resume chunk record byte count does not match")
    if manifest.get("records_sha256") != hashlib.sha256(raw).hexdigest():
        raise GateBValidationError("development resume chunk record digest does not match")
    if not raw.endswith(b"\n"):
        raise GateBValidationError("development resume chunk JSONL is not newline terminated")
    try:
        lines = raw.decode("utf-8", errors="strict").splitlines()
    except UnicodeDecodeError as exc:
        raise GateBValidationError("development resume chunk is not UTF-8") from exc
    if not lines or any(not line for line in lines):
        raise GateBValidationError("development resume chunk JSONL contains an empty record")
    records = tuple(
        _development_baseline_record_from_payload(
            _strict_json_object_from_text(line, "development resume chunk record")
        )
        for line in lines
    )
    if manifest.get("record_count") != len(records) or len(records) != expected_count:
        raise GateBValidationError("development resume chunk record count does not match")
    if _development_identity_sha256(records) != manifest.get("identity_sha256"):
        raise GateBValidationError("development resume chunk identity digest does not match")
    if raw != "".join(f"{record.to_json_line()}\n" for record in records).encode("utf-8"):
        raise GateBValidationError("development resume chunk JSONL is not canonical")
    return records


def _development_baseline_record_from_payload(
    payload: Mapping[str, object],
) -> DevelopmentBaselineRecord:
    expected_keys = {
        "schema_version",
        "problem_id",
        "question_sha256",
        "route",
        "sample_index",
        "seed",
        "model_id",
        "revision",
        "prompt_sha256",
        "checkpoint_sha256",
        "config_sha256",
        "decoding_policy",
        "decoding_policy_sha256",
        "raw_completion",
        "raw_completion_sha256",
        "finish_reason",
        "input_token_count",
        "output_token_count",
        "parse",
        "reference_answer",
        "exact_match",
        "latency_ms",
        "peak_vram_allocated_bytes",
        "split_version",
        "split_sha256",
        "source_groups_sha256",
        "fold",
        "partition",
        "split_partition",
        "group_id",
        "eligibility_ids_sha256",
    }
    if set(payload) != expected_keys:
        raise GateBValidationError("development resume record has an unexpected schema")
    if payload.get("schema_version") != "gate-b1-development-baseline-v2":
        raise GateBValidationError("development resume record schema is unsupported")
    policy_value = payload.get("decoding_policy")
    if not isinstance(policy_value, Mapping):
        raise GateBValidationError("development resume record decoding policy is invalid")
    policy = _decoding_policy_from_payload(policy_value)
    if payload.get("decoding_policy_sha256") != policy.sha256:
        raise GateBValidationError(
            "development resume record decoding policy digest does not match"
        )
    raw_completion = payload.get("raw_completion")
    if not isinstance(raw_completion, str):
        raise GateBValidationError("development resume record raw completion is invalid")
    parse_value = payload.get("parse")
    parsed = parse_answer(raw_completion)
    if parse_value != asdict(parsed):
        raise GateBValidationError("development resume record parser result does not match")
    record = DevelopmentBaselineRecord(
        problem_id=_required_resume_string(payload.get("problem_id"), "resume problem_id"),
        question_sha256=_required_resume_sha256(
            payload.get("question_sha256"), "resume question_sha256"
        ),
        route=_required_resume_string(payload.get("route"), "resume route"),
        sample_index=_required_resume_nonnegative_int(
            payload.get("sample_index"), "resume sample_index"
        ),
        seed=_required_resume_nonnegative_int(payload.get("seed"), "resume seed"),
        model_id=_required_resume_string(payload.get("model_id"), "resume model_id"),
        revision=_required_resume_string(payload.get("revision"), "resume revision"),
        prompt_sha256=_required_resume_sha256(
            payload.get("prompt_sha256"), "resume prompt_sha256"
        ),
        checkpoint_sha256=_required_resume_sha256(
            payload.get("checkpoint_sha256"), "resume checkpoint_sha256"
        ),
        config_sha256=_required_resume_sha256(
            payload.get("config_sha256"), "resume config_sha256"
        ),
        decoding_policy=policy,
        raw_completion=raw_completion,
        raw_completion_sha256=_required_resume_sha256(
            payload.get("raw_completion_sha256"), "resume raw_completion_sha256"
        ),
        finish_reason=_required_resume_string(
            payload.get("finish_reason"), "resume finish_reason"
        ),
        input_token_count=_required_resume_nonnegative_int(
            payload.get("input_token_count"), "resume input_token_count"
        ),
        output_token_count=_required_resume_nonnegative_int(
            payload.get("output_token_count"), "resume output_token_count"
        ),
        parse=parsed,
        reference_answer=_required_resume_int(
            payload.get("reference_answer"), "resume reference_answer"
        ),
        exact_match=payload.get("exact_match"),
        latency_ms=_required_resume_float(payload.get("latency_ms"), "resume latency_ms"),
        peak_vram_allocated_bytes=_required_resume_optional_nonnegative_int(
            payload.get("peak_vram_allocated_bytes"), "resume peak_vram_allocated_bytes"
        ),
        split_version=_required_resume_string(
            payload.get("split_version"), "resume split_version"
        ),
        split_sha256=_required_resume_sha256(
            payload.get("split_sha256"), "resume split_sha256"
        ),
        source_groups_sha256=_required_resume_sha256(
            payload.get("source_groups_sha256"), "resume source_groups_sha256"
        ),
        fold=_required_resume_nonnegative_int(payload.get("fold"), "resume fold"),
        partition=_required_resume_string(payload.get("partition"), "resume partition"),
        split_partition=_required_resume_string(
            payload.get("split_partition"), "resume split_partition"
        ),
        group_id=_required_resume_string(payload.get("group_id"), "resume group_id"),
        eligibility_ids_sha256=_required_resume_sha256(
            payload.get("eligibility_ids_sha256"), "resume eligibility_ids_sha256"
        ),
    )
    if not isinstance(record.exact_match, bool):
        raise GateBValidationError("development resume record exact_match is invalid")
    _validate_baseline_record(record, 1)
    if record.as_dict() != dict(payload):
        raise GateBValidationError("development resume record representation is not canonical")
    return record


def _decoding_policy_from_payload(payload: Mapping[str, object]) -> DecodingPolicy:
    expected_keys = {
        "do_sample",
        "num_beams",
        "max_new_tokens",
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
    }
    if set(payload) != expected_keys:
        raise GateBValidationError("development decoding policy has an unexpected schema")
    if not isinstance(payload.get("do_sample"), bool):
        raise GateBValidationError("development decoding policy do_sample is invalid")
    for field_name in ("num_beams", "max_new_tokens"):
        _required_resume_positive_int(payload.get(field_name), f"development {field_name}")
    for field_name in ("temperature", "top_p", "top_k"):
        value = payload.get(field_name)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise GateBValidationError(f"development decoding policy {field_name} is invalid")
    repetition = payload.get("repetition_penalty")
    if isinstance(repetition, bool) or not isinstance(repetition, (int, float)):
        raise GateBValidationError("development decoding policy repetition_penalty is invalid")
    return DecodingPolicy(
        do_sample=payload["do_sample"],
        num_beams=payload["num_beams"],
        max_new_tokens=payload["max_new_tokens"],
        temperature=payload["temperature"],
        top_p=payload["top_p"],
        top_k=payload["top_k"],
        repetition_penalty=repetition,
    )


def _validate_chunk_records_against_specifications(
    records: Sequence[DevelopmentBaselineRecord],
    specifications: Sequence[_DevelopmentGenerationSpec],
) -> None:
    if len(records) != len(specifications):
        raise GateBValidationError("development resume chunk has an unexpected record count")
    for index, (record, specification) in enumerate(
        zip(records, specifications, strict=True), start=1
    ):
        _validate_baseline_record(record, index)
        expected = {
            "problem_id": specification.problem_id,
            "question_sha256": _sha256_text(specification.record.question_raw),
            "route": specification.config.route,
            "sample_index": specification.sample_index,
            "seed": specification.seed,
            "model_id": specification.config.model_id,
            "revision": specification.config.revision,
            "prompt_sha256": specification.prompt_sha256,
            "checkpoint_sha256": specification.checkpoint_sha256,
            "config_sha256": specification.config.sha256,
            "decoding_policy": specification.config.decoding_policy,
            "reference_answer": specification.reference_answer,
            "split_version": specification.provenance.common["split_version"],
            "split_sha256": specification.provenance.common["split_sha256"],
            "source_groups_sha256": specification.provenance.common["source_groups_sha256"],
            "fold": specification.provenance.common["fold"],
            "partition": specification.provenance.common["partition"],
            "split_partition": specification.provenance.common["split_partition"],
            "group_id": specification.group_id,
            "eligibility_ids_sha256": specification.provenance.common[
                "eligibility_ids_sha256"
            ],
        }
        mismatched = [
            field_name
            for field_name, value in expected.items()
            if getattr(record, field_name) != value
        ]
        if mismatched:
            raise GateBValidationError(
                "development resume chunk record "
                f"{index} does not match its contract: {mismatched!r}"
            )


def _development_identity_sha256(records: Sequence[DevelopmentBaselineRecord]) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            [
                {
                    "problem_id": record.problem_id,
                    "sample_index": record.sample_index,
                    "seed": record.seed,
                }
                for record in records
            ]
        )
    ).hexdigest()


def _development_resume_status(
    *,
    contract_sha256: str,
    state: str,
    chunks: Sequence[Sequence[_DevelopmentGenerationSpec]],
    completed: Mapping[int, Sequence[DevelopmentBaselineRecord]],
    chunk_attempt_count: int,
    invalid_chunk_attempt_count: int,
) -> DevelopmentResumeStatus:
    if state not in {"running", "interrupted", "complete"}:
        raise GateBValidationError("development resume state is invalid")
    completed_rows = tuple(row for rows in completed.values() for row in rows)
    return DevelopmentResumeStatus(
        contract_sha256=contract_sha256,
        state=state,
        process_id=os.getpid() if state == "running" else None,
        total_chunks=len(chunks),
        completed_chunks=len(completed),
        total_generations=sum(len(chunk) for chunk in chunks),
        completed_generations=len(completed_rows),
        chunk_attempt_count=chunk_attempt_count,
        invalid_chunk_attempt_count=invalid_chunk_attempt_count,
        completed_latency_ms=math.fsum(row.latency_ms for row in completed_rows),
    )


def _write_development_resume_progress(root: Path, status: DevelopmentResumeStatus) -> None:
    _write_replacing_json(root / "progress.json", status.as_dict())


def _development_resume_status_from_payload(
    payload: Mapping[str, object],
) -> DevelopmentResumeStatus:
    expected_keys = {
        "schema_version",
        "contract_sha256",
        "state",
        "process_id",
        "total_chunks",
        "completed_chunks",
        "total_generations",
        "completed_generations",
        "chunk_attempt_count",
        "invalid_chunk_attempt_count",
        "completed_latency_ms",
    }
    if set(payload) != expected_keys:
        raise GateBValidationError("development resume progress has an unexpected schema")
    if payload.get("schema_version") != _DEVELOPMENT_RESUME_PROGRESS_SCHEMA:
        raise GateBValidationError("development resume progress schema is unsupported")
    state = payload.get("state")
    if state not in {"planned", "running", "interrupted", "complete"}:
        raise GateBValidationError("development resume progress state is invalid")
    process_id = payload.get("process_id")
    if process_id is not None:
        process_id = _required_resume_positive_int(process_id, "resume progress process_id")
    status = DevelopmentResumeStatus(
        contract_sha256=_required_resume_sha256(
            payload.get("contract_sha256"), "resume progress contract_sha256"
        ),
        state=state,
        process_id=process_id,
        total_chunks=_required_resume_positive_int(
            payload.get("total_chunks"), "resume progress total_chunks"
        ),
        completed_chunks=_required_resume_nonnegative_int(
            payload.get("completed_chunks"), "resume progress completed_chunks"
        ),
        total_generations=_required_resume_positive_int(
            payload.get("total_generations"), "resume progress total_generations"
        ),
        completed_generations=_required_resume_nonnegative_int(
            payload.get("completed_generations"), "resume progress completed_generations"
        ),
        chunk_attempt_count=_required_resume_nonnegative_int(
            payload.get("chunk_attempt_count"), "resume progress chunk_attempt_count"
        ),
        invalid_chunk_attempt_count=_required_resume_nonnegative_int(
            payload.get("invalid_chunk_attempt_count"),
            "resume progress invalid_chunk_attempt_count",
        ),
        completed_latency_ms=_required_resume_nonnegative_float(
            payload.get("completed_latency_ms"), "resume progress completed_latency_ms"
        ),
    )
    if (
        status.completed_chunks > status.total_chunks
        or status.completed_generations > status.total_generations
    ):
        raise GateBValidationError("development resume progress exceeds its total")
    return status


def _validate_final_records_against_resume(
    records: Sequence[DevelopmentBaselineRecord], resume_dir: str | Path
) -> None:
    root = _development_resume_root(resume_dir, create=False)
    contract = _load_development_resume_contract(root)
    chunk_size = _required_resume_positive_int(contract.get("chunk_size"), "resume chunk_size")
    expected_contract = _build_development_resume_contract_from_records(records, chunk_size)
    _validate_development_resume_contract(contract, expected=expected_contract)
    chunks_dir = root / "chunks"
    if not chunks_dir.is_dir() or chunks_dir.is_symlink():
        raise GateBValidationError("development resume chunks path must be a directory")
    samples = _required_resume_positive_int(
        contract.get("samples_per_problem"), "resume samples_per_problem"
    )
    width = chunk_size * samples
    expected_chunks = tuple(
        tuple(records[offset : offset + width]) for offset in range(0, len(records), width)
    )
    if len(expected_chunks) != contract["chunk_count"]:
        raise GateBValidationError(
            "development resume final records have an unexpected chunk count"
        )
    contract_sha256 = _required_resume_sha256(
        contract.get("contract_sha256"), "resume contract_sha256"
    )
    for chunk_index, expected_records in enumerate(expected_chunks):
        candidates = _development_chunk_manifest_candidates(chunks_dir, chunk_index)
        found = False
        for _attempt, manifest_path in reversed(candidates):
            try:
                recovered = _load_development_resume_chunk(
                    chunks_dir,
                    manifest_path=manifest_path,
                    contract_sha256=contract_sha256,
                    chunk_index=chunk_index,
                    expected_count=len(expected_records),
                )
            except (GateBValidationError, OSError, TypeError, ValueError):
                continue
            if recovered == expected_records:
                found = True
                break
        if not found:
            raise GateBValidationError(
                "development resume chunk "
                f"{chunk_index} is incomplete or does not match final records"
            )


def _build_development_resume_contract_from_records(
    records: Sequence[DevelopmentBaselineRecord], chunk_size: int
) -> dict[str, object]:
    first = records[0]
    ordered_ids = tuple(dict.fromkeys(record.problem_id for record in records))
    first_by_id: dict[str, DevelopmentBaselineRecord] = {}
    for record in records:
        first_by_id.setdefault(record.problem_id, record)
    input_rows = [
        {
            "problem_id": problem_id,
            "question_sha256": first_by_id[problem_id].question_sha256,
            "reference_answer": first_by_id[problem_id].reference_answer,
            "group_id": first_by_id[problem_id].group_id,
        }
        for problem_id in ordered_ids
    ]
    samples = len(records) // len(ordered_ids)
    payload: dict[str, object] = {
        "schema_version": _DEVELOPMENT_RESUME_CONTRACT_SCHEMA,
        "model_id": first.model_id,
        "revision": first.revision,
        "route": first.route,
        "checkpoint_sha256": first.checkpoint_sha256,
        "config_sha256": first.config_sha256,
        "decoding_policy": first.decoding_policy.as_dict(),
        "decoding_policy_sha256": first.decoding_policy.sha256,
        "split_version": first.split_version,
        "split_sha256": first.split_sha256,
        "source_groups_sha256": first.source_groups_sha256,
        "fold": first.fold,
        "partition": first.partition,
        "split_partition": first.split_partition,
        "eligible_ids": list(ordered_ids),
        "eligibility_ids_sha256": first.eligibility_ids_sha256,
        "input_records_sha256": hashlib.sha256(canonical_json_bytes(input_rows)).hexdigest(),
        "samples_per_problem": samples,
        "chunk_size": chunk_size,
        "chunk_count": math.ceil(len(ordered_ids) / chunk_size),
        "generation_count": len(records),
    }
    return _with_development_resume_contract_sha256(payload)


def _publish_new_file(target: Path, payload: bytes) -> None:
    temporary = _write_fsynced_temp(target.parent, target.name, payload)
    try:
        os.link(temporary, target)
        _fsync_directory(target.parent)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


def _pretty_json_bytes(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _write_replacing_json(target: Path, value: Mapping[str, object]) -> None:
    temporary = _write_fsynced_temp(target.parent, target.name, _pretty_json_bytes(value))
    try:
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


def _read_strict_json_object(path: Path, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise GateBValidationError(f"{label} must be a regular file")
    try:
        return _strict_json_object_from_text(
            path.read_text(encoding="utf-8", errors="strict"), label
        )
    except (OSError, UnicodeError) as exc:
        raise GateBValidationError(f"cannot read {label}: {exc}") from exc


def _strict_json_object_from_text(text: str, label: str) -> dict[str, object]:
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_unique_development_resume_json_object,
            parse_constant=_reject_development_resume_json_constant,
        )
    except (json.JSONDecodeError, UnicodeError, ValueError) as exc:
        raise GateBValidationError(f"invalid {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise GateBValidationError(f"{label} must contain one JSON object")
    return payload


def _unique_development_resume_json_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON key {key!r}")
        payload[key] = value
    return payload


def _reject_development_resume_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON numeric constant {value!r}")


def _required_resume_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise GateBValidationError(f"{label} must be a non-empty trimmed string")
    return value


def _required_resume_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise GateBValidationError(f"{label} must be a 64-character SHA-256 hex digest")
    return value.lower()


def _required_resume_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise GateBValidationError(f"{label} must be an integer")
    return value


def _required_resume_nonnegative_int(value: object, label: str) -> int:
    parsed = _required_resume_int(value, label)
    if parsed < 0:
        raise GateBValidationError(f"{label} must be non-negative")
    return parsed


def _required_resume_positive_int(value: object, label: str) -> int:
    parsed = _required_resume_nonnegative_int(value, label)
    if parsed <= 0:
        raise GateBValidationError(f"{label} must be positive")
    return parsed


def _required_resume_optional_nonnegative_int(value: object, label: str) -> int | None:
    if value is None:
        return None
    return _required_resume_nonnegative_int(value, label)


def _required_resume_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GateBValidationError(f"{label} must be finite")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise GateBValidationError(f"{label} must be finite")
    return parsed


def _required_resume_nonnegative_float(value: object, label: str) -> float:
    parsed = _required_resume_float(value, label)
    if parsed < 0:
        raise GateBValidationError(f"{label} must be non-negative")
    return parsed


def _derive_eligible_ids(
    manifest: SplitManifest,
    fold: int,
    excluded_ids: Iterable[str],
    *,
    training: bool,
) -> tuple[str, ...]:
    if not isinstance(manifest, SplitManifest):
        raise TypeError("split_manifest must be a SplitManifest")
    exclusions = tuple(excluded_ids)
    try:
        derived = (
            eligible_training_ids(manifest, fold, exclusions)
            if training
            else eligible_validation_ids(manifest, fold, exclusions)
        )
    except SplitValidationError as exc:
        raise GateBValidationError(f"invalid split eligibility boundary: {exc}") from exc
    if not derived:
        role = "training" if training else "validation"
        raise GateBValidationError(f"eligible {role} partition must not be empty")
    return derived


def _partition_provenance(
    manifest: SplitManifest,
    fold: int,
    eligible_ids: Sequence[str],
    partition: str,
) -> _PartitionProvenance:
    assignment_by_id = manifest.assignment_by_id()
    group_ids: dict[str, str] = {}
    for problem_id in eligible_ids:
        assignment = assignment_by_id[problem_id]
        if assignment.partition is not SplitPartition.CROSS_VALIDATION:
            raise GateBValidationError(
                f"{problem_id}: Gate B development/training cannot use locked holdout"
            )
        group_ids[problem_id] = assignment.group_id
    return _PartitionProvenance(
        common={
            "split_version": manifest.version,
            "split_sha256": manifest.sha256,
            "source_groups_sha256": manifest.source_groups_sha256,
            "fold": fold,
            "partition": partition,
            "split_partition": SplitPartition.CROSS_VALIDATION.value,
            "eligibility_ids_sha256": _ids_sha256(eligible_ids),
        },
        group_ids=group_ids,
    )


def _validated_partition_records(
    records: Iterable[MathRecord], eligible_ids: Sequence[str]
) -> tuple[tuple[str, ...], Mapping[str, MathRecord]]:
    ordered_ids = tuple(eligible_ids)
    _reject_non_train_ids(ordered_ids)
    materialized = tuple(records)
    if not materialized:
        raise GateBValidationError("records must not be empty")
    if any(not isinstance(record, MathRecord) for record in materialized):
        raise TypeError("records must contain MathRecord instances")
    record_ids = tuple(record.id for record in materialized)
    if len(set(record_ids)) != len(record_ids):
        raise GateBValidationError("records contain duplicate IDs")
    _reject_non_train_ids(record_ids)
    eligible_set = set(ordered_ids)
    record_set = set(record_ids)
    missing = sorted(eligible_set - record_set)
    extra = sorted(record_set - eligible_set)
    if missing or extra:
        raise GateBValidationError(
            "records must match the split-derived eligible partition exactly; "
            f"missing={missing[:5]!r}, extra={extra[:5]!r}"
        )
    return ordered_ids, {record.id: record for record in materialized}


def _validated_baseline_run(
    records: Iterable[DevelopmentBaselineRecord],
) -> tuple[DevelopmentBaselineRecord, ...]:
    materialized = tuple(records)
    if not materialized:
        raise GateBValidationError("development records must not be empty")
    if any(not isinstance(record, DevelopmentBaselineRecord) for record in materialized):
        raise TypeError("records must contain DevelopmentBaselineRecord instances")

    _reject_non_train_ids(tuple(record.problem_id for record in materialized))
    for index, record in enumerate(materialized, start=1):
        _validate_baseline_record(record, index)

    first = materialized[0]
    consistency_fields = (
        "model_id",
        "revision",
        "route",
        "checkpoint_sha256",
        "config_sha256",
        "decoding_policy",
        "split_version",
        "split_sha256",
        "source_groups_sha256",
        "fold",
        "partition",
        "split_partition",
        "eligibility_ids_sha256",
    )
    for index, record in enumerate(materialized[1:], start=2):
        changed = [
            field_name
            for field_name in consistency_fields
            if getattr(record, field_name) != getattr(first, field_name)
        ]
        if changed:
            raise GateBValidationError(
                f"development record {index} has inconsistent run provenance: {changed!r}"
            )

    identities = [(record.problem_id, record.sample_index) for record in materialized]
    if len(set(identities)) != len(identities):
        raise GateBValidationError("development records contain duplicate generation identities")
    sample_indices_by_id: defaultdict[str, set[int]] = defaultdict(set)
    for record in materialized:
        sample_indices_by_id[record.problem_id].add(record.sample_index)
    expected_samples = set(next(iter(sample_indices_by_id.values())))
    if expected_samples != set(range(len(expected_samples))) or any(
        indices != expected_samples for indices in sample_indices_by_id.values()
    ):
        raise GateBValidationError(
            "every problem must have the same contiguous sample indices starting at zero"
        )
    canonical_problem_ids = tuple(sample_indices_by_id)
    canonical_identities = tuple(
        (problem_id, sample_index)
        for problem_id in canonical_problem_ids
        for sample_index in range(len(expected_samples))
    )
    if tuple(identities) != canonical_identities:
        raise GateBValidationError(
            "development records must use exact canonical validation ID/sample order"
        )
    if _ids_sha256(canonical_problem_ids) != first.eligibility_ids_sha256:
        raise GateBValidationError(
            "development records do not match split-derived validation ID order/coverage "
            "(eligibility_ids_sha256)"
        )
    return materialized


def _validate_baseline_record(record: DevelopmentBaselineRecord, index: int) -> None:
    prefix = f"development record {index}"
    if record.model_id != OFFICIAL_MODEL_ID:
        raise GateBValidationError(f"{prefix} does not use the official fixed base model")
    if record.revision != PINNED_MODEL_REVISION:
        raise GateBValidationError(f"{prefix} does not use the pinned model revision")
    if record.route != "direct_answer":
        raise GateBValidationError(f"{prefix} is not a direct-answer baseline")
    if record.partition != "fold_validation":
        raise GateBValidationError(f"{prefix} is not from the development validation partition")
    if record.split_partition != SplitPartition.CROSS_VALIDATION.value:
        raise GateBValidationError(f"{prefix} is not from the cross-validation split")
    if isinstance(record.fold, bool) or not isinstance(record.fold, int) or record.fold < 0:
        raise GateBValidationError(f"{prefix} has an invalid fold")
    if (
        isinstance(record.sample_index, bool)
        or not isinstance(record.sample_index, int)
        or record.sample_index < 0
    ):
        raise GateBValidationError(f"{prefix} has an invalid sample_index")
    if isinstance(record.seed, bool) or not isinstance(record.seed, int) or record.seed < 0:
        raise GateBValidationError(f"{prefix} has an invalid seed")
    for field_name in (
        "question_sha256",
        "prompt_sha256",
        "checkpoint_sha256",
        "config_sha256",
        "raw_completion_sha256",
        "split_sha256",
        "source_groups_sha256",
        "eligibility_ids_sha256",
    ):
        _validated_sha256(getattr(record, field_name), f"{prefix} {field_name}")
    if _sha256_text(record.raw_completion) != record.raw_completion_sha256:
        raise GateBValidationError(f"{prefix} raw_completion_sha256 does not match")
    if parse_answer(record.raw_completion) != record.parse:
        raise GateBValidationError(f"{prefix} parser result does not match raw completion")
    if isinstance(record.reference_answer, bool) or not isinstance(record.reference_answer, int):
        raise GateBValidationError(f"{prefix} has an invalid reference answer")
    expected_exact_match = record.parse.ok and record.parse.value == record.reference_answer
    if not isinstance(record.exact_match, bool) or record.exact_match != expected_exact_match:
        raise GateBValidationError(f"{prefix} has an inconsistent exact-match result")
    if not math.isfinite(record.latency_ms) or record.latency_ms < 0:
        raise GateBValidationError(f"{prefix} latency_ms must be finite and non-negative")
    for field_name in ("input_token_count", "output_token_count"):
        value = getattr(record, field_name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise GateBValidationError(f"{prefix} {field_name} must be non-negative")
    peak = record.peak_vram_allocated_bytes
    if peak is not None and (
        isinstance(peak, bool) or not isinstance(peak, int) or peak < 0
    ):
        raise GateBValidationError(f"{prefix} peak VRAM must be non-negative or None")
    for field_name in ("finish_reason", "split_version", "group_id"):
        value = getattr(record, field_name)
        if not isinstance(value, str) or not value or value != value.strip():
            raise GateBValidationError(f"{prefix} {field_name} must be non-empty and trimmed")


def _run_manifest_payload(
    records: Sequence[DevelopmentBaselineRecord],
    *,
    records_filename: str,
    records_bytes: int,
    records_sha256: str,
    execution_evidence: DevelopmentExecutionEvidence,
) -> dict[str, object]:
    first = records[0]
    problem_ids = tuple(dict.fromkeys(record.problem_id for record in records))
    parser_counts = Counter(record.parse.status for record in records)
    finish_reason_counts = Counter(record.finish_reason for record in records)
    exact_match_count = sum(record.exact_match for record in records)
    peaks = [
        record.peak_vram_allocated_bytes
        for record in records
        if record.peak_vram_allocated_bytes is not None
    ]
    samples_per_problem = len(records) // len(problem_ids)
    return {
        "schema_version": "gate-b1-development-run-v2",
        "records_file": records_filename,
        "records_bytes": records_bytes,
        "records_sha256": records_sha256,
        "record_count": len(records),
        "problem_count": len(problem_ids),
        "samples_per_problem": samples_per_problem,
        "model_id": first.model_id,
        "revision": first.revision,
        "route": first.route,
        "checkpoint_sha256": first.checkpoint_sha256,
        "config_sha256": first.config_sha256,
        "decoding_policy": first.decoding_policy.as_dict(),
        "decoding_policy_sha256": first.decoding_policy.sha256,
        "split_version": first.split_version,
        "split_sha256": first.split_sha256,
        "source_groups_sha256": first.source_groups_sha256,
        "fold": first.fold,
        "partition": first.partition,
        "split_partition": first.split_partition,
        "eligibility_ids_sha256": first.eligibility_ids_sha256,
        "parser_status_counts": dict(sorted(parser_counts.items())),
        "finish_reason_counts": dict(sorted(finish_reason_counts.items())),
        "exact_match_count": exact_match_count,
        "exact_match_accuracy": exact_match_count / len(records),
        "input_token_count_total": sum(record.input_token_count for record in records),
        "output_token_count_total": sum(record.output_token_count for record in records),
        "peak_vram_allocated_bytes_max": max(peaks) if peaks else None,
        "execution_evidence": execution_evidence.as_dict(),
        "generation_evidence": _generation_evidence_payload(records),
    }


def _validated_execution_evidence(
    value: DevelopmentExecutionEvidence,
    records: Sequence[DevelopmentBaselineRecord],
) -> DevelopmentExecutionEvidence:
    if not isinstance(value, DevelopmentExecutionEvidence):
        raise TypeError("execution_evidence must be DevelopmentExecutionEvidence")
    if not records:  # pragma: no cover - baseline validation already rejects this
        raise GateBValidationError("execution evidence requires at least one record")
    first = records[0]
    if value.config_sha256 != first.config_sha256:
        raise GateBValidationError("execution evidence config_sha256 mismatches records")
    for field_name in (
        "config_file_sha256",
        "preflight_report_sha256",
        "gpu_smoke_report_sha256",
    ):
        _validated_sha256(getattr(value, field_name), f"execution evidence {field_name}")
    for field_name in (
        "config_path",
        "preflight_report_path",
        "gpu_smoke_report_path",
        "gpu_device_name",
    ):
        field_value = getattr(value, field_name)
        if (
            not isinstance(field_value, str)
            or not field_value.strip()
            or field_value != field_value.strip()
        ):
            raise GateBValidationError(
                f"execution evidence {field_name} must be a non-empty trimmed string"
            )
    source = value.source_manifest
    if not isinstance(source, SourceTreeArtifactEvidence):
        raise TypeError("execution evidence source_manifest is invalid")
    _validated_sha256(source.sha256, "execution evidence source manifest sha256")
    _validated_sha256(source.tree_sha256, "execution evidence source tree_sha256")
    if (
        isinstance(source.file_count, bool)
        or not isinstance(source.file_count, int)
        or source.file_count < 1
    ):
        raise GateBValidationError("execution evidence source manifest file_count is invalid")
    if not isinstance(source.path, str) or not source.path.strip():
        raise GateBValidationError("execution evidence source manifest path is invalid")
    return value


def _generation_evidence_payload(
    records: Sequence[DevelopmentBaselineRecord],
) -> dict[str, object]:
    """Summarize seed/prompt/latency evidence without copying raw generations."""

    ordered = tuple(sorted(records, key=lambda record: (record.problem_id, record.sample_index)))
    latencies = tuple(record.latency_ms for record in ordered)
    latency_total = math.fsum(latencies)
    return {
        "schema_version": "gate-b1-generation-evidence-v1",
        "seed_sequence_sha256": hashlib.sha256(
            canonical_json_bytes([record.seed for record in ordered])
        ).hexdigest(),
        "prompt_sha256_sequence_sha256": hashlib.sha256(
            canonical_json_bytes([record.prompt_sha256 for record in ordered])
        ).hexdigest(),
        "latency_ms": {
            "count": len(latencies),
            "total": latency_total,
            "min": min(latencies),
            "max": max(latencies),
            "mean": latency_total / len(latencies),
        },
    }


def _regular_file_identity(path: str | Path, label: str) -> tuple[str, str]:
    supplied = Path(path)
    if supplied.is_symlink():
        raise GateBValidationError(f"{label} must not be a symbolic link")
    source = supplied.resolve(strict=True)
    if not source.is_file():
        raise GateBValidationError(f"{label} must be a regular file: {source}")
    return str(source), sha256_file(source)


def _validate_config_file_semantic_sha(path: str, expected_sha256: str) -> None:
    """Ensure the post-generation config bytes still describe the runtime config."""

    try:
        payload = json.loads(
            Path(path).read_text(encoding="utf-8", errors="strict"),
            object_pairs_hook=_unique_config_json_object,
            parse_constant=_reject_config_json_constant,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise GateBValidationError(f"Gate B config file is invalid: {exc}") from exc
    if not isinstance(payload, dict):
        raise GateBValidationError("Gate B config file must contain one JSON object")
    stored_sha256 = payload.pop("config_sha256", None)
    if not isinstance(stored_sha256, str):
        raise GateBValidationError("Gate B config file is missing config_sha256")
    try:
        parsed = GateBConfig(**payload)
    except (TypeError, GateBValidationError) as exc:
        raise GateBValidationError(f"Gate B config file schema is invalid: {exc}") from exc
    if stored_sha256 != parsed.sha256 or parsed.sha256 != expected_sha256:
        raise GateBValidationError(
            "Gate B config file semantic SHA does not match the runtime config"
        )


def _unique_config_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON key {key!r}")
        payload[key] = value
    return payload


def _reject_config_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON numeric constant {value!r}")


def _validated_artifact_targets(
    jsonl_path: str | Path, manifest_path: str | Path
) -> tuple[Path, Path]:
    records_target = Path(jsonl_path).resolve(strict=False)
    manifest_target = Path(manifest_path).resolve(strict=False)
    if records_target == manifest_target:
        raise GateBValidationError("JSONL and manifest paths must be different")
    if records_target.parent != manifest_target.parent:
        raise GateBValidationError("JSONL and manifest must share one directory")
    parent = records_target.parent
    if not parent.is_dir():
        raise GateBValidationError(f"artifact directory does not exist: {parent}")
    if records_target.name in {"", ".", ".."} or manifest_target.name in {"", ".", ".."}:
        raise GateBValidationError("artifact paths must name regular files")
    return records_target, manifest_target


def _publish_atomic_pair(
    records_target: Path,
    records_bytes: bytes,
    manifest_target: Path,
    manifest_bytes: bytes,
) -> None:
    """Publish a no-overwrite pair with SIGKILL-safe records-orphan recovery.

    POSIX has no atomic operation that creates two arbitrary filenames at once.
    The records JSONL is therefore linked first and the manifest is the commit
    marker.  If a process dies between the two links, a later invocation may
    finish only that records-only orphan after byte-for-byte comparison with the
    newly validated payload.  A different existing file, an existing manifest,
    or a complete pair remains a hard no-overwrite failure.
    """

    parent = records_target.parent
    name_digest = hashlib.sha256(
        f"{records_target.name}\0{manifest_target.name}".encode()
    ).hexdigest()[:16]
    lock_path = parent / f".gate-b-{name_digest}.lock"
    lock_fd: int | None = None
    records_temp: Path | None = None
    manifest_temp: Path | None = None
    records_published = False
    manifest_published = False
    try:
        records_temp = _write_fsynced_temp(parent, records_target.name, records_bytes)
        manifest_temp = _write_fsynced_temp(parent, manifest_target.name, manifest_bytes)
        lock_fd = _acquire_development_artifact_pair_lock(
            lock_path,
            records_target=records_target,
            manifest_target=manifest_target,
        )
        records_exists = _artifact_target_exists(records_target)
        manifest_exists = _artifact_target_exists(manifest_target)
        if records_exists and not manifest_exists:
            _validate_recoverable_records_orphan(records_target, records_bytes)
            try:
                os.link(manifest_temp, manifest_target)
                manifest_published = True
                _fsync_directory(parent)
            except FileExistsError as exc:
                raise GateBArtifactExistsError(
                    "a Gate B manifest destination appeared during orphan recovery"
                ) from exc
            except OSError as exc:
                raise GateBValidationError(
                    f"cannot complete interrupted Gate B artifact publication: {exc}"
                ) from exc
            return
        if records_exists or manifest_exists:
            raise GateBArtifactExistsError(
                "refusing to overwrite existing Gate B artifact: "
                f"{records_target if records_exists else manifest_target}"
            )
        try:
            os.link(records_temp, records_target)
            records_published = True
            os.link(manifest_temp, manifest_target)
            manifest_published = True
            _fsync_directory(parent)
        except FileExistsError as exc:
            raise GateBArtifactExistsError(
                "a Gate B artifact destination appeared during atomic publication"
            ) from exc
        except OSError as exc:
            raise GateBValidationError(
                f"cannot atomically publish Gate B artifacts: {exc}"
            ) from exc
    except BaseException:
        if manifest_published and manifest_temp is not None:
            _unlink_if_same_file(manifest_target, manifest_temp)
        if records_published and records_temp is not None:
            _unlink_if_same_file(records_target, records_temp)
        _fsync_directory(parent)
        raise
    finally:
        for temporary in (records_temp, manifest_temp):
            if temporary is not None:
                with suppress(FileNotFoundError):
                    temporary.unlink()
        if lock_fd is not None:
            _release_development_artifact_pair_lock(lock_path, lock_fd)
        _fsync_directory(parent)


def _artifact_target_exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _validate_recoverable_records_orphan(target: Path, expected: bytes) -> None:
    """Accept only this exact invocation's records-only interrupted state."""

    if target.is_symlink() or not target.is_file():
        raise GateBArtifactExistsError(
            f"refusing to recover a non-regular Gate B records artifact: {target}"
        )
    try:
        actual = target.read_bytes()
    except OSError as exc:
        raise GateBValidationError(
            f"cannot inspect interrupted Gate B records artifact: {exc}"
        ) from exc
    if actual != expected:
        raise GateBArtifactExistsError(
            "refusing to overwrite a different Gate B records artifact after interruption"
        )


def _acquire_development_artifact_pair_lock(
    lock_path: Path,
    *,
    records_target: Path,
    manifest_target: Path,
) -> int:
    """Acquire a fully-written lock and reclaim only a verified dead-owner lock."""

    payload = {
        "schema_version": _DEVELOPMENT_ARTIFACT_PAIR_LOCK_SCHEMA,
        "records_file": records_target.name,
        "manifest_file": manifest_target.name,
        "process_id": os.getpid(),
    }
    parent = lock_path.parent
    for _ in range(2):
        temporary = _write_fsynced_temp(
            parent,
            lock_path.name,
            canonical_json_bytes(payload),
        )
        descriptor: int | None = None
        try:
            descriptor = os.open(
                temporary,
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
            )
            try:
                os.link(temporary, lock_path)
            except FileExistsError:
                os.close(descriptor)
                descriptor = None
                _recover_stale_development_artifact_pair_lock(
                    lock_path,
                    records_target=records_target,
                    manifest_target=manifest_target,
                )
                continue
            _fsync_directory(parent)
            assert descriptor is not None  # local acquisition invariant
            acquired = descriptor
            descriptor = None
            return acquired
        finally:
            if descriptor is not None:
                os.close(descriptor)
            with suppress(FileNotFoundError):
                temporary.unlink()
    raise GateBArtifactExistsError("artifact writer lock changed while recovering stale state")


def _recover_stale_development_artifact_pair_lock(
    lock_path: Path,
    *,
    records_target: Path,
    manifest_target: Path,
) -> None:
    payload = _read_strict_json_object(lock_path, "development artifact writer lock")
    expected = {
        "schema_version": _DEVELOPMENT_ARTIFACT_PAIR_LOCK_SCHEMA,
        "records_file": records_target.name,
        "manifest_file": manifest_target.name,
    }
    if any(payload.get(key) != value for key, value in expected.items()) or set(payload) != {
        *expected,
        "process_id",
    }:
        raise GateBValidationError("development artifact writer lock is incompatible")
    process_id = payload.get("process_id")
    if isinstance(process_id, bool) or not isinstance(process_id, int) or process_id <= 0:
        raise GateBValidationError("development artifact writer lock process ID is invalid")
    if _process_is_alive(process_id):
        raise GateBArtifactExistsError(
            f"artifact writer lock is active for process {process_id}"
        )
    with suppress(FileNotFoundError):
        lock_path.unlink()
    _fsync_directory(lock_path.parent)


def _release_development_artifact_pair_lock(lock_path: Path, descriptor: int) -> None:
    try:
        held = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        current = lock_path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return
    if (held.st_dev, held.st_ino) == (current.st_dev, current.st_ino):
        lock_path.unlink()
        _fsync_directory(lock_path.parent)


def _write_fsynced_temp(parent: Path, target_name: str, payload: bytes) -> Path:
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{target_name}.", suffix=".tmp", dir=parent
    )
    path = Path(raw_path)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        with suppress(FileNotFoundError):
            path.unlink()
        raise
    return path


def _unlink_if_same_file(target: Path, temporary: Path) -> None:
    try:
        target_stat = target.stat(follow_symlinks=False)
        temporary_stat = temporary.stat(follow_symlinks=False)
    except FileNotFoundError:
        return
    if (target_stat.st_dev, target_stat.st_ino) == (
        temporary_stat.st_dev,
        temporary_stat.st_ino,
    ):
        target.unlink()


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _reject_non_train_ids(problem_ids: Sequence[str]) -> None:
    invalid = sorted(
        problem_id for problem_id in problem_ids if _TRAIN_ID_RE.fullmatch(problem_id) is None
    )
    if invalid:
        raise GateBValidationError(
            "organizer-only Gate B inputs require train-XXXXXX IDs; "
            f"leaderboard/test-like IDs are forbidden: {invalid[:5]!r}"
        )


def _required_reference_answer(record: MathRecord) -> int:
    if isinstance(record.answer, bool) or not isinstance(record.answer, int):
        raise GateBValidationError(f"{record.id}: organizer train answer is missing or invalid")
    if not record.question_raw or not record.question_raw.strip():
        raise GateBValidationError(f"{record.id}: organizer train question is empty")
    return record.answer


def _prompt_messages(question: str, config: GateBConfig) -> tuple[ChatMessage, ...]:
    return (ChatMessage("system", config.system_prompt), ChatMessage("user", question))


def _token_ids(values: Sequence[int], *, field_name: str) -> tuple[int, ...]:
    try:
        token_ids = tuple(values)
    except TypeError as exc:
        raise GateBValidationError(f"{field_name} must be a sequence of integers") from exc
    if not token_ids:
        raise GateBValidationError(f"{field_name} must not be empty")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in token_ids
    ):
        raise GateBValidationError(f"{field_name} must contain non-negative integers")
    return token_ids


def _validated_sha256(value: str, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise GateBValidationError(f"{field_name} must be a 64-character SHA-256 hex digest")
    return value.lower()


def _ids_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256(canonical_json_bytes(list(values))).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


__all__ = [
    "ChatMessage",
    "ChatTokenizer",
    "DEFAULT_GATE_B_CONFIG",
    "DecodingPolicy",
    "DevelopmentArtifactWriteResult",
    "DevelopmentBaselineRecord",
    "DevelopmentExecutionEvidence",
    "DevelopmentResumeStatus",
    "DirectAnswerSFTExample",
    "EncodedSFTExample",
    "GateBArtifactExistsError",
    "GateBConfig",
    "GateBPreflightRequiredError",
    "GateBValidationError",
    "GenerationBackend",
    "GenerationRequest",
    "GenerationResult",
    "PINNED_MODEL_REVISION",
    "TransformersGenerationBackend",
    "build_concise_rationale_sft_examples",
    "build_direct_answer_sft_examples",
    "create_development_execution_evidence",
    "encode_response_only_example",
    "run_development_baseline",
    "read_development_resume_status",
    "write_development_artifacts",
]

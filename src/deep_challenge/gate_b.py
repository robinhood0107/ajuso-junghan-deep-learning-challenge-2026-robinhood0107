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
from .provenance import canonical_json_bytes
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
) -> tuple[DevelopmentBaselineRecord, ...]:
    """Generate from exactly one eligible development-validation fold.

    There is no retry, fallback answer, or parser coercion.  Empty, invalid, and
    conflicting generations are retained verbatim.  The full train table and
    locked holdout are rejected because ``records`` must exactly equal the IDs
    derived by :func:`eligible_validation_ids`.
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

    output: list[DevelopmentBaselineRecord] = []
    total_generations = len(ordered_ids) * samples_per_problem
    completed_generations = 0
    decoding_policy = config.decoding_policy
    for problem_id in ordered_ids:
        record = records_by_id[problem_id]
        reference_answer = _required_reference_answer(record)
        messages = _prompt_messages(record.question_raw, config)
        prompt_sha256 = hashlib.sha256(
            canonical_json_bytes(
                {
                    "messages": [message.as_dict() for message in messages],
                    "add_generation_prompt": True,
                }
            )
        ).hexdigest()
        for sample_index in range(samples_per_problem):
            seed = deterministic_seed(
                problem_id,
                sample_index,
                salt=f"gate-b1-v2:{config.seed}:{config.sha256}",
            )
            request = GenerationRequest(
                problem_id=problem_id,
                messages=messages,
                seed=seed,
                sample_index=sample_index,
                model_id=config.model_id,
                revision=config.revision,
                route=config.route,
                prompt_sha256=prompt_sha256,
                config_sha256=config.sha256,
                decoding_policy=decoding_policy,
            )
            started = clock_ns()
            generation = backend.generate(request)
            finished = clock_ns()
            if not isinstance(generation, GenerationResult):
                raise TypeError(
                    "GenerationBackend.generate() must return a GenerationResult"
                )
            if finished < started:
                raise GateBValidationError("monotonic clock moved backwards during generation")
            latency_ms = (finished - started) / 1_000_000
            if not math.isfinite(latency_ms):  # pragma: no cover - integer clock invariant
                raise GateBValidationError("generation latency must be finite")
            parsed = parse_answer(generation.text)
            output.append(
                DevelopmentBaselineRecord(
                    problem_id=problem_id,
                    question_sha256=_sha256_text(record.question_raw),
                    route=config.route,
                    sample_index=sample_index,
                    seed=seed,
                    model_id=config.model_id,
                    revision=config.revision,
                    prompt_sha256=prompt_sha256,
                    checkpoint_sha256=checkpoint_digest,
                    config_sha256=config.sha256,
                    decoding_policy=decoding_policy,
                    raw_completion=generation.text,
                    raw_completion_sha256=_sha256_text(generation.text),
                    finish_reason=generation.finish_reason,
                    input_token_count=generation.input_token_count,
                    output_token_count=generation.output_token_count,
                    parse=parsed,
                    reference_answer=reference_answer,
                    exact_match=parsed.ok and parsed.value == reference_answer,
                    latency_ms=latency_ms,
                    peak_vram_allocated_bytes=generation.peak_vram_allocated_bytes,
                    group_id=provenance.group_ids[problem_id],
                    **provenance.common,
                )
            )
            completed_generations += 1
            if progress_callback is not None:
                progress_callback(completed_generations, total_generations)
    return tuple(output)


def write_development_artifacts(
    records: Iterable[DevelopmentBaselineRecord],
    *,
    jsonl_path: str | Path,
    manifest_path: str | Path,
) -> DevelopmentArtifactWriteResult:
    """Atomically publish no-overwrite JSONL records and a checksum manifest.

    Both files must share an existing directory.  Fully written, fsynced
    temporary files are hard-linked into their final names under an exclusive
    bundle lock.  Hard-link creation fails when a destination already exists,
    so neither a pre-check race nor a concurrent writer can overwrite evidence.
    """

    materialized = _validated_baseline_run(records)
    records_bytes = (
        "".join(f"{record.to_json_line()}\n" for record in materialized).encode("utf-8")
    )
    records_digest = hashlib.sha256(records_bytes).hexdigest()
    manifest_payload = _run_manifest_payload(
        materialized,
        records_filename=Path(jsonl_path).name,
        records_bytes=len(records_bytes),
        records_sha256=records_digest,
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
class _PartitionProvenance:
    common: Mapping[str, object]
    group_ids: Mapping[str, str]


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
    if _ids_sha256(tuple(sample_indices_by_id)) != first.eligibility_ids_sha256:
        raise GateBValidationError(
            "development records do not match their eligibility_ids_sha256"
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
        "schema_version": "gate-b1-development-run-v1",
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
    }


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
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        try:
            lock_fd = os.open(lock_path, flags, 0o600)
        except FileExistsError as exc:
            raise GateBArtifactExistsError(
                f"artifact writer lock already exists: {lock_path}"
            ) from exc
        if records_target.exists() or manifest_target.exists():
            raise GateBArtifactExistsError(
                "refusing to overwrite existing Gate B artifact: "
                f"{records_target if records_target.exists() else manifest_target}"
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
        if lock_fd is not None:
            os.close(lock_fd)
        for temporary in (records_temp, manifest_temp):
            if temporary is not None:
                with suppress(FileNotFoundError):
                    temporary.unlink()
        if lock_fd is not None:
            with suppress(FileNotFoundError):
                lock_path.unlink()
        _fsync_directory(parent)


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
    "build_direct_answer_sft_examples",
    "encode_response_only_example",
    "run_development_baseline",
    "write_development_artifacts",
]

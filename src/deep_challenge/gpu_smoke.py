"""Explicit, fail-closed final GPU smoke test for the fixed competition model.

Importing this module is CPU-only: PyTorch, Transformers, PEFT, and
bitsandbytes are imported only inside :class:`TransformersNF4SmokeRuntime`
when its ``execute`` method is explicitly called.  The public orchestrator
accepts no caller-supplied problem text.  It can therefore run only the fixed,
locally-authored synthetic arithmetic prompt below, never organizer
leaderboard/test content.
"""

from __future__ import annotations

import gc
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import tempfile
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from .answers import AnswerParseResult, parse_answer
from .gate_b import DEFAULT_GATE_B_CONFIG, PINNED_MODEL_REVISION
from .model_preflight import (
    DEFAULT_NF4_MIN_FREE_VRAM_MIB,
    OFFICIAL_MODEL_ID,
    _physical_nvidia_report,
)
from .provenance import canonical_json_bytes, sha256_file

SYNTHETIC_SMOKE_SYSTEM_PROMPT = (
    "You are performing a local arithmetic runtime check. "
    "Return exactly one line in the form: Final answer: <integer>."
)
SYNTHETIC_SMOKE_USER_PROMPT = (
    "Synthetic arithmetic runtime check: add the integers 2 and 3. "
    "This prompt was authored solely for local hardware validation."
)
SYNTHETIC_SMOKE_TARGET = "Final answer: 5"
SYNTHETIC_SMOKE_EXPECTED_ANSWER = 5

_REQUIRED_PACKAGE_NAMES = (
    "torch",
    "transformers",
    "accelerate",
    "peft",
    "bitsandbytes",
    "triton",
)
_MIB = 1024 * 1024
GPU_TOTAL_MEMORY_TOLERANCE_MIB = 64


class GpuSmokeValidationError(ValueError):
    """Raised before publishing when a smoke-test invariant is not met."""


class GpuSmokePreflightError(RuntimeError):
    """Raised when the prerequisite preflight report is not fully green."""


@dataclass(frozen=True, slots=True)
class GpuSmokeConfig:
    """Immutable single-GPU NF4/LoRA smoke profile.

    Every safety-sensitive value is locked.  A materially different experiment
    must use a separately versioned implementation rather than mutating this
    last-gate smoke contract.  The top-level artifact schema remains v1 because
    its structure is backward-compatible; adding the locked Gate B optimizer and
    checkpointing fields intentionally rotates ``config_sha256``, so an older
    smoke artifact cannot authorize the strengthened runtime.
    """

    schema_version: str = "gate-b-final-gpu-smoke-v1"
    model_id: str = OFFICIAL_MODEL_ID
    revision: str = PINNED_MODEL_REVISION
    system_prompt: str = SYNTHETIC_SMOKE_SYSTEM_PROMPT
    user_prompt: str = SYNTHETIC_SMOKE_USER_PROMPT
    target_text: str = SYNTHETIC_SMOKE_TARGET
    expected_answer: int = SYNTHETIC_SMOKE_EXPECTED_ANSWER
    quantization: str = "nf4"
    double_quantization: bool = True
    compute_dtype: str = "bfloat16"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_target_modules: str = "all-linear"
    lora_dropout: float = DEFAULT_GATE_B_CONFIG.lora_dropout
    gradient_checkpointing: bool = DEFAULT_GATE_B_CONFIG.gradient_checkpointing
    gradient_checkpointing_use_reentrant: bool = (
        DEFAULT_GATE_B_CONFIG.gradient_checkpointing_use_reentrant
    )
    optimizer: str = DEFAULT_GATE_B_CONFIG.optimizer
    learning_rate: float = DEFAULT_GATE_B_CONFIG.learning_rate
    response_only_loss: bool = True
    max_new_tokens: int = 16
    generation_use_cache: bool = True
    seed: int = 20_260_804
    minimum_free_vram_mib: int = DEFAULT_NF4_MIN_FREE_VRAM_MIB
    maximum_preexisting_used_vram_mib: int = 1_024

    def __post_init__(self) -> None:
        expected_values: tuple[tuple[str, object, object], ...] = (
            ("schema_version", self.schema_version, "gate-b-final-gpu-smoke-v1"),
            ("model_id", self.model_id, OFFICIAL_MODEL_ID),
            ("revision", self.revision, PINNED_MODEL_REVISION),
            ("system_prompt", self.system_prompt, SYNTHETIC_SMOKE_SYSTEM_PROMPT),
            ("user_prompt", self.user_prompt, SYNTHETIC_SMOKE_USER_PROMPT),
            ("target_text", self.target_text, SYNTHETIC_SMOKE_TARGET),
            ("expected_answer", self.expected_answer, SYNTHETIC_SMOKE_EXPECTED_ANSWER),
            ("quantization", self.quantization, "nf4"),
            ("double_quantization", self.double_quantization, True),
            ("compute_dtype", self.compute_dtype, "bfloat16"),
            ("lora_rank", self.lora_rank, 16),
            ("lora_alpha", self.lora_alpha, 32),
            ("lora_target_modules", self.lora_target_modules, "all-linear"),
            ("lora_dropout", self.lora_dropout, DEFAULT_GATE_B_CONFIG.lora_dropout),
            (
                "gradient_checkpointing",
                self.gradient_checkpointing,
                DEFAULT_GATE_B_CONFIG.gradient_checkpointing,
            ),
            (
                "gradient_checkpointing_use_reentrant",
                self.gradient_checkpointing_use_reentrant,
                DEFAULT_GATE_B_CONFIG.gradient_checkpointing_use_reentrant,
            ),
            ("optimizer", self.optimizer, DEFAULT_GATE_B_CONFIG.optimizer),
            ("learning_rate", self.learning_rate, DEFAULT_GATE_B_CONFIG.learning_rate),
            ("response_only_loss", self.response_only_loss, True),
            ("max_new_tokens", self.max_new_tokens, 16),
            ("generation_use_cache", self.generation_use_cache, True),
            ("seed", self.seed, 20_260_804),
            (
                "minimum_free_vram_mib",
                self.minimum_free_vram_mib,
                DEFAULT_NF4_MIN_FREE_VRAM_MIB,
            ),
            (
                "maximum_preexisting_used_vram_mib",
                self.maximum_preexisting_used_vram_mib,
                1_024,
            ),
        )
        for field_name, value, expected in expected_values:
            if type(value) is not type(expected) or value != expected:
                raise GpuSmokeValidationError(
                    f"{field_name} is locked to exact {type(expected).__name__} "
                    f"value {expected!r} for the final GPU smoke"
                )

    def as_dict(self) -> dict[str, object]:
        """Return the complete canonical configuration."""

        return asdict(self)

    @property
    def sha256(self) -> str:
        """Return the content-derived configuration identity."""

        return hashlib.sha256(canonical_json_bytes(self.as_dict())).hexdigest()


DEFAULT_GPU_SMOKE_CONFIG = GpuSmokeConfig()


@dataclass(frozen=True, slots=True)
class GpuSmokeRequest:
    """Fully fixed request passed through the injectable runtime boundary."""

    model_id: str
    revision: str
    system_prompt: str
    user_prompt: str
    target_text: str
    expected_answer: int
    seed: int
    max_new_tokens: int
    generation_use_cache: bool
    minimum_free_vram_mib: int
    maximum_preexisting_used_vram_mib: int
    lora_rank: int
    lora_alpha: int
    lora_target_modules: str
    lora_dropout: float
    gradient_checkpointing: bool
    gradient_checkpointing_use_reentrant: bool
    optimizer: str
    learning_rate: float

    @classmethod
    def from_config(cls, config: GpuSmokeConfig) -> GpuSmokeRequest:
        """Build the sole permitted synthetic request."""

        return cls(
            model_id=config.model_id,
            revision=config.revision,
            system_prompt=config.system_prompt,
            user_prompt=config.user_prompt,
            target_text=config.target_text,
            expected_answer=config.expected_answer,
            seed=config.seed,
            max_new_tokens=config.max_new_tokens,
            generation_use_cache=config.generation_use_cache,
            minimum_free_vram_mib=config.minimum_free_vram_mib,
            maximum_preexisting_used_vram_mib=config.maximum_preexisting_used_vram_mib,
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            gradient_checkpointing=config.gradient_checkpointing,
            gradient_checkpointing_use_reentrant=(
                config.gradient_checkpointing_use_reentrant
            ),
            optimizer=config.optimizer,
            learning_rate=config.learning_rate,
        )


@dataclass(frozen=True, slots=True)
class GpuSmokeRuntimeEvidence:
    """Raw evidence returned by a real or CPU-fake runtime adapter.

    ``pre_context_physical_*`` comes from a read-only ``nvidia-smi`` probe
    before this process creates a CUDA context.  The later CUDA free-memory
    fields deliberately include the process's own context overhead, so they
    prove usable capacity but cannot safely be interpreted as external use.
    """

    package_versions: Mapping[str, str]
    cuda_device_count: int
    device_index: int
    device_name: str
    pre_context_physical_device_name: str
    pre_context_physical_total_bytes: int
    pre_context_physical_used_bytes: int
    pre_context_physical_free_bytes: int
    physical_total_bytes: int
    physical_free_bytes_before: int
    physical_free_bytes_after_cleanup: int
    allocated_bytes_before: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    training_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    training_loss: float
    optimizer_name: str
    optimizer_step_count: int
    raw_generation: str

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe evidence representation."""

        return {
            "package_versions": dict(sorted(self.package_versions.items())),
            "cuda_device_count": self.cuda_device_count,
            "device_index": self.device_index,
            "device_name": self.device_name,
            "pre_context_physical_device_name": self.pre_context_physical_device_name,
            "pre_context_physical_total_bytes": self.pre_context_physical_total_bytes,
            "pre_context_physical_used_bytes": self.pre_context_physical_used_bytes,
            "pre_context_physical_free_bytes": self.pre_context_physical_free_bytes,
            "physical_total_bytes": self.physical_total_bytes,
            "physical_free_bytes_before": self.physical_free_bytes_before,
            "physical_free_bytes_after_cleanup": self.physical_free_bytes_after_cleanup,
            "allocated_bytes_before": self.allocated_bytes_before,
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "peak_reserved_bytes": self.peak_reserved_bytes,
            "training_latency_ms": self.training_latency_ms,
            "generation_latency_ms": self.generation_latency_ms,
            "total_latency_ms": self.total_latency_ms,
            "training_loss": self.training_loss,
            "optimizer_name": self.optimizer_name,
            "optimizer_step_count": self.optimizer_step_count,
            "raw_generation": self.raw_generation,
        }


class GpuSmokeRuntime(Protocol):
    """Runtime seam that permits complete CPU-only orchestration tests."""

    def execute(self, request: GpuSmokeRequest) -> GpuSmokeRuntimeEvidence: ...


def _live_physical_gpu_before_cuda_context(
    request: GpuSmokeRequest,
) -> dict[str, object]:
    """Fail closed on physical occupancy before this process opens CUDA.

    The fixed 1,024 MiB ceiling is a limit for memory already in use by the
    host or another process.  A CUDA context created by this smoke can itself
    consume memory, especially under WSL, so this probe must happen before
    importing or touching the CUDA runtime.  The later CUDA query still proves
    that enough capacity remains after context creation.
    """

    report = _physical_nvidia_report()
    if report.get("probe_succeeded") is not True or report.get("available") is not True:
        raise GpuSmokePreflightError(
            "live nvidia-smi physical GPU probe is unavailable before CUDA context"
        )
    devices = report.get("devices")
    if report.get("device_count") != 1 or not isinstance(devices, list) or len(devices) != 1:
        raise GpuSmokePreflightError(
            "live nvidia-smi probe must report exactly one physical GPU before CUDA context"
        )
    device = devices[0]
    if not isinstance(device, Mapping) or type(device.get("index")) is not int:
        raise GpuSmokePreflightError("live nvidia-smi physical GPU entry is invalid")
    if device["index"] != 0:
        raise GpuSmokePreflightError("live nvidia-smi physical GPU index must be zero")
    name = device.get("name")
    if not isinstance(name, str) or not name.strip():
        raise GpuSmokePreflightError("live nvidia-smi physical GPU name is missing")

    values: dict[str, int] = {}
    for field_name in ("memory_total_mib", "memory_used_mib", "memory_free_mib"):
        value = device.get(field_name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise GpuSmokePreflightError(
                f"live nvidia-smi physical {field_name} is invalid"
            )
        values[field_name] = value
    total_mib = values["memory_total_mib"]
    used_mib = values["memory_used_mib"]
    free_mib = values["memory_free_mib"]
    if total_mib <= 0 or used_mib > total_mib or free_mib > total_mib:
        raise GpuSmokePreflightError("live nvidia-smi physical VRAM counters are invalid")
    if used_mib + free_mib > total_mib:
        raise GpuSmokePreflightError(
            "live nvidia-smi physical VRAM counters are inconsistent"
        )
    if free_mib < request.minimum_free_vram_mib:
        raise GpuSmokePreflightError(
            "GPU free VRAM fell below the final smoke threshold before CUDA context"
        )
    if used_mib > request.maximum_preexisting_used_vram_mib:
        raise GpuSmokePreflightError(
            "GPU is occupied above the final smoke allowance before CUDA context"
        )
    return {
        "name": name,
        "total_bytes": total_mib * _MIB,
        "used_bytes": used_mib * _MIB,
        "free_bytes": free_mib * _MIB,
    }


@dataclass(frozen=True, slots=True)
class GpuSmokeWriteResult:
    """Checksums and path of one atomically published green artifact."""

    path: Path
    size_bytes: int
    sha256: str
    payload_sha256: str


class TransformersNF4SmokeRuntime:
    """Lazy real adapter for one pinned, local-only NF4 QLoRA smoke.

    Calling :meth:`execute` is the only operation in this module that imports
    or exercises the model stack and CUDA.  It performs one response-only
    forward/backward plus paged AdamW8bit optimizer step and one deterministic
    generation, then drops every model/tensor reference and empties the CUDA
    allocator cache.
    """

    def execute(self, request: GpuSmokeRequest) -> GpuSmokeRuntimeEvidence:
        _validate_runtime_request(request)

        # This must remain ahead of the first CUDA runtime interaction.  A
        # CUDA context has its own overhead and is not evidence of a foreign
        # process occupying the GPU.
        pre_context_physical_gpu = _live_physical_gpu_before_cuda_context(request)

        # These imports are intentionally local.  Merely importing
        # deep_challenge.gpu_smoke cannot initialize CUDA or load native kernels.
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
        peft = importlib.import_module("peft")
        bitsandbytes = importlib.import_module("bitsandbytes")
        importlib.import_module("accelerate")
        importlib.import_module("triton")

        package_versions = _installed_package_versions()
        if not bool(torch.cuda.is_available()):
            raise GpuSmokePreflightError("CUDA became unavailable after the green preflight")
        cuda_device_count = int(torch.cuda.device_count())
        if cuda_device_count != 1:
            raise GpuSmokePreflightError(
                "the final smoke requires exactly one CUDA-visible GPU; "
                f"found {cuda_device_count}"
            )
        if not bool(torch.cuda.is_bf16_supported()):
            raise GpuSmokePreflightError("the sole CUDA device does not support BF16")

        device_index = 0
        torch.cuda.set_device(device_index)
        free_before, total_before = (
            int(value) for value in torch.cuda.mem_get_info(device_index)
        )
        if free_before < request.minimum_free_vram_mib * _MIB:
            raise GpuSmokePreflightError(
                "GPU free VRAM fell below the final smoke threshold after preflight"
            )

        model: Any = None
        tokenizer: Any = None
        input_ids: Any = None
        attention_mask: Any = None
        labels: Any = None
        outputs: Any = None
        loss: Any = None
        optimizer: Any = None
        generation_ids: Any = None
        generation_mask: Any = None
        generated: Any = None
        raw_generation: str | None = None
        training_loss: float | None = None
        training_latency_ms: float | None = None
        generation_latency_ms: float | None = None
        total_started = time.perf_counter_ns()
        execution_succeeded = False
        allocated_bytes_before = int(torch.cuda.memory_allocated(device_index))
        peak_allocated_bytes = 0
        peak_reserved_bytes = 0
        optimizer_step_count = 0
        free_after = 0

        try:
            torch.manual_seed(request.seed)
            torch.cuda.manual_seed_all(request.seed)
            torch.cuda.reset_peak_memory_stats(device_index)

            quantization_config = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                request.model_id,
                revision=request.revision,
                local_files_only=True,
                trust_remote_code=False,
            )
            model = transformers.AutoModelForCausalLM.from_pretrained(
                request.model_id,
                revision=request.revision,
                local_files_only=True,
                trust_remote_code=False,
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16,
                device_map={"": device_index},
            )
            model = peft.prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=request.gradient_checkpointing,
                gradient_checkpointing_kwargs={
                    "use_reentrant": request.gradient_checkpointing_use_reentrant
                },
            )
            model.config.use_cache = False
            lora_config = peft.LoraConfig(
                r=request.lora_rank,
                lora_alpha=request.lora_alpha,
                target_modules=request.lora_target_modules,
                lora_dropout=request.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = peft.get_peft_model(model, lora_config)
            trainable_parameters = [
                parameter for parameter in model.parameters() if parameter.requires_grad
            ]
            if not trainable_parameters:
                raise GpuSmokeValidationError("LoRA smoke produced no trainable parameters")
            optimizer = bitsandbytes.optim.PagedAdamW8bit(
                trainable_parameters,
                lr=request.learning_rate,
            )

            prompt_messages = [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ]
            full_messages = [
                *prompt_messages,
                {"role": "assistant", "content": request.target_text},
            ]
            prompt_ids = _chat_token_ids(
                tokenizer,
                prompt_messages,
                add_generation_prompt=True,
            )
            full_ids = _chat_token_ids(
                tokenizer,
                full_messages,
                add_generation_prompt=False,
            )
            if len(prompt_ids) >= len(full_ids) or full_ids[: len(prompt_ids)] != prompt_ids:
                raise GpuSmokeValidationError(
                    "tokenizer chat template does not preserve the response-only boundary"
                )

            input_ids = torch.tensor([full_ids], dtype=torch.long, device="cuda:0")
            attention_mask = torch.ones_like(input_ids)
            labels = input_ids.clone()
            labels[:, : len(prompt_ids)] = -100

            model.train()
            optimizer.zero_grad(set_to_none=True)
            training_started = time.perf_counter_ns()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            if loss is None or not bool(torch.isfinite(loss).item()):
                raise GpuSmokeValidationError("the tiny training smoke returned non-finite loss")
            loss.backward()
            optimizer.step()
            optimizer_step_count = 1
            torch.cuda.synchronize(device_index)
            training_latency_ms = (time.perf_counter_ns() - training_started) / 1_000_000
            training_loss = float(loss.detach().cpu().item())

            optimizer.zero_grad(set_to_none=True)
            outputs = None
            loss = None
            optimizer = None
            input_ids = None
            attention_mask = None
            labels = None

            generation_ids = torch.tensor(
                [prompt_ids],
                dtype=torch.long,
                device="cuda:0",
            )
            generation_mask = torch.ones_like(generation_ids)
            model.eval()
            generation_started = time.perf_counter_ns()
            with torch.no_grad():
                generated = model.generate(
                    input_ids=generation_ids,
                    attention_mask=generation_mask,
                    do_sample=False,
                    max_new_tokens=request.max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=request.generation_use_cache,
                )
            torch.cuda.synchronize(device_index)
            generation_latency_ms = (
                time.perf_counter_ns() - generation_started
            ) / 1_000_000
            suffix = generated[0, generation_ids.shape[1] :].detach().cpu().tolist()
            raw_generation = tokenizer.decode(suffix, skip_special_tokens=True).strip()
            if not raw_generation:
                raise GpuSmokeValidationError("deterministic generation returned empty text")

            peak_allocated_bytes = int(torch.cuda.max_memory_allocated(device_index))
            peak_reserved_bytes = int(torch.cuda.max_memory_reserved(device_index))
            execution_succeeded = True
        finally:
            generated = None
            generation_mask = None
            generation_ids = None
            outputs = None
            loss = None
            optimizer = None
            labels = None
            attention_mask = None
            input_ids = None
            tokenizer = None
            model = None
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device_index)
            if execution_succeeded:
                free_after, _ = (
                    int(value) for value in torch.cuda.mem_get_info(device_index)
                )

        total_latency_ms = (time.perf_counter_ns() - total_started) / 1_000_000
        if (
            raw_generation is None
            or training_loss is None
            or training_latency_ms is None
            or generation_latency_ms is None
        ):  # pragma: no cover - guarded by execution_succeeded and exceptions
            raise GpuSmokeValidationError("runtime evidence was incomplete")
        return GpuSmokeRuntimeEvidence(
            package_versions=package_versions,
            cuda_device_count=cuda_device_count,
            device_index=device_index,
            device_name=str(torch.cuda.get_device_name(device_index)),
            pre_context_physical_device_name=str(pre_context_physical_gpu["name"]),
            pre_context_physical_total_bytes=int(pre_context_physical_gpu["total_bytes"]),
            pre_context_physical_used_bytes=int(pre_context_physical_gpu["used_bytes"]),
            pre_context_physical_free_bytes=int(pre_context_physical_gpu["free_bytes"]),
            physical_total_bytes=total_before,
            physical_free_bytes_before=free_before,
            physical_free_bytes_after_cleanup=free_after,
            allocated_bytes_before=allocated_bytes_before,
            peak_allocated_bytes=peak_allocated_bytes,
            peak_reserved_bytes=peak_reserved_bytes,
            training_latency_ms=training_latency_ms,
            generation_latency_ms=generation_latency_ms,
            total_latency_ms=total_latency_ms,
            training_loss=training_loss,
            optimizer_name=request.optimizer,
            optimizer_step_count=optimizer_step_count,
            raw_generation=raw_generation,
        )


def run_final_gpu_smoke(
    output_path: str | os.PathLike[str],
    *,
    preflight_report_path: str | os.PathLike[str],
    acknowledge_gpu_execution: bool,
    runtime: GpuSmokeRuntime | None = None,
    config: GpuSmokeConfig = DEFAULT_GPU_SMOKE_CONFIG,
) -> GpuSmokeWriteResult:
    """Run and publish the final gated GPU smoke exactly once.

    No output is created unless the prerequisite report, runtime evidence,
    training loss, deterministic generation, and integer parser all pass.  The
    caller must pass the literal boolean ``True`` to acknowledge that this is
    the final GPU-using step.  Existing output paths are never overwritten.
    """

    if acknowledge_gpu_execution is not True:
        raise GpuSmokePreflightError(
            "explicit acknowledge_gpu_execution=True is required before GPU use"
        )
    if not isinstance(config, GpuSmokeConfig):
        raise TypeError("config must be a GpuSmokeConfig")

    target = Path(output_path)
    if os.path.lexists(target):
        raise FileExistsError(f"refusing to overwrite existing GPU smoke artifact: {target}")

    preflight_path = Path(preflight_report_path)
    preflight = _load_preflight_report(preflight_path)
    selected_physical_gpu = _validate_green_preflight(preflight, config)
    preflight_file_sha256 = sha256_file(preflight_path)

    request = GpuSmokeRequest.from_config(config)
    selected_runtime = TransformersNF4SmokeRuntime() if runtime is None else runtime
    evidence = selected_runtime.execute(request)
    if not isinstance(evidence, GpuSmokeRuntimeEvidence):
        raise TypeError("GpuSmokeRuntime.execute() must return GpuSmokeRuntimeEvidence")
    device_binding = _validate_runtime_evidence(
        evidence,
        request,
        selected_physical_gpu=selected_physical_gpu,
    )

    parsed = parse_answer(evidence.raw_generation)
    if not parsed.ok or parsed.value != config.expected_answer:
        raise GpuSmokeValidationError(
            "deterministic synthetic generation did not parse to the expected integer; "
            "no green artifact was written"
        )

    payload_without_hash: dict[str, object] = {
        "schema_version": config.schema_version,
        "status": "green",
        "model_id": config.model_id,
        "revision": config.revision,
        "config": config.as_dict(),
        "config_sha256": config.sha256,
        "input_provenance": {
            "kind": "locally_authored_synthetic_arithmetic",
            "competition_data_used": False,
            "caller_supplied_prompt_accepted": False,
            "prompt_sha256": hashlib.sha256(
                config.user_prompt.encode("utf-8")
            ).hexdigest(),
        },
        "preflight": {
            "path": str(preflight_path.resolve()),
            "size_bytes": preflight_path.stat().st_size,
            "sha256": preflight_file_sha256,
            "training_ready": True,
            "selected_physical_gpu": selected_physical_gpu,
        },
        "runtime": evidence.as_dict(),
        "device_binding": device_binding,
        "parser": _parse_result_dict(parsed),
        "expected_answer": config.expected_answer,
        "exact_match": True,
    }
    payload_sha256 = hashlib.sha256(
        canonical_json_bytes(payload_without_hash)
    ).hexdigest()
    payload = {**payload_without_hash, "payload_sha256": payload_sha256}
    return _write_green_json_no_overwrite(target, payload, payload_sha256)


def _load_preflight_report(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise GpuSmokePreflightError(f"preflight report is not a regular file: {path}")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise GpuSmokePreflightError(f"preflight report is invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise GpuSmokePreflightError("preflight report root must be a JSON object")
    return payload


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _validate_green_preflight(
    report: Mapping[str, Any], config: GpuSmokeConfig
) -> dict[str, object]:
    if report.get("model_id") != config.model_id:
        raise GpuSmokePreflightError("preflight model_id is not the fixed competition model")
    if report.get("requested_revision") != config.revision:
        raise GpuSmokePreflightError("preflight requested_revision is not the pinned commit")
    if report.get("resolved_commit") != config.revision:
        raise GpuSmokePreflightError("preflight did not resolve the pinned commit")

    required_green_flags = (
        "snapshot_consistent",
        "tokenizer_ready",
        "weights_ready",
        "model_runtime_ready",
        "qlora_dependencies_ready",
        "host_runtime_ready",
        "training_ready",
    )
    blocked_flags = [name for name in required_green_flags if report.get(name) is not True]
    if blocked_flags:
        raise GpuSmokePreflightError(
            f"preflight is not green for required fields: {blocked_flags!r}"
        )
    if report.get("training_profile") != "nf4_qlora_bf16":
        raise GpuSmokePreflightError("preflight training profile is not nf4_qlora_bf16")
    if report.get("blockers") not in ([], ()):
        raise GpuSmokePreflightError("preflight contains blockers")
    if report.get("runtime_blockers") not in ([], ()):
        raise GpuSmokePreflightError("preflight contains runtime blockers")

    packages = report.get("packages")
    if not isinstance(packages, Mapping):
        raise GpuSmokePreflightError("preflight packages report is missing")
    missing_packages = [
        name
        for name in _REQUIRED_PACKAGE_NAMES
        if not isinstance(packages.get(name), str) or not packages[name]
    ]
    if missing_packages:
        raise GpuSmokePreflightError(
            f"preflight lacks installed runtime packages: {missing_packages!r}"
        )

    torch_cuda = report.get("torch_cuda_runtime")
    if not isinstance(torch_cuda, Mapping):
        raise GpuSmokePreflightError("preflight torch CUDA report is missing")
    if torch_cuda.get("available") is not True:
        raise GpuSmokePreflightError("preflight torch CUDA runtime is unavailable")
    if torch_cuda.get("bf16_supported") is not True:
        raise GpuSmokePreflightError("preflight BF16 support is not green")
    if torch_cuda.get("device_count") != 1:
        raise GpuSmokePreflightError("preflight must expose exactly one CUDA device")

    nf4_vram = report.get("nf4_vram")
    if not isinstance(nf4_vram, Mapping) or nf4_vram.get("ready") is not True:
        raise GpuSmokePreflightError("preflight NF4 VRAM gate is not green")
    minimum_reported = _required_nonnegative_int(
        nf4_vram.get("minimum_free_mib"),
        "preflight nf4_vram.minimum_free_mib",
    )
    maximum_observed = _required_nonnegative_int(
        nf4_vram.get("maximum_observed_free_mib"),
        "preflight nf4_vram.maximum_observed_free_mib",
    )
    if minimum_reported < config.minimum_free_vram_mib:
        raise GpuSmokePreflightError("preflight used a weaker free-VRAM threshold")
    if maximum_observed < minimum_reported:
        raise GpuSmokePreflightError("preflight maximum free VRAM is below its threshold")

    physical = report.get("physical_nvidia")
    if not isinstance(physical, Mapping) or physical.get("probe_succeeded") is not True:
        raise GpuSmokePreflightError("preflight physical NVIDIA inventory is not green")
    devices = physical.get("devices")
    if physical.get("device_count") != 1 or not isinstance(devices, list) or len(devices) != 1:
        raise GpuSmokePreflightError("preflight must report exactly one physical NVIDIA GPU")
    device = devices[0]
    if not isinstance(device, Mapping):
        raise GpuSmokePreflightError("preflight physical GPU entry is invalid")
    if device.get("index") != 0:
        raise GpuSmokePreflightError("preflight sole physical GPU must have index 0")
    total_mib = _required_nonnegative_int(
        device.get("memory_total_mib"), "preflight physical total VRAM"
    )
    used_mib = _required_nonnegative_int(
        device.get("memory_used_mib"), "preflight physical used VRAM"
    )
    free_mib = _required_nonnegative_int(
        device.get("memory_free_mib"), "preflight physical free VRAM"
    )
    if used_mib > total_mib or free_mib > total_mib or used_mib + free_mib > total_mib:
        raise GpuSmokePreflightError("preflight physical VRAM counters are inconsistent")
    if free_mib < config.minimum_free_vram_mib:
        raise GpuSmokePreflightError("physical GPU has insufficient free VRAM")
    if used_mib > config.maximum_preexisting_used_vram_mib:
        raise GpuSmokePreflightError("physical GPU is occupied above the smoke allowance")
    if maximum_observed != free_mib:
        raise GpuSmokePreflightError("preflight NF4 and physical free-VRAM evidence disagree")

    name = device.get("name")
    if not isinstance(name, str) or not name.strip():
        raise GpuSmokePreflightError("preflight physical GPU name is missing")
    return {
        "index": 0,
        "name": name,
        "memory_total_mib": total_mib,
        "memory_used_mib": used_mib,
        "memory_free_mib": free_mib,
        "compute_capability": device.get("compute_capability"),
        "driver_version": device.get("driver_version"),
    }


def _validate_runtime_request(request: GpuSmokeRequest) -> None:
    expected = GpuSmokeRequest.from_config(DEFAULT_GPU_SMOKE_CONFIG)
    if request != expected:
        raise GpuSmokeValidationError(
            "runtime request differs from the fixed synthetic GPU smoke contract"
        )


def _validate_runtime_evidence(
    evidence: GpuSmokeRuntimeEvidence,
    request: GpuSmokeRequest,
    *,
    selected_physical_gpu: Mapping[str, object],
) -> dict[str, object]:
    package_versions = evidence.package_versions
    if not isinstance(package_versions, Mapping):
        raise GpuSmokeValidationError("runtime package_versions must be a mapping")
    missing = [
        name
        for name in _REQUIRED_PACKAGE_NAMES
        if not isinstance(package_versions.get(name), str) or not package_versions[name]
    ]
    if missing:
        raise GpuSmokeValidationError(
            f"runtime evidence lacks package versions: {missing!r}"
        )
    if evidence.cuda_device_count != 1 or evidence.device_index != 0:
        raise GpuSmokeValidationError("runtime must use exactly CUDA device 0")
    if not isinstance(evidence.device_name, str) or not evidence.device_name.strip():
        raise GpuSmokeValidationError("runtime device_name must be non-empty")

    integer_fields = {
        "pre_context_physical_total_bytes": evidence.pre_context_physical_total_bytes,
        "pre_context_physical_used_bytes": evidence.pre_context_physical_used_bytes,
        "pre_context_physical_free_bytes": evidence.pre_context_physical_free_bytes,
        "physical_total_bytes": evidence.physical_total_bytes,
        "physical_free_bytes_before": evidence.physical_free_bytes_before,
        "physical_free_bytes_after_cleanup": evidence.physical_free_bytes_after_cleanup,
        "allocated_bytes_before": evidence.allocated_bytes_before,
        "peak_allocated_bytes": evidence.peak_allocated_bytes,
        "peak_reserved_bytes": evidence.peak_reserved_bytes,
    }
    for name, value in integer_fields.items():
        _required_nonnegative_int(value, f"runtime {name}")
    if evidence.physical_total_bytes <= 0:
        raise GpuSmokeValidationError("runtime physical_total_bytes must be positive")
    if evidence.pre_context_physical_total_bytes <= 0:
        raise GpuSmokeValidationError(
            "runtime pre-context physical_total_bytes must be positive"
        )
    if (
        evidence.pre_context_physical_used_bytes
        > evidence.pre_context_physical_total_bytes
        or evidence.pre_context_physical_free_bytes
        > evidence.pre_context_physical_total_bytes
        or (
            evidence.pre_context_physical_used_bytes
            + evidence.pre_context_physical_free_bytes
            > evidence.pre_context_physical_total_bytes
        )
    ):
        raise GpuSmokeValidationError("runtime pre-context physical VRAM is inconsistent")
    if evidence.physical_free_bytes_before > evidence.physical_total_bytes:
        raise GpuSmokeValidationError("runtime free VRAM exceeds total VRAM")
    if evidence.physical_free_bytes_after_cleanup > evidence.physical_total_bytes:
        raise GpuSmokeValidationError("runtime post-cleanup free VRAM exceeds total VRAM")
    if evidence.physical_free_bytes_before < request.minimum_free_vram_mib * _MIB:
        raise GpuSmokeValidationError("runtime began with insufficient free VRAM")
    if evidence.pre_context_physical_free_bytes < request.minimum_free_vram_mib * _MIB:
        raise GpuSmokeValidationError(
            "runtime pre-context physical GPU had insufficient free VRAM"
        )
    if (
        evidence.pre_context_physical_used_bytes
        > request.maximum_preexisting_used_vram_mib * _MIB
    ):
        raise GpuSmokeValidationError("runtime pre-context physical GPU was occupied")
    pre_context_name = evidence.pre_context_physical_device_name
    if not isinstance(pre_context_name, str) or not pre_context_name.strip():
        raise GpuSmokeValidationError("runtime pre-context physical GPU name is missing")
    if " ".join(pre_context_name.split()).casefold() != " ".join(
        evidence.device_name.split()
    ).casefold():
        raise GpuSmokeValidationError(
            "runtime pre-context physical device name does not match CUDA device"
        )
    pre_context_total_difference = abs(
        evidence.pre_context_physical_total_bytes - evidence.physical_total_bytes
    )
    if pre_context_total_difference > GPU_TOTAL_MEMORY_TOLERANCE_MIB * _MIB:
        raise GpuSmokeValidationError(
            "runtime pre-context physical total VRAM does not match CUDA total VRAM"
        )
    runtime_used_before = evidence.physical_total_bytes - evidence.physical_free_bytes_before
    if evidence.allocated_bytes_before > runtime_used_before:
        raise GpuSmokeValidationError("runtime allocator usage exceeds physical used VRAM")
    if evidence.peak_reserved_bytes < evidence.peak_allocated_bytes:
        raise GpuSmokeValidationError("runtime peak reserved VRAM is below peak allocated VRAM")
    device_binding = _validated_runtime_device_binding(
        evidence,
        selected_physical_gpu=selected_physical_gpu,
    )

    finite_fields = {
        "training_latency_ms": evidence.training_latency_ms,
        "generation_latency_ms": evidence.generation_latency_ms,
        "total_latency_ms": evidence.total_latency_ms,
        "training_loss": evidence.training_loss,
    }
    for name, value in finite_fields.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise GpuSmokeValidationError(f"runtime {name} must be numeric")
        if not math.isfinite(value) or value < 0:
            raise GpuSmokeValidationError(f"runtime {name} must be finite and non-negative")
    if evidence.total_latency_ms < (
        evidence.training_latency_ms + evidence.generation_latency_ms
    ):
        raise GpuSmokeValidationError("runtime total latency is below measured substeps")
    if not isinstance(evidence.raw_generation, str) or not evidence.raw_generation.strip():
        raise GpuSmokeValidationError("runtime raw_generation must be non-empty text")
    if evidence.optimizer_name != request.optimizer:
        raise GpuSmokeValidationError(
            "runtime optimizer evidence does not match the locked paged AdamW8bit contract"
        )
    if type(evidence.optimizer_step_count) is not int or evidence.optimizer_step_count != 1:
        raise GpuSmokeValidationError("runtime must prove exactly one optimizer step")
    return device_binding


def _validated_runtime_device_binding(
    evidence: GpuSmokeRuntimeEvidence,
    *,
    selected_physical_gpu: Mapping[str, object],
) -> dict[str, object]:
    preflight_name = selected_physical_gpu.get("name")
    if not isinstance(preflight_name, str) or not preflight_name.strip():
        raise GpuSmokeValidationError("selected preflight GPU name is invalid")
    normalized_preflight_name = " ".join(preflight_name.split()).casefold()
    normalized_runtime_name = " ".join(evidence.device_name.split()).casefold()
    if normalized_runtime_name != normalized_preflight_name:
        raise GpuSmokeValidationError(
            "runtime CUDA device name does not match the selected preflight physical GPU"
        )
    preflight_total_mib = _required_nonnegative_int(
        selected_physical_gpu.get("memory_total_mib"),
        "runtime selected preflight physical total VRAM",
    )
    expected_total_bytes = preflight_total_mib * _MIB
    difference_bytes = abs(evidence.physical_total_bytes - expected_total_bytes)
    tolerance_bytes = GPU_TOTAL_MEMORY_TOLERANCE_MIB * _MIB
    if difference_bytes > tolerance_bytes:
        raise GpuSmokeValidationError(
            "runtime CUDA total VRAM does not match the selected preflight physical GPU "
            f"within {GPU_TOTAL_MEMORY_TOLERANCE_MIB} MiB"
        )
    return {
        "matched": True,
        "preflight_device_name": preflight_name,
        "runtime_device_name": evidence.device_name,
        "preflight_total_mib": preflight_total_mib,
        "runtime_total_bytes": evidence.physical_total_bytes,
        "absolute_difference_bytes": difference_bytes,
        "tolerance_mib": GPU_TOTAL_MEMORY_TOLERANCE_MIB,
    }


def _required_nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        error_type = (
            GpuSmokePreflightError if name.startswith("preflight") else GpuSmokeValidationError
        )
        raise error_type(f"{name} must be a non-negative integer")
    return value


def _installed_package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in _REQUIRED_PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise GpuSmokePreflightError(
                f"required runtime package disappeared after preflight: {name}"
            ) from exc
    return versions


def _chat_token_ids(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    values = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    try:
        token_ids = list(values)
    except TypeError as exc:
        raise GpuSmokeValidationError("tokenizer returned non-iterable input IDs") from exc
    if not token_ids or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in token_ids
    ):
        raise GpuSmokeValidationError(
            "tokenizer must return a non-empty sequence of non-negative integer IDs"
        )
    return token_ids


def _parse_result_dict(result: AnswerParseResult) -> dict[str, object]:
    return {
        "status": result.status,
        "value": result.value,
        "source": result.source,
        "reason": result.reason,
    }


def _write_green_json_no_overwrite(
    target: Path, payload: Mapping[str, object], payload_sha256: str
) -> GpuSmokeWriteResult:
    if payload.get("status") != "green":
        raise GpuSmokeValidationError("only a fully green smoke artifact may be published")
    target.parent.mkdir(parents=True, exist_ok=True)
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

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, target)
        except FileExistsError:
            raise FileExistsError(
                f"refusing to overwrite existing GPU smoke artifact: {target}"
            ) from None
        temporary_path.unlink()
        temporary_path = None
        _fsync_directory(target.parent)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return GpuSmokeWriteResult(
        path=target,
        size_bytes=target.stat().st_size,
        sha256=sha256_file(target),
        payload_sha256=payload_sha256,
    )


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "DEFAULT_GPU_SMOKE_CONFIG",
    "GPU_TOTAL_MEMORY_TOLERANCE_MIB",
    "GpuSmokeConfig",
    "GpuSmokePreflightError",
    "GpuSmokeRequest",
    "GpuSmokeRuntime",
    "GpuSmokeRuntimeEvidence",
    "GpuSmokeValidationError",
    "GpuSmokeWriteResult",
    "SYNTHETIC_SMOKE_EXPECTED_ANSWER",
    "SYNTHETIC_SMOKE_SYSTEM_PROMPT",
    "SYNTHETIC_SMOKE_TARGET",
    "SYNTHETIC_SMOKE_USER_PROMPT",
    "TransformersNF4SmokeRuntime",
    "run_final_gpu_smoke",
]

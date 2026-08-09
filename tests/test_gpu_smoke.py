from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

import deep_challenge.gpu_smoke as gpu_smoke_module
from deep_challenge.gate_b import DEFAULT_GATE_B_CONFIG, PINNED_MODEL_REVISION
from deep_challenge.gpu_smoke import (
    DEFAULT_GPU_SMOKE_CONFIG,
    GPU_TOTAL_MEMORY_TOLERANCE_MIB,
    SYNTHETIC_SMOKE_EXPECTED_ANSWER,
    SYNTHETIC_SMOKE_TARGET,
    SYNTHETIC_SMOKE_USER_PROMPT,
    GpuSmokeConfig,
    GpuSmokePreflightError,
    GpuSmokeRequest,
    GpuSmokeRuntimeEvidence,
    GpuSmokeValidationError,
    run_final_gpu_smoke,
)
from deep_challenge.model_preflight import OFFICIAL_MODEL_ID
from deep_challenge.provenance import canonical_json_bytes

_MIB = 1024 * 1024


def _ready_preflight() -> dict[str, object]:
    packages = {
        "torch": "2.7.0",
        "transformers": "4.51.0",
        "accelerate": "1.7.0",
        "peft": "0.15.0",
        "bitsandbytes": "0.45.5",
        "triton": "3.3.0",
    }
    return {
        "model_id": OFFICIAL_MODEL_ID,
        "requested_revision": PINNED_MODEL_REVISION,
        "resolved_commit": PINNED_MODEL_REVISION,
        "packages": packages,
        "physical_nvidia": {
            "probe_succeeded": True,
            "available": True,
            "device_count": 1,
            "devices": [
                {
                    "index": 0,
                    "name": "NVIDIA Test GPU",
                    "memory_total_mib": 12_288,
                    "memory_used_mib": 512,
                    "memory_free_mib": 11_776,
                    "compute_capability": "8.9",
                    "driver_version": "999.0",
                }
            ],
        },
        "torch_cuda_runtime": {
            "available": True,
            "bf16_supported": True,
            "device_count": 1,
            "devices": [],
        },
        "snapshot_consistent": True,
        "tokenizer_ready": True,
        "weights_ready": True,
        "model_runtime_ready": True,
        "training_profile": "nf4_qlora_bf16",
        "nf4_vram": {
            "minimum_free_mib": 10_240,
            "maximum_observed_free_mib": 11_776,
            "ready": True,
        },
        "host_runtime_ready": True,
        "runtime_blockers": [],
        "qlora_dependencies_ready": True,
        "training_ready": True,
        "blockers": [],
    }


def _write_preflight(tmp_path: Path, payload: dict[str, object] | None = None) -> Path:
    path = tmp_path / "preflight.json"
    path.write_text(
        json.dumps(_ready_preflight() if payload is None else payload),
        encoding="utf-8",
    )
    return path


def _runtime_evidence(raw_generation: str = "Final answer: 5") -> GpuSmokeRuntimeEvidence:
    return GpuSmokeRuntimeEvidence(
        package_versions={
            "torch": "2.7.0",
            "transformers": "4.51.0",
            "accelerate": "1.7.0",
            "peft": "0.15.0",
            "bitsandbytes": "0.45.5",
            "triton": "3.3.0",
        },
        cuda_device_count=1,
        device_index=0,
        device_name="NVIDIA Test GPU",
        pre_context_physical_device_name="NVIDIA Test GPU",
        pre_context_physical_total_bytes=12_288 * _MIB,
        pre_context_physical_used_bytes=512 * _MIB,
        pre_context_physical_free_bytes=11_776 * _MIB,
        physical_total_bytes=12_288 * _MIB,
        physical_free_bytes_before=11_776 * _MIB,
        physical_free_bytes_after_cleanup=11_700 * _MIB,
        allocated_bytes_before=0,
        peak_allocated_bytes=4_000 * _MIB,
        peak_reserved_bytes=4_500 * _MIB,
        training_latency_ms=10.0,
        generation_latency_ms=5.0,
        total_latency_ms=25.0,
        training_loss=0.75,
        optimizer_name="paged_adamw_8bit",
        optimizer_step_count=1,
        raw_generation=raw_generation,
    )


def _live_physical_report(*, used_mib: int, free_mib: int) -> dict[str, object]:
    return {
        "probe_succeeded": True,
        "available": True,
        "device_count": 1,
        "devices": [
            {
                "index": 0,
                "name": "NVIDIA Test GPU",
                "memory_total_mib": 12_288,
                "memory_used_mib": used_mib,
                "memory_free_mib": free_mib,
            }
        ],
    }


class FakeRuntime:
    def __init__(self, evidence: GpuSmokeRuntimeEvidence | None = None) -> None:
        self.evidence = _runtime_evidence() if evidence is None else evidence
        self.requests: list[GpuSmokeRequest] = []

    def execute(self, request: GpuSmokeRequest) -> GpuSmokeRuntimeEvidence:
        self.requests.append(request)
        return self.evidence


def test_module_import_does_not_import_gpu_model_stack() -> None:
    script = """
import sys
import deep_challenge.gpu_smoke
for name in ('torch', 'transformers', 'peft', 'bitsandbytes'):
    assert name not in sys.modules, name
"""
    environment = dict(os.environ)
    environment["PYTHONNOUSERSITE"] = "1"

    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr


def test_config_locks_model_revision_and_synthetic_prompt() -> None:
    assert DEFAULT_GPU_SMOKE_CONFIG.model_id == OFFICIAL_MODEL_ID
    assert DEFAULT_GPU_SMOKE_CONFIG.revision == PINNED_MODEL_REVISION
    assert DEFAULT_GPU_SMOKE_CONFIG.user_prompt == SYNTHETIC_SMOKE_USER_PROMPT
    assert DEFAULT_GPU_SMOKE_CONFIG.target_text == SYNTHETIC_SMOKE_TARGET
    assert DEFAULT_GPU_SMOKE_CONFIG.expected_answer == SYNTHETIC_SMOKE_EXPECTED_ANSWER
    assert DEFAULT_GPU_SMOKE_CONFIG.schema_version == "gate-b-final-gpu-smoke-v1"
    assert DEFAULT_GPU_SMOKE_CONFIG.lora_dropout == DEFAULT_GATE_B_CONFIG.lora_dropout
    assert (
        DEFAULT_GPU_SMOKE_CONFIG.gradient_checkpointing
        == DEFAULT_GATE_B_CONFIG.gradient_checkpointing
    )
    assert (
        DEFAULT_GPU_SMOKE_CONFIG.gradient_checkpointing_use_reentrant
        == DEFAULT_GATE_B_CONFIG.gradient_checkpointing_use_reentrant
    )
    assert DEFAULT_GPU_SMOKE_CONFIG.optimizer == DEFAULT_GATE_B_CONFIG.optimizer
    assert DEFAULT_GPU_SMOKE_CONFIG.learning_rate == DEFAULT_GATE_B_CONFIG.learning_rate
    assert DEFAULT_GPU_SMOKE_CONFIG.generation_use_cache is True
    assert DEFAULT_GPU_SMOKE_CONFIG.sha256 == (
        "50e3090f45f6b459c0f8e97aca83a38c9ff93792e8670046b9c4495c9f601086"
    )
    assert DEFAULT_GPU_SMOKE_CONFIG.sha256 != (
        "1f6132e365c9bf944cb8fc674fbf65fb4fbf485c202a56da1940e05b0c842900"
    )

    with pytest.raises(GpuSmokeValidationError, match="model_id is locked"):
        GpuSmokeConfig(model_id="Qwen/Qwen2.5-Math-7B-Instruct")
    with pytest.raises(GpuSmokeValidationError, match="user_prompt is locked"):
        GpuSmokeConfig(user_prompt="leaderboard question")
    with pytest.raises(GpuSmokeValidationError, match="lora_dropout is locked"):
        GpuSmokeConfig(lora_dropout=0.0)
    with pytest.raises(GpuSmokeValidationError, match="optimizer is locked"):
        GpuSmokeConfig(optimizer="adamw_torch")
    with pytest.raises(GpuSmokeValidationError, match="generation_use_cache is locked"):
        GpuSmokeConfig(generation_use_cache=False)


def test_config_rejects_equal_values_with_wrong_exact_types() -> None:
    with pytest.raises(GpuSmokeValidationError, match="lora_rank is locked to exact int"):
        GpuSmokeConfig(lora_rank=16.0)  # type: ignore[arg-type]
    with pytest.raises(
        GpuSmokeValidationError,
        match="double_quantization is locked to exact bool",
    ):
        GpuSmokeConfig(double_quantization=1)  # type: ignore[arg-type]


def test_explicit_acknowledgement_is_required_before_runtime_call(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    output = tmp_path / "smoke.json"

    with pytest.raises(GpuSmokePreflightError, match="explicit acknowledge"):
        run_final_gpu_smoke(
            output,
            preflight_report_path=_write_preflight(tmp_path),
            acknowledge_gpu_execution=False,
            runtime=runtime,
        )

    assert runtime.requests == []
    assert not output.exists()


def test_live_physical_recheck_keeps_external_occupancy_separate_from_cuda_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = GpuSmokeRequest.from_config(DEFAULT_GPU_SMOKE_CONFIG)
    monkeypatch.setattr(
        gpu_smoke_module,
        "_physical_nvidia_report",
        lambda: _live_physical_report(used_mib=1_024, free_mib=11_264),
    )

    checked = gpu_smoke_module._live_physical_gpu_before_cuda_context(request)

    assert checked == {
        "name": "NVIDIA Test GPU",
        "total_bytes": 12_288 * _MIB,
        "used_bytes": 1_024 * _MIB,
        "free_bytes": 11_264 * _MIB,
    }

    monkeypatch.setattr(
        gpu_smoke_module,
        "_physical_nvidia_report",
        lambda: _live_physical_report(used_mib=1_025, free_mib=11_263),
    )
    with pytest.raises(GpuSmokePreflightError, match="occupied"):
        gpu_smoke_module._live_physical_gpu_before_cuda_context(request)


@pytest.mark.parametrize(
    "field",
    [
        "snapshot_consistent",
        "tokenizer_ready",
        "weights_ready",
        "model_runtime_ready",
        "qlora_dependencies_ready",
        "host_runtime_ready",
        "training_ready",
    ],
)
def test_every_required_preflight_flag_must_be_green(tmp_path: Path, field: str) -> None:
    report = _ready_preflight()
    report[field] = False
    runtime = FakeRuntime()
    output = tmp_path / "smoke.json"

    with pytest.raises(GpuSmokePreflightError, match="not green"):
        run_final_gpu_smoke(
            output,
            preflight_report_path=_write_preflight(tmp_path, report),
            acknowledge_gpu_execution=True,
            runtime=runtime,
        )

    assert runtime.requests == []
    assert not output.exists()


def test_preflight_rejects_insufficient_or_occupied_vram(tmp_path: Path) -> None:
    insufficient = _ready_preflight()
    insufficient["physical_nvidia"]["devices"][0].update(  # type: ignore[index]
        memory_used_mib=4_096,
        memory_free_mib=8_192,
    )
    insufficient["nf4_vram"] = {
        "minimum_free_mib": 10_240,
        "maximum_observed_free_mib": 8_192,
        "ready": True,
    }
    with pytest.raises(GpuSmokePreflightError, match="below its threshold"):
        run_final_gpu_smoke(
            tmp_path / "insufficient.json",
            preflight_report_path=_write_preflight(tmp_path, insufficient),
            acknowledge_gpu_execution=True,
            runtime=FakeRuntime(),
        )

    occupied = _ready_preflight()
    occupied["physical_nvidia"]["devices"][0].update(  # type: ignore[index]
        memory_total_mib=16_384,
        memory_used_mib=2_048,
        memory_free_mib=14_336,
    )
    occupied["nf4_vram"] = {
        "minimum_free_mib": 10_240,
        "maximum_observed_free_mib": 14_336,
        "ready": True,
    }
    with pytest.raises(GpuSmokePreflightError, match="occupied"):
        run_final_gpu_smoke(
            tmp_path / "occupied.json",
            preflight_report_path=_write_preflight(tmp_path, occupied),
            acknowledge_gpu_execution=True,
            runtime=FakeRuntime(),
        )


def test_preflight_rejects_model_identity_mismatch(tmp_path: Path) -> None:
    report = _ready_preflight()
    report["model_id"] = "Qwen/Qwen2.5-Math-3B-Instruct"
    runtime = FakeRuntime()

    with pytest.raises(GpuSmokePreflightError, match="fixed competition model"):
        run_final_gpu_smoke(
            tmp_path / "smoke.json",
            preflight_report_path=_write_preflight(tmp_path, report),
            acknowledge_gpu_execution=True,
            runtime=runtime,
        )

    assert runtime.requests == []


def test_green_fake_runtime_writes_atomic_checksummed_artifact(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    preflight_path = _write_preflight(tmp_path)
    output = tmp_path / "artifacts" / "gpu-smoke.json"

    result = run_final_gpu_smoke(
        output,
        preflight_report_path=preflight_path,
        acknowledge_gpu_execution=True,
        runtime=runtime,
    )

    assert len(runtime.requests) == 1
    request = runtime.requests[0]
    assert request.model_id == OFFICIAL_MODEL_ID
    assert request.revision == PINNED_MODEL_REVISION
    assert request.user_prompt == SYNTHETIC_SMOKE_USER_PROMPT
    assert request.expected_answer == 5
    assert request.lora_dropout == 0.05
    assert request.gradient_checkpointing is True
    assert request.gradient_checkpointing_use_reentrant is False
    assert request.optimizer == "paged_adamw_8bit"
    assert request.learning_rate == 0.0001
    assert result.path == output
    assert result.size_bytes == output.stat().st_size
    assert result.sha256 == hashlib.sha256(output.read_bytes()).hexdigest()

    payload = json.loads(output.read_text(encoding="utf-8"))
    stored_payload_sha256 = payload.pop("payload_sha256")
    assert stored_payload_sha256 == hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    assert result.payload_sha256 == stored_payload_sha256
    assert payload["status"] == "green"
    assert payload["model_id"] == OFFICIAL_MODEL_ID
    assert payload["revision"] == PINNED_MODEL_REVISION
    assert payload["input_provenance"]["competition_data_used"] is False
    assert payload["input_provenance"]["caller_supplied_prompt_accepted"] is False
    assert payload["runtime"]["raw_generation"] == "Final answer: 5"
    assert payload["runtime"]["pre_context_physical_used_bytes"] == 512 * _MIB
    assert payload["runtime"]["optimizer_name"] == "paged_adamw_8bit"
    assert payload["runtime"]["optimizer_step_count"] == 1
    assert payload["device_binding"] == {
        "matched": True,
        "preflight_device_name": "NVIDIA Test GPU",
        "runtime_device_name": "NVIDIA Test GPU",
        "preflight_total_mib": 12_288,
        "runtime_total_bytes": 12_288 * _MIB,
        "absolute_difference_bytes": 0,
        "tolerance_mib": GPU_TOTAL_MEMORY_TOLERANCE_MIB,
    }
    assert payload["parser"]["status"] == "ok"
    assert payload["parser"]["value"] == 5
    assert payload["exact_match"] is True


def test_invalid_or_conflicting_generation_never_writes_green_artifact(
    tmp_path: Path,
) -> None:
    output = tmp_path / "smoke.json"
    runtime = FakeRuntime(_runtime_evidence("Final answer: 5\nFinal answer: 6"))

    with pytest.raises(GpuSmokeValidationError, match="did not parse"):
        run_final_gpu_smoke(
            output,
            preflight_report_path=_write_preflight(tmp_path),
            acknowledge_gpu_execution=True,
            runtime=runtime,
        )

    assert not output.exists()


def test_runtime_evidence_must_show_clean_single_gpu_and_valid_packages(
    tmp_path: Path,
) -> None:
    output = tmp_path / "smoke.json"
    invalid = replace(_runtime_evidence(), cuda_device_count=2)

    with pytest.raises(GpuSmokeValidationError, match="exactly CUDA device 0"):
        run_final_gpu_smoke(
            output,
            preflight_report_path=_write_preflight(tmp_path),
            acknowledge_gpu_execution=True,
            runtime=FakeRuntime(invalid),
        )

    assert not output.exists()


@pytest.mark.parametrize(
    "invalid,match",
    [
        (
            replace(_runtime_evidence(), device_name="Different NVIDIA GPU"),
            "device name does not match",
        ),
        (
            replace(
                _runtime_evidence(),
                physical_total_bytes=13_000 * _MIB,
                physical_free_bytes_before=12_488 * _MIB,
                physical_free_bytes_after_cleanup=12_400 * _MIB,
            ),
            "total VRAM does not match",
        ),
        (
            replace(_runtime_evidence(), optimizer_name="adamw_torch"),
            "optimizer evidence",
        ),
        (
            replace(_runtime_evidence(), optimizer_step_count=0),
            "exactly one optimizer step",
        ),
        (
            replace(_runtime_evidence(), optimizer_step_count=True),
            "exactly one optimizer step",
        ),
        (
            replace(
                _runtime_evidence(),
                pre_context_physical_used_bytes=1_025 * _MIB,
                pre_context_physical_free_bytes=11_263 * _MIB,
            ),
            "pre-context physical GPU was occupied",
        ),
    ],
)
def test_runtime_must_match_preflight_device_and_prove_locked_optimizer_step(
    tmp_path: Path,
    invalid: GpuSmokeRuntimeEvidence,
    match: str,
) -> None:
    output = tmp_path / "smoke.json"
    with pytest.raises(GpuSmokeValidationError, match=match):
        run_final_gpu_smoke(
            output,
            preflight_report_path=_write_preflight(tmp_path),
            acknowledge_gpu_execution=True,
            runtime=FakeRuntime(invalid),
        )
    assert not output.exists()


def test_runtime_total_vram_small_driver_rounding_difference_is_recorded(
    tmp_path: Path,
) -> None:
    runtime_total_mib = 12_288 + GPU_TOTAL_MEMORY_TOLERANCE_MIB // 2
    evidence = replace(
        _runtime_evidence(),
        device_name="  NVIDIA   Test GPU  ",
        physical_total_bytes=runtime_total_mib * _MIB,
    )
    output = tmp_path / "smoke.json"

    run_final_gpu_smoke(
        output,
        preflight_report_path=_write_preflight(tmp_path),
        acknowledge_gpu_execution=True,
        runtime=FakeRuntime(evidence),
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["device_binding"]["matched"] is True
    assert payload["device_binding"]["absolute_difference_bytes"] == (
        GPU_TOTAL_MEMORY_TOLERANCE_MIB // 2 * _MIB
    )


def test_existing_output_is_rejected_before_preflight_or_runtime(tmp_path: Path) -> None:
    output = tmp_path / "smoke.json"
    original = b"user-owned\n"
    output.write_bytes(original)
    runtime = FakeRuntime()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_final_gpu_smoke(
            output,
            preflight_report_path=tmp_path / "missing-preflight.json",
            acknowledge_gpu_execution=True,
            runtime=runtime,
        )

    assert output.read_bytes() == original
    assert runtime.requests == []

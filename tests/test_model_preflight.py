from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import deep_challenge.model_preflight as preflight
from deep_challenge.model_preflight import (
    OFFICIAL_MODEL_ID,
    OFFICIAL_REVISION,
    run_model_preflight,
)

REVISION = OFFICIAL_REVISION


def _physical_gpu_report(
    *,
    total_mib: int = 12_288,
    used_mib: int = 512,
    free_mib: int = 11_776,
) -> dict[str, object]:
    return {
        "probe": "nvidia-smi",
        "query_fields": [
            "name",
            "memory.total",
            "memory.used",
            "memory.free",
            "compute_cap",
            "driver_version",
        ],
        "command_available": True,
        "probe_succeeded": True,
        "device_count": 1,
        "devices": [
            {
                "index": 0,
                "name": "NVIDIA Test GPU",
                "memory_total_mib": total_mib,
                "memory_used_mib": used_mib,
                "memory_free_mib": free_mib,
                "compute_capability": "8.9",
                "driver_version": "999.0",
            }
        ],
        "diagnostic": None,
        "detail": None,
    }


def _snapshot_files(tmp_path: Path, *, include_second_shard: bool = True) -> dict[str, Path]:
    root = tmp_path / "models--Qwen--Qwen2.5-3B-Instruct" / "snapshots" / REVISION
    root.mkdir(parents=True)
    files: dict[str, Path] = {}
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        path = root / name
        path.write_text("{}", encoding="utf-8")
        files[name] = path
    index = root / "model.safetensors.index.json"
    index.write_text(
        json.dumps(
            {
                "weight_map": {
                    "layer.0": "model-00001-of-00002.safetensors",
                    "layer.1": "model-00002-of-00002.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    files[index.name] = index
    for name in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
        if name.endswith("00002.safetensors") and not include_second_shard:
            continue
        path = root / name
        path.write_bytes(name.encode())
        files[name] = path
    return files


def _patch_ready_host(monkeypatch: pytest.MonkeyPatch, files: dict[str, Path]) -> None:
    monkeypatch.setattr(
        preflight,
        "_cached_file_path",
        lambda _model, _revision, filename: files.get(filename),
    )
    monkeypatch.setattr(preflight, "_package_version", lambda _name: "1.0")
    monkeypatch.setattr(
        preflight,
        "_cuda_report",
        lambda _version: {
            "available": True,
            "bf16_supported": True,
            "device_count": 1,
            "devices": [],
        },
    )
    monkeypatch.setattr(preflight, "_physical_nvidia_report", _physical_gpu_report)
    monkeypatch.setattr(preflight, "_runtime_import_report", _ready_runtime_import_report)
    _patch_test_weight_contract(monkeypatch, files)


def _patch_test_weight_contract(
    monkeypatch: pytest.MonkeyPatch, files: dict[str, Path]
) -> None:
    expected: dict[str, dict[str, str | int]] = {}
    for filename, official in preflight.PINNED_WEIGHT_ARTIFACTS.items():
        path = files.get(filename)
        if path is None:
            expected[filename] = dict(official)
        else:
            expected[filename] = {
                "size_bytes": path.stat().st_size,
                "sha256": preflight.sha256_file(path),
            }
    monkeypatch.setattr(preflight, "PINNED_WEIGHT_ARTIFACTS", expected)


def _ready_runtime_import_report() -> dict[str, object]:
    packages = {
        package: {
            "module": module,
            "succeeded": True,
            "diagnostic": None,
            "detail": None,
        }
        for package, module in preflight._RUNTIME_IMPORT_MODULES
    }
    return {
        "probe": "isolated_cpu_only_python_imports",
        "python_executable": "test-python",
        "timeout_seconds_per_package": 45.0,
        "cuda_devices_hidden": True,
        "all_imports_ready": True,
        "packages": packages,
        "proof_scope": "python_imports_only_no_cuda_kernel_or_model_load",
    }


def test_runtime_import_probe_is_isolated_bounded_and_hides_cuda() -> None:
    observed: list[tuple[list[str], dict[str, object]]] = []

    def successful_runner(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        observed.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    report = preflight._runtime_import_report(runner=successful_runner)

    assert report["all_imports_ready"] is True
    assert len(observed) == len(preflight._RUNTIME_IMPORT_MODULES)
    assert [command[-1] for command, _kwargs in observed] == [
        module for _package, module in preflight._RUNTIME_IMPORT_MODULES
    ]
    for command, kwargs in observed:
        assert command[:2] == [preflight.sys.executable, "-I"]
        assert kwargs["shell"] is False
        assert kwargs["timeout"] == 45.0
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        assert environment["CUDA_VISIBLE_DEVICES"] == ""
        assert environment["NVIDIA_VISIBLE_DEVICES"] == ""


def test_runtime_import_probe_captures_broken_import_and_timeout() -> None:
    def failing_runner(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        module = command[-1]
        if module == "bitsandbytes":
            return subprocess.CompletedProcess(
                command,
                1,
                stdout="",
                stderr="ImportError: undefined symbol: broken_abi\n",
            )
        if module == "triton":
            raise subprocess.TimeoutExpired(command, timeout=3.0, stderr="hung import")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    report = preflight._runtime_import_report(
        runner=failing_runner,
        timeout_seconds=3.0,
    )

    assert report["all_imports_ready"] is False
    packages = report["packages"]
    assert packages["bitsandbytes"]["diagnostic"] == "import_nonzero_exit"
    assert "broken_abi" in packages["bitsandbytes"]["detail"]
    assert packages["triton"]["diagnostic"] == "import_timeout"
    assert "hung import" in packages["triton"]["detail"]


def test_nvidia_smi_probe_is_bounded_and_fails_closed_when_missing() -> None:
    def missing_runner(
        *_args: object, **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError

    report = preflight._physical_nvidia_report(runner=missing_runner)

    assert report["command_available"] is False
    assert report["probe_succeeded"] is False
    assert report["devices"] == []
    assert report["diagnostic"] == "nvidia_smi_not_found"


def test_training_readiness_fails_closed_without_nvidia_smi(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = _snapshot_files(tmp_path)
    _patch_ready_host(monkeypatch, files)

    report = run_model_preflight(
        model_id=OFFICIAL_MODEL_ID,
        revision=REVISION,
        nvidia_probe=lambda: {
            "probe": "nvidia-smi",
            "command_available": False,
            "probe_succeeded": False,
            "device_count": 0,
            "devices": [],
            "diagnostic": "nvidia_smi_not_found",
            "detail": None,
        },
    )

    assert report["torch_cuda_runtime"]["available"] is True
    assert report["host_runtime_ready"] is False
    assert report["training_ready"] is False
    assert "physical_nvidia_probe_failed" in report["runtime_blockers"]


def test_nvidia_smi_probe_uses_fixed_query_and_parses_devices() -> None:
    observed: dict[str, object] = {}

    def successful_runner(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        observed["command"] = command
        observed.update(kwargs)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "NVIDIA GeForce RTX 4070, 12282, 781, 11501, 8.9, 555.42\n"
                "NVIDIA A100-SXM4-40GB, 40960, 2048, 38912, 8.0, 555.42\n"
            ),
            stderr="",
        )

    report = preflight._physical_nvidia_report(runner=successful_runner)

    assert observed["command"] == [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,memory.free,compute_cap,driver_version",
        "--format=csv,noheader,nounits",
    ]
    assert observed["shell"] is False
    assert observed["timeout"] == 5.0
    assert report["probe_succeeded"] is True
    assert report["device_count"] == 2
    assert report["devices"][0] == {
        "index": 0,
        "name": "NVIDIA GeForce RTX 4070",
        "memory_total_mib": 12_282,
        "memory_used_mib": 781,
        "memory_free_mib": 11_501,
        "compute_capability": "8.9",
        "driver_version": "555.42",
    }


def test_model_preflight_requires_official_model_and_immutable_commit() -> None:
    with pytest.raises(ValueError, match="fixed competition base model"):
        run_model_preflight(model_id="other/model", revision=REVISION)
    with pytest.raises(ValueError, match="40-hex"):
        run_model_preflight(model_id=OFFICIAL_MODEL_ID, revision="main")
    with pytest.raises(ValueError, match="pinned competition commit"):
        run_model_preflight(model_id=OFFICIAL_MODEL_ID, revision="a" * 40)


def test_complete_pinned_sharded_snapshot_is_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = _snapshot_files(tmp_path)
    _patch_ready_host(monkeypatch, files)

    report = run_model_preflight(model_id=OFFICIAL_MODEL_ID, revision=REVISION)

    assert report["snapshot_consistent"] is True
    assert report["tokenizer_ready"] is True
    assert report["weights_ready"] is True
    assert report["model_runtime_ready"] is True
    assert report["host_runtime_ready"] is True
    assert report["nf4_vram"]["ready"] is True
    assert report["training_ready"] is True
    assert report["training_ready_scope"] == "pre_gpu_smoke_prerequisites"
    assert report["execution_ready"] is False
    assert report["final_gpu_smoke"] == {
        "required": True,
        "performed": False,
        "execution_ready": False,
        "blocker": "final_gpu_kernel_and_nf4_model_load_smoke_required",
        "required_checks": [
            "bitsandbytes_cuda_kernel_smoke",
            "pinned_model_nf4_load_smoke",
        ],
        "note": (
            "CPU-safe import success does not prove bitsandbytes CUDA-kernel "
            "compatibility or that the pinned model loads in NF4 on this GPU."
        ),
    }
    assert set(report["cached_weight_files"]) >= {
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    }


def test_pinned_weight_sha_mismatch_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = _snapshot_files(tmp_path)
    _patch_ready_host(monkeypatch, files)
    files["model-00001-of-00002.safetensors"].write_bytes(b"corrupted")

    report = run_model_preflight(model_id=OFFICIAL_MODEL_ID, revision=REVISION)

    assert report["pinned_weight_contract"]["ready"] is False
    assert report["weights_ready"] is False
    assert report["training_ready"] is False
    assert any(
        "pinned_weight_sha256_mismatch:model-00001-of-00002.safetensors" in blocker
        for blocker in report["blockers"]
    )


def test_training_readiness_requires_qlora_packages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = _snapshot_files(tmp_path)
    _patch_ready_host(monkeypatch, files)
    monkeypatch.setattr(
        preflight,
        "_package_version",
        lambda name: None if name == "bitsandbytes" else "1.0",
    )

    report = run_model_preflight(model_id=OFFICIAL_MODEL_ID, revision=REVISION)

    assert report["model_runtime_ready"] is True
    assert report["qlora_dependencies_ready"] is False
    assert report["training_ready"] is False
    assert "qlora_dependency_not_installed:bitsandbytes" in report["blockers"]


def test_broken_qlora_import_fails_closed_despite_installed_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = _snapshot_files(tmp_path)
    _patch_ready_host(monkeypatch, files)
    imports = _ready_runtime_import_report()
    packages = imports["packages"]
    assert isinstance(packages, dict)
    packages["bitsandbytes"] = {
        "module": "bitsandbytes",
        "succeeded": False,
        "diagnostic": "import_nonzero_exit",
        "detail": "returncode=1; ImportError: undefined symbol",
    }
    monkeypatch.setattr(preflight, "_runtime_import_report", lambda: imports)

    report = run_model_preflight(model_id=OFFICIAL_MODEL_ID, revision=REVISION)

    assert report["packages"]["bitsandbytes"] == "1.0"
    assert report["model_runtime_ready"] is True
    assert report["qlora_dependencies_ready"] is False
    assert report["training_ready"] is False
    expected = "runtime_import_failed:bitsandbytes:import_nonzero_exit"
    assert expected in report["runtime_blockers"]
    assert expected in report["blockers"]


def test_broken_torch_import_blocks_model_runtime_without_main_process_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = _snapshot_files(tmp_path)
    _patch_ready_host(monkeypatch, files)
    imports = _ready_runtime_import_report()
    packages = imports["packages"]
    assert isinstance(packages, dict)
    packages["torch"] = {
        "module": "torch",
        "succeeded": False,
        "diagnostic": "import_timeout",
        "detail": "timeout_seconds=20",
    }
    monkeypatch.setattr(preflight, "_runtime_import_report", lambda: imports)
    cuda_versions: list[str | None] = []

    def cuda_report(version: str | None) -> dict[str, object]:
        cuda_versions.append(version)
        return {
            "available": False,
            "bf16_supported": False,
            "device_count": 0,
            "devices": [],
        }

    monkeypatch.setattr(preflight, "_cuda_report", cuda_report)

    report = run_model_preflight(model_id=OFFICIAL_MODEL_ID, revision=REVISION)

    assert cuda_versions == [None]
    assert report["model_runtime_ready"] is False
    assert report["training_ready"] is False
    assert "runtime_import_failed:torch:import_timeout" in report["blockers"]


def test_training_readiness_requires_bf16_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = _snapshot_files(tmp_path)
    _patch_ready_host(monkeypatch, files)
    monkeypatch.setattr(
        preflight,
        "_cuda_report",
        lambda _version: {
            "available": True,
            "bf16_supported": False,
            "device_count": 1,
            "devices": [],
        },
    )

    report = run_model_preflight(model_id=OFFICIAL_MODEL_ID, revision=REVISION)

    assert report["model_runtime_ready"] is True
    assert report["training_ready"] is False
    assert "cuda_bf16_not_supported" in report["blockers"]


def test_training_readiness_rejects_saturated_12gb_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = _snapshot_files(tmp_path)
    _patch_ready_host(monkeypatch, files)
    monkeypatch.setattr(
        preflight,
        "_physical_nvidia_report",
        lambda: _physical_gpu_report(used_mib=4_096, free_mib=8_192),
    )

    report = run_model_preflight(model_id=OFFICIAL_MODEL_ID, revision=REVISION)

    assert report["torch_cuda_runtime"]["available"] is True
    assert report["nf4_vram"] == {
        "minimum_free_mib": preflight.DEFAULT_NF4_MIN_FREE_VRAM_MIB,
        "maximum_observed_free_mib": 8_192,
        "ready": False,
    }
    assert report["host_runtime_ready"] is False
    assert report["training_ready"] is False
    assert "insufficient_physical_gpu_free_vram" in report["runtime_blockers"]
    assert "insufficient_physical_gpu_free_vram" in report["blockers"]


def test_training_readiness_accepts_adequate_12gb_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = _snapshot_files(tmp_path)
    _patch_ready_host(monkeypatch, files)
    monkeypatch.setattr(
        preflight,
        "_physical_nvidia_report",
        lambda: _physical_gpu_report(used_mib=1_024, free_mib=11_264),
    )

    report = run_model_preflight(model_id=OFFICIAL_MODEL_ID, revision=REVISION)

    assert report["nf4_vram"]["minimum_free_mib"] == 10_240
    assert report["nf4_vram"]["maximum_observed_free_mib"] == 11_264
    assert report["host_runtime_ready"] is True
    assert report["training_ready"] is True


def test_physical_gpu_is_distinct_when_torch_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = _snapshot_files(tmp_path)
    monkeypatch.setattr(
        preflight,
        "_cached_file_path",
        lambda _model, _revision, filename: files.get(filename),
    )
    monkeypatch.setattr(
        preflight,
        "_package_version",
        lambda name: None if name == "torch" else "1.0",
    )
    monkeypatch.setattr(preflight, "_physical_nvidia_report", _physical_gpu_report)
    monkeypatch.setattr(preflight, "_runtime_import_report", _ready_runtime_import_report)
    _patch_test_weight_contract(monkeypatch, files)

    report = run_model_preflight(model_id=OFFICIAL_MODEL_ID, revision=REVISION)

    assert report["physical_nvidia"]["probe_succeeded"] is True
    assert report["physical_nvidia"]["device_count"] == 1
    assert report["nf4_vram"]["ready"] is True
    assert report["torch_cuda_runtime"]["available"] is False
    assert report["host_runtime_ready"] is False
    assert report["training_ready"] is False
    assert "torch_not_installed" in report["blockers"]
    assert "physical_nvidia_probe_failed" not in report["blockers"]


def test_index_without_every_shard_is_not_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = _snapshot_files(tmp_path, include_second_shard=False)
    _patch_ready_host(monkeypatch, files)

    report = run_model_preflight(model_id=OFFICIAL_MODEL_ID, revision=REVISION)

    assert report["weights_ready"] is False
    assert report["model_runtime_ready"] is False
    assert any("missing shards" in error for error in report["weight_index_errors"])


def test_mixed_snapshot_blocks_tokenizer_and_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = _snapshot_files(tmp_path)
    other_root = tmp_path / "models--Qwen--Qwen2.5-3B-Instruct" / "snapshots" / ("b" * 40)
    other_root.mkdir(parents=True)
    mixed_config = other_root / "config.json"
    mixed_config.write_text("{}", encoding="utf-8")
    files["config.json"] = mixed_config
    _patch_ready_host(monkeypatch, files)

    report = run_model_preflight(model_id=OFFICIAL_MODEL_ID, revision=REVISION)

    assert report["tokenizer_ready"] is False
    assert report["snapshot_consistent"] is False
    assert report["model_runtime_ready"] is False

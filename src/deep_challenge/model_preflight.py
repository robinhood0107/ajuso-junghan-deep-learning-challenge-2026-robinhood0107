"""Read-only readiness checks for the fixed model and local accelerator state."""

from __future__ import annotations

import csv
import importlib.metadata
import importlib.util
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from .provenance import sha256_file

OFFICIAL_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
OFFICIAL_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
_IMMUTABLE_REVISION_RE = re.compile(r"[0-9a-fA-F]{40}\Z")
_REQUIRED_TOKENIZER_FILES = frozenset(
    {"config.json", "tokenizer.json", "tokenizer_config.json"}
)
_REQUIRED_QLORA_PACKAGES = ("accelerate", "peft", "bitsandbytes", "triton")
_RUNTIME_IMPORT_MODULES = (
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("accelerate", "accelerate"),
    ("peft", "peft"),
    ("bitsandbytes", "bitsandbytes"),
    ("triton", "triton"),
)
_REQUIRED_MODEL_RUNTIME_IMPORTS = frozenset({"torch", "transformers"})
_REQUIRED_QLORA_RUNTIME_IMPORTS = frozenset(
    {"accelerate", "peft", "bitsandbytes", "triton"}
)
_REPORTED_PACKAGES = (
    "accelerate",
    "bitsandbytes",
    "huggingface-hub",
    "numpy",
    "peft",
    "safetensors",
    "tokenizers",
    "torch",
    "transformers",
    "triton",
)
_NVIDIA_SMI_QUERY_FIELDS = (
    "name",
    "memory.total",
    "memory.used",
    "memory.free",
    "compute_cap",
    "driver_version",
)
_NVIDIA_SMI_TIMEOUT_SECONDS = 5.0
_RUNTIME_IMPORT_TIMEOUT_SECONDS = 45.0
PINNED_WEIGHT_ARTIFACTS: dict[str, dict[str, str | int]] = {
    "model.safetensors.index.json": {
        "size_bytes": 35_581,
        "sha256": "bc8aaa0c87d4335177e01c765f1de0db81661c67c1a72fbfb0d521b09f5ddc56",
    },
    "model-00001-of-00002.safetensors": {
        "size_bytes": 3_968_658_944,
        "sha256": "67347b23fb4165b652eb6611f5e1f2a06dfcddba8e909df1b2b0b1857bee06c2",
    },
    "model-00002-of-00002.safetensors": {
        "size_bytes": 2_203_268_048,
        "sha256": "a40d941d0e7e0b966ad8b62bb6d6b7c88cce1299197b599d9d0a4ce59aabfc1d",
    },
}
# Backward-compatible private alias for existing injected preflight tests.
_PINNED_WEIGHT_ARTIFACTS = PINNED_WEIGHT_ARTIFACTS

# A nominal 12 GiB board normally retains some driver/display allocation. Requiring
# 10 GiB free rejects a busy 12 GiB host while still allowing a clean host to reach
# a separately monitored, low-batch QLoRA smoke test. This is a startup gate, not an
# assertion that every sequence length or batch size will fit.
DEFAULT_NF4_MIN_FREE_VRAM_MIB = 10 * 1024


def _parse_nvidia_smi_output(output: str) -> list[dict[str, str | int]]:
    """Parse the fixed no-units NVIDIA inventory query or fail closed."""

    devices: list[dict[str, str | int]] = []
    for row_number, row in enumerate(
        csv.reader(io.StringIO(output), skipinitialspace=True), start=1
    ):
        if not row or all(not value.strip() for value in row):
            continue
        values = [value.strip() for value in row]
        if len(values) != len(_NVIDIA_SMI_QUERY_FIELDS):
            raise ValueError(
                f"row {row_number} has {len(values)} fields; "
                f"expected {len(_NVIDIA_SMI_QUERY_FIELDS)}"
            )
        name, total_text, used_text, free_text, compute_capability, driver_version = values
        if not name or not compute_capability or not driver_version:
            raise ValueError(f"row {row_number} has an empty identity field")
        try:
            total_mib = int(total_text)
            used_mib = int(used_text)
            free_mib = int(free_text)
        except ValueError as exc:
            raise ValueError(f"row {row_number} has non-integer memory fields") from exc
        if min(total_mib, used_mib, free_mib) < 0:
            raise ValueError(f"row {row_number} has negative memory fields")
        if used_mib > total_mib or free_mib > total_mib:
            raise ValueError(f"row {row_number} has memory usage above total memory")
        devices.append(
            {
                "index": len(devices),
                "name": name,
                "memory_total_mib": total_mib,
                "memory_used_mib": used_mib,
                "memory_free_mib": free_mib,
                "compute_capability": compute_capability,
                "driver_version": driver_version,
            }
        )
    return devices


def _compact_diagnostic(value: str | None, *, limit: int = 400) -> str | None:
    if value is None:
        return None
    compact = " ".join(value.split())
    if not compact:
        return None
    return compact[:limit]


def _runtime_import_report(
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    timeout_seconds: float = _RUNTIME_IMPORT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Import model-runtime packages in bounded CPU-only child processes.

    One isolated process is used per package so a native-extension crash, hang, or
    broken ABI cannot take down or poison the preflight process. CUDA devices are
    hidden from every child. This verifies Python imports only: it deliberately
    does not create tensors, load weights, or exercise a bitsandbytes CUDA kernel.
    """

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    execute = subprocess.run if runner is None else runner
    child_environment = dict(os.environ)
    child_environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "NVIDIA_VISIBLE_DEVICES": "",
            "PYTHONNOUSERSITE": "1",
        }
    )
    packages: dict[str, dict[str, Any]] = {}
    import_script = (
        "import importlib, sys; "
        "importlib.import_module(sys.argv[1])"
    )
    for package, module in _RUNTIME_IMPORT_MODULES:
        command = [sys.executable, "-I", "-c", import_script, module]
        entry: dict[str, Any] = {
            "module": module,
            "succeeded": False,
            "diagnostic": None,
            "detail": None,
        }
        try:
            completed = execute(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                shell=False,
                env=child_environment,
            )
        except subprocess.TimeoutExpired as exc:
            entry["diagnostic"] = "import_timeout"
            detail = _compact_diagnostic(
                exc.stderr if isinstance(exc.stderr, str) else None
            )
            entry["detail"] = detail or f"timeout_seconds={timeout_seconds:g}"
        except FileNotFoundError:
            entry["diagnostic"] = "python_executable_not_found"
        except OSError as exc:
            entry["diagnostic"] = "import_probe_os_error"
            entry["detail"] = _compact_diagnostic(f"{type(exc).__name__}: {exc}")
        else:
            if completed.returncode == 0:
                entry["succeeded"] = True
            else:
                entry["diagnostic"] = "import_nonzero_exit"
                detail = _compact_diagnostic(completed.stderr)
                if detail is None:
                    detail = _compact_diagnostic(completed.stdout)
                entry["detail"] = (
                    f"returncode={completed.returncode}"
                    if detail is None
                    else f"returncode={completed.returncode}; {detail}"
                )
        packages[package] = entry
    return {
        "probe": "isolated_cpu_only_python_imports",
        "python_executable": sys.executable,
        "timeout_seconds_per_package": timeout_seconds,
        "cuda_devices_hidden": True,
        "all_imports_ready": all(
            entry["succeeded"] for entry in packages.values()
        ),
        "packages": packages,
        "proof_scope": "python_imports_only_no_cuda_kernel_or_model_load",
    }


def _physical_nvidia_report(
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    timeout_seconds: float = _NVIDIA_SMI_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Inventory physical NVIDIA devices with a bounded, read-only subprocess.

    This probe does not import PyTorch or initialize a CUDA context. A missing,
    timed-out, non-zero, or malformed ``nvidia-smi`` result is represented as a
    failed probe with no trusted devices, so VRAM readiness fails closed.
    """

    command = [
        "nvidia-smi",
        f"--query-gpu={','.join(_NVIDIA_SMI_QUERY_FIELDS)}",
        "--format=csv,noheader,nounits",
    ]
    execute = subprocess.run if runner is None else runner
    base: dict[str, Any] = {
        "probe": "nvidia-smi",
        "query_fields": list(_NVIDIA_SMI_QUERY_FIELDS),
        "timeout_seconds": timeout_seconds,
        "command_available": False,
        "probe_succeeded": False,
        "available": False,
        "device_count": 0,
        "devices": [],
        "diagnostic": None,
        "detail": None,
    }
    try:
        completed = execute(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            shell=False,
        )
    except FileNotFoundError:
        base["diagnostic"] = "nvidia_smi_not_found"
        return base
    except subprocess.TimeoutExpired:
        base["command_available"] = True
        base["diagnostic"] = "nvidia_smi_timeout"
        base["detail"] = f"timeout_seconds={timeout_seconds:g}"
        return base
    except OSError as exc:
        base["diagnostic"] = "nvidia_smi_os_error"
        base["detail"] = _compact_diagnostic(str(exc))
        return base

    base["command_available"] = True
    if completed.returncode != 0:
        base["diagnostic"] = "nvidia_smi_nonzero_exit"
        stderr = _compact_diagnostic(completed.stderr)
        base["detail"] = f"returncode={completed.returncode}"
        if stderr is not None:
            base["detail"] += f"; stderr={stderr}"
        return base
    try:
        devices = _parse_nvidia_smi_output(completed.stdout)
    except ValueError as exc:
        base["diagnostic"] = "nvidia_smi_parse_error"
        base["detail"] = _compact_diagnostic(str(exc))
        return base
    base.update(
        {
            "probe_succeeded": True,
            "available": bool(devices),
            "device_count": len(devices),
            "devices": devices,
            "diagnostic": None if devices else "nvidia_smi_no_devices",
        }
    )
    return base


def _disk_report(
    paths: Mapping[str, str | Path] | None = None,
) -> dict[str, dict[str, str | int | bool | None]]:
    """Report free space for configurable filesystems without writing to them."""

    selected: Mapping[str, str | Path] = (
        {"linux_root": Path("/"), "windows_c": Path("/mnt/c")} if paths is None else paths
    )
    report: dict[str, dict[str, str | int | bool | None]] = {}
    for label in sorted(selected):
        path = Path(selected[label])
        entry: dict[str, str | int | bool | None] = {
            "path": str(path),
            "probe_succeeded": False,
            "total_bytes": None,
            "used_bytes": None,
            "free_bytes": None,
            "diagnostic": None,
        }
        try:
            usage = shutil.disk_usage(path)
        except OSError as exc:
            entry["diagnostic"] = _compact_diagnostic(str(exc))
        else:
            entry.update(
                {
                    "probe_succeeded": True,
                    "total_bytes": usage.total,
                    "used_bytes": usage.used,
                    "free_bytes": usage.free,
                }
            )
        report[label] = entry
    return report


def _memory_report(meminfo_path: Path = Path("/proc/meminfo")) -> dict[str, Any]:
    """Return RAM/swap capacity where the host exposes portable read-only counters."""

    report: dict[str, Any] = {
        "probe_succeeded": False,
        "source": None,
        "total_bytes": None,
        "available_bytes": None,
        "swap_total_bytes": None,
        "swap_free_bytes": None,
        "diagnostic": None,
    }
    try:
        values: dict[str, int] = {}
        for line in meminfo_path.read_text(encoding="utf-8").splitlines():
            key, separator, remainder = line.partition(":")
            if not separator:
                continue
            fields = remainder.split()
            if not fields:
                continue
            multiplier = 1024 if len(fields) > 1 and fields[1].lower() == "kb" else 1
            values[key] = int(fields[0]) * multiplier
        total = values.get("MemTotal")
        available = values.get("MemAvailable", values.get("MemFree"))
        if total is not None:
            report.update(
                {
                    "probe_succeeded": True,
                    "source": str(meminfo_path),
                    "total_bytes": total,
                    "available_bytes": available,
                    "swap_total_bytes": values.get("SwapTotal"),
                    "swap_free_bytes": values.get("SwapFree"),
                }
            )
            return report
        report["diagnostic"] = "meminfo_missing_MemTotal"
    except (OSError, UnicodeError, ValueError) as exc:
        report["diagnostic"] = _compact_diagnostic(str(exc))

    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        physical_pages = int(os.sysconf("SC_PHYS_PAGES"))
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        if min(page_size, physical_pages, available_pages) < 0:
            raise ValueError("sysconf returned a negative memory counter")
    except (AttributeError, OSError, ValueError):
        return report
    report.update(
        {
            "probe_succeeded": True,
            "source": "os.sysconf",
            "total_bytes": page_size * physical_pages,
            "available_bytes": page_size * available_pages,
            "diagnostic": None,
        }
    )
    return report


def _host_inventory(
    *, disk_paths: Mapping[str, str | Path] | None = None
) -> dict[str, Any]:
    release = platform.release()
    return {
        "platform": {
            "system": platform.system(),
            "release": release,
            "version": platform.version(),
            "machine": platform.machine(),
            "os_name": os.name,
            "is_wsl": "microsoft" in release.lower(),
        },
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "memory": _memory_report(),
        "disks": _disk_report(disk_paths),
    }


def validate_model_identity(model_id: str, revision: str) -> str:
    """Validate the competition model and return a normalized immutable commit."""

    if model_id != OFFICIAL_MODEL_ID:
        raise ValueError(
            f"model_id must be the fixed competition base model {OFFICIAL_MODEL_ID!r}"
        )
    if not isinstance(revision, str) or _IMMUTABLE_REVISION_RE.fullmatch(revision) is None:
        raise ValueError("revision must be an immutable 40-hex Hugging Face commit")
    normalized_revision = revision.lower()
    if normalized_revision != OFFICIAL_REVISION:
        raise ValueError(
            f"revision must be the pinned competition commit {OFFICIAL_REVISION}"
        )
    return normalized_revision


def _snapshot_commit_from_path(path: Path) -> str | None:
    parts = path.parts
    try:
        snapshot_index = parts.index("snapshots")
    except ValueError:
        return None
    commit_index = snapshot_index + 1
    if commit_index >= len(parts):
        return None
    candidate = parts[commit_index]
    return candidate.lower() if _IMMUTABLE_REVISION_RE.fullmatch(candidate) else None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _cached_file_path(model_id: str, revision: str, filename: str) -> Path | None:
    if importlib.util.find_spec("transformers") is None:
        return None
    from transformers.utils.hub import cached_file

    try:
        located = cached_file(
            model_id,
            filename,
            revision=revision,
            local_files_only=True,
            _raise_exceptions_for_gated_repo=False,
            _raise_exceptions_for_missing_entries=False,
        )
    except OSError:
        located = None
    return None if located is None else Path(located)


def _artifact(path: Path) -> dict[str, str | int | None]:
    return {
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "snapshot_commit": _snapshot_commit_from_path(path),
    }


def _inspect_cached_files(
    model_id: str, revision: str, filenames: tuple[str, ...]
) -> tuple[dict[str, dict[str, str | int | None]], dict[str, Path]]:
    artifacts: dict[str, dict[str, str | int | None]] = {}
    paths: dict[str, Path] = {}
    for filename in filenames:
        path = _cached_file_path(model_id, revision, filename)
        if path is None:
            continue
        paths[filename] = path
        artifacts[filename] = _artifact(path)
    return artifacts, paths


def _index_shard_names(path: Path) -> tuple[str, ...]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid model weight index {path}: {exc}") from exc
    weight_map = payload.get("weight_map") if isinstance(payload, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"model weight index has no non-empty weight_map: {path}")
    shard_names: set[str] = set()
    for shard in weight_map.values():
        if not isinstance(shard, str) or not shard:
            raise ValueError(f"model weight index contains an invalid shard name: {path}")
        parsed = PurePosixPath(shard)
        if parsed.is_absolute() or ".." in parsed.parts:
            raise ValueError(f"model weight index contains an unsafe shard path: {shard!r}")
        shard_names.add(parsed.as_posix())
    return tuple(sorted(shard_names))


def _pinned_weight_contract_report(
    weight_files: Mapping[str, Mapping[str, str | int | None]],
) -> dict[str, Any]:
    """Verify exact size and SHA-256 for every pinned official weight artifact."""

    errors: list[str] = []
    observed: dict[str, dict[str, str | int | None]] = {}
    for filename, expected in PINNED_WEIGHT_ARTIFACTS.items():
        artifact = weight_files.get(filename)
        if artifact is None:
            errors.append(f"missing_pinned_weight_artifact:{filename}")
            continue
        observed[filename] = {
            "size_bytes": artifact.get("size_bytes"),
            "sha256": artifact.get("sha256"),
        }
        expected_size = expected["size_bytes"]
        expected_sha256 = expected["sha256"]
        if artifact.get("size_bytes") != expected_size:
            errors.append(
                f"pinned_weight_size_mismatch:{filename}:"
                f"expected={expected_size}:observed={artifact.get('size_bytes')}"
            )
        if artifact.get("sha256") != expected_sha256:
            errors.append(f"pinned_weight_sha256_mismatch:{filename}")
    return {
        "revision": OFFICIAL_REVISION,
        "algorithm": "sha256",
        "expected": PINNED_WEIGHT_ARTIFACTS,
        "observed": observed,
        "ready": not errors,
        "errors": errors,
    }


def _cuda_report(torch_version: str | None) -> dict[str, Any]:
    cuda: dict[str, Any] = {
        "available": False,
        "bf16_supported": False,
        "device_count": 0,
        "devices": [],
    }
    if torch_version is None:
        return cuda
    import torch

    cuda["available"] = torch.cuda.is_available()
    if cuda["available"]:
        cuda["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
    cuda["device_count"] = torch.cuda.device_count()
    cuda["compiled_cuda_version"] = torch.version.cuda
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        cuda["devices"].append(
            {
                "index": index,
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
            }
        )
    return cuda


def run_model_preflight(
    *,
    model_id: str = OFFICIAL_MODEL_ID,
    revision: str,
    min_nf4_free_vram_mib: int = DEFAULT_NF4_MIN_FREE_VRAM_MIB,
    nvidia_probe: Callable[[], dict[str, Any]] | None = None,
    runtime_import_probe: Callable[[], dict[str, Any]] | None = None,
    disk_paths: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Inspect a pinned local snapshot and host without downloads or weight loading.

    ``training_ready`` retains all previous model, CUDA BF16, and QLoRA package
    requirements and additionally requires trusted physical-GPU evidence with at
    least ``min_nf4_free_vram_mib`` currently free on one device. The physical
    inventory uses ``nvidia-smi`` and is reported separately from the PyTorch CUDA
    runtime. No tensor, model, or CUDA workload is created by the inventory probe.
    """

    revision = validate_model_identity(model_id, revision)
    if (
        isinstance(min_nf4_free_vram_mib, bool)
        or not isinstance(min_nf4_free_vram_mib, int)
        or min_nf4_free_vram_mib <= 0
    ):
        raise ValueError("min_nf4_free_vram_mib must be a positive integer")
    metadata_files = (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "LICENSE",
    )
    descriptor_files = (
        "model.safetensors.index.json",
        "model.safetensors",
        "pytorch_model.bin.index.json",
        "pytorch_model.bin",
    )
    metadata, _ = _inspect_cached_files(model_id, revision, metadata_files)
    descriptors, descriptor_paths = _inspect_cached_files(model_id, revision, descriptor_files)

    weight_files: dict[str, dict[str, str | int | None]] = {}
    weight_errors: list[str] = []
    complete_weight_layout = False
    for full_name in ("model.safetensors", "pytorch_model.bin"):
        artifact = descriptors.get(full_name)
        if artifact is not None and artifact["size_bytes"] > 0:
            complete_weight_layout = True
            weight_files[full_name] = artifact

    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = descriptor_paths.get(index_name)
        if index_path is None:
            continue
        try:
            shard_names = _index_shard_names(index_path)
        except ValueError as exc:
            weight_errors.append(str(exc))
            continue
        shard_artifacts, _ = _inspect_cached_files(model_id, revision, shard_names)
        missing = [name for name in shard_names if name not in shard_artifacts]
        empty = [
            name
            for name, artifact in shard_artifacts.items()
            if artifact["size_bytes"] <= 0
        ]
        if missing:
            weight_errors.append(f"{index_name}: missing shards {missing!r}")
        if empty:
            weight_errors.append(f"{index_name}: empty shards {empty!r}")
        if not missing and not empty:
            complete_weight_layout = True
            weight_files[index_name] = descriptors[index_name]
            weight_files.update(shard_artifacts)

    all_ready_artifacts = {**metadata, **descriptors, **weight_files}
    snapshot_commits = {
        value["snapshot_commit"]
        for value in all_ready_artifacts.values()
        if value["snapshot_commit"] is not None
    }
    snapshot_consistent = bool(all_ready_artifacts) and snapshot_commits == {revision}
    tokenizer_ready = (
        _REQUIRED_TOKENIZER_FILES.issubset(metadata)
        and all(metadata[name]["size_bytes"] > 0 for name in _REQUIRED_TOKENIZER_FILES)
        and all(metadata[name]["snapshot_commit"] == revision for name in _REQUIRED_TOKENIZER_FILES)
    )
    pinned_weight_contract = _pinned_weight_contract_report(weight_files)
    weights_ready = (
        complete_weight_layout
        and not weight_errors
        and pinned_weight_contract["ready"]
        and all(artifact["snapshot_commit"] == revision for artifact in weight_files.values())
    )

    dependency_versions = {package: _package_version(package) for package in _REPORTED_PACKAGES}
    torch_version = dependency_versions["torch"]
    transformers_version = dependency_versions["transformers"]
    qlora_packages = {
        package: dependency_versions[package] for package in _REQUIRED_QLORA_PACKAGES
    }
    import_probe = _runtime_import_report if runtime_import_probe is None else runtime_import_probe
    try:
        probed_imports = import_probe()
        if not isinstance(probed_imports, dict):
            raise TypeError("runtime import probe must return a dictionary")
        runtime_imports = dict(probed_imports)
    except Exception as exc:  # Import-probe failures must make readiness fail closed.
        runtime_imports = {
            "probe": "isolated_cpu_only_python_imports",
            "python_executable": sys.executable,
            "timeout_seconds_per_package": _RUNTIME_IMPORT_TIMEOUT_SECONDS,
            "cuda_devices_hidden": True,
            "all_imports_ready": False,
            "packages": {},
            "proof_scope": "python_imports_only_no_cuda_kernel_or_model_load",
            "diagnostic": "runtime_import_probe_exception",
            "detail": _compact_diagnostic(f"{type(exc).__name__}: {exc}"),
        }
    reported_runtime_packages = runtime_imports.get("packages")
    if not isinstance(reported_runtime_packages, dict):
        reported_runtime_packages = {}
    normalized_runtime_packages: dict[str, dict[str, Any]] = {}
    runtime_import_ready: dict[str, bool] = {}
    for package, module in _RUNTIME_IMPORT_MODULES:
        raw_entry = reported_runtime_packages.get(package)
        if isinstance(raw_entry, dict):
            entry = dict(raw_entry)
        else:
            entry = {
                "module": module,
                "succeeded": False,
                "diagnostic": "import_probe_result_missing",
                "detail": None,
            }
        entry["module"] = module
        succeeded = entry.get("succeeded") is True
        entry["succeeded"] = succeeded
        if not succeeded and not entry.get("diagnostic"):
            entry["diagnostic"] = "import_failed_without_diagnostic"
        normalized_runtime_packages[package] = entry
        runtime_import_ready[package] = succeeded
    runtime_imports["packages"] = normalized_runtime_packages
    runtime_imports["all_imports_ready"] = all(runtime_import_ready.values())
    runtime_imports.setdefault(
        "proof_scope", "python_imports_only_no_cuda_kernel_or_model_load"
    )

    # Avoid importing a known-broken torch runtime into this process after the
    # isolated ABI probe has failed. A successful import probe is still followed
    # by the existing read-only CUDA capability inventory.
    if runtime_import_ready["torch"] and torch_version is not None:
        try:
            torch_cuda = _cuda_report(torch_version)
        except Exception as exc:
            torch_cuda = _cuda_report(None)
            torch_cuda["diagnostic"] = "torch_cuda_inventory_exception"
            torch_cuda["detail"] = _compact_diagnostic(f"{type(exc).__name__}: {exc}")
            runtime_import_ready["torch"] = False
            normalized_runtime_packages["torch"].update(
                {
                    "succeeded": False,
                    "diagnostic": "post_import_cuda_inventory_failed",
                    "detail": torch_cuda["detail"],
                }
            )
            runtime_imports["all_imports_ready"] = False
    else:
        torch_cuda = _cuda_report(None)
    probe = _physical_nvidia_report if nvidia_probe is None else nvidia_probe
    try:
        probed_nvidia = probe()
        if not isinstance(probed_nvidia, dict):
            raise TypeError("NVIDIA probe must return a dictionary")
        physical_nvidia = dict(probed_nvidia)
    except Exception as exc:  # A probe failure must never make a host look ready.
        physical_nvidia = {
            "probe": "nvidia-smi",
            "query_fields": list(_NVIDIA_SMI_QUERY_FIELDS),
            "timeout_seconds": _NVIDIA_SMI_TIMEOUT_SECONDS,
            "command_available": False,
            "probe_succeeded": False,
            "available": False,
            "device_count": 0,
            "devices": [],
            "diagnostic": "nvidia_probe_exception",
            "detail": _compact_diagnostic(f"{type(exc).__name__}: {exc}"),
        }

    physical_devices = physical_nvidia.get("devices")
    trusted_free_vram_mib: list[int] = []
    physical_inventory_valid = bool(physical_nvidia.get("probe_succeeded"))
    if not isinstance(physical_devices, list):
        physical_inventory_valid = False
    else:
        for device in physical_devices:
            if not isinstance(device, dict):
                physical_inventory_valid = False
                break
            free_mib = device.get("memory_free_mib")
            if isinstance(free_mib, bool) or not isinstance(free_mib, int) or free_mib < 0:
                physical_inventory_valid = False
                break
            trusted_free_vram_mib.append(free_mib)
    physical_nvidia["available"] = bool(
        physical_inventory_valid and trusted_free_vram_mib
    )
    physical_nvidia["device_count"] = (
        len(physical_devices) if isinstance(physical_devices, list) else 0
    )
    maximum_free_vram_mib = (
        max(trusted_free_vram_mib) if physical_inventory_valid and trusted_free_vram_mib else None
    )
    nf4_vram_ready = (
        maximum_free_vram_mib is not None
        and maximum_free_vram_mib >= min_nf4_free_vram_mib
    )
    model_runtime_ready = (
        torch_version is not None
        and transformers_version is not None
        and all(
            runtime_import_ready[package]
            for package in _REQUIRED_MODEL_RUNTIME_IMPORTS
        )
        and tokenizer_ready
        and weights_ready
        and snapshot_consistent
    )
    qlora_dependencies_ready = all(qlora_packages.values()) and all(
        runtime_import_ready[package]
        for package in _REQUIRED_QLORA_RUNTIME_IMPORTS
    )
    runtime_blockers: list[str] = []
    if torch_version is None:
        runtime_blockers.append("torch_not_installed")
    if not torch_cuda["available"]:
        runtime_blockers.append("torch_cuda_runtime_unavailable")
    if torch_cuda["available"] and not torch_cuda.get("bf16_supported"):
        runtime_blockers.append("torch_cuda_bf16_not_supported")
    if not physical_nvidia.get("probe_succeeded"):
        runtime_blockers.append("physical_nvidia_probe_failed")
    elif not physical_inventory_valid:
        runtime_blockers.append("physical_nvidia_inventory_invalid")
    elif not trusted_free_vram_mib:
        runtime_blockers.append("physical_nvidia_gpu_unavailable")
    elif not nf4_vram_ready:
        runtime_blockers.append("insufficient_physical_gpu_free_vram")
    for package, ready in runtime_import_ready.items():
        if not ready:
            diagnostic = normalized_runtime_packages[package].get("diagnostic")
            runtime_blockers.append(f"runtime_import_failed:{package}:{diagnostic}")
    host_runtime_ready = not runtime_blockers
    training_ready = (
        model_runtime_ready
        and host_runtime_ready
        and qlora_dependencies_ready
    )
    blockers: list[str] = []
    if not tokenizer_ready:
        blockers.append("tokenizer_or_config_not_ready_at_pinned_commit")
    if not weights_ready:
        blockers.append("model_weights_not_complete_at_pinned_commit")
    blockers.extend(
        f"pinned_weight_contract_error:{error}"
        for error in pinned_weight_contract["errors"]
    )
    if not snapshot_consistent:
        blockers.append("cached_artifacts_not_from_one_pinned_snapshot")
    if torch_version is None:
        blockers.append("torch_not_installed")
    if transformers_version is None:
        blockers.append("transformers_not_installed")
    if not torch_cuda["available"]:
        blockers.append("cuda_gpu_unavailable")
    if torch_cuda["available"] and not torch_cuda.get("bf16_supported"):
        blockers.append("cuda_bf16_not_supported")
    if not physical_nvidia.get("probe_succeeded"):
        blockers.append("physical_nvidia_probe_failed")
    elif not physical_inventory_valid:
        blockers.append("physical_nvidia_inventory_invalid")
    elif not trusted_free_vram_mib:
        blockers.append("physical_nvidia_gpu_unavailable")
    elif not nf4_vram_ready:
        blockers.append("insufficient_physical_gpu_free_vram")
    for package, version in qlora_packages.items():
        if version is None:
            blockers.append(f"qlora_dependency_not_installed:{package}")
    for package, ready in runtime_import_ready.items():
        if not ready:
            diagnostic = normalized_runtime_packages[package].get("diagnostic")
            blockers.append(f"runtime_import_failed:{package}:{diagnostic}")
    blockers.extend(f"weight_index_error:{error}" for error in weight_errors)
    final_gpu_smoke_blocker = "final_gpu_kernel_and_nf4_model_load_smoke_required"
    return {
        "model_id": model_id,
        "requested_revision": revision,
        "resolved_commit": revision if snapshot_consistent else None,
        "packages": {
            "python_transformers": transformers_version,
            **dependency_versions,
        },
        "host_inventory": _host_inventory(disk_paths=disk_paths),
        "physical_nvidia": physical_nvidia,
        "cached_metadata": metadata,
        "cached_weight_descriptors": descriptors,
        "cached_weight_files": weight_files,
        "pinned_weight_contract": pinned_weight_contract,
        "weight_index_errors": weight_errors,
        "snapshot_commits": sorted(snapshot_commits),
        "snapshot_consistent": snapshot_consistent,
        # ``cuda`` remains as a compatibility alias for existing report consumers.
        "cuda": torch_cuda,
        "torch_cuda_runtime": torch_cuda,
        "runtime_imports": runtime_imports,
        "tokenizer_ready": tokenizer_ready,
        "weights_ready": weights_ready,
        "model_runtime_ready": model_runtime_ready,
        "training_profile": "nf4_qlora_bf16",
        "nf4_vram": {
            "minimum_free_mib": min_nf4_free_vram_mib,
            "maximum_observed_free_mib": maximum_free_vram_mib,
            "ready": nf4_vram_ready,
        },
        "host_runtime_ready": host_runtime_ready,
        "runtime_blockers": runtime_blockers,
        "qlora_dependencies_ready": qlora_dependencies_ready,
        "training_ready_scope": "pre_gpu_smoke_prerequisites",
        "training_ready": training_ready,
        "final_gpu_smoke": {
            "required": True,
            "performed": False,
            "execution_ready": False,
            "blocker": final_gpu_smoke_blocker,
            "required_checks": [
                "bitsandbytes_cuda_kernel_smoke",
                "pinned_model_nf4_load_smoke",
            ],
            "note": (
                "CPU-safe import success does not prove bitsandbytes CUDA-kernel "
                "compatibility or that the pinned model loads in NF4 on this GPU."
            ),
        },
        "execution_ready": False,
        "execution_blockers": [final_gpu_smoke_blocker],
        "blockers": blockers,
    }


__all__ = [
    "DEFAULT_NF4_MIN_FREE_VRAM_MIB",
    "OFFICIAL_MODEL_ID",
    "OFFICIAL_REVISION",
    "PINNED_WEIGHT_ARTIFACTS",
    "run_model_preflight",
    "validate_model_identity",
]

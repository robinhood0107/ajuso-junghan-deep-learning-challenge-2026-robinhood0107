"""Immutable CV-only train shard used after the locked split is created."""

from __future__ import annotations

import csv
import errno
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .data import CsvDataset, DatasetValidationError, load_train_csv
from .gate_b import GateBArtifactExistsError, GateBValidationError
from .provenance import canonical_json_bytes, sha256_file
from .splits import SplitManifest, SplitPartition

_SCHEMA = "gate-b-development-cv-shard-v1"
_CSV_NAME = "development-train.csv"
_MANIFEST_NAME = "manifest.json"
_CHECKSUMS_NAME = "CHECKSUMS.sha256"
_SHA256_CHARS = frozenset("0123456789abcdef")


@dataclass(frozen=True, slots=True)
class DevelopmentShardEvidence:
    path: str
    bundle_sha256: str
    csv_sha256: str
    row_count: int
    dataset: CsvDataset


def build_development_cv_shard(
    train: CsvDataset,
    *,
    split_manifest: SplitManifest,
    split_artifact_sha256: str,
    output_dir: str | Path,
) -> DevelopmentShardEvidence:
    """Create one atomic shard containing CV rows and no locked-holdout rows."""

    if not isinstance(train, CsvDataset):
        raise TypeError("train must be a CsvDataset")
    split_manifest.validate()
    split_file_sha = _required_sha256(
        split_artifact_sha256, "split_artifact_sha256"
    )
    source_ids = tuple(record.id for record in train)
    split_ids = tuple(assignment.record_id for assignment in split_manifest.assignments)
    if tuple(sorted(source_ids)) != split_ids:
        raise GateBValidationError("source train IDs do not match the locked split")
    cv_ids = _development_ids(split_manifest)
    by_id = {record.id: record for record in train}
    selected = tuple(by_id[problem_id] for problem_id in cv_ids)
    target = _new_directory_target(output_dir)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.build-", dir=target.parent))
    try:
        csv_path = staging / _CSV_NAME
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(("id", "question", "answer"))
            for record in selected:
                assert record.answer_raw is not None
                writer.writerow((record.id, record.question_raw, record.answer_raw))
            stream.flush()
            os.fsync(stream.fileno())
        csv_sha = sha256_file(csv_path)
        payload_without_hash = {
            "schema_version": _SCHEMA,
            "purpose": "sealed_development_cv_only_after_split_v4",
            "source_train_content_accessed_during_one_time_shard_build": True,
            "locked_holdout_rows_emitted": False,
            "source_train_sha256": train.manifest.sha256,
            "split_artifact_sha256": split_file_sha,
            "split_sha256": split_manifest.sha256,
            "source_groups_sha256": split_manifest.source_groups_sha256,
            "row_count": len(selected),
            "ids_sha256": _ids_sha256(cv_ids),
            "csv": {
                "name": _CSV_NAME,
                "size_bytes": csv_path.stat().st_size,
                "sha256": csv_sha,
            },
        }
        payload_sha = hashlib.sha256(
            canonical_json_bytes(payload_without_hash)
        ).hexdigest()
        manifest_path = staging / _MANIFEST_NAME
        _write_fsynced(
            manifest_path,
            (
                json.dumps(
                    {**payload_without_hash, "payload_sha256": payload_sha},
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    indent=2,
                )
                + "\n"
            ).encode("utf-8"),
        )
        checksums = (
            f"{csv_sha}  {_CSV_NAME}\n"
            f"{sha256_file(manifest_path)}  {_MANIFEST_NAME}\n"
        ).encode()
        _write_fsynced(staging / _CHECKSUMS_NAME, checksums)
        _fsync_directory(staging)
        _publish_directory_noreplace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return load_development_cv_shard(
        target,
        source_train_sha256=train.manifest.sha256,
        split_manifest=split_manifest,
        split_artifact_sha256=split_file_sha,
        expected_bundle_sha256=sha256_file(target / _CHECKSUMS_NAME),
    )


def load_development_cv_shard(
    path: str | Path,
    *,
    source_train_sha256: str,
    split_manifest: SplitManifest,
    split_artifact_sha256: str,
    expected_bundle_sha256: str,
) -> DevelopmentShardEvidence:
    """Validate a CV-only shard without opening the original train contents."""

    split_manifest.validate()
    expected_source = _required_sha256(source_train_sha256, "source_train_sha256")
    expected_split_file = _required_sha256(
        split_artifact_sha256, "split_artifact_sha256"
    )
    expected_bundle = _required_sha256(
        expected_bundle_sha256, "expected_bundle_sha256"
    )
    supplied = Path(path)
    if supplied.is_symlink():
        raise GateBValidationError("development shard refuses symlinks")
    root = supplied.resolve(strict=True)
    if not root.is_dir():
        raise GateBValidationError("development shard must be a real directory")
    files = tuple(sorted(item.name for item in root.iterdir()))
    if files != tuple(sorted((_CHECKSUMS_NAME, _CSV_NAME, _MANIFEST_NAME))):
        raise GateBValidationError("development shard has an unexpected file inventory")
    for name in files:
        item = root / name
        if item.is_symlink() or not item.is_file() or item.stat().st_size <= 0:
            raise GateBValidationError("development shard contains an unsafe file")
    bundle_sha = sha256_file(root / _CHECKSUMS_NAME)
    if bundle_sha != expected_bundle:
        raise GateBValidationError("development shard bundle SHA-256 changed")
    manifest = _load_manifest(root / _MANIFEST_NAME)
    expected_scalars = {
        "schema_version": _SCHEMA,
        "purpose": "sealed_development_cv_only_after_split_v4",
        "source_train_content_accessed_during_one_time_shard_build": True,
        "locked_holdout_rows_emitted": False,
        "source_train_sha256": expected_source,
        "split_artifact_sha256": expected_split_file,
        "split_sha256": split_manifest.sha256,
        "source_groups_sha256": split_manifest.source_groups_sha256,
    }
    mismatched = [key for key, value in expected_scalars.items() if manifest.get(key) != value]
    if mismatched:
        raise GateBValidationError(
            f"development shard provenance mismatch: {mismatched!r}"
        )
    csv_path = root / _CSV_NAME
    csv_evidence = manifest.get("csv")
    if not isinstance(csv_evidence, Mapping):
        raise GateBValidationError("development shard manifest lacks CSV evidence")
    csv_sha = sha256_file(csv_path)
    if (
        csv_evidence.get("name") != _CSV_NAME
        or csv_evidence.get("size_bytes") != csv_path.stat().st_size
        or csv_evidence.get("sha256") != csv_sha
    ):
        raise GateBValidationError("development shard CSV evidence does not match")
    expected_checksums = {
        _CSV_NAME: csv_sha,
        _MANIFEST_NAME: sha256_file(root / _MANIFEST_NAME),
    }
    if _parse_checksums(root / _CHECKSUMS_NAME) != expected_checksums:
        raise GateBValidationError("development shard checksums do not close")
    dataset = load_train_csv(csv_path)
    cv_ids = _development_ids(split_manifest)
    actual_ids = tuple(record.id for record in dataset)
    if actual_ids != cv_ids:
        raise GateBValidationError("development shard IDs/order do not match the locked split")
    if manifest.get("row_count") != len(cv_ids) or manifest.get(
        "ids_sha256"
    ) != _ids_sha256(cv_ids):
        raise GateBValidationError("development shard row/ID summary does not match")
    return DevelopmentShardEvidence(
        path=str(root),
        bundle_sha256=bundle_sha,
        csv_sha256=csv_sha,
        row_count=len(dataset),
        dataset=dataset,
    )


def _development_ids(manifest: SplitManifest) -> tuple[str, ...]:
    return tuple(
        assignment.record_id
        for assignment in manifest.assignments
        if assignment.partition is SplitPartition.CROSS_VALIDATION
    )


def _load_manifest(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8", errors="strict"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GateBValidationError(f"invalid development shard manifest: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise GateBValidationError("development shard manifest must be an object")
    stored = _required_sha256(payload.get("payload_sha256"), "payload_sha256")
    without_hash = dict(payload)
    without_hash.pop("payload_sha256")
    computed = hashlib.sha256(canonical_json_bytes(without_hash)).hexdigest()
    if stored != computed:
        raise GateBValidationError("development shard payload hash does not match")
    return payload


def _parse_checksums(path: Path) -> dict[str, str]:
    output: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    except (OSError, UnicodeError) as exc:
        raise GateBValidationError(f"cannot read development shard checksums: {exc}") from exc
    for line in lines:
        if "  " not in line:
            raise GateBValidationError("development shard checksum format is invalid")
        digest, name = line.split("  ", 1)
        _required_sha256(digest, "checksum digest")
        if name in output or name not in {_CSV_NAME, _MANIFEST_NAME}:
            raise GateBValidationError("development shard checksum path is invalid")
        output[name] = digest
    return output


def _new_directory_target(path: str | Path) -> Path:
    raw = Path(path)
    if raw.is_symlink() or raw.exists() or raw.parent.is_symlink():
        raise GateBArtifactExistsError("development shard output must be a new directory")
    target = raw.resolve(strict=False)
    if not target.parent.resolve(strict=True).is_dir():
        raise GateBValidationError("development shard parent must be a directory")
    return target


def _publish_directory_noreplace(staging: Path, target: Path) -> None:
    lock = target.parent / f".{target.name}.publish.lock"
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise GateBArtifactExistsError("development shard publication lock exists") from exc
    try:
        os.close(descriptor)
        if target.exists() or target.is_symlink():
            raise GateBArtifactExistsError("development shard destination exists")
        try:
            os.rename(staging, target)
        except OSError as exc:
            if exc.errno in {errno.EEXIST, errno.ENOTEMPTY}:
                raise GateBArtifactExistsError(
                    "development shard destination exists"
                ) from exc
            raise
        _fsync_directory(target.parent)
    finally:
        with suppress(FileNotFoundError):
            lock.unlink()


def _write_fsynced(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ids_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256(canonical_json_bytes(list(values))).hexdigest()


def _required_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_CHARS for character in value)
    ):
        raise GateBValidationError(f"{name} must be a lowercase SHA-256")
    return value


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise GateBValidationError(f"duplicate JSON key {key!r}")
        output[key] = value
    return output


def _reject_constant(value: str) -> None:
    raise DatasetValidationError(f"non-finite JSON constant is forbidden: {value}")


__all__ = [
    "DevelopmentShardEvidence",
    "build_development_cv_shard",
    "load_development_cv_shard",
]

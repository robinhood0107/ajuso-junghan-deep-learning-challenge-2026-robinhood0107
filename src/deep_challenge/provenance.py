"""Content-addressed provenance helpers.

The workspace is not assumed to be a Git repository.  A deterministic source-tree
manifest therefore provides a reproducible fallback identifier for experiment runs.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_DEFAULT_EXCLUDED_PARTS = frozenset(
    {
        ".git",
        ".gstack",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "artifacts",
    }
)
_DEFAULT_INCLUDED_NAMES = frozenset({".gitignore", ".python-version"})


@dataclass(frozen=True, slots=True)
class FileDigest:
    """Digest of one file relative to the manifest root."""

    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class SourceTreeManifest:
    """Deterministic manifest and its content-derived identifier."""

    algorithm: str
    files: tuple[FileDigest, ...]
    tree_sha256: str

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "algorithm": self.algorithm,
            "files": [asdict(entry) for entry in self.files],
            "tree_sha256": self.tree_sha256,
        }


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading it entirely into memory."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    """Encode JSON deterministically for hashing."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _iter_manifest_files(
    root: Path,
    *,
    include_suffixes: frozenset[str] | None,
    excluded_parts: frozenset[str],
    excluded_paths: frozenset[Path],
) -> Iterable[Path]:
    for candidate in root.rglob("*"):
        relative = candidate.relative_to(root)
        if any(part in excluded_parts for part in relative.parts):
            continue
        if candidate.is_symlink():
            raise ValueError(f"source manifest refuses symbolic links: {relative.as_posix()}")
        if not candidate.is_file():
            continue
        resolved_candidate = candidate.resolve(strict=True)
        if not resolved_candidate.is_relative_to(root):
            raise ValueError(
                f"source manifest file resolves outside root: {relative.as_posix()}"
            )
        if resolved_candidate in excluded_paths:
            continue
        if (
            include_suffixes is not None
            and candidate.suffix not in include_suffixes
            and candidate.name not in _DEFAULT_INCLUDED_NAMES
        ):
            continue
        yield candidate


def build_source_tree_manifest(
    root: str | Path,
    *,
    include_suffixes: frozenset[str] | None = frozenset(
        {".py", ".toml", ".lock", ".md", ".yaml", ".yml", ".json"}
    ),
    excluded_parts: frozenset[str] = _DEFAULT_EXCLUDED_PARTS,
    excluded_paths: Iterable[str | Path] = (),
) -> SourceTreeManifest:
    """Build a stable manifest for source, configuration, tests, and documentation."""

    resolved_root = Path(root).resolve(strict=True)
    if not resolved_root.is_dir():
        raise NotADirectoryError(resolved_root)
    resolved_exclusions = frozenset(
        Path(path).resolve(strict=False) for path in excluded_paths
    )

    entries = tuple(
        _manifest_file_digest(path, resolved_root)
        for path in sorted(
            _iter_manifest_files(
                resolved_root,
                include_suffixes=include_suffixes,
                excluded_parts=excluded_parts,
                excluded_paths=resolved_exclusions,
            ),
            key=lambda item: item.relative_to(resolved_root).as_posix(),
        )
    )
    payload = {
        "algorithm": "sha256-path-size-content-v1",
        "files": [asdict(entry) for entry in entries],
    }
    return SourceTreeManifest(
        algorithm=payload["algorithm"],
        files=entries,
        tree_sha256=hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
    )


def _manifest_file_digest(path: Path, root: Path) -> FileDigest:
    """Hash one regular file through a no-follow descriptor with race checks."""

    relative = path.relative_to(root).as_posix()
    before = os.lstat(path)
    if stat.S_ISLNK(before.st_mode):
        raise ValueError(f"source manifest refuses symbolic links: {relative}")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError(f"source manifest accepts regular files only: {relative}")
        if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise ValueError(f"source manifest file changed before open: {relative}")

        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
        stable_fields_before = (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
        )
        stable_fields_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if stable_fields_before != stable_fields_after:
            raise ValueError(f"source manifest file changed while hashing: {relative}")
        return FileDigest(
            path=relative,
            size_bytes=opened.st_size,
            sha256=digest.hexdigest(),
        )
    except OSError as exc:
        raise ValueError(f"source manifest could not safely open {relative}: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def write_json_atomic(path: str | Path, value: Any) -> None:
    """Write pretty JSON atomically on the target filesystem."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    ) + "\n"
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_name = handle.name
        os.replace(temporary_name, target)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)

"""Fail-closed checks for files that may be published in this repository.

This module intentionally has no competition-data dependency.  It is used by
the local Git pre-commit hook and can also inspect the complete tracked tree.
It catches the two mistakes that ``.gitignore`` alone cannot prevent:
``git add -f`` of a private path and staging a recognizable access token.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Iterable, Sequence
from pathlib import PurePosixPath

_FORBIDDEN_TOP_LEVEL = frozenset(
    {
        ".gstack",
        ".kaggle",
        ".venv",
        "artifacts",
        "cache",
        "checkpoints",
        "data",
        "datasets",
        "deep-learning-challenge-2026",
        "downloads",
        "env",
        "mlruns",
        "models",
        "outputs",
        "predictions",
        "runs",
        "submissions",
        "wandb",
    }
)
_FORBIDDEN_BASENAMES = frozenset(
    {
        "NUL",
        "NU_",
        "access_token",
        "adapter_model.bin",
        "credentials",
        "id_ed25519",
        "id_rsa",
        "kaggle.json",
        "submission.csv",
        "이어서 하기 프롬프트.txt",
    }
)
_FORBIDDEN_SUFFIXES = frozenset(
    {
        ".bin",
        ".ckpt",
        ".csv",
        ".gz",
        ".jsonl",
        ".kdbx",
        ".key",
        ".npy",
        ".npz",
        ".onnx",
        ".parquet",
        ".pickle",
        ".pkl",
        ".p12",
        ".pfx",
        ".pem",
        ".pth",
        ".pt",
        ".safetensors",
        ".tar",
        ".tsv",
        ".xz",
        ".zip",
    }
)
_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[bytes]], ...] = (
    (
        "named-access-token",
        re.compile(
            rb"(?i)(?:KAGGLE_API_TOKEN|HF_TOKEN|HUGGINGFACE_HUB_TOKEN|"
            rb"GITHUB_TOKEN)\s*[:=]\s*[\"']?[A-Za-z0-9_-]{12,}"
        ),
    ),
    ("github-token", re.compile(rb"(?:ghp|github_pat)_[A-Za-z0-9_-]{20,}")),
    ("huggingface-token", re.compile(rb"hf_[A-Za-z0-9_-]{20,}")),
    ("aws-access-key", re.compile(rb"AKIA[0-9A-Z]{16}")),
    ("private-key", re.compile(rb"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----")),
)


def find_forbidden_paths(paths: Iterable[str]) -> tuple[str, ...]:
    """Return normalized tracked paths that are never public repository inputs."""

    violations: list[str] = []
    for raw_path in paths:
        if not isinstance(raw_path, str) or not raw_path:
            violations.append(repr(raw_path))
            continue
        path = PurePosixPath(raw_path)
        parts = path.parts
        if path.is_absolute() or ".." in parts or not parts:
            violations.append(raw_path)
            continue
        basename = parts[-1]
        if (
            any(part in _FORBIDDEN_TOP_LEVEL for part in parts)
            or parts[:2] == ("reports", "generated")
            or basename in _FORBIDDEN_BASENAMES
            or basename.startswith(".env")
            or PurePosixPath(basename).suffix.lower() in _FORBIDDEN_SUFFIXES
        ):
            violations.append(raw_path)
    return tuple(sorted(set(violations)))


def find_secret_markers(payload: bytes) -> tuple[str, ...]:
    """Return category names only; never expose a matched secret in diagnostics."""

    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    return tuple(label for label, pattern in _SECRET_PATTERNS if pattern.search(payload))


def _git_paths(*, staged: bool) -> tuple[str, ...]:
    command = (
        ["git", "diff", "--cached", "--name-only", "-z", "--diff-filter=ACMRT"]
        if staged
        else ["git", "ls-files", "-z"]
    )
    result = subprocess.run(command, check=True, capture_output=True)
    return tuple(
        entry.decode("utf-8", "surrogateescape")
        for entry in result.stdout.split(b"\0")
        if entry
    )


def _staged_blob(path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f":{path}"], check=True, capture_output=True
    ).stdout


def audit_git_tree(
    *, staged: bool
) -> tuple[tuple[str, ...], tuple[tuple[str, tuple[str, ...]], ...]]:
    """Inspect staged changes or the complete index tree for public-safety issues."""

    paths = _git_paths(staged=staged)
    forbidden = find_forbidden_paths(paths)
    secrets: list[tuple[str, tuple[str, ...]]] = []
    for path in paths:
        markers = find_secret_markers(_staged_blob(path))
        if markers:
            secrets.append((path, markers))
    return forbidden, tuple(secrets)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fail when a private competition path or recognizable token is tracked."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--staged",
        action="store_true",
        help="inspect the index (the pre-commit default)",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="inspect every file in the complete Git index, including staged additions",
    )
    args = parser.parse_args(argv)
    try:
        forbidden, secrets = audit_git_tree(staged=not args.all)
    except subprocess.CalledProcessError as exc:
        print(f"public-repo guard could not read Git state: {exc}", file=sys.stderr)
        return 2
    if not forbidden and not secrets:
        scope = "staged" if not args.all else "indexed"
        print(f"public-repo guard: {scope} tree is safe")
        return 0
    for path in forbidden:
        print(f"forbidden public path: {path}", file=sys.stderr)
    for path, markers in secrets:
        print(
            f"recognizable secret marker in {path}: {', '.join(markers)}",
            file=sys.stderr,
        )
    return 1


if __name__ == "__main__":  # pragma: no cover - exercised through the entrypoint
    raise SystemExit(main())

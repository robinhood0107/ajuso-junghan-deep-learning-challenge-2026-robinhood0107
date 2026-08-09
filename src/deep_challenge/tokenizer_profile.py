"""Versioned Qwen tokenizer length profiling without loading model weights."""

from __future__ import annotations

import importlib.metadata
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from .audit import quantile_r7
from .data import MathRecord
from .model_preflight import (
    OFFICIAL_MODEL_ID,
    _snapshot_commit_from_path,
    validate_model_identity,
)
from .provenance import sha256_file

DEFAULT_SYSTEM_PROMPT = (
    "Solve the math problem carefully. Return the final signed integer as "
    "`Final answer: <integer>`."
)


class TokenizerLike(Protocol):
    """Small protocol that keeps profiling logic unit-testable."""

    name_or_path: str
    init_kwargs: dict[str, Any]

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, Sequence[int]]: ...

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> Sequence[int]: ...


def _length_summary(values: Sequence[int]) -> dict[str, int | float]:
    if not values:
        raise ValueError("cannot summarize an empty record set")
    return {
        "min": min(values),
        "p50": quantile_r7(values, 0.50),
        "p90": quantile_r7(values, 0.90),
        "p95": quantile_r7(values, 0.95),
        "p99": quantile_r7(values, 0.99),
        "max": max(values),
        "over_1024_count": sum(value > 1024 for value in values),
        "over_2048_count": sum(value > 2048 for value in values),
        "over_4096_count": sum(value > 4096 for value in values),
    }


def profile_records(
    records: Sequence[MathRecord],
    tokenizer: TokenizerLike,
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> dict[str, Any]:
    """Measure raw and chat-formatted lengths with the exact supplied prompt."""

    if not records:
        raise ValueError("records must not be empty")
    raw_lengths: list[int] = []
    chat_lengths: list[int] = []
    longest: list[tuple[int, str]] = []
    for record in records:
        raw = tokenizer(record.question_raw, add_special_tokens=False)["input_ids"]
        chat = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": record.question_raw},
            ],
            tokenize=True,
            add_generation_prompt=True,
        )
        raw_length = len(raw)
        chat_length = len(chat)
        raw_lengths.append(raw_length)
        chat_lengths.append(chat_length)
        longest.append((chat_length, record.id))
    longest.sort(key=lambda item: (-item[0], item[1]))
    return {
        "system_prompt": system_prompt,
        "record_count": len(records),
        "raw_tokens": _length_summary(raw_lengths),
        "chat_input_tokens": _length_summary(chat_lengths),
        "longest_chat_inputs": [
            {"id": record_id, "tokens": tokens} for tokens, record_id in longest[:20]
        ],
    }


def load_and_profile(
    records: Sequence[MathRecord],
    *,
    model_id: str = OFFICIAL_MODEL_ID,
    revision: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    local_files_only: bool = True,
) -> dict[str, Any]:
    """Load the official tokenizer and return a provenance-rich profile."""

    return load_and_profile_datasets(
        {"records": records},
        model_id=model_id,
        revision=revision,
        system_prompt=system_prompt,
        local_files_only=local_files_only,
    )["records"]


def load_pinned_tokenizer(
    *,
    model_id: str = OFFICIAL_MODEL_ID,
    revision: str,
    local_files_only: bool = True,
) -> tuple[TokenizerLike, dict[str, Any]]:
    """Load one exact tokenizer snapshot and return verified file provenance."""

    revision = validate_model_identity(model_id, revision)

    from transformers import AutoTokenizer
    from transformers.utils.hub import cached_file

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        local_files_only=local_files_only,
        trust_remote_code=False,
        use_fast=True,
    )
    resolved_commit = tokenizer.init_kwargs.get("_commit_hash")
    file_digests: dict[str, dict[str, str | int]] = {}
    snapshot_commits: set[str] = set()
    required_files = {"config.json", "tokenizer.json", "tokenizer_config.json"}
    for filename in (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "LICENSE",
    ):
        try:
            located = cached_file(
                model_id,
                filename,
                revision=revision,
                local_files_only=local_files_only,
                _raise_exceptions_for_gated_repo=False,
                _raise_exceptions_for_missing_entries=False,
            )
        except OSError:
            located = None
        if located is not None:
            path = Path(located)
            inferred_commit = _snapshot_commit_from_path(path)
            if inferred_commit is not None:
                snapshot_commits.add(inferred_commit)
            size_bytes = path.stat().st_size
            if size_bytes <= 0:
                raise RuntimeError(f"cached tokenizer artifact is empty: {filename}")
            file_digests[filename] = {
                "size_bytes": size_bytes,
                "sha256": sha256_file(path),
                "snapshot_commit": inferred_commit or "",
            }
    missing = sorted(required_files - set(file_digests))
    if missing:
        raise RuntimeError(f"required tokenizer/config files are not cached: {missing!r}")
    normalized_resolved = resolved_commit.lower() if isinstance(resolved_commit, str) else None
    if snapshot_commits != {revision}:
        raise RuntimeError(
            "tokenizer/config artifacts are not from exactly the requested snapshot: "
            f"requested={revision!r}, observed={sorted(snapshot_commits)!r}"
        )
    if normalized_resolved is not None and normalized_resolved != revision:
        raise RuntimeError(
            "loaded tokenizer commit does not match requested immutable revision: "
            f"loaded={normalized_resolved!r}, requested={revision!r}"
        )
    provenance: dict[str, Any] = {
        "model_id": model_id,
        "requested_revision": revision,
        "resolved_commit": revision,
        "tokenizer_class": type(tokenizer).__name__,
        "transformers_version": importlib.metadata.version("transformers"),
        "tokenizers_version": importlib.metadata.version("tokenizers"),
        "local_files_only": local_files_only,
        "files": file_digests,
    }
    return tokenizer, provenance


def load_and_profile_datasets(
    datasets: Mapping[str, Sequence[MathRecord]],
    *,
    model_id: str = OFFICIAL_MODEL_ID,
    revision: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    local_files_only: bool = True,
) -> dict[str, dict[str, Any]]:
    """Profile multiple datasets with one pinned tokenizer and identical provenance."""

    if not datasets:
        raise ValueError("datasets must not be empty")
    tokenizer, provenance = load_pinned_tokenizer(
        model_id=model_id,
        revision=revision,
        local_files_only=local_files_only,
    )
    reports: dict[str, dict[str, Any]] = {}
    for name, records in datasets.items():
        if not isinstance(name, str) or not name:
            raise ValueError("dataset names must be non-empty strings")
        report = profile_records(records, tokenizer, system_prompt=system_prompt)
        report["provenance"] = provenance.copy()
        reports[name] = report
    return reports


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "load_and_profile",
    "load_and_profile_datasets",
    "load_pinned_tokenizer",
    "profile_records",
]

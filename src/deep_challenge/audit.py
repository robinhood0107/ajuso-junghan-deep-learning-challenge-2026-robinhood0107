"""Reproducible, dependency-free audit metrics for the competition CSV files."""

from __future__ import annotations

import platform
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .data import CsvDataset, MathRecord, load_leaderboard_csv, load_train_csv
from .provenance import build_source_tree_manifest
from .quality import (
    QualityFlag,
    assess_question,
    math_aware_fingerprint,
    number_masked_template_fingerprint,
    source_format_fingerprint,
)

AUDIT_VERSION = "data-audit-v3"
FILTERED_AUDIT_VERSION = "data-audit-v4-filtered"
_WORD_RE = re.compile(r"\S+")


def quantile_r7(values: Sequence[int | float], probability: float) -> float:
    """Return the Hyndman-Fan type 7 / NumPy-default linear quantile."""

    if not values:
        raise ValueError("values must not be empty")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between zero and one")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _distribution(values: Sequence[int | float]) -> dict[str, float | int]:
    return {
        "min": min(values),
        "p50": quantile_r7(values, 0.50),
        "p75": quantile_r7(values, 0.75),
        "p90": quantile_r7(values, 0.90),
        "p95": quantile_r7(values, 0.95),
        "p99": quantile_r7(values, 0.99),
        "max": max(values),
    }


def _question_summary(dataset: CsvDataset) -> dict[str, Any]:
    characters = [len(record.question_raw) for record in dataset]
    words = [len(_WORD_RE.findall(record.question_raw)) for record in dataset]
    flag_counts: Counter[str] = Counter()
    for record in dataset:
        flag_counts.update(flag.value for flag in assess_question(record.question_raw).flags)
    return {
        "characters": _distribution(characters),
        "words": _distribution(words),
        "multiline_count": sum(
            "\n" in record.question_raw or "\r" in record.question_raw
            for record in dataset
        ),
        "non_ascii_count": sum(
            any(ord(character) > 127 for character in record.question_raw)
            for record in dataset
        ),
        "quality_flag_counts": dict(sorted(flag_counts.items())),
    }


def _answer_summary(train: CsvDataset) -> dict[str, Any]:
    answers = [record.answer for record in train]
    if any(answer is None for answer in answers):
        raise ValueError("train audit requires every answer")
    integers = [int(answer) for answer in answers if answer is not None]
    absolute = [abs(answer) for answer in integers]
    frequencies = Counter(integers)
    most_common_answer, most_common_count = min(
        frequencies.items(), key=lambda item: (-item[1], item[0])
    )
    return {
        "unique_count": len(frequencies),
        "minimum": min(integers),
        "maximum": max(integers),
        "negative_count": sum(answer < 0 for answer in integers),
        "zero_count": sum(answer == 0 for answer in integers),
        "positive_count": sum(answer > 0 for answer in integers),
        "most_common_answer": most_common_answer,
        "most_common_count": most_common_count,
        "absolute_distribution": _distribution(absolute),
        "absolute_le_10_count": sum(answer <= 10 for answer in absolute),
        "absolute_le_100_count": sum(answer <= 100 for answer in absolute),
        "absolute_le_1000_count": sum(answer <= 1000 for answer in absolute),
        "absolute_ge_1e6_count": sum(answer >= 1_000_000 for answer in absolute),
        "absolute_ge_1e12_count": sum(answer >= 1_000_000_000_000 for answer in absolute),
    }


def _groups_by(
    records: Iterable[MathRecord], key: Callable[[str], str]
) -> dict[str, tuple[str, ...]]:
    groups: defaultdict[str, list[str]] = defaultdict(list)
    for record in records:
        groups[key(record.question_raw)].append(record.id)
    return {
        fingerprint: tuple(sorted(ids))
        for fingerprint, ids in sorted(groups.items())
    }


def _duplicate_summary(groups: dict[str, tuple[str, ...]]) -> dict[str, Any]:
    duplicate_groups = [ids for ids in groups.values() if len(ids) > 1]
    return {
        "group_count": len(duplicate_groups),
        "row_count": sum(len(ids) for ids in duplicate_groups),
        "max_group_size": max((len(ids) for ids in duplicate_groups), default=1),
        "groups": duplicate_groups,
    }


def _cross_overlap(
    train: CsvDataset,
    leaderboard: CsvDataset,
    key: Callable[[str], str],
) -> dict[str, Any]:
    train_groups = _groups_by(train, key)
    leaderboard_groups = _groups_by(leaderboard, key)
    shared = sorted(set(train_groups) & set(leaderboard_groups))
    return {
        "shared_fingerprint_count": len(shared),
        "leaderboard_row_count": sum(len(leaderboard_groups[value]) for value in shared),
        "pairs": [
            {
                "train_ids": train_groups[value],
                "leaderboard_ids": leaderboard_groups[value],
            }
            for value in shared
        ],
    }


def build_data_audit_report(
    train_path: str | Path,
    leaderboard_path: str | Path,
    *,
    source_tree_root: str | Path | None = None,
    source_tree_excluded_paths: Iterable[str | Path] = (),
) -> dict[str, Any]:
    """Load both official files and return a content-addressed audit report."""

    train = load_train_csv(train_path)
    leaderboard = load_leaderboard_csv(leaderboard_path)
    return build_data_audit_report_from_datasets(
        train,
        leaderboard,
        source_tree_root=source_tree_root,
        source_tree_excluded_paths=source_tree_excluded_paths,
    )


def build_data_audit_report_from_datasets(
    train: CsvDataset,
    leaderboard: CsvDataset,
    *,
    audit_version: str = AUDIT_VERSION,
    eligibility: dict[str, Any] | None = None,
    source_tree_root: str | Path | None = None,
    source_tree_excluded_paths: Iterable[str | Path] = (),
) -> dict[str, Any]:
    """Audit already-validated datasets, including an optional eligibility overlay.

    The official 2026-08-03 correction is represented as an immutable exclusion
    overlay instead of rewriting the organizer train CSV.  Callers can therefore
    pass a filtered :class:`CsvDataset` while retaining the raw file manifest and
    recording the exclusion manifest in ``eligibility``.
    """

    if train.kind.value != "train":
        raise ValueError("train dataset must have kind='train'")
    if leaderboard.kind.value != "leaderboard":
        raise ValueError("leaderboard dataset must have kind='leaderboard'")
    if not train.records or not leaderboard.records:
        raise ValueError("audit datasets must not be empty")
    if not isinstance(audit_version, str) or not audit_version:
        raise ValueError("audit_version must be a non-empty string")
    canonical_groups = _groups_by(train, math_aware_fingerprint)
    source_format_groups = _groups_by(train, source_format_fingerprint)
    template_groups = _groups_by(train, number_masked_template_fingerprint)
    report: dict[str, Any] = {
        "audit_version": audit_version,
        "method": {
            "python": platform.python_version(),
            "csv_parser": "python-csv-excel-strict-newline-empty",
            "quantile": "Hyndman-Fan type 7; h=(n-1)q; linear interpolation",
            "normalization": (
                "raw preserved; comparison view applies CRLF/CR->LF, NBSP->space, "
                "Unicode NFKC; canonical_math_text additionally casefolds and normalizes "
                "representational TeX delimiters/spacing while preserving operators"
            ),
            "source_format_normalization": (
                "canonical math text plus narrowly scoped removal of leading exercise "
                "numbers, trailing standalone # markers, and spaces before ?.!"
            ),
            "number_masked_template": (
                "numeric magnitudes are masked but signs are preserved; this is a soft "
                "candidate signal and must not be used for automatic hard clustering"
            ),
            "fuzzy_similarity": "not computed in deterministic audit v3",
            "tokenizer": "not computed in data audit; use tokenizer-profile command",
        },
        "train": {
            "manifest": asdict(train.manifest),
            "raw_header": train.raw_header,
            "row_count": len(train),
            "questions": _question_summary(train),
            "answers": _answer_summary(train),
        },
        "leaderboard": {
            "manifest": asdict(leaderboard.manifest),
            "raw_header": leaderboard.raw_header,
            "row_count": len(leaderboard),
            "questions": _question_summary(leaderboard),
        },
        "duplicates": {
            "train_math_aware": _duplicate_summary(canonical_groups),
            "train_source_format": _duplicate_summary(source_format_groups),
            "train_number_masked_template": _duplicate_summary(template_groups),
            "train_leaderboard_math_aware": _cross_overlap(
                train, leaderboard, math_aware_fingerprint
            ),
            "train_leaderboard_source_format": _cross_overlap(
                train, leaderboard, source_format_fingerprint
            ),
            "train_leaderboard_number_masked_template": _cross_overlap(
                train, leaderboard, number_masked_template_fingerprint
            ),
        },
        "quality_contract": {
            "flags_are_triage_not_auto_delete": True,
            "number_masked_templates_are_soft_candidates_only": True,
            "supported_flags": [flag.value for flag in QualityFlag],
        },
    }
    if eligibility is not None:
        report["eligibility"] = eligibility
    if source_tree_root is not None:
        report["source_tree"] = build_source_tree_manifest(
            source_tree_root,
            excluded_paths=source_tree_excluded_paths,
        ).as_dict()
    return report


__all__ = [
    "AUDIT_VERSION",
    "FILTERED_AUDIT_VERSION",
    "build_data_audit_report",
    "build_data_audit_report_from_datasets",
    "quantile_r7",
]

from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path

import pytest

from deep_challenge.data import MathRecord
from deep_challenge.rationale_corpus import (
    DEFAULT_CONCISE_RATIONALE_CONFIG,
    RationaleCorpusArtifactExistsError,
    RationaleCorpusValidationError,
    audit_rationale_corpus,
    build_rationale_corpus,
    load_concise_rationale_config,
    load_verified_rationale_corpus,
)
from deep_challenge.splits import (
    SplitManifest,
    eligible_training_ids,
    make_grouped_split_manifest,
)


def _record(identifier: str) -> MathRecord:
    answer = int(identifier.removeprefix("train-"))
    question = f"Return the integer associated with {identifier}."
    return MathRecord(
        id=identifier,
        question_raw=question,
        question_normalized=question,
        answer_raw=str(answer),
        answer=answer,
        row_number=answer + 1,
    )


def _split_manifest() -> SplitManifest:
    identifiers = tuple(f"train-{index:06d}" for index in range(1, 13))
    return make_grouped_split_manifest(
        identifiers,
        dict(zip(identifiers, identifiers, strict=True)),
        n_folds=2,
        holdout_fraction=0.25,
        seed=20_260_731,
        version="splits-v4-rationale-test",
    )


def _training_records(manifest: SplitManifest, fold: int = 0) -> tuple[MathRecord, ...]:
    return tuple(_record(identifier) for identifier in eligible_training_ids(manifest, fold, ()))


def _row(record: MathRecord) -> dict[str, object]:
    assert isinstance(record.answer, int)
    target = (
        "The identifier directly determines the requested integer.\n"
        f"Final answer: {record.answer}"
    )
    return {
        "schema_version": "gate-b-concise-rationale-row-v1",
        "problem_id": record.id,
        "question_sha256": hashlib.sha256(record.question_raw.encode()).hexdigest(),
        "target_text": target,
        "target_sha256": hashlib.sha256(target.encode()).hexdigest(),
        "teacher": {
            "provider": "local-test-provider",
            "model_id": "teacher-test-model",
            "model_revision": "teacher-test-revision",
            "prompt_sha256": "1" * 64,
            "generation_config_sha256": "2" * 64,
            "seed": 7,
            "sample_index": 0,
            "raw_generation_sha256": "3" * 64,
            "reference_answer_in_prompt": False,
            "network_scope": "training_only",
        },
        "verification": {
            "status": "accepted",
            "method": "reference_answer_exact_match",
            "leaderboard_or_test_used": False,
            "locked_holdout_accessed": False,
            "tool_used": False,
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _build_pair(
    tmp_path: Path,
) -> tuple[SplitManifest, tuple[MathRecord, ...], Path, Path, str]:
    manifest = _split_manifest()
    records = _training_records(manifest)
    source = tmp_path / "teacher.jsonl"
    _write_jsonl(source, [_row(record) for record in reversed(records)])
    output_dir = tmp_path / "corpus"
    output_dir.mkdir()
    corpus = output_dir / "rationales.jsonl"
    corpus_manifest = output_dir / "manifest.json"
    config_file_sha = "c" * 64
    result = build_rationale_corpus(
        source,
        records,
        split_manifest=manifest,
        fold=0,
        excluded_ids=(),
        candidate_config_file_sha256=config_file_sha,
        output_jsonl=corpus,
        output_manifest=corpus_manifest,
    )
    assert result.record_count == len(records)
    return manifest, records, corpus, corpus_manifest, config_file_sha


def test_tracked_concise_rationale_config_has_exact_semantic_hash() -> None:
    config, file_sha = load_concise_rationale_config(
        Path("configs/gate_b/rtx4070-super-12gb-concise-rationale-v1.json")
    )

    assert config == DEFAULT_CONCISE_RATIONALE_CONFIG
    assert config.sha256 == "75a315b638481a0c8213c413aa3a1253d269776d08bd2252b68654fb38c3f053"
    assert len(file_sha) == 64


def test_build_audit_and_reload_rationale_corpus_without_raw_audit_fields(
    tmp_path: Path,
) -> None:
    manifest, records, corpus, corpus_manifest, config_file_sha = _build_pair(tmp_path)
    audit_path = tmp_path / "rationale-audit.json"

    audit_result = audit_rationale_corpus(
        corpus,
        corpus_manifest,
        records,
        split_manifest=manifest,
        fold=0,
        excluded_ids=(),
        candidate_config_file_sha256=config_file_sha,
        output_path=audit_path,
    )
    evidence = load_verified_rationale_corpus(
        corpus,
        corpus_manifest,
        records,
        split_manifest=manifest,
        fold=0,
        excluded_ids=(),
        candidate_config_file_sha256=config_file_sha,
        audit_path=audit_path,
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))

    assert audit_result.record_count == len(records)
    assert evidence.record_count == len(records)
    assert tuple(row.problem_id for row in evidence.rows) == eligible_training_ids(
        manifest, 0, ()
    )
    assert evidence.audit_sha256 == audit_result.sha256
    assert audit["status"] == "green"
    assert audit["answer_verified_count"] == len(records)
    assert audit["raw_rationale_serialized"] is False
    assert audit["problem_id_serialized"] is False
    assert audit["reference_answer_serialized"] is False
    serialized_audit = audit_path.read_text(encoding="utf-8")
    assert records[0].id not in serialized_audit
    assert records[0].question_raw not in serialized_audit


@pytest.mark.parametrize(
    "mutation,expected",
    [
        ("wrong_answer", "final line does not exactly match"),
        ("marker_conflict", "exactly one final-answer marker"),
        ("tool_used", "verification.tool_used must remain false"),
        ("leaderboard_used", "verification.leaderboard_or_test_used must remain false"),
        ("holdout_used", "verification.locked_holdout_accessed must remain false"),
        ("network_scope", "network_scope must be exactly"),
        ("answer_visible", "reference answer must remain hidden"),
        ("question_sha", "question SHA does not match"),
        ("unverified", "verification.status must be accepted"),
        ("too_long", "rationale character count is outside"),
    ],
)
def test_rationale_rows_fail_closed_on_safety_or_quality_drift(
    tmp_path: Path,
    mutation: str,
    expected: str,
) -> None:
    manifest = _split_manifest()
    records = _training_records(manifest)
    rows = [_row(record) for record in records]
    row = deepcopy(rows[0])
    if mutation == "wrong_answer":
        row["target_text"] = "A sufficiently long explanation.\nFinal answer: 999"
    elif mutation == "marker_conflict":
        answer = records[0].answer
        row["target_text"] = (
            f"Final answer: {answer}\nA sufficiently long explanation.\n"
            f"Final answer: {answer}"
        )
    elif mutation == "tool_used":
        row["verification"]["tool_used"] = True  # type: ignore[index]
    elif mutation == "leaderboard_used":
        row["verification"]["leaderboard_or_test_used"] = True  # type: ignore[index]
    elif mutation == "holdout_used":
        row["verification"]["locked_holdout_accessed"] = True  # type: ignore[index]
    elif mutation == "network_scope":
        row["teacher"]["network_scope"] = "test"  # type: ignore[index]
    elif mutation == "answer_visible":
        row["teacher"]["reference_answer_in_prompt"] = True  # type: ignore[index]
    elif mutation == "question_sha":
        row["question_sha256"] = "f" * 64
    elif mutation == "unverified":
        row["verification"]["status"] = "rejected"  # type: ignore[index]
    elif mutation == "too_long":
        answer = records[0].answer
        row["target_text"] = f"{'x' * 1501}\nFinal answer: {answer}"
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(mutation)
    if isinstance(row.get("target_text"), str):
        row["target_sha256"] = hashlib.sha256(row["target_text"].encode()).hexdigest()
    rows[0] = row
    source = tmp_path / "teacher.jsonl"
    _write_jsonl(source, rows)
    output = tmp_path / "output"
    output.mkdir()

    with pytest.raises(RationaleCorpusValidationError, match=expected):
        build_rationale_corpus(
            source,
            records,
            split_manifest=manifest,
            fold=0,
            excluded_ids=(),
            candidate_config_file_sha256="c" * 64,
            output_jsonl=output / "rationales.jsonl",
            output_manifest=output / "manifest.json",
        )


def test_rationale_corpus_requires_exact_fold_training_coverage(tmp_path: Path) -> None:
    manifest = _split_manifest()
    records = _training_records(manifest)
    source = tmp_path / "teacher.jsonl"
    _write_jsonl(source, [_row(record) for record in records[:-1]])
    output = tmp_path / "output"
    output.mkdir()

    with pytest.raises(RationaleCorpusValidationError, match="exactly cover"):
        build_rationale_corpus(
            source,
            records,
            split_manifest=manifest,
            fold=0,
            excluded_ids=(),
            candidate_config_file_sha256="c" * 64,
            output_jsonl=output / "rationales.jsonl",
            output_manifest=output / "manifest.json",
        )


def test_rationale_pair_and_audit_refuse_overwrite_or_tampering(tmp_path: Path) -> None:
    manifest, records, corpus, corpus_manifest, config_file_sha = _build_pair(tmp_path)
    audit_path = tmp_path / "rationale-audit.json"
    audit_rationale_corpus(
        corpus,
        corpus_manifest,
        records,
        split_manifest=manifest,
        fold=0,
        excluded_ids=(),
        candidate_config_file_sha256=config_file_sha,
        output_path=audit_path,
    )

    with pytest.raises(RationaleCorpusArtifactExistsError, match="overwrite"):
        audit_rationale_corpus(
            corpus,
            corpus_manifest,
            records,
            split_manifest=manifest,
            fold=0,
            excluded_ids=(),
            candidate_config_file_sha256=config_file_sha,
            output_path=audit_path,
        )

    tampered = json.loads(audit_path.read_text(encoding="utf-8"))
    tampered["record_count"] += 1
    audit_path.write_text(json.dumps(tampered) + "\n", encoding="utf-8")
    with pytest.raises(RationaleCorpusValidationError, match="audit binding mismatch"):
        load_verified_rationale_corpus(
            corpus,
            corpus_manifest,
            records,
            split_manifest=manifest,
            fold=0,
            excluded_ids=(),
            candidate_config_file_sha256=config_file_sha,
            audit_path=audit_path,
        )


def test_rationale_pair_rolls_back_first_file_on_second_publish_io_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _split_manifest()
    records = _training_records(manifest)
    source = tmp_path / "teacher.jsonl"
    _write_jsonl(source, [_row(record) for record in records])
    output = tmp_path / "output"
    output.mkdir()
    corpus = output / "rationales.jsonl"
    corpus_manifest = output / "manifest.json"
    original_link = os.link

    def fail_second_link(source_path: object, target_path: object) -> None:
        if Path(target_path) == corpus_manifest.resolve():
            raise OSError("simulated second-link I/O failure")
        original_link(source_path, target_path)

    monkeypatch.setattr(os, "link", fail_second_link)

    with pytest.raises(OSError, match="second-link"):
        build_rationale_corpus(
            source,
            records,
            split_manifest=manifest,
            fold=0,
            excluded_ids=(),
            candidate_config_file_sha256="4" * 64,
            output_jsonl=corpus,
            output_manifest=corpus_manifest,
        )

    assert not corpus.exists()
    assert not corpus_manifest.exists()
    assert not tuple(output.glob(".*.tmp"))


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("teacher_identity_counts", []),
        ("rationale_character_count", {"min": 0, "max": 0, "mean": 0.0}),
        ("verification_method_counts", {}),
    ],
)
def test_rationale_manifest_recomputes_every_derived_aggregate(
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    manifest, records, corpus, corpus_manifest, config_file_sha = _build_pair(tmp_path)
    payload = json.loads(corpus_manifest.read_text(encoding="utf-8"))
    payload[field] = replacement
    corpus_manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(RationaleCorpusValidationError, match="derived manifest fields"):
        load_verified_rationale_corpus(
            corpus,
            corpus_manifest,
            records,
            split_manifest=manifest,
            fold=0,
            excluded_ids=(),
            candidate_config_file_sha256=config_file_sha,
        )


def test_rationale_audit_recomputes_redacted_aggregate_fields(tmp_path: Path) -> None:
    manifest, records, corpus, corpus_manifest, config_file_sha = _build_pair(tmp_path)
    audit_path = tmp_path / "rationale-audit.json"
    audit_rationale_corpus(
        corpus,
        corpus_manifest,
        records,
        split_manifest=manifest,
        fold=0,
        excluded_ids=(),
        candidate_config_file_sha256=config_file_sha,
        output_path=audit_path,
    )
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    payload["teacher_identity_count"] += 1
    without_hash = dict(payload)
    without_hash.pop("payload_sha256")
    payload["payload_sha256"] = hashlib.sha256(
        json.dumps(
            without_hash,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    audit_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(RationaleCorpusValidationError, match="audit binding mismatch"):
        load_verified_rationale_corpus(
            corpus,
            corpus_manifest,
            records,
            split_manifest=manifest,
            fold=0,
            excluded_ids=(),
            candidate_config_file_sha256=config_file_sha,
            audit_path=audit_path,
        )

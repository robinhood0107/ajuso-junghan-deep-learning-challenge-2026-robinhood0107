from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

import deep_challenge.cli as cli_module
from deep_challenge.cli import main
from deep_challenge.data import MathRecord
from deep_challenge.rationale_corpus import build_rationale_corpus
from deep_challenge.rationale_materialization import (
    FinalizedTeacherBank,
    TeacherBankMaterializationResult,
    TeacherBankMaterializationValidationError,
    load_teacher_bank_materialization_manifest,
    materialize_teacher_bank_source,
)
from deep_challenge.splits import (
    SplitManifest,
    eligible_training_ids,
    eligible_validation_ids,
    make_grouped_split_manifest,
)
from deep_challenge.teacher_rationale import (
    CodexCommandResult,
    TeacherExecutionConfig,
    create_teacher_plan,
    finalize_teacher_bank,
    run_teacher_plan,
)


def _record(index: int) -> MathRecord:
    identifier = f"train-{index:06d}"
    question = f"Synthetic materialization question {index}."
    return MathRecord(
        id=identifier,
        question_raw=question,
        question_normalized=question,
        answer_raw=str(700_000 + index),
        answer=700_000 + index,
        row_number=index + 1,
    )


def _target(answer: int) -> str:
    return f"The condition determines one integer.\nFinal answer: {answer}"


def _events(items: list[dict[str, str]]) -> str:
    message = json.dumps({"items": items}, ensure_ascii=False, separators=(",", ":"))
    rows = (
        {"type": "thread.started", "thread_id": "synthetic-materialization"},
        {"type": "turn.started"},
        {"type": "item.completed", "item": {"type": "agent_message", "text": message}},
        {
            "type": "turn.completed",
            "usage": {"input_tokens": 9, "output_tokens": 5},
        },
    )
    return "\n".join(json.dumps(row, separators=(",", ":")) for row in rows) + "\n"


def _finalized_bank(
    tmp_path: Path,
    label: str,
    records: tuple[MathRecord, ...],
    *,
    seed: int = 20_260_731,
) -> FinalizedTeacherBank:
    plan = create_teacher_plan(
        records,
        tuple(record.id for record in records),
        tmp_path / f"{label}-plan",
        chunk_size=64,
        execution=TeacherExecutionConfig(seed=seed),
    )
    answers = {record.id: record.answer for record in records}

    def runner(_command: tuple[str, ...]) -> CodexCommandResult:
        items = [
            {
                "problem_id": problem_id,
                "target_text": _target(int(answers[problem_id])),
            }
            for problem_id in plan.problem_ids
        ]
        return CodexCommandResult(
            stdout=_events(items),
            stderr="",
            returncode=0,
            latency_ms=2,
        )

    run_teacher_plan(plan.plan_dir, runner)
    source = tmp_path / f"{label}-source.jsonl"
    manifest = tmp_path / f"{label}-source.manifest.json"
    result = finalize_teacher_bank(
        plan.plan_dir,
        records,
        output_jsonl=source,
        output_manifest=manifest,
    )
    assert result.complete
    return FinalizedTeacherBank(plan.plan_dir, source, manifest)


def _split_fixture() -> tuple[tuple[MathRecord, ...], SplitManifest]:
    records = tuple(_record(index) for index in range(1, 37))
    identifiers = tuple(record.id for record in records)
    manifest = make_grouped_split_manifest(
        identifiers,
        dict(zip(identifiers, identifiers, strict=True)),
        n_folds=3,
        holdout_fraction=0.25,
        seed=20_260_731,
        version="teacher-materialization-test-v1",
    )
    return records, manifest


def _records_for_ids(
    records: tuple[MathRecord, ...], identifiers: tuple[str, ...]
) -> tuple[MathRecord, ...]:
    by_id = {record.id: record for record in records}
    return tuple(by_id[identifier] for identifier in identifiers)


def _read_ids(path: Path) -> tuple[str, ...]:
    return tuple(
        json.loads(line)["problem_id"]
        for line in path.read_text(encoding="utf-8").splitlines()
    )


def test_v1_fold_zero_materialization_is_private_and_rationale_builder_compatible(
    tmp_path: Path,
) -> None:
    records, manifest = _split_fixture()
    v1_ids = eligible_training_ids(manifest, 0, ())
    v1 = _finalized_bank(tmp_path, "v1", _records_for_ids(records, v1_ids))
    output_dir = tmp_path / "materialized"
    output_dir.mkdir()
    source = output_dir / "fold-0-source.jsonl"
    source_manifest = output_dir / "fold-0-source.manifest.json"

    result = materialize_teacher_bank_source(
        (v1,),
        _records_for_ids(records, v1_ids),
        split_manifest=manifest,
        fold=0,
        excluded_ids=(),
        output_jsonl=source,
        output_manifest=source_manifest,
        allow_unqualified_synthetic_banks=True,
    )

    assert result.record_count == len(v1_ids)
    assert result.promotion_authorization_verified is False
    assert _read_ids(source) == v1_ids
    raw_free = source_manifest.read_text(encoding="utf-8")
    assert records[0].id not in raw_free
    assert records[0].question_raw not in raw_free
    assert str(records[0].answer) not in raw_free
    loaded_manifest = load_teacher_bank_materialization_manifest(source_manifest)
    assert loaded_manifest["training_ids_sha256"] == result.training_ids_sha256
    assert loaded_manifest["source_bank_count"] == 1
    assert loaded_manifest["raw_rationale_serialized"] is False
    assert loaded_manifest["promotion_authorization_verified"] is False

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    corpus = build_rationale_corpus(
        source,
        _records_for_ids(records, v1_ids),
        split_manifest=manifest,
        fold=0,
        excluded_ids=(),
        candidate_config_file_sha256="c" * 64,
        output_jsonl=corpus_dir / "rationales.jsonl",
        output_manifest=corpus_dir / "manifest.json",
    )
    assert corpus.record_count == len(v1_ids)


def test_materializer_rejects_unqualified_finalized_bank_outside_explicit_fixture_mode(
    tmp_path: Path,
) -> None:
    records, manifest = _split_fixture()
    v1_ids = eligible_training_ids(manifest, 0, ())
    v1 = _finalized_bank(tmp_path, "v1", _records_for_ids(records, v1_ids))
    output_dir = tmp_path / "materialized"
    output_dir.mkdir()

    with pytest.raises(
        TeacherBankMaterializationValidationError,
        match="pilot promotion authorization",
    ):
        materialize_teacher_bank_source(
            (v1,),
            _records_for_ids(records, v1_ids),
            split_manifest=manifest,
            fold=0,
            excluded_ids=(),
            output_jsonl=output_dir / "must-not-publish.jsonl",
            output_manifest=output_dir / "must-not-publish.manifest.json",
        )


def test_later_fold_materialization_unions_v1_v2_and_excludes_its_validation(
    tmp_path: Path,
) -> None:
    records, manifest = _split_fixture()
    v1_ids = eligible_training_ids(manifest, 0, ())
    v2_ids = eligible_validation_ids(manifest, 0, ())
    v1 = _finalized_bank(tmp_path, "v1", _records_for_ids(records, v1_ids))
    v2 = _finalized_bank(tmp_path, "v2", _records_for_ids(records, v2_ids))
    expected = eligible_training_ids(manifest, 1, ())
    output_dir = tmp_path / "materialized"
    output_dir.mkdir()
    source = output_dir / "fold-1-source.jsonl"
    source_manifest = output_dir / "fold-1-source.manifest.json"

    result = materialize_teacher_bank_source(
        (v2, v1),
        _records_for_ids(records, expected),
        split_manifest=manifest,
        fold=1,
        excluded_ids=(),
        output_jsonl=source,
        output_manifest=source_manifest,
        allow_unqualified_synthetic_banks=True,
    )

    selected_ids = _read_ids(source)
    assert selected_ids == expected
    assert set(selected_ids) & set(eligible_validation_ids(manifest, 1, ())) == set()
    assert set(selected_ids) & set(v2_ids)
    assert result.source_bank_count == 2


def test_materialization_rejects_missing_duplicate_and_scope_crossing_banks(
    tmp_path: Path,
) -> None:
    records, manifest = _split_fixture()
    v1_ids = eligible_training_ids(manifest, 0, ())
    v2_ids = eligible_validation_ids(manifest, 0, ())
    v1 = _finalized_bank(tmp_path, "v1", _records_for_ids(records, v1_ids))
    v1_overlap = _finalized_bank(
        tmp_path,
        "v1-overlap",
        _records_for_ids(records, v1_ids),
        seed=20_260_732,
    )
    v2 = _finalized_bank(tmp_path, "v2", _records_for_ids(records, v2_ids))
    expected_later = eligible_training_ids(manifest, 1, ())
    output_dir = tmp_path / "materialized"
    output_dir.mkdir()

    with pytest.raises(TeacherBankMaterializationValidationError, match="do not cover"):
        materialize_teacher_bank_source(
            (v1,),
            _records_for_ids(records, expected_later),
            split_manifest=manifest,
            fold=1,
            excluded_ids=(),
            output_jsonl=output_dir / "missing.jsonl",
            output_manifest=output_dir / "missing.manifest.json",
            allow_unqualified_synthetic_banks=True,
        )
    with pytest.raises(
        TeacherBankMaterializationValidationError,
        match="duplicate or conflicting",
    ):
        materialize_teacher_bank_source(
            (v1, v1_overlap, v2),
            _records_for_ids(records, expected_later),
            split_manifest=manifest,
            fold=1,
            excluded_ids=(),
            output_jsonl=output_dir / "duplicate.jsonl",
            output_manifest=output_dir / "duplicate.manifest.json",
            allow_unqualified_synthetic_banks=True,
        )

    holdout_ids = manifest.final_holdout_ids()
    holdout = _finalized_bank(
        tmp_path,
        "holdout",
        _records_for_ids(records, holdout_ids),
    )
    with pytest.raises(
        TeacherBankMaterializationValidationError,
        match="crosses the eligible development-CV scope",
    ):
        materialize_teacher_bank_source(
            (v1, holdout),
            _records_for_ids(records, v1_ids),
            split_manifest=manifest,
            fold=0,
            excluded_ids=(),
            output_jsonl=output_dir / "scope.jsonl",
            output_manifest=output_dir / "scope.manifest.json",
            allow_unqualified_synthetic_banks=True,
        )


def test_materialization_rejects_tampered_finalized_source_provenance(
    tmp_path: Path,
) -> None:
    records, manifest = _split_fixture()
    v1_ids = eligible_training_ids(manifest, 0, ())
    v1 = _finalized_bank(tmp_path, "v1", _records_for_ids(records, v1_ids))
    Path(v1.source_jsonl).write_text("{}\n", encoding="utf-8")
    output_dir = tmp_path / "materialized"
    output_dir.mkdir()

    with pytest.raises(
        TeacherBankMaterializationValidationError,
        match="plan/source/manifest provenance",
    ):
        materialize_teacher_bank_source(
            (v1,),
            _records_for_ids(records, v1_ids),
            split_manifest=manifest,
            fold=0,
            excluded_ids=(),
            output_jsonl=output_dir / "tampered.jsonl",
            output_manifest=output_dir / "tampered.manifest.json",
            allow_unqualified_synthetic_banks=True,
        )


def _write_csv(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)


def test_cli_materializer_rederives_exact_fold_contract_without_raw_stdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    train = tmp_path / "train.csv"
    _write_csv(
        train,
        [
            ["id", "question", "answer"],
            *[
                [f"train-{index:06d}", f"CLI materialization question {index}", str(index)]
                for index in range(1, 25)
            ],
        ],
    )
    split = tmp_path / "splits.json"
    assert (
        main(
            [
                "build-splits",
                "--train",
                str(train),
                "--folds",
                "2",
                "--seed",
                "13",
                "--output",
                str(split),
            ]
        )
        == 0
    )
    split_payload = json.loads(split.read_text(encoding="utf-8"))
    manifest = SplitManifest.from_dict(split_payload["split"])
    shard = tmp_path / "development-shard"
    assert (
        main(
            [
                "build-development-shard",
                "--train",
                str(train),
                "--split-artifact",
                str(split),
                "--expected-train-sha256",
                hashlib.sha256(train.read_bytes()).hexdigest(),
                "--expected-split-sha256",
                manifest.sha256,
                "--output-dir",
                str(shard),
            ]
        )
        == 0
    )
    exclusions = tmp_path / "exclusions.csv"
    _write_csv(exclusions, [["id"]])
    captured: dict[str, object] = {}

    def fake_materialize(
        banks: list[FinalizedTeacherBank],
        records: tuple[MathRecord, ...],
        **kwargs: object,
    ) -> TeacherBankMaterializationResult:
        captured["banks"] = banks
        captured["ids"] = tuple(record.id for record in records)
        captured["kwargs"] = kwargs
        assert kwargs.get("allow_unqualified_synthetic_banks", False) is False
        return TeacherBankMaterializationResult(
            records_path=str(tmp_path / "private-source.jsonl"),
            records_sha256="a" * 64,
            manifest_path=str(tmp_path / "private-source.manifest.json"),
            manifest_sha256="b" * 64,
            record_count=len(records),
            source_bank_count=len(banks),
            training_ids_sha256="c" * 64,
        )

    monkeypatch.setattr(cli_module, "materialize_teacher_bank_source", fake_materialize)
    capsys.readouterr()
    assert (
        main(
            [
                "gate-b-materialize-teacher-bank",
                "--train",
                str(train),
                "--train-exclusions",
                str(exclusions),
                "--split-artifact",
                str(split),
                "--fold",
                "0",
                "--expected-train-sha256",
                hashlib.sha256(train.read_bytes()).hexdigest(),
                "--expected-exclusions-sha256",
                hashlib.sha256(exclusions.read_bytes()).hexdigest(),
                "--expected-exclusion-count",
                "0",
                "--expected-split-sha256",
                manifest.sha256,
                "--development-shard",
                str(shard),
                "--expected-development-shard-sha256",
                hashlib.sha256((shard / "CHECKSUMS.sha256").read_bytes()).hexdigest(),
                "--teacher-bank",
                str(tmp_path / "v1-plan"),
                str(tmp_path / "v1-source.jsonl"),
                str(tmp_path / "v1-source.manifest.json"),
                "--output-jsonl",
                str(tmp_path / "output.jsonl"),
                "--output-manifest",
                str(tmp_path / "output.manifest.json"),
            ]
        )
        == 0
    )
    status = json.loads(capsys.readouterr().out)
    assert captured["ids"] == eligible_training_ids(manifest, 0, ())
    assert len(captured["banks"]) == 1
    assert status["raw_rationale_serialized"] is False
    assert status["torch_or_cuda_used"] is False
    assert "CLI materialization question" not in json.dumps(status)
    assert "train-000001" not in json.dumps(status)

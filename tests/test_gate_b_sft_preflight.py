from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from deep_challenge.data import MathRecord
from deep_challenge.gate_b import DEFAULT_GATE_B_CONFIG, GateBValidationError
from deep_challenge.gate_b_sft_preflight import run_sft_encoding_preflight
from deep_challenge.model_preflight import OFFICIAL_MODEL_ID, OFFICIAL_REVISION
from deep_challenge.rationale_corpus import (
    DEFAULT_CONCISE_RATIONALE_CONFIG,
    audit_rationale_corpus,
    build_rationale_corpus,
    load_verified_rationale_corpus,
)
from deep_challenge.splits import (
    eligible_training_ids,
    eligible_validation_ids,
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


def _split():
    identifiers = tuple(f"train-{index:06d}" for index in range(1, 13))
    return make_grouped_split_manifest(
        identifiers,
        dict(zip(identifiers, identifiers, strict=True)),
        n_folds=2,
        holdout_fraction=0.25,
        seed=20_260_731,
        version="splits-v4-sft-preflight-test",
    )


class _Tokenizer:
    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        assert tokenize is True
        assert [item["role"] for item in conversation[:2]] == ["system", "user"]
        prefix = [11, 12, 13]
        return prefix if add_generation_prompt else [*prefix, 99, 100]


class _OverflowTokenizer(_Tokenizer):
    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        del conversation, tokenize
        return [1] if add_generation_prompt else list(range(2_049))


def _provenance() -> dict[str, object]:
    return {
        "model_id": OFFICIAL_MODEL_ID,
        "requested_revision": OFFICIAL_REVISION,
        "resolved_commit": OFFICIAL_REVISION,
        "local_files_only": True,
        "files": {},
    }


def _cv_records(manifest, fold: int = 0) -> tuple[MathRecord, ...]:
    identifiers = {
        *eligible_training_ids(manifest, fold, ()),
        *eligible_validation_ids(manifest, fold, ()),
    }
    return tuple(_record(identifier) for identifier in sorted(identifiers))


def _rationale_evidence(tmp_path: Path, manifest, fold: int = 0):
    training_ids = eligible_training_ids(manifest, fold, ())
    records = tuple(_record(identifier) for identifier in training_ids)
    rows = []
    for record in records:
        target = (
            "The identifier explicitly supplies the requested integer.\n"
            f"Final answer: {record.answer}"
        )
        rows.append(
            {
                "schema_version": "gate-b-concise-rationale-row-v1",
                "problem_id": record.id,
                "question_sha256": hashlib.sha256(
                    record.question_raw.encode("utf-8")
                ).hexdigest(),
                "target_text": target,
                "target_sha256": hashlib.sha256(target.encode("utf-8")).hexdigest(),
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
        )
    source = tmp_path / "teacher.jsonl"
    source.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    corpus = corpus_dir / "rationales.jsonl"
    corpus_manifest = corpus_dir / "manifest.json"
    config_file_sha256 = "4" * 64
    build_rationale_corpus(
        source,
        records,
        split_manifest=manifest,
        fold=fold,
        excluded_ids=(),
        candidate_config_file_sha256=config_file_sha256,
        output_jsonl=corpus,
        output_manifest=corpus_manifest,
    )
    audit = tmp_path / "audit.json"
    audit_rationale_corpus(
        corpus,
        corpus_manifest,
        records,
        split_manifest=manifest,
        fold=fold,
        excluded_ids=(),
        candidate_config_file_sha256=config_file_sha256,
        output_path=audit,
    )
    return load_verified_rationale_corpus(
        corpus,
        corpus_manifest,
        records,
        split_manifest=manifest,
        fold=fold,
        excluded_ids=(),
        candidate_config_file_sha256=config_file_sha256,
        audit_path=audit,
    )


def test_sft_encoding_preflight_is_cpu_only_and_excludes_locked_holdout(
    tmp_path: Path,
) -> None:
    manifest = _split()
    holdout_ids = set(manifest.final_holdout_ids())
    output = tmp_path / "sft-preflight.json"

    result = run_sft_encoding_preflight(
        _cv_records(manifest),
        split_manifest=manifest,
        excluded_ids=(),
        tokenizer=_Tokenizer(),
        tokenizer_provenance=_provenance(),
        train_file_sha256="1" * 64,
        exclusions_file_sha256="2" * 64,
        split_artifact_sha256="3" * 64,
        development_shard_sha256="4" * 64,
        output_path=output,
        folds=(0,),
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert result.sha256 and payload["status"] == "green"
    assert payload["model_weights_loaded"] is False
    assert payload["torch_or_cuda_used"] is False
    assert payload["locked_holdout_accessed"] is False
    assert payload["folds"]["0"]["training"]["count"] == len(
        eligible_training_ids(manifest, 0, ())
    )
    assert payload["folds"]["0"]["validation"]["count"] == len(
        eligible_validation_ids(manifest, 0, ())
    )
    serialized = output.read_text(encoding="utf-8")
    assert not any(problem_id in serialized for problem_id in holdout_ids)
    with pytest.raises(GateBValidationError, match="overwrite"):
        run_sft_encoding_preflight(
            _cv_records(manifest),
            split_manifest=manifest,
            excluded_ids=(),
            tokenizer=_Tokenizer(),
            tokenizer_provenance=_provenance(),
            train_file_sha256="1" * 64,
            exclusions_file_sha256="2" * 64,
            split_artifact_sha256="3" * 64,
            development_shard_sha256="4" * 64,
            output_path=output,
            folds=(0,),
        )


def test_sft_encoding_preflight_rejects_holdout_and_overflow(tmp_path: Path) -> None:
    manifest = _split()
    records = _cv_records(manifest)
    with pytest.raises(GateBValidationError, match="development-CV scope"):
        run_sft_encoding_preflight(
            (*records, _record(manifest.final_holdout_ids()[0])),
            split_manifest=manifest,
            excluded_ids=(),
            tokenizer=_Tokenizer(),
            tokenizer_provenance=_provenance(),
            train_file_sha256="1" * 64,
            exclusions_file_sha256="2" * 64,
            split_artifact_sha256="3" * 64,
            development_shard_sha256="4" * 64,
            output_path=tmp_path / "holdout.json",
            folds=(0,),
        )
    with pytest.raises(GateBValidationError, match="truncation is forbidden"):
        run_sft_encoding_preflight(
            records,
            split_manifest=manifest,
            excluded_ids=(),
            tokenizer=_OverflowTokenizer(),
            tokenizer_provenance=_provenance(),
            train_file_sha256="1" * 64,
            exclusions_file_sha256="2" * 64,
            split_artifact_sha256="3" * 64,
            development_shard_sha256="4" * 64,
            output_path=tmp_path / "overflow.json",
            folds=(0,),
            config=DEFAULT_GATE_B_CONFIG,
        )


def test_sft_encoding_preflight_requires_pinned_local_tokenizer(tmp_path: Path) -> None:
    manifest = _split()
    provenance = _provenance()
    provenance["local_files_only"] = False
    with pytest.raises(GateBValidationError, match="local-files-only"):
        run_sft_encoding_preflight(
            _cv_records(manifest),
            split_manifest=manifest,
            excluded_ids=(),
            tokenizer=_Tokenizer(),
            tokenizer_provenance=provenance,
            train_file_sha256="1" * 64,
            exclusions_file_sha256="2" * 64,
            split_artifact_sha256="3" * 64,
            development_shard_sha256="4" * 64,
            output_path=tmp_path / "bad-provenance.json",
            folds=(0,),
        )


def test_sft_encoding_preflight_binds_audited_concise_rationale_target(
    tmp_path: Path,
) -> None:
    manifest = _split()
    evidence = _rationale_evidence(tmp_path, manifest)
    output = tmp_path / "rationale-sft-preflight.json"

    run_sft_encoding_preflight(
        _cv_records(manifest),
        split_manifest=manifest,
        excluded_ids=(),
        tokenizer=_Tokenizer(),
        tokenizer_provenance=_provenance(),
        train_file_sha256="1" * 64,
        exclusions_file_sha256="2" * 64,
        split_artifact_sha256="3" * 64,
        development_shard_sha256="4" * 64,
        output_path=output,
        folds=(0,),
        rationale_corpus=evidence,
        rationale_config=DEFAULT_CONCISE_RATIONALE_CONFIG,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "gate-b-sft-encoding-preflight-v4"
    assert payload["training_target"] == {
        "kind": "verified_concise_rationale",
        "candidate_config_sha256": evidence.candidate_config_sha256,
        "candidate_config_file_sha256": evidence.candidate_config_file_sha256,
        "corpus_records_sha256": evidence.records_sha256,
        "corpus_manifest_sha256": evidence.manifest_sha256,
        "corpus_audit_sha256": evidence.audit_sha256,
    }
    assert payload["torch_or_cuda_used"] is False
    assert payload["locked_holdout_accessed"] is False

    with pytest.raises(GateBValidationError, match="corpus-bound fold"):
        run_sft_encoding_preflight(
            tuple(_record(assignment.record_id) for assignment in manifest.assignments),
            split_manifest=manifest,
            excluded_ids=(),
            tokenizer=_Tokenizer(),
            tokenizer_provenance=_provenance(),
            train_file_sha256="1" * 64,
            exclusions_file_sha256="2" * 64,
            split_artifact_sha256="3" * 64,
            development_shard_sha256="4" * 64,
            output_path=tmp_path / "all-folds.json",
            rationale_corpus=evidence,
            rationale_config=DEFAULT_CONCISE_RATIONALE_CONFIG,
        )

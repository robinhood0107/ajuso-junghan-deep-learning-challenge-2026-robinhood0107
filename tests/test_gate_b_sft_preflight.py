from __future__ import annotations

import json
from pathlib import Path

import pytest

from deep_challenge.data import MathRecord
from deep_challenge.gate_b import DEFAULT_GATE_B_CONFIG, GateBValidationError
from deep_challenge.gate_b_sft_preflight import run_sft_encoding_preflight
from deep_challenge.model_preflight import OFFICIAL_MODEL_ID, OFFICIAL_REVISION
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

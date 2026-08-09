from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

import deep_challenge.gate_b_prediction as prediction_module
from deep_challenge.data import DatasetValidationError, MathRecord
from deep_challenge.gate_b import GateBValidationError, GenerationRequest, GenerationResult
from deep_challenge.gate_b_prediction import run_frozen_evaluation_inference
from deep_challenge.gate_b_selection import FrozenSelectionMethods
from deep_challenge.splits import make_grouped_split_manifest


class _Backend:
    def __init__(self, checkpoint: str, outputs: dict[str, str]) -> None:
        self.checkpoint_sha256 = checkpoint
        self.outputs = outputs
        self.requests: list[GenerationRequest] = []
        self.closed = False

    def generate(self, request: GenerationRequest) -> GenerationResult:
        assert not self.closed
        self.requests.append(request)
        return GenerationResult(self.outputs[request.problem_id], "stop", 10, 3, 123)

    def close(self) -> None:
        self.closed = True


def _split():
    identifiers = tuple(f"train-{index:06d}" for index in range(1, 13))
    return make_grouped_split_manifest(
        identifiers,
        dict(zip(identifiers, identifiers, strict=True)),
        n_folds=2,
        holdout_fraction=0.25,
        seed=7,
        version="prediction-test-v1",
    )


def _records() -> tuple[MathRecord, ...]:
    return tuple(
        MathRecord(
            id=f"prob_{index:04d}",
            question_raw=f"Return {index}.",
            question_normalized=f"Return {index}.",
            answer_raw=None,
            answer=None,
            row_number=index + 1,
        )
        for index in range(1, 4)
    )


def _methods(manifest) -> FrozenSelectionMethods:
    return FrozenSelectionMethods(
        freeze_path="/frozen.json",
        freeze_sha256="f" * 64,
        split_sha256=manifest.sha256,
        fold=0,
        train_file_sha256="1" * 64,
        exclusions_file_sha256="2" * 64,
        excluded_ids_sha256=hashlib.sha256(b"[]").hexdigest(),
        split_artifact_sha256="4" * 64,
        development_shard_sha256="5" * 64,
        primary_label="primary",
        primary_checkpoint_sha256="a" * 64,
        fallback_label="fallback",
        fallback_checkpoint_sha256="b" * 64,
        routing_policy="primary_then_fallback_only_on_primary_parse_failure",
    )


def _scope_args() -> dict[str, object]:
    return {
        "train_file_sha256": "1" * 64,
        "exclusions_file_sha256": "2" * 64,
        "excluded_ids": (),
        "excluded_ids_sha256": hashlib.sha256(b"[]").hexdigest(),
        "split_artifact_sha256": "4" * 64,
        "development_shard_sha256": "5" * 64,
        "fold": 0,
    }


def _write_evaluation(
    path: Path,
    records: tuple[MathRecord, ...],
    *,
    answered_first: bool = False,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("id", "question", "answer"))
        for index, record in enumerate(records):
            writer.writerow(
                (record.id, record.question_raw, "1" if answered_first and index == 0 else "")
            )


def test_frozen_prediction_uses_fallback_only_for_invalid_primary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _split()
    monkeypatch.setattr(
        prediction_module,
        "validate_frozen_selection_methods",
        lambda *_args, **_kwargs: _methods(manifest),
    )
    records = _records()
    evaluation = tmp_path / "leaderboard.csv"
    _write_evaluation(evaluation, records)
    primary = _Backend(
        "a" * 64,
        {
            "prob_0001": "Final answer: 1",
            "prob_0002": "no integer",
            "prob_0003": "Final answer: 3",
        },
    )
    fallback = _Backend("b" * 64, {"prob_0002": "Final answer: 2"})
    artifact = tmp_path / "raw.json"
    predictions = tmp_path / "predictions.json"
    clocks = iter([0, 1_000_000] * 4)

    result = run_frozen_evaluation_inference(
        dataset_role="leaderboard",
        evaluation_file_path=evaluation,
        expected_evaluation_sha256=hashlib.sha256(evaluation.read_bytes()).hexdigest(),
        split_manifest=manifest,
        **_scope_args(),
        freeze_artifact=tmp_path / "ignored-freeze.json",
        primary_backend=primary,
        fallback_backend=fallback,
        artifact_path=artifact,
        predictions_path=predictions,
        clock_ns=lambda: next(clocks),
    )

    assert result.invalid_count == 0
    assert json.loads(predictions.read_text(encoding="utf-8")) == {
        "prob_0001": 1,
        "prob_0002": 2,
        "prob_0003": 3,
    }
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["status"] == "complete"
    assert payload["fallback_invocation_count"] == 1
    assert payload["model_residency_policy"] == "sequential_single_backend"
    assert payload["internet_or_external_api_used"] is False
    assert primary.closed is True
    assert [request.problem_id for request in fallback.requests] == ["prob_0002"]


def test_invalid_answer_is_omitted_without_silent_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _split()
    methods = _methods(manifest)
    methods = FrozenSelectionMethods(
        freeze_path=methods.freeze_path,
        freeze_sha256=methods.freeze_sha256,
        split_sha256=methods.split_sha256,
        fold=methods.fold,
        train_file_sha256=methods.train_file_sha256,
        exclusions_file_sha256=methods.exclusions_file_sha256,
        excluded_ids_sha256=methods.excluded_ids_sha256,
        split_artifact_sha256=methods.split_artifact_sha256,
        development_shard_sha256=methods.development_shard_sha256,
        primary_label=methods.primary_label,
        primary_checkpoint_sha256=methods.primary_checkpoint_sha256,
        fallback_label=None,
        fallback_checkpoint_sha256=None,
        routing_policy="primary_only",
    )
    monkeypatch.setattr(
        prediction_module,
        "validate_frozen_selection_methods",
        lambda *_args, **_kwargs: methods,
    )
    evaluation = tmp_path / "test.csv"
    _write_evaluation(evaluation, _records())
    primary = _Backend("a" * 64, {record.id: "no integer" for record in _records()})

    result = run_frozen_evaluation_inference(
        dataset_role="test",
        evaluation_file_path=evaluation,
        expected_evaluation_sha256=hashlib.sha256(evaluation.read_bytes()).hexdigest(),
        split_manifest=manifest,
        **_scope_args(),
        freeze_artifact=tmp_path / "ignored.json",
        primary_backend=primary,
        fallback_backend=None,
        artifact_path=tmp_path / "raw.json",
        predictions_path=tmp_path / "predictions.json",
        clock_ns=lambda: 0,
    )

    assert result.invalid_count == 3
    assert json.loads((tmp_path / "predictions.json").read_text(encoding="utf-8")) == {}
    payload = json.loads((tmp_path / "raw.json").read_text(encoding="utf-8"))
    assert payload["status"] == "incomplete_invalid_answers"
    assert payload["predictions_file"]["silent_zero_fallback"] is False
    assert payload["invalid_ids"] == ["prob_0001", "prob_0002", "prob_0003"]


def test_prediction_pair_publishes_manifest_last_and_rolls_back_normal_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    predictions = tmp_path / "predictions.json"
    manifest = tmp_path / "manifest.json"
    real_link = prediction_module.os.link
    linked_targets: list[str] = []

    def fail_second_link(source: Path, target: Path) -> None:
        linked_targets.append(Path(target).name)
        if len(linked_targets) == 2:
            raise OSError("simulated manifest publication failure")
        real_link(source, target)

    monkeypatch.setattr(prediction_module.os, "link", fail_second_link)
    with pytest.raises(OSError, match="simulated manifest"):
        prediction_module._publish_atomic_pair(
            predictions,
            b"{}\n",
            manifest,
            b'{"status":"complete"}\n',
        )

    assert linked_targets == ["predictions.json", "manifest.json"]
    assert not predictions.exists()
    assert not manifest.exists()
    assert not tuple(tmp_path.glob(".*.tmp"))


def test_prediction_rejects_answered_or_changed_evaluation_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _split()
    monkeypatch.setattr(
        prediction_module,
        "validate_frozen_selection_methods",
        lambda *_args, **_kwargs: _methods(manifest),
    )
    evaluation = tmp_path / "leaderboard.csv"
    _write_evaluation(evaluation, _records())
    backend = _Backend("a" * 64, {})
    with pytest.raises(GateBValidationError, match="SHA-256"):
        run_frozen_evaluation_inference(
            dataset_role="leaderboard",
            evaluation_file_path=evaluation,
            expected_evaluation_sha256="0" * 64,
            split_manifest=manifest,
            **_scope_args(),
            freeze_artifact=tmp_path / "ignored.json",
            primary_backend=backend,
            fallback_backend=_Backend("b" * 64, {}),
            artifact_path=tmp_path / "raw.json",
            predictions_path=tmp_path / "predictions.json",
        )
    answered_evaluation = tmp_path / "answered.csv"
    _write_evaluation(answered_evaluation, _records(), answered_first=True)
    with pytest.raises(DatasetValidationError, match="leaderboard answer must be empty"):
        run_frozen_evaluation_inference(
            dataset_role="leaderboard",
            evaluation_file_path=answered_evaluation,
            expected_evaluation_sha256=hashlib.sha256(
                answered_evaluation.read_bytes()
            ).hexdigest(),
            split_manifest=manifest,
            **_scope_args(),
            freeze_artifact=tmp_path / "ignored.json",
            primary_backend=backend,
            fallback_backend=_Backend("b" * 64, {}),
            artifact_path=tmp_path / "raw2.json",
            predictions_path=tmp_path / "predictions2.json",
        )


def test_prediction_rejects_forged_exclusion_digest_before_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _split()
    monkeypatch.setattr(
        prediction_module,
        "validate_frozen_selection_methods",
        lambda *_args, **_kwargs: _methods(manifest),
    )
    evaluation = tmp_path / "leaderboard.csv"
    _write_evaluation(evaluation, _records())
    scope = _scope_args()
    scope["excluded_ids_sha256"] = "f" * 64
    primary = _Backend("a" * 64, {})

    with pytest.raises(GateBValidationError, match="excluded_ids do not match"):
        run_frozen_evaluation_inference(
            dataset_role="leaderboard",
            evaluation_file_path=evaluation,
            expected_evaluation_sha256=hashlib.sha256(
                evaluation.read_bytes()
            ).hexdigest(),
            split_manifest=manifest,
            **scope,
            freeze_artifact=tmp_path / "ignored.json",
            primary_backend=primary,
            fallback_backend=None,
            artifact_path=tmp_path / "raw.json",
            predictions_path=tmp_path / "predictions.json",
        )
    assert primary.requests == []

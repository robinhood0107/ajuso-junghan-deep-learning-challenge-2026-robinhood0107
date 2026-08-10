from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

import deep_challenge.gate_b_holdout as holdout_module
import deep_challenge.gate_b_selection as selection_module
from deep_challenge.data import MathRecord
from deep_challenge.gate_b import (
    DEFAULT_GATE_B_CONFIG,
    DevelopmentExecutionEvidence,
    GateBValidationError,
    GenerationRequest,
    GenerationResult,
    create_development_execution_evidence,
    run_development_baseline,
    write_development_artifacts,
)
from deep_challenge.gate_b_holdout import evaluate_locked_holdout_once
from deep_challenge.gate_b_runtime import (
    BASE_MODEL_CHECKPOINT_SHA256,
    AdapterArtifactEvidence,
    build_fold_sft_plan,
)
from deep_challenge.gate_b_selection import (
    ADAPTER_METHOD_KIND,
    FIXED_BASE_METHOD_KIND,
    HOLDOUT_ACCESS_ACKNOWLEDGEMENT,
    FoldDevelopmentRun,
    compare_cross_fold_development_runs,
    freeze_development_selection,
)
from deep_challenge.provenance import (
    build_source_tree_manifest,
    canonical_json_bytes,
    validate_source_tree_manifest_artifact,
    write_json_atomic,
)
from deep_challenge.splits import (
    eligible_training_ids,
    eligible_validation_ids,
    make_grouped_split_manifest,
)


class _Backend:
    def __init__(
        self, checkpoint_sha256: str, *, invalid_ids: tuple[str, ...] = ()
    ) -> None:
        self.checkpoint_sha256 = checkpoint_sha256
        self.invalid_ids = set(invalid_ids)
        self.requests: list[GenerationRequest] = []
        self.closed = False

    def generate(self, request: GenerationRequest) -> GenerationResult:
        assert not self.closed
        self.requests.append(request)
        if request.problem_id in self.invalid_ids:
            return GenerationResult("no integer", "stop", 10, 3, 123)
        answer = int(request.problem_id.removeprefix("train-"))
        return GenerationResult(f"Final answer: {answer}", "stop", 10, 3, 123)

    def close(self) -> None:
        self.closed = True


def _adapter_digest(fold: int) -> str:
    return hashlib.sha256(f"holdout-adapter:{fold}".encode()).hexdigest()


def _execution_evidence(tmp_path: Path) -> DevelopmentExecutionEvidence:
    source_root = tmp_path / "source"
    source_root.mkdir(exist_ok=True)
    (source_root / "runtime.py").write_text("VALUE = 1\n", encoding="utf-8")
    source_manifest_path = tmp_path / "source-manifest.json"
    write_json_atomic(
        source_manifest_path,
        build_source_tree_manifest(source_root, excluded_paths=(source_manifest_path,)).as_dict(),
    )
    source_manifest = validate_source_tree_manifest_artifact(
        source_manifest_path,
        root=source_root,
    )
    config = tmp_path / "config.json"
    preflight = tmp_path / "preflight.json"
    smoke = tmp_path / "smoke.json"
    config.write_text(
        json.dumps(
            {
                **DEFAULT_GATE_B_CONFIG.as_dict(),
                "config_sha256": DEFAULT_GATE_B_CONFIG.sha256,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    preflight.write_text("{}\n", encoding="utf-8")
    smoke.write_text("{}\n", encoding="utf-8")
    return create_development_execution_evidence(
        source_manifest=source_manifest,
        config_path=config,
        config_sha256=DEFAULT_GATE_B_CONFIG.sha256,
        preflight_report_path=preflight,
        preflight_report_sha256=hashlib.sha256(preflight.read_bytes()).hexdigest(),
        gpu_smoke_report_path=smoke,
        gpu_smoke_report_sha256=hashlib.sha256(smoke.read_bytes()).hexdigest(),
        gpu_device_name="NVIDIA Test GPU",
    )


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    with_fallback: bool = False,
):
    identifiers = tuple(f"train-{index:06d}" for index in range(1, 13))
    manifest = make_grouped_split_manifest(
        identifiers,
        dict(zip(identifiers, identifiers, strict=True)),
        n_folds=2,
        holdout_fraction=0.25,
        seed=7,
        version="holdout-test-v1",
    )
    records = tuple(
        MathRecord(
            id=identifier,
            question_raw=f"Return {index}.",
            question_normalized=f"Return {index}.",
            answer_raw=str(index),
            answer=index,
            row_number=index + 1,
        )
        for index, identifier in enumerate(identifiers, start=1)
    )
    by_id = {record.id: record for record in records}
    fold_runs: list[FoldDevelopmentRun] = []
    adapter_by_path: dict[Path, AdapterArtifactEvidence] = {}
    for fold in range(manifest.n_folds):
        validation_ids = eligible_validation_ids(manifest, fold, ())
        validation = tuple(by_id[identifier] for identifier in validation_ids)
        for label, checkpoint, method_kind in (
            ("base", BASE_MODEL_CHECKPOINT_SHA256, FIXED_BASE_METHOD_KIND),
            ("candidate", _adapter_digest(fold), ADAPTER_METHOD_KIND),
        ):
            generated = run_development_baseline(
                validation,
                split_manifest=manifest,
                fold=fold,
                excluded_ids=(),
                backend=_Backend(checkpoint),
                checkpoint_sha256=checkpoint,
                clock_ns=lambda: 0,
            )
            jsonl = tmp_path / f"{label}.fold{fold}.jsonl"
            run_manifest = tmp_path / f"{label}.fold{fold}.manifest.json"
            write_development_artifacts(
                generated,
                jsonl_path=jsonl,
                manifest_path=run_manifest,
                execution_evidence=_execution_evidence(tmp_path),
            )
            adapter_path = None
            if method_kind == ADAPTER_METHOD_KIND:
                training_ids = eligible_training_ids(manifest, fold, ())
                plan = build_fold_sft_plan(
                    tuple(by_id[problem_id] for problem_id in training_ids),
                    validation,
                    split_manifest=manifest,
                    fold=fold,
                    excluded_ids=(),
                )
                adapter_dir = tmp_path / f"candidate.fold{fold}.adapter"
                adapter_dir.mkdir()
                adapter = AdapterArtifactEvidence(
                    path=str(adapter_dir.resolve()),
                    artifact_sha256=checkpoint,
                    manifest_sha256=hashlib.sha256(
                        f"manifest:{fold}".encode()
                    ).hexdigest(),
                    checksums_sha256=hashlib.sha256(
                        f"checksums:{fold}".encode()
                    ).hexdigest(),
                    file_count=6,
                    config_sha256=plan.config_sha256,
                    split_version=manifest.version,
                    split_sha256=manifest.sha256,
                    source_groups_sha256=manifest.source_groups_sha256,
                    fold=fold,
                    excluded_ids_sha256=plan.excluded_ids_sha256,
                    training_count=len(plan.training_ids),
                    training_ids_sha256=plan.training_ids_sha256,
                    validation_count=len(plan.validation_ids),
                    validation_ids_sha256=plan.validation_ids_sha256,
                    training_examples_sha256=plan.training_examples_sha256,
                    validation_examples_sha256=plan.validation_examples_sha256,
                    train_file_sha256="1" * 64,
                    exclusions_file_sha256="2" * 64,
                    split_artifact_sha256="3" * 64,
                    development_shard_sha256="4" * 64,
                    preflight_sha256="5" * 64,
                    gpu_smoke_sha256="6" * 64,
                    source_manifest_sha256="7" * 64,
                    source_tree_sha256="8" * 64,
                    source_file_count=1,
                )
                adapter_by_path[adapter_dir.resolve()] = adapter
                adapter_path = adapter_dir
            fold_runs.append(
                FoldDevelopmentRun(
                    fold=fold,
                    label=label,
                    records_path=jsonl,
                    manifest_path=run_manifest,
                    method_kind=method_kind,
                    adapter_path=adapter_path,
                )
            )

    def validate_adapter(
        path: str | Path, **_kwargs: object
    ) -> AdapterArtifactEvidence:
        return adapter_by_path[Path(path).resolve(strict=True)]

    monkeypatch.setattr(selection_module, "validate_adapter_artifact", validate_adapter)
    comparison = tmp_path / "comparison.json"
    eligible_ids = tuple(
        sorted(
            problem_id
            for fold in range(manifest.n_folds)
            for problem_id in eligible_validation_ids(manifest, fold, ())
        )
    )
    compare_cross_fold_development_runs(
        "base",
        ("candidate",),
        fold_runs,
        tuple(by_id[problem_id] for problem_id in eligible_ids),
        split_manifest=manifest,
        deployment_fold=0,
        excluded_ids=(),
        train_file_sha256="1" * 64,
        exclusions_file_sha256="2" * 64,
        split_artifact_sha256="3" * 64,
        development_shard_sha256="4" * 64,
        output_path=comparison,
        bootstrap_samples=20,
    )
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps({"tree_sha256": "c" * 64, "files": [{"path": "x"}]}) + "\n",
        encoding="utf-8",
    )
    lock = tmp_path / "uv.lock"
    lock.write_text("version = 1\n", encoding="utf-8")
    freeze = tmp_path / "freeze.json"
    freeze_development_selection(
        comparison,
        primary_label="base",
        fallback_label="candidate" if with_fallback else None,
        decision_note="Development evidence selected the base method.",
        source_manifest_path=source,
        lockfile_path=lock,
        output_path=freeze,
        now=lambda: datetime(2026, 8, 4, tzinfo=UTC),
    )
    return manifest, records, freeze


def _provenance_args() -> dict[str, object]:
    return {
        "train_file_sha256": "1" * 64,
        "exclusions_file_sha256": "2" * 64,
        "excluded_ids_sha256": hashlib.sha256(canonical_json_bytes([])).hexdigest(),
        "split_artifact_sha256": "3" * 64,
        "development_shard_sha256": "4" * 64,
        "fold": 0,
        "holdout_acknowledgement": HOLDOUT_ACCESS_ACKNOWLEDGEMENT,
    }


def test_holdout_claim_is_coupled_to_exactly_one_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, records, freeze = _fixture(tmp_path, monkeypatch)
    ledger = tmp_path / "canonical-ledger"
    monkeypatch.setattr(holdout_module, "CANONICAL_HOLDOUT_LEDGER_ROOT", ledger)
    backend = _Backend(BASE_MODEL_CHECKPOINT_SHA256)
    clock_values = iter([0, 1_000_000] * len(manifest.final_holdout_ids()))
    output = tmp_path / "holdout-evaluation.json"

    result = evaluate_locked_holdout_once(
        lambda: records,
        split_manifest=manifest,
        excluded_ids=(),
        **_provenance_args(),
        freeze_artifact=freeze,
        primary_backend=backend,
        fallback_backend=None,
        output_path=output,
        clock_ns=lambda: next(clock_values),
        now=lambda: datetime(2026, 8, 4, tzinfo=UTC),
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert result.sha256 and payload["status"] == "complete"
    assert payload["policy_result"]["exact_match_accuracy"] == 1.0
    assert payload["frozen_methods"]["fallback"] is None
    assert payload["routing_policy"] == "primary_only"
    assert payload["leaderboard_or_test_used"] is False
    assert len(payload["policy_result"]["records"]) == len(
        manifest.final_holdout_ids()
    )
    assert len(list(ledger.glob("*.claim.json"))) == 1
    assert len(list(ledger.glob("*.receipt.json"))) == 1

    second_backend = _Backend(BASE_MODEL_CHECKPOINT_SHA256)
    with pytest.raises(GateBValidationError, match="already exists|overwrite"):
        evaluate_locked_holdout_once(
            lambda: records,
            split_manifest=manifest,
            excluded_ids=(),
            **_provenance_args(),
            freeze_artifact=freeze,
            primary_backend=second_backend,
            fallback_backend=None,
            output_path=tmp_path / "second.json",
        )
    assert second_backend.requests == []


def test_holdout_scores_only_the_development_frozen_fallback_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, records, freeze = _fixture(tmp_path, monkeypatch, with_fallback=True)
    ledger = tmp_path / "canonical-ledger"
    monkeypatch.setattr(holdout_module, "CANONICAL_HOLDOUT_LEDGER_ROOT", ledger)
    first_holdout_id = manifest.final_holdout_ids()[0]
    primary = _Backend(BASE_MODEL_CHECKPOINT_SHA256, invalid_ids=(first_holdout_id,))
    fallback = _Backend(_adapter_digest(0))
    clock_values = iter([0, 1_000_000] * (len(manifest.final_holdout_ids()) + 1))
    output = tmp_path / "holdout-policy.json"

    evaluate_locked_holdout_once(
        lambda: records,
        split_manifest=manifest,
        excluded_ids=(),
        **_provenance_args(),
        freeze_artifact=freeze,
        primary_backend=primary,
        fallback_backend=fallback,
        output_path=output,
        clock_ns=lambda: next(clock_values),
        now=lambda: datetime(2026, 8, 4, tzinfo=UTC),
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    result = payload["policy_result"]
    assert payload["schema_version"] == "gate-b-locked-holdout-evaluation-v2"
    assert (
        payload["routing_policy"]
        == "primary_then_fallback_only_on_primary_parse_failure"
    )
    assert result["fallback_invocation_count"] == 1
    assert result["model_residency_policy"] == "sequential_single_backend"
    assert result["exact_match_accuracy"] == 1.0
    assert primary.closed is True
    assert [request.problem_id for request in fallback.requests] == [first_holdout_id]
    assert "exact_match_accuracy" not in payload["frozen_methods"]["fallback"]


def test_holdout_static_errors_do_not_consume_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, records, freeze = _fixture(tmp_path, monkeypatch)
    ledger = tmp_path / "canonical-ledger"
    monkeypatch.setattr(holdout_module, "CANONICAL_HOLDOUT_LEDGER_ROOT", ledger)
    existing = tmp_path / "existing.json"
    existing.write_text("do not overwrite\n", encoding="utf-8")

    with pytest.raises(GateBValidationError, match="new non-symlink"):
        evaluate_locked_holdout_once(
            lambda: records,
            split_manifest=manifest,
            excluded_ids=(),
            **_provenance_args(),
            freeze_artifact=freeze,
            primary_backend=_Backend(BASE_MODEL_CHECKPOINT_SHA256),
            fallback_backend=None,
            output_path=existing,
        )
    assert not ledger.exists()

    bad_provenance = _provenance_args()
    bad_provenance["excluded_ids_sha256"] = "f" * 64
    with pytest.raises(GateBValidationError, match="excluded_ids do not match"):
        evaluate_locked_holdout_once(
            lambda: records,
            split_manifest=manifest,
            excluded_ids=(),
            **bad_provenance,
            freeze_artifact=freeze,
            primary_backend=_Backend(BASE_MODEL_CHECKPOINT_SHA256),
            fallback_backend=None,
            output_path=tmp_path / "bad-exclusions.json",
        )
    assert not ledger.exists()

    with pytest.raises(GateBValidationError, match="frozen checkpoint"):
        evaluate_locked_holdout_once(
            lambda: records,
            split_manifest=manifest,
            excluded_ids=(),
            **_provenance_args(),
            freeze_artifact=freeze,
            primary_backend=_Backend("f" * 64),
            fallback_backend=None,
            output_path=tmp_path / "new.json",
        )
    assert not ledger.exists()


def test_holdout_loader_failure_after_claim_permanently_consumes_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, _, freeze = _fixture(tmp_path, monkeypatch)
    ledger = tmp_path / "canonical-ledger"
    monkeypatch.setattr(holdout_module, "CANONICAL_HOLDOUT_LEDGER_ROOT", ledger)
    output = tmp_path / "holdout-failed.json"

    def fail_after_claim() -> tuple[MathRecord, ...]:
        raise RuntimeError("simulated organizer train read failure")

    with pytest.raises(RuntimeError, match="simulated organizer"):
        evaluate_locked_holdout_once(
            fail_after_claim,
            split_manifest=manifest,
            excluded_ids=(),
            **_provenance_args(),
            freeze_artifact=freeze,
            primary_backend=_Backend(BASE_MODEL_CHECKPOINT_SHA256),
            fallback_backend=None,
            output_path=output,
        )
    assert len(list(ledger.glob("*.claim.json"))) == 1
    assert len(list(ledger.glob("*.receipt.json"))) == 1
    assert not output.exists()

    with pytest.raises(GateBValidationError, match="already exists|overwrite"):
        evaluate_locked_holdout_once(
            lambda: (),
            split_manifest=manifest,
            excluded_ids=(),
            **_provenance_args(),
            freeze_artifact=freeze,
            primary_backend=_Backend(BASE_MODEL_CHECKPOINT_SHA256),
            fallback_backend=None,
            output_path=tmp_path / "retry.json",
        )

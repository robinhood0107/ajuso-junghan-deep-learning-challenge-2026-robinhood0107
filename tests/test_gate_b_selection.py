from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

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
    NamedDevelopmentRun,
    authorize_locked_holdout_once,
    compare_cross_fold_development_runs,
    compare_development_runs,
    decide_candidate_probe_promotion,
    freeze_base_development_selection,
    freeze_development_selection,
    require_base_development_artifact,
    validate_frozen_selection_methods,
    validate_locked_holdout_access,
    verify_base_development_oof,
)
from deep_challenge.provenance import (
    build_source_tree_manifest,
    canonical_json_bytes,
    validate_source_tree_manifest_artifact,
    write_json_atomic,
)
from deep_challenge.rationale_corpus import DEFAULT_CONCISE_RATIONALE_CONFIG
from deep_challenge.splits import (
    SplitManifest,
    build_group_clusters,
    eligible_training_ids,
    eligible_validation_ids,
    make_grouped_split_manifest,
)


class _Backend:
    def __init__(self, answers: dict[str, int | None]) -> None:
        self.answers = answers

    def generate(self, request: GenerationRequest) -> GenerationResult:
        answer = self.answers.get(request.problem_id)
        text = "no integer" if answer is None else f"Final answer: {answer}"
        return GenerationResult(text, "stop", 10, 3, 1234)


def _fixture(*, n_folds: int = 2) -> tuple[SplitManifest, tuple[MathRecord, ...]]:
    ids = tuple(f"train-{index:06d}" for index in range(1, 13))
    groups = build_group_clusters(ids)
    manifest = make_grouped_split_manifest(
        ids,
        groups,
        n_folds=n_folds,
        holdout_fraction=0.25,
        seed=7,
        version="test-v1",
    )
    records = tuple(
        MathRecord(
            id=problem_id,
            question_raw=f"What is {index} plus zero?",
            question_normalized=f"What is {index} plus zero?",
            answer_raw=str(index),
            answer=index,
            row_number=index + 1,
        )
        for index, problem_id in enumerate(ids, start=1)
    )
    return manifest, records


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


def _write_run(
    tmp_path: Path,
    *,
    label: str,
    manifest: SplitManifest,
    all_records: tuple[MathRecord, ...],
    wrong_first: bool = False,
    invalid_first: bool = False,
    samples_per_problem: int = 1,
    fold: int = 0,
    checkpoint_sha256: str | None = None,
) -> NamedDevelopmentRun:
    validation_ids = eligible_validation_ids(manifest, fold, ())
    records_by_id = {record.id: record for record in all_records}
    validation = tuple(records_by_id[problem_id] for problem_id in validation_ids)
    answers = {record.id: record.answer for record in validation}
    if wrong_first:
        answers[validation_ids[0]] = int(answers[validation_ids[0]]) + 1
    if invalid_first:
        answers[validation_ids[0]] = None
    generated = run_development_baseline(
        validation,
        split_manifest=manifest,
        fold=fold,
        excluded_ids=(),
        backend=_Backend(answers),
        checkpoint_sha256=(
            checkpoint_sha256
            if checkpoint_sha256 is not None
            else BASE_MODEL_CHECKPOINT_SHA256
            if label == "base"
            else "b" * 64
        ),
        samples_per_problem=samples_per_problem,
        clock_ns=lambda: 0,
    )
    fold_suffix = "" if fold == 0 else f".fold{fold}"
    records_path = tmp_path / f"{label}{fold_suffix}.jsonl"
    manifest_path = tmp_path / f"{label}{fold_suffix}.manifest.json"
    write_development_artifacts(
        generated,
        jsonl_path=records_path,
        manifest_path=manifest_path,
        execution_evidence=_execution_evidence(tmp_path),
    )
    return NamedDevelopmentRun(label, records_path, manifest_path)


def _fake_adapter_evidence(
    tmp_path: Path,
    *,
    label: str,
    fold: int,
    manifest: SplitManifest,
    records: tuple[MathRecord, ...],
    method_salt: str = "shared-method",
    rationale: bool = False,
) -> AdapterArtifactEvidence:
    adapter_path = tmp_path / f"{label}.fold{fold}.adapter"
    adapter_path.mkdir()
    by_id = {record.id: record for record in records}
    training_ids = eligible_training_ids(manifest, fold, ())
    validation_ids = eligible_validation_ids(manifest, fold, ())
    plan = build_fold_sft_plan(
        tuple(by_id[problem_id] for problem_id in training_ids),
        tuple(by_id[problem_id] for problem_id in validation_ids),
        split_manifest=manifest,
        fold=fold,
        excluded_ids=(),
    )

    def digest(value: str) -> str:
        return hashlib.sha256(value.encode()).hexdigest()

    target_evidence: dict[str, object] = {}
    if rationale:
        target_evidence = {
            "training_target_kind": "verified_concise_rationale",
            "rationale_candidate_config_sha256": (
                DEFAULT_CONCISE_RATIONALE_CONFIG.sha256
            ),
            "rationale_candidate_config_file_sha256": "5" * 64,
            "rationale_corpus_records_sha256": digest(
                f"rationale-records:{fold}"
            ),
            "rationale_corpus_manifest_sha256": digest(
                f"rationale-manifest:{fold}"
            ),
            "rationale_corpus_audit_sha256": digest(f"rationale-audit:{fold}"),
        }
    return AdapterArtifactEvidence(
        path=str(adapter_path.resolve()),
        artifact_sha256=digest(f"artifact:{label}:{fold}"),
        manifest_sha256=digest(f"manifest:{label}:{fold}"),
        checksums_sha256=digest(f"checksums:{label}:{fold}"),
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
        preflight_sha256=digest(f"preflight:{method_salt}"),
        gpu_smoke_sha256=digest(f"smoke:{method_salt}"),
        source_manifest_sha256=digest(f"source-manifest:{method_salt}"),
        source_tree_sha256=digest(f"source-tree:{method_salt}"),
        source_file_count=1,
        **target_evidence,
    )


def _install_fake_adapter_validator(
    monkeypatch: pytest.MonkeyPatch,
    evidence: tuple[AdapterArtifactEvidence, ...],
) -> None:
    by_path = {Path(item.path): item for item in evidence}

    def validate(path: str | Path, **_kwargs: object) -> AdapterArtifactEvidence:
        return by_path[Path(path).resolve(strict=True)]

    monkeypatch.setattr(selection_module, "validate_adapter_artifact", validate)


def test_direct_adapter_method_fingerprint_retains_v1_contract(tmp_path: Path) -> None:
    manifest, records = _fixture()
    adapter = _fake_adapter_evidence(
        tmp_path,
        label="qlora",
        fold=0,
        manifest=manifest,
        records=records,
    )

    assert selection_module._adapter_method_fingerprint(adapter) == (
        "91bf72f9f75c4c29355f833c7c2acc094faa57c601aaa14f17101233b6cae7b2"
    )


def _validation_records(
    manifest: SplitManifest, records: tuple[MathRecord, ...]
) -> tuple[MathRecord, ...]:
    by_id = {record.id: record for record in records}
    return tuple(by_id[problem_id] for problem_id in eligible_validation_ids(manifest, 0, ()))


def _comparison(tmp_path: Path) -> tuple[Path, SplitManifest, tuple[MathRecord, ...]]:
    manifest, records = _fixture()
    base = _write_run(tmp_path, label="base", manifest=manifest, all_records=records)
    candidate = _write_run(
        tmp_path,
        label="qlora",
        manifest=manifest,
        all_records=records,
        wrong_first=True,
    )
    output = tmp_path / "comparison.json"
    compare_development_runs(
        base,
        (candidate,),
        _validation_records(manifest, records),
        split_manifest=manifest,
        fold=0,
        excluded_ids=(),
        train_file_sha256="1" * 64,
        exclusions_file_sha256="2" * 64,
        split_artifact_sha256="3" * 64,
        development_shard_sha256="4" * 64,
        output_path=output,
        bootstrap_samples=200,
        seed=9,
    )
    return output, manifest, records


def _rewrite_probe_statistics(
    source: Path,
    target: Path,
    *,
    delta: float,
    ci_low: float,
    ci_high: float,
    raw_p: float,
    reject: bool,
) -> Path:
    payload = json.loads(source.read_text(encoding="utf-8"))
    comparison = payload["comparisons"][0]
    assert comparison["delta_b_minus_a"] == delta
    comparison["bootstrap_delta_ci_low"] = ci_low
    comparison["bootstrap_delta_ci_high"] = ci_high
    comparison["mcnemar_exact_p_value"] = raw_p
    raw_p_values = {
        item["candidate_label"]: item["mcnemar_exact_p_value"]
        for item in payload["comparisons"]
    }
    raw_p_values.update(
        {
            item["hypothesis"]: item["mcnemar_exact_p_value"]
            for item in payload["fallback_routing_comparisons"]
        }
    )
    adjustments = {
        item.hypothesis: asdict(item)
        for item in selection_module.holm_bonferroni(
            raw_p_values,
            alpha=payload["statistics"]["alpha"],
        )
    }
    for item in payload["comparisons"]:
        item["holm"] = adjustments[item["candidate_label"]]
    for item in payload["fallback_routing_comparisons"]:
        item["holm"] = adjustments[item["hypothesis"]]
    comparison["holm"]["reject"] = reject
    payload.pop("payload_sha256")
    payload["payload_sha256"] = hashlib.sha256(
        canonical_json_bytes(payload)
    ).hexdigest()
    target.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return target


def _oof_comparison(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    rationale: bool = False,
) -> tuple[
    Path,
    SplitManifest,
    tuple[MathRecord, ...],
    tuple[AdapterArtifactEvidence, ...],
]:
    manifest, records = _fixture()
    fold_runs: list[FoldDevelopmentRun] = []
    adapters: list[AdapterArtifactEvidence] = []
    for fold in range(manifest.n_folds):
        base = _write_run(
            tmp_path,
            label="base",
            manifest=manifest,
            all_records=records,
            fold=fold,
        )
        adapter = _fake_adapter_evidence(
            tmp_path,
            label="qlora",
            fold=fold,
            manifest=manifest,
            records=records,
            rationale=rationale,
        )
        candidate = _write_run(
            tmp_path,
            label="qlora",
            manifest=manifest,
            all_records=records,
            fold=fold,
            checkpoint_sha256=adapter.artifact_sha256,
        )
        adapters.append(adapter)
        fold_runs.extend(
            (
                FoldDevelopmentRun(
                    fold,
                    base.label,
                    base.records_path,
                    base.manifest_path,
                    FIXED_BASE_METHOD_KIND,
                ),
                FoldDevelopmentRun(
                    fold,
                    candidate.label,
                    candidate.records_path,
                    candidate.manifest_path,
                    ADAPTER_METHOD_KIND,
                    adapter.path,
                ),
            )
        )
    _install_fake_adapter_validator(monkeypatch, tuple(adapters))
    eligible_ids = tuple(
        sorted(
            problem_id
            for fold in range(manifest.n_folds)
            for problem_id in eligible_validation_ids(manifest, fold, ())
        )
    )
    by_id = {record.id: record for record in records}
    output = tmp_path / "oof-comparison.json"
    compare_cross_fold_development_runs(
        "base",
        ("qlora",),
        fold_runs,
        tuple(by_id[problem_id] for problem_id in eligible_ids),
        split_manifest=manifest,
        deployment_fold=0,
        excluded_ids=(),
        train_file_sha256="1" * 64,
        exclusions_file_sha256="2" * 64,
        split_artifact_sha256="3" * 64,
        development_shard_sha256="4" * 64,
        output_path=output,
        bootstrap_samples=20,
        seed=9,
    )
    return output, manifest, records, tuple(adapters)


def _existing_oof_runs(
    tmp_path: Path,
    manifest: SplitManifest,
    adapters: tuple[AdapterArtifactEvidence, ...],
) -> tuple[FoldDevelopmentRun, ...]:
    output: list[FoldDevelopmentRun] = []
    for fold in range(manifest.n_folds):
        suffix = "" if fold == 0 else f".fold{fold}"
        output.extend(
            (
                FoldDevelopmentRun(
                    fold=fold,
                    label="base",
                    records_path=tmp_path / f"base{suffix}.jsonl",
                    manifest_path=tmp_path / f"base{suffix}.manifest.json",
                    method_kind=FIXED_BASE_METHOD_KIND,
                ),
                FoldDevelopmentRun(
                    fold=fold,
                    label="qlora",
                    records_path=tmp_path / f"qlora{suffix}.jsonl",
                    manifest_path=tmp_path / f"qlora{suffix}.manifest.json",
                    method_kind=ADAPTER_METHOD_KIND,
                    adapter_path=adapters[fold].path,
                ),
            )
        )
    return tuple(output)


def test_comparison_validates_runs_and_writes_cluster_statistics(tmp_path: Path) -> None:
    output, manifest, records = _comparison(tmp_path)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "gate-b-development-comparison-v2"
    assert payload["locked_holdout_accessed"] is False
    assert payload["leaderboard_or_test_used"] is False
    assert payload["statistics"]["bootstrap_unit"] == "duplicate_cluster"
    assert payload["statistics"]["family_size"] == 3
    assert payload["comparisons"][0]["bootstrap_unit"] == "duplicate_cluster"
    assert payload["comparisons"][0]["holm"]["hypothesis"] == "qlora"
    assert len(payload["fallback_routing_comparisons"]) == 2
    assert {
        item["holm"]["hypothesis"]
        for item in payload["fallback_routing_comparisons"]
    } == {
        "fallback_on_invalid:base->qlora",
        "fallback_on_invalid:qlora->base",
    }
    assert payload["runs"]["base"]["exact_match_accuracy"] == 1.0
    assert payload["runs"]["qlora"]["exact_match_accuracy"] < 1.0
    assert payload["split_sha256"] == manifest.sha256
    assert payload["problem_count"] == len(_validation_records(manifest, records))

    with pytest.raises(GateBValidationError, match="overwrite"):
        compare_development_runs(
            NamedDevelopmentRun(
                "base",
                tmp_path / "base.jsonl",
                tmp_path / "base.manifest.json",
            ),
            (
                NamedDevelopmentRun(
                    "qlora",
                    tmp_path / "qlora.jsonl",
                    tmp_path / "qlora.manifest.json",
                ),
            ),
            _validation_records(manifest, records),
            split_manifest=manifest,
            fold=0,
            excluded_ids=(),
            train_file_sha256="1" * 64,
            exclusions_file_sha256="2" * 64,
            split_artifact_sha256="3" * 64,
            development_shard_sha256="4" * 64,
            output_path=output,
            bootstrap_samples=10,
        )


def test_candidate_probe_harm_screen_stops_only_significant_regression(
    tmp_path: Path,
) -> None:
    comparison, _, _ = _comparison(tmp_path)
    continue_output = tmp_path / "continue-decision.json"
    decide_candidate_probe_promotion(
        comparison,
        candidate_label="qlora",
        output_path=continue_output,
    )
    continued = json.loads(continue_output.read_text(encoding="utf-8"))
    assert continued["candidate_action"] == "continue_to_complete_oof"
    assert continued["candidate_full_oof_authorized"] is True
    assert continued["final_selection_eligible"] is False
    assert continued["complete_oof_required_before_freeze"] is True
    assert continued["locked_holdout_accessed"] is False

    significant = _rewrite_probe_statistics(
        comparison,
        tmp_path / "significant-regression.json",
        delta=-0.25,
        ci_low=-0.30,
        ci_high=-0.10,
        raw_p=0.001,
        reject=True,
    )
    stop_output = tmp_path / "stop-decision.json"
    decide_candidate_probe_promotion(
        significant,
        candidate_label="qlora",
        output_path=stop_output,
    )
    stopped = json.loads(stop_output.read_text(encoding="utf-8"))
    assert stopped["significant_regression"] is True
    assert stopped["candidate_action"] == "stop_before_remaining_folds"
    assert stopped["candidate_full_oof_authorized"] is False
    assert stopped["selection_frozen"] is False
    assert stopped["leaderboard_or_test_used"] is False

    with pytest.raises(GateBValidationError, match="overwrite"):
        decide_candidate_probe_promotion(
            significant,
            candidate_label="qlora",
            output_path=stop_output,
        )


def test_candidate_probe_rejects_inconsistent_holm_evidence(tmp_path: Path) -> None:
    comparison, _, _ = _comparison(tmp_path)
    inconsistent = _rewrite_probe_statistics(
        comparison,
        tmp_path / "inconsistent-regression.json",
        delta=-0.25,
        ci_low=-0.30,
        ci_high=-0.10,
        raw_p=0.001,
        reject=False,
    )
    with pytest.raises(GateBValidationError, match="full comparison family"):
        decide_candidate_probe_promotion(
            inconsistent,
            candidate_label="qlora",
            output_path=tmp_path / "forbidden-decision.json",
        )


def test_candidate_probe_rejects_accuracy_inconsistent_with_contingency(
    tmp_path: Path,
) -> None:
    comparison, _, _ = _comparison(tmp_path)
    payload = json.loads(comparison.read_text(encoding="utf-8"))
    payload["comparisons"][0]["accuracy_a"] = 0.5
    payload["comparisons"][0]["accuracy_b"] = 0.25
    payload.pop("payload_sha256")
    payload["payload_sha256"] = hashlib.sha256(
        canonical_json_bytes(payload)
    ).hexdigest()
    inconsistent = tmp_path / "inconsistent-contingency.json"
    inconsistent.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )

    with pytest.raises(GateBValidationError, match="paired contingency"):
        decide_candidate_probe_promotion(
            inconsistent,
            candidate_label="qlora",
            output_path=tmp_path / "forbidden-contingency-decision.json",
        )


def test_comparison_measures_frozen_fallback_routing_policy(tmp_path: Path) -> None:
    manifest, records = _fixture()
    base = _write_run(tmp_path, label="base", manifest=manifest, all_records=records)
    candidate = _write_run(
        tmp_path,
        label="qlora",
        manifest=manifest,
        all_records=records,
        invalid_first=True,
    )
    output = tmp_path / "comparison.json"
    compare_development_runs(
        base,
        (candidate,),
        _validation_records(manifest, records),
        split_manifest=manifest,
        fold=0,
        excluded_ids=(),
        train_file_sha256="1" * 64,
        exclusions_file_sha256="2" * 64,
        split_artifact_sha256="3" * 64,
        development_shard_sha256="4" * 64,
        output_path=output,
        bootstrap_samples=20,
        seed=9,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    route = next(
        item
        for item in payload["fallback_routing_comparisons"]
        if item["primary_label"] == "qlora" and item["fallback_label"] == "base"
    )
    assert route["fallback_invocation_count"] == 1
    assert route["fallback_resolved_count"] == 1
    assert route["unresolved_count"] == 0
    assert route["accuracy_b"] > route["accuracy_a"]


def test_cross_fold_comparison_pools_every_fold_and_freezes_deployment_fold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, records = _fixture()
    supplied: list[FoldDevelopmentRun] = []
    adapters: list[AdapterArtifactEvidence] = []
    for fold in range(manifest.n_folds):
        base = _write_run(
            tmp_path,
            label="base",
            manifest=manifest,
            all_records=records,
            fold=fold,
        )
        adapter = _fake_adapter_evidence(
            tmp_path,
            label="qlora",
            fold=fold,
            manifest=manifest,
            records=records,
        )
        adapters.append(adapter)
        qlora = _write_run(
            tmp_path,
            label="qlora",
            manifest=manifest,
            all_records=records,
            wrong_first=fold == 1,
            fold=fold,
            checkpoint_sha256=adapter.artifact_sha256,
        )
        supplied.extend(
            (
                FoldDevelopmentRun(
                    fold=fold,
                    label=base.label,
                    records_path=base.records_path,
                    manifest_path=base.manifest_path,
                    method_kind=FIXED_BASE_METHOD_KIND,
                ),
                FoldDevelopmentRun(
                    fold=fold,
                    label=qlora.label,
                    records_path=qlora.records_path,
                    manifest_path=qlora.manifest_path,
                    method_kind=ADAPTER_METHOD_KIND,
                    adapter_path=adapter.path,
                ),
            )
        )
    _install_fake_adapter_validator(monkeypatch, tuple(adapters))
    eligible_ids = tuple(
        sorted(
            problem_id
            for fold in range(manifest.n_folds)
            for problem_id in eligible_validation_ids(manifest, fold, ())
        )
    )
    by_id = {record.id: record for record in records}
    eligible_records = tuple(by_id[problem_id] for problem_id in eligible_ids)
    output = tmp_path / "oof-comparison.json"

    compare_cross_fold_development_runs(
        "base",
        ("qlora",),
        supplied,
        eligible_records,
        split_manifest=manifest,
        deployment_fold=0,
        excluded_ids=(),
        train_file_sha256="1" * 64,
        exclusions_file_sha256="2" * 64,
        split_artifact_sha256="3" * 64,
        development_shard_sha256="4" * 64,
        output_path=output,
        bootstrap_samples=40,
        seed=9,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "gate-b-development-oof-comparison-v1"
    assert payload["partition"] == "out_of_fold_cross_validation"
    assert payload["folds"] == list(range(manifest.n_folds))
    assert payload["deployment_fold"] == 0
    assert payload["problem_count"] == len(eligible_ids)
    assert payload["statistics"]["evidence_scope"] == "complete_out_of_fold_union"
    assert payload["statistics"]["family_size"] == 3
    assert set(payload["runs"]["base"]["fold_runs"]) == {"0", "1"}
    assert payload["runs"]["qlora"]["oof_exact_match_count"] == len(eligible_ids) - 1
    assert payload["runs"]["base"]["method_kind"] == FIXED_BASE_METHOD_KIND
    assert payload["runs"]["base"]["adapter_artifact"] is None
    assert payload["runs"]["qlora"]["method_kind"] == ADAPTER_METHOD_KIND
    assert payload["runs"]["qlora"]["adapter_artifact"]["path"] == adapters[0].path
    assert len(payload["runs"]["qlora"]["training_method_fingerprint"]) == 64

    source_manifest = tmp_path / "source.json"
    source_manifest.write_text(
        json.dumps({"tree_sha256": "c" * 64, "files": [{"path": "x"}]}) + "\n",
        encoding="utf-8",
    )
    lockfile = tmp_path / "uv.lock"
    lockfile.write_text("version = 1\n", encoding="utf-8")
    freeze = tmp_path / "oof-freeze.json"
    freeze_development_selection(
        output,
        primary_label="qlora",
        fallback_label="base",
        decision_note="Complete OOF evidence selected a fold-zero deployment checkpoint.",
        source_manifest_path=source_manifest,
        lockfile_path=lockfile,
        output_path=freeze,
        now=lambda: datetime(2026, 8, 4, tzinfo=UTC),
    )
    frozen = json.loads(freeze.read_text(encoding="utf-8"))
    assert frozen["schema_version"] == "gate-b-selection-freeze-v3"
    assert frozen["comparison_scope"] == "complete_out_of_fold_union"
    assert frozen["fold"] == 0
    assert frozen["primary"]["checkpoint_sha256"] == adapters[0].artifact_sha256
    validated = validate_frozen_selection_methods(
        freeze,
        split_manifest=manifest,
        train_file_sha256="1" * 64,
        exclusions_file_sha256="2" * 64,
        excluded_ids_sha256=hashlib.sha256(b"[]").hexdigest(),
        split_artifact_sha256="3" * 64,
        development_shard_sha256="4" * 64,
        fold=0,
    )
    assert validated.primary_label == "qlora"


def test_base_only_oof_qualifies_freezes_and_authorizes_holdout(
    tmp_path: Path,
) -> None:
    manifest, records = _fixture(n_folds=5)
    supplied: list[FoldDevelopmentRun] = []
    for fold in range(manifest.n_folds):
        base = _write_run(
            tmp_path,
            label="base",
            manifest=manifest,
            all_records=records,
            fold=fold,
        )
        supplied.append(
            FoldDevelopmentRun(
                fold=fold,
                label="base",
                records_path=base.records_path,
                manifest_path=base.manifest_path,
                method_kind=FIXED_BASE_METHOD_KIND,
            )
        )
    eligible_ids = tuple(
        sorted(
            problem_id
            for fold in range(manifest.n_folds)
            for problem_id in eligible_validation_ids(manifest, fold, ())
        )
    )
    by_id = {record.id: record for record in records}
    eligible_records = tuple(by_id[problem_id] for problem_id in eligible_ids)
    oof = tmp_path / "base-oof.json"
    with pytest.raises(GateBValidationError, match="exactly one"):
        verify_base_development_oof(
            "base",
            supplied[:-1],
            eligible_records,
            split_manifest=manifest,
            deployment_fold=0,
            excluded_ids=(),
            train_file_sha256="1" * 64,
            exclusions_file_sha256="2" * 64,
            split_artifact_sha256="3" * 64,
            development_shard_sha256="4" * 64,
            output_path=tmp_path / "missing-fold.json",
        )
    with pytest.raises(GateBValidationError, match="duplicate"):
        verify_base_development_oof(
            "base",
            (*supplied[:-1], supplied[0]),
            eligible_records,
            split_manifest=manifest,
            deployment_fold=0,
            excluded_ids=(),
            train_file_sha256="1" * 64,
            exclusions_file_sha256="2" * 64,
            split_artifact_sha256="3" * 64,
            development_shard_sha256="4" * 64,
            output_path=tmp_path / "duplicate-fold.json",
        )
    verify_base_development_oof(
        "base",
        supplied,
        eligible_records,
        split_manifest=manifest,
        deployment_fold=0,
        excluded_ids=(),
        train_file_sha256="1" * 64,
        exclusions_file_sha256="2" * 64,
        split_artifact_sha256="3" * 64,
        development_shard_sha256="4" * 64,
        output_path=oof,
    )
    qualified = json.loads(oof.read_text(encoding="utf-8"))
    assert qualified["schema_version"] == "gate-b-base-development-oof-v1"
    assert qualified["problem_count"] == len(eligible_ids)
    assert qualified["run_order"] == ["base"]
    assert qualified["qualification"] == {
        "evidence_scope": "complete_out_of_fold_union",
        "fixed_base_only": True,
        "exact_fold_coverage": True,
        "validation_ids_unique": True,
        "run_evidence_unique": True,
        "checkpoint_sha256": BASE_MODEL_CHECKPOINT_SHA256,
        "training_method_fingerprint": qualified["runs"]["base"][
            "training_method_fingerprint"
        ],
    }

    source_manifest = tmp_path / "source.json"
    source_manifest.write_text(
        json.dumps({"tree_sha256": "c" * 64, "files": [{"path": "x"}]}) + "\n",
        encoding="utf-8",
    )
    lockfile = tmp_path / "uv.lock"
    lockfile.write_text("version = 1\n", encoding="utf-8")
    freeze = tmp_path / "base-freeze.json"
    freeze_base_development_selection(
        oof,
        primary_label="base",
        decision_note="Complete fixed-base OOF evidence selected the fallback path.",
        source_manifest_path=source_manifest,
        lockfile_path=lockfile,
        output_path=freeze,
        now=lambda: datetime(2026, 8, 11, tzinfo=UTC),
    )
    frozen = json.loads(freeze.read_text(encoding="utf-8"))
    assert frozen["schema_version"] == "gate-b-base-selection-freeze-v1"
    assert frozen["selection_kind"] == "base_only_complete_oof"
    assert frozen["primary"]["label"] == "base"
    assert frozen["fallback"] is None
    assert frozen["routing_policy"] == "primary_only"
    methods = validate_frozen_selection_methods(
        freeze,
        split_manifest=manifest,
        train_file_sha256="1" * 64,
        exclusions_file_sha256="2" * 64,
        excluded_ids_sha256=hashlib.sha256(b"[]").hexdigest(),
        split_artifact_sha256="3" * 64,
        development_shard_sha256="4" * 64,
        fold=0,
    )
    assert methods.primary_label == "base"
    assert methods.fallback_label is None

    ledger = tmp_path / "holdout-ledger"
    ledger.mkdir()
    receipt = authorize_locked_holdout_once(
        freeze,
        split_manifest=manifest,
        excluded_ids=(),
        acknowledgement=HOLDOUT_ACCESS_ACKNOWLEDGEMENT,
        ledger_root=ledger,
        now=lambda: datetime(2026, 8, 11, tzinfo=UTC),
    )
    access = validate_locked_holdout_access(
        receipt.path,
        freeze_artifact=freeze,
        split_manifest=manifest,
        excluded_ids=(),
    )
    assert access.primary_label == "base"
    assert access.fallback_label is None

    tampered_oof_payload = json.loads(oof.read_text(encoding="utf-8"))
    tampered_oof_payload["run_order"] = []
    tampered_oof_payload.pop("payload_sha256")
    tampered_oof_payload["payload_sha256"] = hashlib.sha256(
        canonical_json_bytes(tampered_oof_payload)
    ).hexdigest()
    tampered_oof = tmp_path / "tampered-base-oof.json"
    tampered_oof.write_text(
        json.dumps(tampered_oof_payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(GateBValidationError, match="run order"):
        freeze_base_development_selection(
            tampered_oof,
            primary_label="base",
            decision_note="Tampered OOF evidence must not be frozen.",
            source_manifest_path=source_manifest,
            lockfile_path=lockfile,
            output_path=tmp_path / "tampered-oof-freeze.json",
        )

    tampered_freeze_payload = json.loads(freeze.read_text(encoding="utf-8"))
    tampered_freeze_payload["primary"]["checkpoint_sha256"] = "f" * 64
    tampered_freeze_payload.pop("payload_sha256")
    tampered_freeze_payload["payload_sha256"] = hashlib.sha256(
        canonical_json_bytes(tampered_freeze_payload)
    ).hexdigest()
    tampered_freeze = tmp_path / "tampered-base-freeze.json"
    tampered_freeze.write_text(
        json.dumps(tampered_freeze_payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(GateBValidationError, match="primary provenance"):
        validate_frozen_selection_methods(
            tampered_freeze,
            split_manifest=manifest,
            train_file_sha256="1" * 64,
            exclusions_file_sha256="2" * 64,
            excluded_ids_sha256=hashlib.sha256(b"[]").hexdigest(),
            split_artifact_sha256="3" * 64,
            development_shard_sha256="4" * 64,
            fold=0,
        )

    with pytest.raises(GateBValidationError, match="schema"):
        freeze_development_selection(
            oof,
            primary_label="base",
            fallback_label=None,
            decision_note="Cross-load must fail.",
            source_manifest_path=source_manifest,
            lockfile_path=lockfile,
            output_path=tmp_path / "must-not-exist.json",
        )


def test_rationale_adapter_provenance_survives_oof_comparison_and_freeze(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comparison, manifest, _, adapters = _oof_comparison(
        tmp_path,
        monkeypatch,
        rationale=True,
    )
    payload = json.loads(comparison.read_text(encoding="utf-8"))
    deployment_evidence = payload["runs"]["qlora"]["adapter_artifact"]
    assert deployment_evidence == {
        "path": adapters[0].path,
        "artifact_sha256": adapters[0].artifact_sha256,
        "manifest_sha256": adapters[0].manifest_sha256,
        "checksums_sha256": adapters[0].checksums_sha256,
        "training_target_kind": "verified_concise_rationale",
        "rationale_candidate_config_sha256": (
            DEFAULT_CONCISE_RATIONALE_CONFIG.sha256
        ),
        "rationale_candidate_config_file_sha256": "5" * 64,
        "rationale_corpus_records_sha256": (
            adapters[0].rationale_corpus_records_sha256
        ),
        "rationale_corpus_manifest_sha256": (
            adapters[0].rationale_corpus_manifest_sha256
        ),
        "rationale_corpus_audit_sha256": adapters[0].rationale_corpus_audit_sha256,
    }

    source_manifest = tmp_path / "source.json"
    source_manifest.write_text(
        json.dumps({"tree_sha256": "c" * 64, "files": [{"path": "x"}]}) + "\n",
        encoding="utf-8",
    )
    lockfile = tmp_path / "uv.lock"
    lockfile.write_text("version = 1\n", encoding="utf-8")
    freeze = tmp_path / "rationale-oof-freeze.json"
    freeze_development_selection(
        comparison,
        primary_label="qlora",
        fallback_label="base",
        decision_note="Complete OOF evidence selected the rationale adapter.",
        source_manifest_path=source_manifest,
        lockfile_path=lockfile,
        output_path=freeze,
        now=lambda: datetime(2026, 8, 11, tzinfo=UTC),
    )

    frozen = json.loads(freeze.read_text(encoding="utf-8"))
    assert frozen["primary"]["adapter_artifact"] == deployment_evidence
    validated = validate_frozen_selection_methods(
        freeze,
        split_manifest=manifest,
        train_file_sha256="1" * 64,
        exclusions_file_sha256="2" * 64,
        excluded_ids_sha256=hashlib.sha256(b"[]").hexdigest(),
        split_artifact_sha256="3" * 64,
        development_shard_sha256="4" * 64,
        fold=0,
    )
    assert validated.primary_label == "qlora"


def test_cross_fold_comparison_rejects_incomplete_fold_matrix(tmp_path: Path) -> None:
    manifest, records = _fixture()
    base = _write_run(
        tmp_path,
        label="base",
        manifest=manifest,
        all_records=records,
        fold=0,
    )
    qlora = _write_run(
        tmp_path,
        label="qlora",
        manifest=manifest,
        all_records=records,
        fold=0,
    )
    supplied = (
        FoldDevelopmentRun(
            0,
            base.label,
            base.records_path,
            base.manifest_path,
            FIXED_BASE_METHOD_KIND,
        ),
        FoldDevelopmentRun(
            0,
            qlora.label,
            qlora.records_path,
            qlora.manifest_path,
            ADAPTER_METHOD_KIND,
            tmp_path / "missing-adapter",
        ),
    )
    eligible_ids = tuple(
        sorted(
            problem_id
            for fold in range(manifest.n_folds)
            for problem_id in eligible_validation_ids(manifest, fold, ())
        )
    )
    by_id = {record.id: record for record in records}

    with pytest.raises(GateBValidationError, match="every label on every fold"):
        compare_cross_fold_development_runs(
            "base",
            ("qlora",),
            supplied,
            tuple(by_id[problem_id] for problem_id in eligible_ids),
            split_manifest=manifest,
            deployment_fold=0,
            excluded_ids=(),
            train_file_sha256="1" * 64,
            exclusions_file_sha256="2" * 64,
            split_artifact_sha256="3" * 64,
            development_shard_sha256="4" * 64,
            output_path=tmp_path / "incomplete.json",
        )


def test_final_freeze_rejects_single_fold_probe_artifact(tmp_path: Path) -> None:
    comparison, _, _ = _comparison(tmp_path)
    source_manifest = tmp_path / "source.json"
    source_manifest.write_text(
        json.dumps({"tree_sha256": "c" * 64, "files": [{"path": "x"}]}) + "\n",
        encoding="utf-8",
    )
    lockfile = tmp_path / "uv.lock"
    lockfile.write_text("version = 1\n", encoding="utf-8")

    with pytest.raises(GateBValidationError, match="schema"):
        freeze_development_selection(
            comparison,
            primary_label="base",
            fallback_label=None,
            decision_note="A single-fold probe must not be promoted.",
            source_manifest_path=source_manifest,
            lockfile_path=lockfile,
            output_path=tmp_path / "forbidden-freeze.json",
        )


def test_cross_fold_comparison_rejects_reused_run_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, manifest, records, adapters = _oof_comparison(tmp_path, monkeypatch)
    runs = list(_existing_oof_runs(tmp_path, manifest, adapters))
    for run in tuple(runs):
        if run.label == "qlora":
            runs.append(
                FoldDevelopmentRun(
                    fold=run.fold,
                    label="qlora-copy",
                    records_path=run.records_path,
                    manifest_path=run.manifest_path,
                    method_kind=ADAPTER_METHOD_KIND,
                    adapter_path=run.adapter_path,
                )
            )
    eligible_records = tuple(
        record
        for record in records
        if record.id
        in {
            problem_id
            for fold in range(manifest.n_folds)
            for problem_id in eligible_validation_ids(manifest, fold, ())
        }
    )

    with pytest.raises(GateBValidationError, match="run evidence is reused"):
        compare_cross_fold_development_runs(
            "base",
            ("qlora", "qlora-copy"),
            runs,
            eligible_records,
            split_manifest=manifest,
            deployment_fold=0,
            excluded_ids=(),
            train_file_sha256="1" * 64,
            exclusions_file_sha256="2" * 64,
            split_artifact_sha256="3" * 64,
            development_shard_sha256="4" * 64,
            output_path=tmp_path / "reused.json",
            bootstrap_samples=20,
        )


def test_cross_fold_comparison_rejects_mixed_method_fingerprints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, manifest, records, adapters = _oof_comparison(tmp_path, monkeypatch)
    changed = (
        adapters[0],
        replace(adapters[1], preflight_sha256="7" * 64),
    )
    _install_fake_adapter_validator(monkeypatch, changed)
    runs = _existing_oof_runs(tmp_path, manifest, changed)
    eligible_ids = {
        problem_id
        for fold in range(manifest.n_folds)
        for problem_id in eligible_validation_ids(manifest, fold, ())
    }

    with pytest.raises(GateBValidationError, match="training-method fingerprints"):
        compare_cross_fold_development_runs(
            "base",
            ("qlora",),
            runs,
            tuple(record for record in records if record.id in eligible_ids),
            split_manifest=manifest,
            deployment_fold=0,
            excluded_ids=(),
            train_file_sha256="1" * 64,
            exclusions_file_sha256="2" * 64,
            split_artifact_sha256="3" * 64,
            development_shard_sha256="4" * 64,
            output_path=tmp_path / "mixed-method.json",
            bootstrap_samples=20,
        )


def test_freeze_rejects_adapter_swap_after_oof_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    comparison, _, _, adapters = _oof_comparison(tmp_path, monkeypatch)
    _install_fake_adapter_validator(
        monkeypatch,
        (replace(adapters[0], artifact_sha256="f" * 64), adapters[1]),
    )
    source_manifest = tmp_path / "source.json"
    source_manifest.write_text(
        json.dumps({"tree_sha256": "c" * 64, "files": [{"path": "x"}]}) + "\n",
        encoding="utf-8",
    )
    lockfile = tmp_path / "uv.lock"
    lockfile.write_text("version = 1\n", encoding="utf-8")

    with pytest.raises(GateBValidationError, match="adapter bytes/scope changed"):
        freeze_development_selection(
            comparison,
            primary_label="qlora",
            fallback_label=None,
            decision_note="The selected adapter must remain byte-identical.",
            source_manifest_path=source_manifest,
            lockfile_path=lockfile,
            output_path=tmp_path / "swapped-freeze.json",
        )


def test_base_development_artifact_is_required_by_exact_checkpoint(
    tmp_path: Path,
) -> None:
    manifest, records = _fixture()
    base = _write_run(tmp_path, label="base", manifest=manifest, all_records=records)
    validation = _validation_records(manifest, records)

    require_base_development_artifact(
        base.manifest_path,
        validation,
        split_manifest=manifest,
        fold=0,
        excluded_ids=(),
        expected_checkpoint_sha256=BASE_MODEL_CHECKPOINT_SHA256,
    )
    with pytest.raises(GateBValidationError, match="pinned base checkpoint"):
        require_base_development_artifact(
            base.manifest_path,
            validation,
            split_manifest=manifest,
            fold=0,
            excluded_ids=(),
            expected_checkpoint_sha256="f" * 64,
        )


def test_base_development_artifact_rejects_changed_b0_evidence(tmp_path: Path) -> None:
    manifest, records = _fixture()
    base = _write_run(tmp_path, label="base", manifest=manifest, all_records=records)
    validation = _validation_records(manifest, records)
    (tmp_path / "preflight.json").write_text('{"changed": true}\n', encoding="utf-8")

    with pytest.raises(GateBValidationError, match="model preflight report evidence bytes"):
        require_base_development_artifact(
            base.manifest_path,
            validation,
            split_manifest=manifest,
            fold=0,
            excluded_ids=(),
            expected_checkpoint_sha256=BASE_MODEL_CHECKPOINT_SHA256,
        )


def test_comparison_recomputes_parser_and_rejects_multisample(tmp_path: Path) -> None:
    manifest, records = _fixture()
    base = _write_run(tmp_path, label="base", manifest=manifest, all_records=records)
    candidate = _write_run(tmp_path, label="qlora", manifest=manifest, all_records=records)

    rows = candidate.records_path.read_text(encoding="utf-8").splitlines()
    first = json.loads(rows[0])
    first["parse"]["value"] = int(first["parse"]["value"]) + 10
    rows[0] = json.dumps(first, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    changed = ("\n".join(rows) + "\n").encode()
    candidate.records_path.write_bytes(changed)
    run_manifest = json.loads(candidate.manifest_path.read_text(encoding="utf-8"))
    run_manifest["records_bytes"] = len(changed)
    run_manifest["records_sha256"] = hashlib.sha256(changed).hexdigest()
    candidate.manifest_path.write_text(
        json.dumps(run_manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(GateBValidationError, match="parser result was changed"):
        compare_development_runs(
            base,
            (candidate,),
            _validation_records(manifest, records),
            split_manifest=manifest,
            fold=0,
            excluded_ids=(),
            train_file_sha256="1" * 64,
            exclusions_file_sha256="2" * 64,
            split_artifact_sha256="3" * 64,
            development_shard_sha256="4" * 64,
            output_path=tmp_path / "bad.json",
        )

    other = tmp_path / "multi"
    other.mkdir()
    multi = _write_run(
        other,
        label="multi",
        manifest=manifest,
        all_records=records,
        samples_per_problem=2,
    )
    with pytest.raises(GateBValidationError, match="sample_index=0"):
        compare_development_runs(
            base,
            (multi,),
            _validation_records(manifest, records),
            split_manifest=manifest,
            fold=0,
            excluded_ids=(),
            train_file_sha256="1" * 64,
            exclusions_file_sha256="2" * 64,
            split_artifact_sha256="3" * 64,
            development_shard_sha256="4" * 64,
            output_path=tmp_path / "multi.json",
        )


def test_freeze_and_split_keyed_holdout_claim_are_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    comparison, manifest, _, _ = _oof_comparison(tmp_path, monkeypatch)
    source_manifest = tmp_path / "source.json"
    source_manifest.write_text(
        json.dumps({"tree_sha256": "c" * 64, "files": [{"path": "x"}]}) + "\n",
        encoding="utf-8",
    )
    lockfile = tmp_path / "uv.lock"
    lockfile.write_text("version = 1\n", encoding="utf-8")
    freeze = tmp_path / "freeze.json"
    frozen = freeze_development_selection(
        comparison,
        primary_label="base",
        fallback_label="qlora",
        decision_note="Development evidence reviewed before opening holdout.",
        source_manifest_path=source_manifest,
        lockfile_path=lockfile,
        output_path=freeze,
        now=lambda: datetime(2026, 8, 4, tzinfo=UTC),
    )
    assert Path(frozen.path) == freeze.resolve()
    frozen_payload = json.loads(freeze.read_text(encoding="utf-8"))
    assert frozen_payload["selection_frozen"] is True
    assert frozen_payload["locked_holdout_accessed"] is False
    assert frozen_payload["leaderboard_used_for_selection"] is False
    assert (
        frozen_payload["routing_policy"]
        == "primary_then_fallback_only_on_primary_parse_failure"
    )
    assert frozen_payload["fallback_routing_evidence"]["primary_label"] == "base"
    methods = validate_frozen_selection_methods(freeze, split_manifest=manifest)
    assert methods.routing_policy == frozen_payload["routing_policy"]
    assert "holdout_ids" not in json.dumps(frozen_payload)

    ledger = tmp_path / "locked-holdout-access-v1"
    ledger.mkdir()
    with pytest.raises(GateBValidationError, match="acknowledgement"):
        authorize_locked_holdout_once(
            freeze,
            split_manifest=manifest,
            excluded_ids=(),
            acknowledgement="no",
            ledger_root=ledger,
        )
    assert not list(ledger.iterdir())

    receipt = authorize_locked_holdout_once(
        freeze,
        split_manifest=manifest,
        excluded_ids=(),
        acknowledgement=HOLDOUT_ACCESS_ACKNOWLEDGEMENT,
        ledger_root=ledger,
        now=lambda: datetime(2026, 8, 4, tzinfo=UTC),
    )
    claim_path = ledger / f"{manifest.sha256}.claim.json"
    assert claim_path.is_file()
    assert Path(receipt.path).name == f"{manifest.sha256}.receipt.json"
    receipt_payload = json.loads(Path(receipt.path).read_text(encoding="utf-8"))
    rendered = json.dumps(receipt_payload)
    assert receipt_payload["ids_questions_answers_emitted"] is False
    assert "train-" not in rendered
    access = validate_locked_holdout_access(
        receipt.path,
        freeze_artifact=freeze,
        split_manifest=manifest,
        excluded_ids=(),
    )
    assert access.eligible_ids == manifest.final_holdout_ids()

    with pytest.raises(GateBValidationError, match="already exists|overwrite"):
        authorize_locked_holdout_once(
            freeze,
            split_manifest=manifest,
            excluded_ids=(),
            acknowledgement=HOLDOUT_ACCESS_ACKNOWLEDGEMENT,
            ledger_root=ledger,
        )


def test_comparison_rejects_symlink_and_changed_freeze_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    comparison, manifest, records, _ = _oof_comparison(tmp_path, monkeypatch)
    linked = tmp_path / "linked.jsonl"
    linked.symlink_to(tmp_path / "base.jsonl")
    with pytest.raises(GateBValidationError, match="symlink"):
        compare_development_runs(
            NamedDevelopmentRun("linked", linked, tmp_path / "base.manifest.json"),
            (
                NamedDevelopmentRun(
                    "qlora", tmp_path / "qlora.jsonl", tmp_path / "qlora.manifest.json"
                ),
            ),
            _validation_records(manifest, records),
            split_manifest=manifest,
            fold=0,
            excluded_ids=(),
            train_file_sha256="1" * 64,
            exclusions_file_sha256="2" * 64,
            split_artifact_sha256="3" * 64,
            development_shard_sha256="4" * 64,
            output_path=tmp_path / "linked-out.json",
        )

    source_manifest = tmp_path / "source.json"
    source_manifest.write_text(
        json.dumps({"tree_sha256": "c" * 64, "files": [{"path": "x"}]}) + "\n",
        encoding="utf-8",
    )
    lockfile = tmp_path / "uv.lock"
    lockfile.write_text("version = 1\n", encoding="utf-8")
    base_path = tmp_path / "base.jsonl"
    base_path.write_bytes(base_path.read_bytes() + b" ")
    with pytest.raises(GateBValidationError, match="changed after comparison"):
        freeze_development_selection(
            comparison,
            primary_label="base",
            fallback_label=None,
            decision_note="Evidence-only decision.",
            source_manifest_path=source_manifest,
            lockfile_path=lockfile,
            output_path=tmp_path / "freeze.json",
        )

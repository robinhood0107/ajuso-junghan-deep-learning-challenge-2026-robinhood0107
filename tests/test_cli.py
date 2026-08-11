from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

import deep_challenge.cli as cli_module
from deep_challenge.cli import main
from deep_challenge.data import load_train_csv
from deep_challenge.gate_b import (
    DEFAULT_GATE_B_CONFIG,
    DevelopmentResumeStatus,
    GenerationResult,
)
from deep_challenge.gate_b_prediction import PredictionArtifactWriteResult
from deep_challenge.gate_b_runtime import (
    RuntimeGateEvidence,
    TrainingArtifact,
    TrainingResumeStatus,
)
from deep_challenge.gate_b_selection import (
    HOLDOUT_ACCESS_ACKNOWLEDGEMENT,
    GateBSelectionWriteResult,
)
from deep_challenge.model_preflight import OFFICIAL_MODEL_ID, OFFICIAL_REVISION
from deep_challenge.provenance import (
    build_source_tree_manifest,
    canonical_json_bytes,
    write_json_atomic,
)
from deep_challenge.splits import SplitManifest, eligible_training_ids


def _write_csv(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)


def _datasets(tmp_path: Path) -> tuple[Path, Path]:
    train = tmp_path / "train.csv"
    leaderboard = tmp_path / "leaderboard.csv"
    _write_csv(
        train,
        [
            ["id", "question", "answer"],
            *[
                [
                    f"train-{index}",
                    f"Compute symbol-{chr(97 + index)} when its value is {index}+1.",
                    str(index + 1),
                ]
                for index in range(20)
            ],
        ],
    )
    _write_csv(
        leaderboard,
        [
            ["id", "question", " answer"],
            ["val-0", "What is 5+5?"],
            ["val-1", "What is 8+8?"],
        ],
    )
    return train, leaderboard


def _locked_gate_b_config() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "gate_b"
        / "rtx4070-super-12gb-direct-answer-v1.json"
    )


def _locked_rationale_config() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "gate_b"
        / "rtx4070-super-12gb-concise-rationale-v1.json"
    )


def _write_cli_rationale_source(
    path: Path,
    *,
    train: Path,
    split: Path,
    fold: int,
    excluded_ids: tuple[str, ...],
) -> None:
    split_payload = json.loads(split.read_text(encoding="utf-8"))
    manifest = SplitManifest.from_dict(split_payload["split"])
    records = {record.id: record for record in load_train_csv(train)}
    rows: list[dict[str, object]] = []
    for problem_id in eligible_training_ids(manifest, fold, excluded_ids):
        record = records[problem_id]
        target = (
            "The problem statement directly identifies the requested integer.\n"
            f"Final answer: {record.answer}"
        )
        rows.append(
            {
                "schema_version": "gate-b-concise-rationale-row-v1",
                "problem_id": problem_id,
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
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _gate_b_cli_fixture(tmp_path: Path) -> tuple[list[str], Path, Path, Path]:
    train = tmp_path / "gate-b-train.csv"
    _write_csv(
        train,
        [
            ["id", "question", "answer"],
            *[
                [
                    f"train-{index:06d}",
                    f"Return integer {index} for unique symbol {chr(96 + index)}.",
                    str(index),
                ]
                for index in range(1, 21)
            ],
        ],
    )
    split = tmp_path / "gate-b-split.json"
    assert (
        main(
            [
                "build-splits",
                "--train",
                str(train),
                "--folds",
                "2",
                "--seed",
                "7",
                "--output",
                str(split),
            ]
        )
        == 0
    )
    split_payload = json.loads(split.read_text(encoding="utf-8"))
    development_shard = tmp_path / "development-shard"
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
                split_payload["split"]["sha256"],
                "--output-dir",
                str(development_shard),
            ]
        )
        == 0
    )
    exclusions = tmp_path / "gate-b-exclusions.csv"
    _write_csv(exclusions, [["id"], ["train-000001"]])
    contract = [
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
        "1",
        "--expected-split-sha256",
        split_payload["split"]["sha256"],
        "--development-shard",
        str(development_shard),
        "--expected-development-shard-sha256",
        hashlib.sha256(
            (development_shard / "CHECKSUMS.sha256").read_bytes()
        ).hexdigest(),
    ]
    return contract, train, exclusions, split


def _gate_b_runtime_source_args(
    tmp_path: Path,
) -> tuple[list[str], Path, Path, RuntimeGateEvidence]:
    source_root = tmp_path / "runtime-source"
    source_root.mkdir(exist_ok=True)
    (source_root / "runtime.py").write_text("VALUE = 1\n", encoding="utf-8")
    source_manifest = tmp_path / "runtime-source-manifest.json"
    write_json_atomic(
        source_manifest,
        build_source_tree_manifest(source_root, excluded_paths=(source_manifest,)).as_dict(),
    )
    preflight = tmp_path / "preflight.json"
    smoke = tmp_path / "smoke.json"
    preflight.write_text("{}\n", encoding="utf-8")
    smoke.write_text("{}\n", encoding="utf-8")
    runtime_gate = RuntimeGateEvidence(
        preflight_path=str(preflight.resolve()),
        preflight_sha256=hashlib.sha256(preflight.read_bytes()).hexdigest(),
        smoke_path=str(smoke.resolve()),
        smoke_sha256=hashlib.sha256(smoke.read_bytes()).hexdigest(),
        model_id=OFFICIAL_MODEL_ID,
        revision=OFFICIAL_REVISION,
        config_sha256=DEFAULT_GATE_B_CONFIG.sha256,
        device_name="NVIDIA Test GPU",
    )
    return (
        [
            "--source-root",
            str(source_root),
            "--source-manifest",
            str(source_manifest),
        ],
        preflight,
        smoke,
        runtime_gate,
    )


def test_audit_and_source_manifest_commands(tmp_path: Path) -> None:
    train, leaderboard = _datasets(tmp_path)
    audit = tmp_path / "audit.json"
    assert (
        main(
            [
                "audit-data",
                "--train",
                str(train),
                "--leaderboard",
                str(leaderboard),
                "--source-root",
                str(tmp_path),
                "--output",
                str(audit),
            ]
        )
        == 0
    )
    report = json.loads(audit.read_text(encoding="utf-8"))
    assert report["train"]["row_count"] == 20

    source = tmp_path / "source.json"
    assert main(["source-manifest", "--root", str(tmp_path), "--output", str(source)]) == 0
    assert len(json.loads(source.read_text(encoding="utf-8"))["tree_sha256"]) == 64


def test_source_manifest_command_excludes_output_and_is_repeatable(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    (root / "module.py").write_text("value = 1\n", encoding="utf-8")
    output = root / "source-manifest.json"
    assert main(["source-manifest", "--root", str(root), "--output", str(output)]) == 0
    first = json.loads(output.read_text(encoding="utf-8"))
    assert main(["source-manifest", "--root", str(root), "--output", str(output)]) == 0
    second = json.loads(output.read_text(encoding="utf-8"))
    assert first == second
    assert [entry["path"] for entry in second["files"]] == ["module.py"]


def test_audit_command_excludes_its_output_from_source_manifest(tmp_path: Path) -> None:
    root = tmp_path / "audit-source"
    root.mkdir()
    train, leaderboard = _datasets(root)
    (root / "module.py").write_text("value = 1\n", encoding="utf-8")
    output = root / "audit.json"
    arguments = [
        "audit-data",
        "--train",
        str(train),
        "--leaderboard",
        str(leaderboard),
        "--source-root",
        str(root),
        "--output",
        str(output),
    ]
    assert main(arguments) == 0
    first = json.loads(output.read_text(encoding="utf-8"))
    assert main(arguments) == 0
    second = json.loads(output.read_text(encoding="utf-8"))
    assert first["source_tree"] == second["source_tree"]
    assert [entry["path"] for entry in second["source_tree"]["files"]] == ["module.py"]


def test_split_command_is_deterministic(tmp_path: Path) -> None:
    train, _ = _datasets(tmp_path)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    arguments = ["build-splits", "--train", str(train), "--folds", "3", "--seed", "7"]
    assert main([*arguments, "--output", str(first)]) == 0
    assert main([*arguments, "--output", str(second)]) == 0
    first_report = json.loads(first.read_text(encoding="utf-8"))
    second_report = json.loads(second.read_text(encoding="utf-8"))
    assert first_report["split"]["sha256"] == second_report["split"]["sha256"]
    assert first_report["strata_summary"]["overall"]["row_count"] == 20
    assert first_report["strata_summary"]["final_locked_holdout"]["row_count"] == 2
    assert sum(
        fold["row_count"] for fold in first_report["strata_summary"]["folds"].values()
    ) == 18


def test_split_command_keeps_number_masked_templates_as_soft_signals(tmp_path: Path) -> None:
    train = tmp_path / "train.csv"
    rows = [
        ["train-0", "What is 2+3?", "5"],
        ["train-1", "What is 7+8?", "15"],
        *[
            [
                f"train-{index}",
                f"Compute unique-symbol-{chr(97 + index)}.",
                str(index),
            ]
            for index in range(2, 20)
        ],
    ]
    _write_csv(train, [["id", "question", "answer"], *rows])
    output = tmp_path / "split.json"
    assert (
        main(
            [
                "build-splits",
                "--train",
                str(train),
                "--folds",
                "3",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["cluster_count"] == 20
    assert "soft audit candidates" in report["cluster_method"]


def test_parse_answer_command_returns_nonzero_for_invalid(capsys) -> None:
    assert main(["parse-answer", "--text", "Final answer: -42"]) == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["value"] == -42
    assert main(["parse-answer", "--text", "No result."]) == 1


def test_gpu_smoke_command_requires_explicit_acknowledgement(tmp_path: Path) -> None:
    output = tmp_path / "gpu-smoke.json"
    assert (
        main(
            [
                "gpu-smoke",
                "--preflight-report",
                str(tmp_path / "not-read-without-ack.json"),
                "--output",
                str(output),
            ]
        )
        == 2
    )
    assert not output.exists()


def test_validate_submission_command(tmp_path: Path, capsys) -> None:
    _, expected = _datasets(tmp_path)
    submission = tmp_path / "submission.csv"
    _write_csv(submission, [["ID", "answer"], ["val-0", "10"], ["val-1", "16"]])
    assert (
        main(
            [
                "validate-submission",
                "--submission",
                str(submission),
                "--expected",
                str(expected),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["valid"] is True


def test_write_submission_command_is_strict_and_content_addressed(
    tmp_path: Path, capsys
) -> None:
    _, expected = _datasets(tmp_path)
    predictions = tmp_path / "predictions.json"
    predictions.write_text('{"val-0": "10", "val-1": 16}', encoding="utf-8")
    output = tmp_path / "submission.csv"
    assert (
        main(
            [
                "write-submission",
                "--predictions",
                str(predictions),
                "--expected",
                str(expected),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["row_count"] == 2
    assert report["fallback_count"] == 0
    assert len(report["sha256"]) == 64
    assert output.read_text(encoding="utf-8") == "ID,answer\nval-0,10\nval-1,16\n"

    incomplete = tmp_path / "incomplete.json"
    incomplete.write_text('{"val-0": "10"}', encoding="utf-8")
    assert (
        main(
            [
                "write-submission",
                "--predictions",
                str(incomplete),
                "--expected",
                str(expected),
                "--output",
                str(tmp_path / "invalid.csv"),
            ]
        )
        == 2
    )


def test_filtered_audit_uses_organizer_exclusion_ids_without_rewriting_train(
    tmp_path: Path,
) -> None:
    train, _ = _datasets(tmp_path)
    leaderboard = tmp_path / "leaderboard-filtered.csv"
    _write_csv(
        leaderboard,
        [["id", "question"], ["val-0", "What is 5+5?"], ["val-1", "What is 8+8?"]],
    )
    exclusions = tmp_path / "train_filtered_ids.csv"
    _write_csv(
        exclusions,
        [
            ["id", "answer", "question"],
            ["train-0", "1", "ignored historical question"],
            ["train-3", "4", "ignored historical question"],
        ],
    )
    output = tmp_path / "filtered-audit.json"

    assert (
        main(
            [
                "audit-data",
                "--train",
                str(train),
                "--train-exclusions",
                str(exclusions),
                "--train-scope",
                "all",
                "--allow-all-train-scope",
                "--leaderboard",
                str(leaderboard),
                "--source-root",
                str(tmp_path),
                "--output",
                str(output),
            ]
        )
        == 0
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["audit_version"] == "data-audit-v4-filtered"
    assert report["train"]["row_count"] == 18
    assert report["leaderboard"]["raw_header"] == ["id", "question"]
    assert report["eligibility"]["excluded_count"] == 2
    assert report["eligibility"]["eligible_count"] == 18
    assert report["eligibility"]["hard_group_expansion_applied"] is False
    assert report["train"]["manifest"]["sha256"] == hashlib.sha256(
        train.read_bytes()
    ).hexdigest()


def test_eligibility_overlay_reuses_split_and_never_emits_holdout_ids(
    tmp_path: Path,
) -> None:
    train, leaderboard = _datasets(tmp_path)
    split = tmp_path / "split.json"
    assert (
        main(
            [
                "build-splits",
                "--train",
                str(train),
                "--folds",
                "3",
                "--seed",
                "7",
                "--output",
                str(split),
            ]
        )
        == 0
    )
    split_payload = json.loads(split.read_text(encoding="utf-8"))
    assignments = split_payload["split"]["assignments"]
    direct_id = next(
        item["record_id"]
        for item in assignments
        if item["partition"] == "cross_validation"
    )
    holdout_ids = {
        item["record_id"]
        for item in assignments
        if item["partition"] == "final_locked_holdout"
    }
    exclusions = tmp_path / "train_filtered_ids.csv"
    _write_csv(exclusions, [["id"], [direct_id]])
    output = tmp_path / "eligibility.json"

    assert (
        main(
            [
                "build-eligibility-overlay",
                "--train",
                str(train),
                "--train-exclusions",
                str(exclusions),
                "--split-artifact",
                str(split),
                "--output",
                str(output),
            ]
        )
        == 0
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema_version"] == "train-eligibility-v1"
    assert report["direct_excluded_count"] == 1
    assert report["expanded_excluded_count"] == 1
    assert report["locked_holdout"]["accessed_for_model_selection"] is False
    assert report["locked_holdout"]["ids_or_answers_emitted"] is False
    serialized = output.read_text(encoding="utf-8")
    assert "expanded_excluded_ids" not in report
    assert not any(holdout_id in serialized for holdout_id in holdout_ids)

    audit = tmp_path / "cv-only-audit.json"
    assert (
        main(
            [
                "audit-data",
                "--train",
                str(train),
                "--train-exclusions",
                str(exclusions),
                "--split-artifact",
                str(split),
                "--train-scope",
                "cv-only",
                "--leaderboard",
                str(leaderboard),
                "--source-root",
                str(tmp_path),
                "--output",
                str(audit),
            ]
        )
        == 0
    )
    audit_report = json.loads(audit.read_text(encoding="utf-8"))
    assert audit_report["train"]["row_count"] == 17
    assert audit_report["eligibility"]["hard_group_expansion_applied"] is True
    assert audit_report["eligibility"]["scope"]["train_scope"] == "cv-only"
    assert (
        audit_report["eligibility"]["scope"][
            "locked_holdout_used_for_metrics_or_examples"
        ]
        is False
    )


def test_eligibility_overlay_rejects_train_sha_mismatch(tmp_path: Path) -> None:
    train, _ = _datasets(tmp_path)
    split = tmp_path / "split.json"
    assert main(["build-splits", "--train", str(train), "--output", str(split)]) == 0
    split_payload = json.loads(split.read_text(encoding="utf-8"))
    split_payload["dataset_manifest"]["sha256"] = "0" * 64
    split.write_text(json.dumps(split_payload), encoding="utf-8")
    exclusions = tmp_path / "train_filtered_ids.csv"
    _write_csv(exclusions, [["id"], ["train-0"]])

    assert (
        main(
            [
                "build-eligibility-overlay",
                "--train",
                str(train),
                "--train-exclusions",
                str(exclusions),
                "--split-artifact",
                str(split),
                "--output",
                str(tmp_path / "eligibility.json"),
            ]
        )
        == 2
    )


def test_eligibility_overlay_rejects_tampered_cluster_metadata(tmp_path: Path) -> None:
    train, _ = _datasets(tmp_path)
    split = tmp_path / "split.json"
    assert main(["build-splits", "--train", str(train), "--output", str(split)]) == 0
    split_payload = json.loads(split.read_text(encoding="utf-8"))
    split_payload["cluster_count"] += 1
    split.write_text(json.dumps(split_payload), encoding="utf-8")
    exclusions = tmp_path / "train_filtered_ids.csv"
    _write_csv(exclusions, [["id"], ["train-0"]])

    assert (
        main(
            [
                "build-eligibility-overlay",
                "--train",
                str(train),
                "--train-exclusions",
                str(exclusions),
                "--split-artifact",
                str(split),
                "--output",
                str(tmp_path / "eligibility.json"),
            ]
        )
        == 2
    )


class _GateBTokenizer:
    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        assert tokenize is True
        assert [item["role"] for item in conversation[:2]] == ["system", "user"]
        return [1, 2] if add_generation_prompt else [1, 2, 3]


def test_gate_b_sft_preflight_cli_is_cpu_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract, _, _, _ = _gate_b_cli_fixture(tmp_path)
    provenance = {
        "model_id": OFFICIAL_MODEL_ID,
        "requested_revision": OFFICIAL_REVISION,
        "resolved_commit": OFFICIAL_REVISION,
        "local_files_only": True,
        "files": {},
    }
    monkeypatch.setattr(
        cli_module,
        "load_pinned_tokenizer",
        lambda **_kwargs: (_GateBTokenizer(), provenance),
    )
    output = tmp_path / "sft-preflight.json"

    assert (
        main(
            [
                "gate-b-sft-preflight",
                *contract,
                "--revision",
                OFFICIAL_REVISION,
                "--config",
                str(_locked_gate_b_config()),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "green"
    assert payload["torch_or_cuda_used"] is False
    assert payload["locked_holdout_accessed"] is False


def test_rationale_corpus_cli_roundtrip_and_sft_preflight_are_cpu_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract, train, _, split = _gate_b_cli_fixture(tmp_path)
    source = tmp_path / "private-teacher.jsonl"
    _write_cli_rationale_source(
        source,
        train=train,
        split=split,
        fold=0,
        excluded_ids=("train-000001",),
    )
    corpus_dir = tmp_path / "private-corpus"
    corpus_dir.mkdir()
    corpus = corpus_dir / "rationales.jsonl"
    corpus_manifest = corpus_dir / "manifest.json"
    audit = tmp_path / "rationale-audit.json"

    assert (
        main(
            [
                "build-rationale-corpus",
                *contract,
                "--source-jsonl",
                str(source),
                "--rationale-config",
                str(_locked_rationale_config()),
                "--output-jsonl",
                str(corpus),
                "--output-manifest",
                str(corpus_manifest),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "audit-rationale-corpus",
                *contract,
                "--rationale-corpus",
                str(corpus),
                "--rationale-manifest",
                str(corpus_manifest),
                "--rationale-config",
                str(_locked_rationale_config()),
                "--output",
                str(audit),
            ]
        )
        == 0
    )
    provenance = {
        "model_id": OFFICIAL_MODEL_ID,
        "requested_revision": OFFICIAL_REVISION,
        "resolved_commit": OFFICIAL_REVISION,
        "local_files_only": True,
        "files": {},
    }
    monkeypatch.setattr(
        cli_module,
        "load_pinned_tokenizer",
        lambda **_kwargs: (_GateBTokenizer(), provenance),
    )
    preflight = tmp_path / "rationale-sft-preflight.json"
    rationale_inputs = [
        "--rationale-corpus",
        str(corpus),
        "--rationale-manifest",
        str(corpus_manifest),
        "--rationale-audit",
        str(audit),
        "--rationale-config",
        str(_locked_rationale_config()),
    ]
    assert (
        main(
            [
                "gate-b-sft-preflight",
                *contract,
                "--revision",
                OFFICIAL_REVISION,
                "--config",
                str(_locked_gate_b_config()),
                *rationale_inputs,
                "--output",
                str(preflight),
            ]
        )
        == 0
    )

    audit_payload = json.loads(audit.read_text(encoding="utf-8"))
    preflight_payload = json.loads(preflight.read_text(encoding="utf-8"))
    assert audit_payload["raw_rationale_serialized"] is False
    assert audit_payload["locked_holdout_accessed"] is False
    assert preflight_payload["schema_version"] == "gate-b-sft-encoding-preflight-v4"
    assert preflight_payload["training_target"]["kind"] == (
        "verified_concise_rationale"
    )
    assert preflight_payload["training_target"]["corpus_audit_sha256"] == (
        hashlib.sha256(audit.read_bytes()).hexdigest()
    )
    assert preflight_payload["torch_or_cuda_used"] is False
    assert preflight_payload["locked_holdout_accessed"] is False


def test_rationale_training_cli_inputs_are_all_or_none(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    contract, _, _, _ = _gate_b_cli_fixture(tmp_path)
    output = tmp_path / "must-not-exist.json"

    assert (
        main(
            [
                "gate-b-sft-preflight",
                *contract,
                "--revision",
                OFFICIAL_REVISION,
                "--config",
                str(_locked_gate_b_config()),
                "--rationale-corpus",
                str(tmp_path / "unread-corpus.jsonl"),
                "--output",
                str(output),
            ]
        )
        == 2
    )
    assert "all-or-none" in capsys.readouterr().err
    assert not output.exists()


def test_compare_development_oof_cli_wires_all_folds_from_development_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract, _, _, _ = _gate_b_cli_fixture(tmp_path)
    fold_index = contract.index("--fold")
    del contract[fold_index : fold_index + 2]
    captured: dict[str, object] = {}

    def fake_compare(*args: object, **kwargs: object) -> GateBSelectionWriteResult:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return GateBSelectionWriteResult(
            path=str(tmp_path / "oof.json"),
            size_bytes=123,
            sha256="a" * 64,
            payload_sha256="b" * 64,
        )

    monkeypatch.setattr(
        cli_module, "compare_cross_fold_development_runs", fake_compare
    )
    output = tmp_path / "oof.json"
    assert (
        main(
            [
                "compare-development-oof",
                *contract,
                "--deployment-fold",
                "0",
                "--reference-label",
                "base",
                "--candidate-label",
                "qlora",
                "--base-run",
                "0",
                str(tmp_path / "base0.jsonl"),
                str(tmp_path / "base0.json"),
                "--adapter-run",
                "0",
                "qlora",
                str(tmp_path / "qlora0.jsonl"),
                str(tmp_path / "qlora0.json"),
                str(tmp_path / "adapter0"),
                "--base-run",
                "1",
                str(tmp_path / "base1.jsonl"),
                str(tmp_path / "base1.json"),
                "--adapter-run",
                "1",
                "qlora",
                str(tmp_path / "qlora1.jsonl"),
                str(tmp_path / "qlora1.json"),
                str(tmp_path / "adapter1"),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    positional = captured["args"]
    assert isinstance(positional, tuple)
    assert positional[0:2] == ("base", ("qlora",))
    fold_runs = positional[2]
    assert isinstance(fold_runs, tuple)
    assert {(item.fold, item.label) for item in fold_runs} == {
        (0, "base"),
        (0, "qlora"),
        (1, "base"),
        (1, "qlora"),
    }
    assert {
        (item.fold, item.label, item.method_kind, item.adapter_path)
        for item in fold_runs
    } == {
        (0, "base", "fixed_base", None),
        (0, "qlora", "adapter", tmp_path / "adapter0"),
        (1, "base", "fixed_base", None),
        (1, "qlora", "adapter", tmp_path / "adapter1"),
    }
    development_records = positional[3]
    assert isinstance(development_records, tuple)
    assert all(record.id != "train-000001" for record in development_records)
    keyword = captured["kwargs"]
    assert isinstance(keyword, dict)
    assert keyword["deployment_fold"] == 0


def test_verify_base_development_oof_cli_wires_complete_base_folds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    contract, _, _, _ = _gate_b_cli_fixture(tmp_path)
    fold_index = contract.index("--fold")
    del contract[fold_index : fold_index + 2]
    captured: dict[str, object] = {}

    def fake_verify(*args: object, **kwargs: object) -> GateBSelectionWriteResult:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return GateBSelectionWriteResult(
            path=str(tmp_path / "base-oof.json"),
            size_bytes=123,
            sha256="a" * 64,
            payload_sha256="b" * 64,
        )

    monkeypatch.setattr(cli_module, "verify_base_development_oof", fake_verify)
    output = tmp_path / "base-oof.json"
    assert (
        main(
            [
                "verify-base-development-oof",
                *contract,
                "--deployment-fold",
                "0",
                "--base-label",
                "fixed-base",
                "--base-run",
                "0",
                str(tmp_path / "base0.jsonl"),
                str(tmp_path / "base0.json"),
                "--base-run",
                "1",
                str(tmp_path / "base1.jsonl"),
                str(tmp_path / "base1.json"),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    positional = captured["args"]
    assert isinstance(positional, tuple)
    assert positional[0] == "fixed-base"
    fold_runs = positional[1]
    assert isinstance(fold_runs, tuple)
    assert [(item.fold, item.label, item.method_kind) for item in fold_runs] == [
        (0, "fixed-base", "fixed_base"),
        (1, "fixed-base", "fixed_base"),
    ]
    assert all(item.adapter_path is None for item in fold_runs)
    development_records = positional[2]
    assert isinstance(development_records, tuple)
    assert all(record.id != "train-000001" for record in development_records)
    keyword = captured["kwargs"]
    assert isinstance(keyword, dict)
    assert keyword["deployment_fold"] == 0
    rendered = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert rendered["qualified"] is True
    assert rendered["holdout_accessed"] is False


def test_freeze_development_base_cli_requires_ack_and_wires_primary_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    captured: dict[str, object] = {}

    def fake_freeze(*args: object, **kwargs: object) -> GateBSelectionWriteResult:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return GateBSelectionWriteResult(
            path=str(tmp_path / "freeze.json"),
            size_bytes=123,
            sha256="a" * 64,
            payload_sha256="b" * 64,
        )

    monkeypatch.setattr(cli_module, "freeze_base_development_selection", fake_freeze)
    common = [
        "freeze-development-base",
        "--base-oof-artifact",
        str(tmp_path / "base-oof.json"),
        "--primary-label",
        "fixed-base",
        "--decision-note",
        "candidate unavailable; freeze qualified complete base OOF",
        "--source-manifest",
        str(tmp_path / "source.json"),
        "--lockfile",
        str(tmp_path / "uv.lock"),
        "--output",
        str(tmp_path / "freeze.json"),
    ]
    assert main(common) == 2
    assert "required" in capsys.readouterr().err
    assert captured == {}

    assert main([*common, "--confirm-no-leaderboard-selection"]) == 0
    assert captured["args"] == (tmp_path / "base-oof.json",)
    assert captured["kwargs"] == {
        "primary_label": "fixed-base",
        "decision_note": "candidate unavailable; freeze qualified complete base OOF",
        "source_manifest_path": tmp_path / "source.json",
        "lockfile_path": tmp_path / "uv.lock",
        "output_path": tmp_path / "freeze.json",
    }
    rendered = json.loads(capsys.readouterr().out)
    assert rendered["fallback_label"] is None
    assert rendered["routing_policy"] == "primary_only"
    assert rendered["selection_frozen"] is True


def test_gate_b_status_clis_emit_shared_raw_free_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    development = DevelopmentResumeStatus(
        contract_sha256="a" * 64,
        state="interrupted",
        process_id=None,
        total_chunks=2,
        completed_chunks=1,
        total_generations=10,
        completed_generations=4,
        chunk_attempt_count=2,
        invalid_chunk_attempt_count=1,
        completed_latency_ms=12.5,
    )
    training = TrainingResumeStatus(
        contract_sha256="b" * 64,
        state="retryable",
        process_id=None,
        latest_checkpoint_step=7,
        completion_artifact_sha256=None,
    )
    monkeypatch.setattr(
        cli_module, "read_development_resume_status", lambda _path: development
    )
    monkeypatch.setattr(
        cli_module, "read_training_resume_status", lambda _path: training
    )
    development_output = tmp_path / "development-status.json"
    assert (
        main(
            [
                "gate-b-development-status",
                "--resume-dir",
                str(tmp_path / "private-development"),
                "--output",
                str(development_output),
            ]
        )
        == 0
    )
    development_payload = json.loads(capsys.readouterr().out)
    assert development_payload == json.loads(
        development_output.read_text(encoding="utf-8")
    )
    assert development_payload["schema_version"] == "gate-b-workflow-status-v1"
    assert development_payload["state"] == "retryable"
    assert development_payload["next_action"] == "resume_development"

    training_output = tmp_path / "training-status.json"
    assert (
        main(
            [
                "gate-b-training-status",
                "--resume-dir",
                str(tmp_path / "private-training"),
                "--output",
                str(training_output),
            ]
        )
        == 0
    )
    training_payload = json.loads(capsys.readouterr().out)
    assert training_payload == json.loads(training_output.read_text(encoding="utf-8"))
    assert training_payload["schema_version"] == "gate-b-workflow-status-v1"
    assert training_payload["state"] == "retryable"
    assert training_payload["next_action"] == "resume_training"
    serialized = json.dumps([development_payload, training_payload])
    assert "/mnt/" not in serialized
    assert not any(
        word in serialized for word in ("problem_id", "question", "answer", "prompt")
    )

    assert (
        main(
            [
                "gate-b-development-status",
                "--resume-dir",
                str(tmp_path / "private-development"),
            ]
        )
        == 0
    )
    stdout_only_development = json.loads(capsys.readouterr().out)
    assert stdout_only_development == development_payload
    assert (
        main(
            [
                "gate-b-training-status",
                "--resume-dir",
                str(tmp_path / "private-training"),
            ]
        )
        == 0
    )
    stdout_only_training = json.loads(capsys.readouterr().out)
    assert stdout_only_training == training_payload


def test_decide_candidate_probe_cli_writes_cost_control_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    comparison = tmp_path / "comparison.json"
    comparison.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "decision.json"
    captured: dict[str, object] = {}

    def fake_decide(
        comparison_artifact: Path,
        *,
        candidate_label: str,
        output_path: Path,
    ) -> GateBSelectionWriteResult:
        captured.update(
            comparison_artifact=comparison_artifact,
            candidate_label=candidate_label,
            output_path=output_path,
        )
        output_path.write_text(
            json.dumps(
                {
                    "candidate_action": "stop_before_remaining_folds",
                    "candidate_full_oof_authorized": False,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return GateBSelectionWriteResult(
            path=str(output_path),
            size_bytes=output_path.stat().st_size,
            sha256="a" * 64,
            payload_sha256="b" * 64,
        )

    monkeypatch.setattr(cli_module, "decide_candidate_probe_promotion", fake_decide)
    assert (
        main(
            [
                "decide-candidate-probe",
                "--comparison-artifact",
                str(comparison),
                "--candidate-label",
                "qlora-direct",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert captured == {
        "comparison_artifact": comparison,
        "candidate_label": "qlora-direct",
        "output_path": output,
    }
    rendered = json.loads(capsys.readouterr().out)
    assert rendered["candidate_action"] == "stop_before_remaining_folds"
    assert rendered["candidate_full_oof_authorized"] is False
    assert rendered["selection_frozen"] is False
    assert rendered["holdout_accessed"] is False


@pytest.mark.parametrize(
    "command",
    [
        "gate-b-development",
        "gate-b-train-fold",
        "gate-b-predict-evaluation",
    ],
)
def test_gate_b_gpu_clis_fail_before_reading_files_without_ack(
    tmp_path: Path, command: str
) -> None:
    common = [
        command,
        "--train",
        str(tmp_path / "unread-train.csv"),
        "--train-exclusions",
        str(tmp_path / "unread-exclusions.csv"),
        "--split-artifact",
        str(tmp_path / "unread-split.json"),
        "--fold",
        "0",
        "--expected-train-sha256",
        "1" * 64,
        "--expected-exclusions-sha256",
        "2" * 64,
        "--expected-exclusion-count",
        "1",
        "--expected-split-sha256",
        "3" * 64,
        "--expected-development-shard-sha256",
        "5" * 64,
        "--preflight-report",
        str(tmp_path / "unread-preflight.json"),
        "--gpu-smoke-report",
        str(tmp_path / "unread-smoke.json"),
        "--config",
        str(tmp_path / "unread-config.json"),
    ]
    if command == "gate-b-development":
        common.extend(
            [
                "--development-shard",
                str(tmp_path / "unread-development-shard"),
                "--source-root",
                str(tmp_path / "unread-source-root"),
                "--source-manifest",
                str(tmp_path / "unread-source-manifest.json"),
            ]
        )
        common.extend(
            [
                "--adapter",
                str(tmp_path / "unread-adapter"),
                "--base-baseline-manifest",
                str(tmp_path / "unread-base.json"),
                "--output-jsonl",
                str(tmp_path / "not-created.jsonl"),
                "--output-manifest",
                str(tmp_path / "not-created.manifest.json"),
            ]
        )
    elif command == "gate-b-train-fold":
        common.extend(
            [
                "--development-shard",
                str(tmp_path / "unread-development-shard"),
                "--source-root",
                str(tmp_path / "unread-source-root"),
                "--source-manifest",
                str(tmp_path / "unread-source-manifest.json"),
            ]
        )
        common.extend(
            [
                "--base-baseline-manifest",
                str(tmp_path / "unread-base.json"),
                "--output-dir",
                str(tmp_path / "not-created-adapter"),
            ]
        )
    else:
        common.extend(
            [
                "--evaluation",
                str(tmp_path / "unread-evaluation.csv"),
                "--dataset-role",
                "leaderboard",
                "--expected-evaluation-sha256",
                "4" * 64,
                "--freeze-artifact",
                str(tmp_path / "unread-freeze.json"),
                "--primary-kind",
                "base",
                "--fallback-kind",
                "none",
                "--output-artifact",
                str(tmp_path / "not-created-evaluation.json"),
                "--output-predictions",
                str(tmp_path / "not-created-predictions.json"),
            ]
        )

    assert main(common) == 2
    assert not any(path.name.startswith("not-created") for path in tmp_path.iterdir())


def test_locked_holdout_cli_requires_gpu_and_separate_one_time_acknowledgements(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    common = [
        "gate-b-locked-holdout-evaluate",
        "--train",
        str(tmp_path / "unread-train.csv"),
        "--train-exclusions",
        str(tmp_path / "unread-exclusions.csv"),
        "--split-artifact",
        str(tmp_path / "unread-split.json"),
        "--fold",
        "0",
        "--expected-train-sha256",
        "1" * 64,
        "--expected-exclusions-sha256",
        "2" * 64,
        "--expected-exclusion-count",
        "1",
        "--expected-split-sha256",
        "3" * 64,
        "--expected-development-shard-sha256",
        "5" * 64,
        "--preflight-report",
        str(tmp_path / "unread-preflight.json"),
        "--gpu-smoke-report",
        str(tmp_path / "unread-smoke.json"),
        "--config",
        str(tmp_path / "unread-config.json"),
        "--freeze-artifact",
        str(tmp_path / "unread-freeze.json"),
        "--primary-kind",
        "base",
        "--fallback-kind",
        "none",
        "--output",
        str(tmp_path / "not-created-holdout.json"),
    ]

    assert main([*common, "--acknowledge-one-time-locked-holdout"]) == 2
    assert "--acknowledge-gpu-use is required" in capsys.readouterr().err

    assert main([*common, "--acknowledge-gpu-use"]) == 2
    assert "--acknowledge-one-time-locked-holdout is required" in capsys.readouterr().err
    assert not (tmp_path / "not-created-holdout.json").exists()


def test_locked_holdout_cli_wires_success_path_without_loading_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract, train, exclusions, split = _gate_b_cli_fixture(tmp_path)
    development_sha256 = contract[
        contract.index("--expected-development-shard-sha256") + 1
    ]
    development_index = contract.index("--development-shard")
    del contract[development_index : development_index + 2]
    preflight = tmp_path / "preflight.json"
    smoke = tmp_path / "smoke.json"
    freeze = tmp_path / "freeze.json"
    primary_adapter = tmp_path / "primary-adapter"
    output = tmp_path / "holdout.json"
    backend_calls: list[dict[str, object]] = []
    close_order: list[str] = []
    captured: dict[str, object] = {}

    class FakeBackend:
        def __init__(self, role: str) -> None:
            self.role = role

        def close(self) -> None:
            close_order.append(self.role)

    def fake_backend_factory(**kwargs: object) -> FakeBackend:
        backend_calls.append(dict(kwargs))
        return FakeBackend(str(kwargs["role"]))

    def fake_evaluate(records_loader: object, **kwargs: object) -> GateBSelectionWriteResult:
        captured["records_loader"] = records_loader
        captured["kwargs"] = kwargs
        return GateBSelectionWriteResult(
            path=str(output),
            size_bytes=123,
            sha256="a" * 64,
            payload_sha256="b" * 64,
        )

    monkeypatch.setattr(cli_module, "_create_gate_b_backend", fake_backend_factory)
    monkeypatch.setattr(cli_module, "evaluate_locked_holdout_once", fake_evaluate)
    monkeypatch.setattr(
        cli_module,
        "load_train_csv",
        lambda *_args, **_kwargs: pytest.fail(
            "holdout train rows must remain behind the records_loader callback"
        ),
    )

    assert (
        main(
            [
                "gate-b-locked-holdout-evaluate",
                *contract,
                "--preflight-report",
                str(preflight),
                "--gpu-smoke-report",
                str(smoke),
                "--config",
                str(_locked_gate_b_config()),
                "--freeze-artifact",
                str(freeze),
                "--primary-kind",
                "adapter",
                "--primary-adapter",
                str(primary_adapter),
                "--fallback-kind",
                "base",
                "--output",
                str(output),
                "--acknowledge-gpu-use",
                "--acknowledge-one-time-locked-holdout",
            ]
        )
        == 0
    )

    assert callable(captured["records_loader"])
    keyword = captured["kwargs"]
    assert isinstance(keyword, dict)
    assert keyword["freeze_artifact"] == freeze
    assert keyword["output_path"] == output
    assert keyword["holdout_acknowledgement"] == HOLDOUT_ACCESS_ACKNOWLEDGEMENT
    assert keyword["config"] == DEFAULT_GATE_B_CONFIG
    assert keyword["fold"] == 0
    assert keyword["excluded_ids"] == ("train-000001",)
    assert keyword["excluded_ids_sha256"] == hashlib.sha256(
        canonical_json_bytes(["train-000001"])
    ).hexdigest()
    assert keyword["train_file_sha256"] == hashlib.sha256(train.read_bytes()).hexdigest()
    assert keyword["exclusions_file_sha256"] == hashlib.sha256(
        exclusions.read_bytes()
    ).hexdigest()
    assert keyword["split_artifact_sha256"] == hashlib.sha256(split.read_bytes()).hexdigest()
    assert keyword["development_shard_sha256"] == development_sha256
    assert keyword["primary_backend"].role == "primary"
    assert keyword["fallback_backend"].role == "fallback"
    assert len(backend_calls) == 2
    assert backend_calls[0]["kind"] == "adapter"
    assert backend_calls[0]["adapter_path"] == primary_adapter
    assert backend_calls[1]["kind"] == "base"
    assert backend_calls[1]["adapter_path"] is None
    for call in backend_calls:
        assert call["preflight_report"] == preflight
        assert call["gpu_smoke_report"] == smoke
        assert call["config"] == DEFAULT_GATE_B_CONFIG
        assert call["fold"] == 0
        assert call["excluded_ids"] == ("train-000001",)
        assert call["development_shard_sha256"] == development_sha256
        assert call["manifest"] is keyword["split_manifest"]
    assert close_order == ["fallback", "primary"]


def test_predict_evaluation_cli_wires_success_path_without_loading_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract, train, exclusions, split = _gate_b_cli_fixture(tmp_path)
    development_sha256 = contract[
        contract.index("--expected-development-shard-sha256") + 1
    ]
    development_index = contract.index("--development-shard")
    del contract[development_index : development_index + 2]
    evaluation = tmp_path / "filtered-leaderboard.csv"
    _write_csv(evaluation, [["id", "question"], ["val-000001", "What is 1+1?"]])
    evaluation_sha256 = hashlib.sha256(evaluation.read_bytes()).hexdigest()
    preflight = tmp_path / "preflight.json"
    smoke = tmp_path / "smoke.json"
    freeze = tmp_path / "freeze.json"
    primary_adapter = tmp_path / "primary-adapter"
    predictions = tmp_path / "predictions.json"
    artifact = tmp_path / "prediction-manifest.json"
    backend_calls: list[dict[str, object]] = []
    close_order: list[str] = []
    captured: dict[str, object] = {}

    class FakeBackend:
        def __init__(self, role: str) -> None:
            self.role = role

        def close(self) -> None:
            close_order.append(self.role)

    def fake_backend_factory(**kwargs: object) -> FakeBackend:
        backend_calls.append(dict(kwargs))
        return FakeBackend(str(kwargs["role"]))

    def fake_predict(**kwargs: object) -> PredictionArtifactWriteResult:
        captured.update(kwargs)
        return PredictionArtifactWriteResult(
            artifact_path=str(artifact),
            predictions_path=str(predictions),
            artifact_sha256="a" * 64,
            predictions_sha256="b" * 64,
            problem_count=1,
            invalid_count=0,
        )

    monkeypatch.setattr(cli_module, "_create_gate_b_backend", fake_backend_factory)
    monkeypatch.setattr(cli_module, "run_frozen_evaluation_inference", fake_predict)

    assert (
        main(
            [
                "gate-b-predict-evaluation",
                *contract,
                "--preflight-report",
                str(preflight),
                "--gpu-smoke-report",
                str(smoke),
                "--config",
                str(_locked_gate_b_config()),
                "--evaluation",
                str(evaluation),
                "--dataset-role",
                "leaderboard",
                "--expected-evaluation-sha256",
                evaluation_sha256,
                "--freeze-artifact",
                str(freeze),
                "--primary-kind",
                "adapter",
                "--primary-adapter",
                str(primary_adapter),
                "--fallback-kind",
                "base",
                "--output-artifact",
                str(artifact),
                "--output-predictions",
                str(predictions),
                "--acknowledge-gpu-use",
            ]
        )
        == 0
    )

    assert captured["dataset_role"] == "leaderboard"
    assert captured["evaluation_file_path"] == evaluation
    assert captured["expected_evaluation_sha256"] == evaluation_sha256
    assert captured["freeze_artifact"] == freeze
    assert captured["artifact_path"] == artifact
    assert captured["predictions_path"] == predictions
    assert captured["config"] == DEFAULT_GATE_B_CONFIG
    assert captured["fold"] == 0
    assert captured["excluded_ids"] == ("train-000001",)
    assert captured["excluded_ids_sha256"] == hashlib.sha256(
        canonical_json_bytes(["train-000001"])
    ).hexdigest()
    assert captured["train_file_sha256"] == hashlib.sha256(train.read_bytes()).hexdigest()
    assert captured["exclusions_file_sha256"] == hashlib.sha256(
        exclusions.read_bytes()
    ).hexdigest()
    assert captured["split_artifact_sha256"] == hashlib.sha256(split.read_bytes()).hexdigest()
    assert captured["development_shard_sha256"] == development_sha256
    assert captured["primary_backend"].role == "primary"
    assert captured["fallback_backend"].role == "fallback"
    assert len(backend_calls) == 2
    assert backend_calls[0]["kind"] == "adapter"
    assert backend_calls[0]["adapter_path"] == primary_adapter
    assert backend_calls[1]["kind"] == "base"
    assert backend_calls[1]["adapter_path"] is None
    for call in backend_calls:
        assert call["preflight_report"] == preflight
        assert call["gpu_smoke_report"] == smoke
        assert call["config"] == DEFAULT_GATE_B_CONFIG
        assert call["fold"] == 0
        assert call["excluded_ids"] == ("train-000001",)
        assert call["development_shard_sha256"] == development_sha256
        assert call["manifest"] is captured["split_manifest"]
    assert close_order == ["fallback", "primary"]


def test_gate_b_base_development_uses_backend_checkpoint_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract, _, _, _ = _gate_b_cli_fixture(tmp_path)
    source_args, preflight, smoke, runtime_gate = _gate_b_runtime_source_args(tmp_path)

    class Backend:
        checkpoint_sha256 = "d" * 64

        @property
        def runtime_gate_evidence(self) -> RuntimeGateEvidence:
            return runtime_gate

        def generate(self, _request: object) -> GenerationResult:
            return GenerationResult("Final answer: 1", "stop", 10, 3, 123)

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        cli_module,
        "create_base_development_backend",
        lambda **_kwargs: Backend(),
    )
    records = tmp_path / "base.jsonl"
    manifest = tmp_path / "base.manifest.json"
    assert (
        main(
            [
                "gate-b-development",
                *contract,
                "--preflight-report",
                str(preflight),
                "--gpu-smoke-report",
                str(smoke),
                "--config",
                str(_locked_gate_b_config()),
                *source_args,
                "--output-jsonl",
                str(records),
                "--output-manifest",
                str(manifest),
                "--acknowledge-gpu-use",
            ]
        )
        == 0
    )
    assert {
        json.loads(line)["checkpoint_sha256"]
        for line in records.read_text(encoding="utf-8").splitlines()
    } == {"d" * 64}
    run_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    assert run_manifest["schema_version"] == "gate-b1-development-run-v2"
    assert run_manifest["execution_evidence"]["runtime_gate"]["gpu_device_name"] == (
        "NVIDIA Test GPU"
    )


def test_gate_b_training_checks_base_before_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract, _, _, _ = _gate_b_cli_fixture(tmp_path)
    source_args, _, _, _ = _gate_b_runtime_source_args(tmp_path)
    ordered: list[str] = []

    def require_base(*_args: object, **_kwargs: object) -> None:
        ordered.append("base")

    def train(*_args: object, **_kwargs: object) -> TrainingArtifact:
        assert ordered == ["base"]
        ordered.append("train")
        return TrainingArtifact(
            path=str(tmp_path / "adapter"),
            artifact_sha256="a" * 64,
            manifest_sha256="b" * 64,
            checksums_sha256="c" * 64,
            file_count=5,
            training_count=10,
            validation_count=5,
        )

    monkeypatch.setattr(cli_module, "require_base_development_artifact", require_base)
    monkeypatch.setattr(cli_module, "train_qlora_fold", train)
    assert (
        main(
            [
                "gate-b-train-fold",
                *contract,
                "--preflight-report",
                str(tmp_path / "not-read-preflight.json"),
                "--gpu-smoke-report",
                str(tmp_path / "not-read-smoke.json"),
                "--config",
                str(_locked_gate_b_config()),
                *source_args,
                "--base-baseline-manifest",
                str(tmp_path / "base.manifest.json"),
                "--output-dir",
                str(tmp_path / "adapter"),
                "--acknowledge-gpu-use",
            ]
        )
        == 0
    )
    assert ordered == ["base", "train"]

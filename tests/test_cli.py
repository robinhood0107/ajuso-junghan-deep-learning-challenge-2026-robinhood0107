from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

import deep_challenge.cli as cli_module
from deep_challenge.cli import main
from deep_challenge.gate_b import DEFAULT_GATE_B_CONFIG, GenerationResult
from deep_challenge.gate_b_prediction import PredictionArtifactWriteResult
from deep_challenge.gate_b_runtime import TrainingArtifact
from deep_challenge.gate_b_selection import (
    HOLDOUT_ACCESS_ACKNOWLEDGEMENT,
    GateBSelectionWriteResult,
)
from deep_challenge.model_preflight import OFFICIAL_MODEL_ID, OFFICIAL_REVISION
from deep_challenge.provenance import canonical_json_bytes


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

    class Backend:
        checkpoint_sha256 = "d" * 64

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
                str(tmp_path / "not-read-preflight.json"),
                "--gpu-smoke-report",
                str(tmp_path / "not-read-smoke.json"),
                "--config",
                str(_locked_gate_b_config()),
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


def test_gate_b_training_checks_base_before_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract, _, _, _ = _gate_b_cli_fixture(tmp_path)
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

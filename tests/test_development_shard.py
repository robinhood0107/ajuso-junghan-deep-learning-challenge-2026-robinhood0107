from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from deep_challenge.data import load_train_csv
from deep_challenge.development_shard import (
    build_development_cv_shard,
    load_development_cv_shard,
)
from deep_challenge.gate_b import GateBArtifactExistsError, GateBValidationError
from deep_challenge.splits import make_grouped_split_manifest


def _fixture(tmp_path: Path):
    train_path = tmp_path / "train.csv"
    with train_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("id", "question", "answer"))
        for index in range(1, 13):
            writer.writerow((f"train-{index:06d}", f"Return {index}.", index))
    train = load_train_csv(train_path)
    ids = tuple(record.id for record in train)
    split = make_grouped_split_manifest(
        ids,
        dict(zip(ids, ids, strict=True)),
        n_folds=2,
        holdout_fraction=0.25,
        seed=7,
        version="development-shard-test-v1",
    )
    return train, split


def test_development_shard_is_atomic_exact_and_contains_no_holdout(
    tmp_path: Path,
) -> None:
    train, split = _fixture(tmp_path)
    output = tmp_path / "development-shard"
    result = build_development_cv_shard(
        train,
        split_manifest=split,
        split_artifact_sha256="4" * 64,
        output_dir=output,
    )

    assert result.row_count == 9
    assert set(record.id for record in result.dataset).isdisjoint(
        split.final_holdout_ids()
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["locked_holdout_rows_emitted"] is False
    assert manifest["source_train_sha256"] == train.manifest.sha256
    assert not any(
        problem_id.encode("utf-8") in (output / "development-train.csv").read_bytes()
        for problem_id in split.final_holdout_ids()
    )

    loaded = load_development_cv_shard(
        output,
        source_train_sha256=train.manifest.sha256,
        split_manifest=split,
        split_artifact_sha256="4" * 64,
        expected_bundle_sha256=result.bundle_sha256,
    )
    assert loaded.csv_sha256 == result.csv_sha256

    with pytest.raises(GateBArtifactExistsError, match="new directory"):
        build_development_cv_shard(
            train,
            split_manifest=split,
            split_artifact_sha256="4" * 64,
            output_dir=output,
        )


def test_development_shard_rejects_changed_bundle_before_loading_rows(
    tmp_path: Path,
) -> None:
    train, split = _fixture(tmp_path)
    output = tmp_path / "development-shard"
    result = build_development_cv_shard(
        train,
        split_manifest=split,
        split_artifact_sha256="4" * 64,
        output_dir=output,
    )
    checksums = output / "CHECKSUMS.sha256"
    checksums.write_text("0" * 64 + "  development-train.csv\n", encoding="utf-8")

    with pytest.raises(GateBValidationError, match="bundle SHA-256"):
        load_development_cv_shard(
            output,
            source_train_sha256=train.manifest.sha256,
            split_manifest=split,
            split_artifact_sha256="4" * 64,
            expected_bundle_sha256=result.bundle_sha256,
        )
    assert hashlib.sha256(checksums.read_bytes()).hexdigest() != result.bundle_sha256

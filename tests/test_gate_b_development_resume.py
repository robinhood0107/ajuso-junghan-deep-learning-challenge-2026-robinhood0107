from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

import deep_challenge.gate_b as gate_b_module
from deep_challenge.data import MathRecord
from deep_challenge.gate_b import (
    DEFAULT_GATE_B_CONFIG,
    DevelopmentExecutionEvidence,
    GateBArtifactExistsError,
    GateBValidationError,
    GenerationRequest,
    GenerationResult,
    create_development_execution_evidence,
    read_development_resume_status,
    run_development_baseline,
    write_development_artifacts,
)
from deep_challenge.provenance import (
    build_source_tree_manifest,
    validate_source_tree_manifest_artifact,
    write_json_atomic,
)
from deep_challenge.splits import (
    SplitManifest,
    eligible_validation_ids,
    make_grouped_split_manifest,
)


def _record(identifier: str) -> MathRecord:
    answer = int(identifier.removeprefix("train-"))
    question = f"What integer is associated with {identifier}?"
    return MathRecord(
        id=identifier,
        question_raw=question,
        question_normalized=question,
        answer_raw=str(answer),
        answer=answer,
        row_number=2,
    )


def _manifest() -> SplitManifest:
    ids = tuple(f"train-{index:06d}" for index in range(1, 25))
    return make_grouped_split_manifest(
        ids,
        dict(zip(ids, ids, strict=True)),
        n_folds=2,
        holdout_fraction=0.25,
        seed=20_260_731,
        version="splits-v4-development-resume-test",
    )


class _Backend:
    def __init__(self, *, fail_after: int | None = None) -> None:
        self.fail_after = fail_after
        self.requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> GenerationResult:
        if self.fail_after is not None and len(self.requests) >= self.fail_after:
            raise RuntimeError("synthetic interruption")
        self.requests.append(request)
        answer = int(request.problem_id.removeprefix("train-"))
        return GenerationResult(
            text=f"Reasoning.\nFinal answer: {answer}",
            finish_reason="stop",
            input_token_count=11,
            output_token_count=7,
            peak_vram_allocated_bytes=12_345,
        )


def _run(
    resume_dir: Path | None,
    *,
    backend: _Backend,
    checkpoint_sha256: str = "a" * 64,
    chunk_size: int = 2,
    samples_per_problem: int = 1,
):
    manifest = _manifest()
    validation_ids = eligible_validation_ids(manifest, 0, ())
    return run_development_baseline(
        [_record(problem_id) for problem_id in validation_ids],
        split_manifest=manifest,
        fold=0,
        excluded_ids=(),
        backend=backend,
        checkpoint_sha256=checkpoint_sha256,
        samples_per_problem=samples_per_problem,
        clock_ns=lambda: 0,
        resume_dir=resume_dir,
        chunk_size=chunk_size,
    )


def _execution_evidence(tmp_path: Path) -> DevelopmentExecutionEvidence:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "runtime.py").write_text("VALUE = 1\n", encoding="utf-8")
    source_manifest_path = tmp_path / "source-manifest.json"
    write_json_atomic(
        source_manifest_path,
        build_source_tree_manifest(
            source_root,
            excluded_paths=(source_manifest_path,),
        ).as_dict(),
    )
    source_manifest = validate_source_tree_manifest_artifact(
        source_manifest_path,
        root=source_root,
    )
    config_path = tmp_path / "config.json"
    preflight_path = tmp_path / "preflight.json"
    smoke_path = tmp_path / "smoke.json"
    config_path.write_text(
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
    preflight_path.write_text("{}\n", encoding="utf-8")
    smoke_path.write_text("{}\n", encoding="utf-8")
    return create_development_execution_evidence(
        source_manifest=source_manifest,
        config_path=config_path,
        config_sha256=DEFAULT_GATE_B_CONFIG.sha256,
        preflight_report_path=preflight_path,
        preflight_report_sha256=hashlib.sha256(preflight_path.read_bytes()).hexdigest(),
        gpu_smoke_report_path=smoke_path,
        gpu_smoke_report_sha256=hashlib.sha256(smoke_path.read_bytes()).hexdigest(),
        gpu_device_name="NVIDIA Synthetic GPU",
    )


def test_development_resume_recovers_only_verified_chunks_and_stays_raw_free(
    tmp_path: Path,
) -> None:
    resume_dir = tmp_path / "resume"
    first_backend = _Backend(fail_after=2)

    with pytest.raises(RuntimeError, match="synthetic interruption"):
        _run(resume_dir, backend=first_backend)

    interrupted = read_development_resume_status(resume_dir)
    assert interrupted.state == "interrupted"
    assert interrupted.completed_chunks == 1
    assert interrupted.completed_generations == 2
    progress_text = (resume_dir / "progress.json").read_text(encoding="utf-8")
    assert "train-" not in progress_text
    assert "What integer" not in progress_text
    assert "Final answer" not in progress_text

    resumed_backend = _Backend()
    records = _run(resume_dir, backend=resumed_backend)
    validation_ids = eligible_validation_ids(_manifest(), 0, ())
    assert [record.problem_id for record in records] == list(validation_ids)
    assert len(resumed_backend.requests) == len(validation_ids) - 2
    assert read_development_resume_status(resume_dir).state == "complete"


def test_development_resume_reruns_a_corrupt_chunk_as_a_new_attempt(tmp_path: Path) -> None:
    resume_dir = tmp_path / "resume"
    first_records = _run(resume_dir, backend=_Backend())
    chunks_dir = resume_dir / "chunks"
    first_chunk = chunks_dir / "chunk-000000-attempt-000001.jsonl"
    original = first_chunk.read_bytes()
    first_chunk.write_bytes(b"corrupted\n")

    backend = _Backend()
    recovered = _run(resume_dir, backend=backend)

    assert recovered == first_records
    assert len(backend.requests) == 2
    assert first_chunk.read_bytes() == b"corrupted\n"
    assert (chunks_dir / "chunk-000000-attempt-000002.jsonl").is_file()
    assert read_development_resume_status(resume_dir).invalid_chunk_attempt_count >= 1
    assert original != first_chunk.read_bytes()


def test_development_resume_chunks_never_split_samples_of_one_problem(tmp_path: Path) -> None:
    resume_dir = tmp_path / "resume"
    first_backend = _Backend(fail_after=4)

    with pytest.raises(RuntimeError, match="synthetic interruption"):
        _run(
            resume_dir,
            backend=first_backend,
            samples_per_problem=2,
        )

    interrupted = read_development_resume_status(resume_dir)
    assert interrupted.completed_chunks == 1
    assert interrupted.completed_generations == 4

    backend = _Backend()
    records = _run(resume_dir, backend=backend, samples_per_problem=2)
    assert records[0].problem_id == records[1].problem_id
    assert records[2].problem_id == records[3].problem_id
    assert records[0].problem_id != records[2].problem_id
    assert [record.sample_index for record in records[:4]] == [0, 1, 0, 1]
    assert len(backend.requests) == len(records) - 4


def test_development_resume_rejects_tampered_or_incompatible_contract(tmp_path: Path) -> None:
    resume_dir = tmp_path / "resume"
    _run(resume_dir, backend=_Backend())

    with pytest.raises(GateBValidationError, match="incompatible"):
        _run(resume_dir, backend=_Backend(), checkpoint_sha256="b" * 64)

    contract_path = resume_dir / "contract.json"
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    payload["chunk_size"] = 3
    contract_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(GateBValidationError, match="digest does not match"):
        _run(resume_dir, backend=_Backend())


def test_resumed_writer_requires_exact_canonical_coverage_and_keeps_v2_output(
    tmp_path: Path,
) -> None:
    resume_dir = tmp_path / "resume"
    records = _run(resume_dir, backend=_Backend())
    evidence = _execution_evidence(tmp_path)
    jsonl_path = tmp_path / "development.jsonl"
    manifest_path = tmp_path / "development.manifest.json"

    result = write_development_artifacts(
        records,
        jsonl_path=jsonl_path,
        manifest_path=manifest_path,
        execution_evidence=evidence,
        resume_dir=resume_dir,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert result.record_count == len(records)
    assert manifest["schema_version"] == "gate-b1-development-run-v2"
    assert all(
        json.loads(line)["schema_version"] == "gate-b1-development-baseline-v2"
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
    )

    reordered = (records[1], records[0], *records[2:])
    with pytest.raises(GateBValidationError, match="validation ID order/coverage"):
        write_development_artifacts(
            reordered,
            jsonl_path=tmp_path / "bad.jsonl",
            manifest_path=tmp_path / "bad.manifest.json",
            execution_evidence=evidence,
            resume_dir=resume_dir,
        )


def test_resumed_writer_completes_exact_records_orphan_after_dead_owner_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = _run(None, backend=_Backend())
    evidence = _execution_evidence(tmp_path)
    jsonl_path = tmp_path / "development.jsonl"
    manifest_path = tmp_path / "development.manifest.json"
    records_bytes = "".join(f"{record.to_json_line()}\n" for record in records).encode(
        "utf-8"
    )
    jsonl_path.write_bytes(records_bytes)
    name_digest = hashlib.sha256(
        f"{jsonl_path.name}\0{manifest_path.name}".encode()
    ).hexdigest()[:16]
    lock_path = tmp_path / f".gate-b-{name_digest}.lock"
    lock_path.write_text(
        json.dumps(
            {
                "schema_version": "gate-b1-development-artifact-pair-lock-v1",
                "records_file": jsonl_path.name,
                "manifest_file": manifest_path.name,
                "process_id": os.getpid() + 10_000,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(gate_b_module, "_process_is_alive", lambda _pid: False)

    written = write_development_artifacts(
        records,
        jsonl_path=jsonl_path,
        manifest_path=manifest_path,
        execution_evidence=evidence,
    )

    assert jsonl_path.read_bytes() == records_bytes
    assert written.records_sha256 == hashlib.sha256(records_bytes).hexdigest()
    assert not lock_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "gate-b1-development-run-v2"
    assert manifest["records_sha256"] == written.records_sha256
    with pytest.raises(GateBArtifactExistsError, match="overwrite"):
        write_development_artifacts(
            records,
            jsonl_path=jsonl_path,
            manifest_path=manifest_path,
            execution_evidence=evidence,
        )


def test_resumed_writer_refuses_different_records_orphan(tmp_path: Path) -> None:
    records = _run(None, backend=_Backend())
    evidence = _execution_evidence(tmp_path)
    jsonl_path = tmp_path / "development.jsonl"
    manifest_path = tmp_path / "development.manifest.json"
    jsonl_path.write_bytes(b"different private generation bytes\n")

    with pytest.raises(GateBArtifactExistsError, match="different Gate B records"):
        write_development_artifacts(
            records,
            jsonl_path=jsonl_path,
            manifest_path=manifest_path,
            execution_evidence=evidence,
        )

    assert jsonl_path.read_bytes() == b"different private generation bytes\n"
    assert not manifest_path.exists()


def test_existing_nonresumable_generation_api_remains_available() -> None:
    backend = _Backend()
    records = _run(None, backend=backend)

    assert records
    assert len(backend.requests) == len(records)

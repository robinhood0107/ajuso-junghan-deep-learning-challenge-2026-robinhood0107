from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

import deep_challenge.gate_b_runtime as runtime_module
from deep_challenge.data import MathRecord
from deep_challenge.gate_b import (
    DEFAULT_GATE_B_CONFIG,
    PINNED_MODEL_REVISION,
    ChatMessage,
    GateBArtifactExistsError,
    GateBConfig,
    GateBPreflightRequiredError,
    GateBValidationError,
    GenerationRequest,
)
from deep_challenge.gate_b_runtime import (
    GPU_EXECUTION_ACKNOWLEDGEMENT,
    PINNED_QWEN_ALL_LINEAR_TARGET_MODULES,
    RuntimeTrainingResult,
    build_fold_sft_plan,
    create_adapted_development_backend,
    create_base_development_backend,
    train_qlora_fold,
    validate_adapter_artifact,
    validate_runtime_gate,
)
from deep_challenge.gpu_smoke import (
    GpuSmokeRequest,
    GpuSmokeRuntimeEvidence,
    run_final_gpu_smoke,
)
from deep_challenge.model_preflight import OFFICIAL_MODEL_ID
from deep_challenge.provenance import (
    build_source_tree_manifest,
    canonical_json_bytes,
    validate_source_tree_manifest_artifact,
    write_json_atomic,
)
from deep_challenge.rationale_corpus import (
    DEFAULT_CONCISE_RATIONALE_CONFIG,
    audit_rationale_corpus,
    build_rationale_corpus,
    load_concise_rationale_config,
    load_verified_rationale_corpus,
)
from deep_challenge.splits import (
    SplitManifest,
    eligible_training_ids,
    eligible_validation_ids,
    make_grouped_split_manifest,
)

_MIB = 1024 * 1024
_FAKE_TOKENIZER_CONFIG_BYTES = (
    json.dumps(
        {
            "chat_template": "test-chat-template",
            "eos_token": "<|im_end|>",
            "pad_token": "<|endoftext|>",
            "model_max_length": 131_072,
        }
    )
    + "\n"
).encode("utf-8")


@pytest.fixture(autouse=True)
def _tiny_adapter_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime_module, "_PINNED_QWEN_LAYER_COUNT", 1)
    monkeypatch.setattr(
        runtime_module,
        "_PINNED_QWEN_PROJECTION_DIMS",
        {
            "q_proj": (32, 32),
            "k_proj": (16, 32),
            "v_proj": (16, 32),
            "o_proj": (32, 32),
            "gate_proj": (64, 32),
            "up_proj": (64, 32),
            "down_proj": (32, 64),
        },
    )
    monkeypatch.setattr(
        runtime_module,
        "_PINNED_TOKENIZER_JSON_SHA256",
        hashlib.sha256(b"{}\n").hexdigest(),
    )
    monkeypatch.setattr(
        runtime_module,
        "_PINNED_TOKENIZER_CHAT_TEMPLATE_SHA256",
        hashlib.sha256(b"test-chat-template").hexdigest(),
    )
    monkeypatch.setattr(
        runtime_module,
        "_PINNED_TOKENIZER_CONFIG_JSON_SHA256",
        hashlib.sha256(_FAKE_TOKENIZER_CONFIG_BYTES).hexdigest(),
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


def _split_manifest() -> SplitManifest:
    identifiers = tuple(f"train-{index:06d}" for index in range(1, 13))
    return make_grouped_split_manifest(
        identifiers,
        dict(zip(identifiers, identifiers, strict=True)),
        n_folds=2,
        holdout_fraction=0.25,
        seed=20_260_731,
        version="splits-v4-runtime-test",
    )


def _records(identifiers: tuple[str, ...]) -> tuple[MathRecord, ...]:
    return tuple(_record(identifier) for identifier in identifiers)


def _rationale_corpus_evidence(
    tmp_path: Path,
    manifest: SplitManifest,
    records: tuple[MathRecord, ...],
    *,
    fold: int,
    excluded_ids: tuple[str, ...],
):
    candidate_config_path = Path(
        "configs/gate_b/rtx4070-super-12gb-concise-rationale-v1.json"
    )
    candidate_config, config_file_sha256 = load_concise_rationale_config(
        candidate_config_path
    )
    source = tmp_path / "teacher-rationales.jsonl"
    rows = []
    for record in records:
        target = (
            "The identifier directly determines the requested integer.\n"
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
                    "prompt_sha256": "5" * 64,
                    "generation_config_sha256": "6" * 64,
                    "seed": 7,
                    "sample_index": 0,
                    "raw_generation_sha256": "7" * 64,
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
    source.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    corpus_root = tmp_path / "rationale-corpus"
    corpus_root.mkdir()
    corpus = corpus_root / "rationales.jsonl"
    corpus_manifest = corpus_root / "manifest.json"
    build_rationale_corpus(
        source,
        records,
        split_manifest=manifest,
        fold=fold,
        excluded_ids=excluded_ids,
        candidate_config_file_sha256=config_file_sha256,
        output_jsonl=corpus,
        output_manifest=corpus_manifest,
        config=candidate_config,
    )
    audit = tmp_path / "rationale-audit.json"
    audit_rationale_corpus(
        corpus,
        corpus_manifest,
        records,
        split_manifest=manifest,
        fold=fold,
        excluded_ids=excluded_ids,
        candidate_config_file_sha256=config_file_sha256,
        output_path=audit,
        config=candidate_config,
    )
    evidence = load_verified_rationale_corpus(
        corpus,
        corpus_manifest,
        records,
        split_manifest=manifest,
        fold=fold,
        excluded_ids=excluded_ids,
        candidate_config_file_sha256=config_file_sha256,
        audit_path=audit,
        config=candidate_config,
    )
    return evidence, candidate_config


def _data_provenance_args() -> dict[str, str]:
    return {
        "train_file_sha256": "1" * 64,
        "exclusions_file_sha256": "2" * 64,
        "split_artifact_sha256": "3" * 64,
        "development_shard_sha256": "4" * 64,
    }


def _source_manifest_evidence(tmp_path: Path):
    source_root = tmp_path / "source"
    source_root.mkdir(exist_ok=True)
    (source_root / "runtime.py").write_text("VALUE = 1\n", encoding="utf-8")
    output = tmp_path / "source-manifest.json"
    write_json_atomic(
        output,
        build_source_tree_manifest(source_root, excluded_paths=(output,)).as_dict(),
    )
    return validate_source_tree_manifest_artifact(output, root=source_root)


def _ready_preflight() -> dict[str, object]:
    packages = {
        "torch": "2.7.0",
        "transformers": "4.51.0",
        "accelerate": "1.7.0",
        "peft": "0.15.0",
        "bitsandbytes": "0.45.5",
        "triton": "3.3.0",
    }
    return {
        "model_id": OFFICIAL_MODEL_ID,
        "requested_revision": PINNED_MODEL_REVISION,
        "resolved_commit": PINNED_MODEL_REVISION,
        "packages": packages,
        "physical_nvidia": {
            "probe_succeeded": True,
            "available": True,
            "device_count": 1,
            "devices": [
                {
                    "index": 0,
                    "name": "NVIDIA Test GPU",
                    "memory_total_mib": 12_288,
                    "memory_used_mib": 512,
                    "memory_free_mib": 11_776,
                    "compute_capability": "8.9",
                    "driver_version": "999.0",
                }
            ],
        },
        "torch_cuda_runtime": {
            "available": True,
            "bf16_supported": True,
            "device_count": 1,
            "devices": [],
        },
        "snapshot_consistent": True,
        "tokenizer_ready": True,
        "weights_ready": True,
        "model_runtime_ready": True,
        "training_profile": "nf4_qlora_bf16",
        "nf4_vram": {
            "minimum_free_mib": 10_240,
            "maximum_observed_free_mib": 11_776,
            "ready": True,
        },
        "host_runtime_ready": True,
        "runtime_blockers": [],
        "qlora_dependencies_ready": True,
        "training_ready_scope": "pre_gpu_smoke_prerequisites",
        "training_ready": True,
        "blockers": [],
    }


class _SmokeRuntime:
    def __init__(self) -> None:
        self.requests: list[GpuSmokeRequest] = []

    def execute(self, request: GpuSmokeRequest) -> GpuSmokeRuntimeEvidence:
        self.requests.append(request)
        return GpuSmokeRuntimeEvidence(
            package_versions={
                "torch": "2.7.0",
                "transformers": "4.51.0",
                "accelerate": "1.7.0",
                "peft": "0.15.0",
                "bitsandbytes": "0.45.5",
                "triton": "3.3.0",
            },
            cuda_device_count=1,
            device_index=0,
            device_name="NVIDIA Test GPU",
            pre_context_physical_device_name="NVIDIA Test GPU",
            pre_context_physical_total_bytes=12_288 * _MIB,
            pre_context_physical_used_bytes=512 * _MIB,
            pre_context_physical_free_bytes=11_776 * _MIB,
            physical_total_bytes=12_288 * _MIB,
            physical_free_bytes_before=11_776 * _MIB,
            physical_free_bytes_after_cleanup=11_700 * _MIB,
            allocated_bytes_before=0,
            peak_allocated_bytes=4_000 * _MIB,
            peak_reserved_bytes=4_500 * _MIB,
            training_latency_ms=10.0,
            generation_latency_ms=5.0,
            total_latency_ms=25.0,
            training_loss=0.75,
            optimizer_name="paged_adamw_8bit",
            optimizer_step_count=1,
            raw_generation="Final answer: 5",
        )


def _gate_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    preflight = tmp_path / "preflight.json"
    preflight.write_text(json.dumps(_ready_preflight()), encoding="utf-8")
    smoke = tmp_path / "gpu-smoke.json"
    run_final_gpu_smoke(
        smoke,
        preflight_report_path=preflight,
        acknowledge_gpu_execution=True,
        runtime=_SmokeRuntime(),
    )
    return preflight, smoke


class _Tokenizer:
    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        assert tokenize is True
        assert [message["role"] for message in conversation[:2]] == ["system", "user"]
        return [11, 12] if add_generation_prompt else [11, 12, 99]


class _TrainingRuntime:
    def __init__(self, *, incomplete: str | None = None) -> None:
        self._tokenizer = _Tokenizer()
        self.incomplete = incomplete
        self.closed = False
        self.training_ids: tuple[str, ...] = ()
        self.validation_ids: tuple[str, ...] = ()
        self.config: GateBConfig | None = None

    @property
    def tokenizer(self) -> _Tokenizer:
        return self._tokenizer

    def train(
        self,
        *,
        training_examples: tuple[object, ...],
        validation_examples: tuple[object, ...],
        work_dir: Path,
        export_dir: Path,
        plan: object,
        config: GateBConfig,
    ) -> RuntimeTrainingResult:
        del work_dir
        self.training_ids = tuple(example.problem_id for example in training_examples)
        self.validation_ids = tuple(example.problem_id for example in validation_examples)
        self.config = config
        assert self.training_ids == plan.training_ids
        assert self.validation_ids == plan.validation_ids
        assert all(example.labels[:2] == (-100, -100) for example in training_examples)
        assert all(example.sequence_token_count <= 2_048 for example in training_examples)
        (export_dir / "adapter_config.json").write_text(
            json.dumps(
                {
                    "base_model_name_or_path": OFFICIAL_MODEL_ID,
                    "peft_type": "LORA",
                    "task_type": "CAUSAL_LM",
                    "r": 16,
                    "lora_alpha": 32,
                    "lora_dropout": 0.05,
                    "bias": "none",
                    "target_modules": list(reversed(PINNED_QWEN_ALL_LINEAR_TARGET_MODULES)),
                    "modules_to_save": None,
                    "use_dora": False,
                    "use_rslora": False,
                    "trainable_token_indices": None,
                    "target_parameters": None,
                    "rank_pattern": {},
                    "alpha_pattern": {},
                    "inference_mode": True,
                    "fan_in_fan_out": False,
                }
            ),
            encoding="utf-8",
        )
        (export_dir / "tokenizer_config.json").write_bytes(
            _FAKE_TOKENIZER_CONFIG_BYTES
        )
        if self.incomplete != "tokenizer":
            (export_dir / "tokenizer.json").write_text("{}\n", encoding="utf-8")
        if self.incomplete == "shard":
            (export_dir / "adapter_model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"x": "adapter_model-00001-of-00002.safetensors"}}),
                encoding="utf-8",
            )
        else:
            save_file(
                {
                    name: np.zeros(shape, dtype=np.float32)
                    for name, shape in runtime_module._expected_lora_tensor_shapes().items()
                },
                export_dir / "adapter_model.safetensors",
            )
        return RuntimeTrainingResult(
            global_step=7,
            metrics={"train_loss": 0.25},
            package_versions={
                "torch": "2.7.0",
                "transformers": "4.51.0",
                "peft": "0.15.0",
                "accelerate": "1.7.0",
                "bitsandbytes": "0.45.5",
            },
        )

    def close(self) -> None:
        self.closed = True


def _write_complete_trainer_checkpoint(
    root: Path,
    *,
    global_step: int = 7,
    contract_sha256: str,
) -> None:
    root.mkdir(parents=True)
    (root / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": OFFICIAL_MODEL_ID,
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "bias": "none",
                "target_modules": list(PINNED_QWEN_ALL_LINEAR_TARGET_MODULES),
                "modules_to_save": None,
                "use_dora": False,
                "use_rslora": False,
                "trainable_token_indices": None,
                "target_parameters": None,
                "rank_pattern": {},
                "alpha_pattern": {},
                "inference_mode": True,
                "fan_in_fan_out": False,
            }
        ),
        encoding="utf-8",
    )
    save_file(
        {
            name: np.zeros(shape, dtype=np.float32)
            for name, shape in runtime_module._expected_lora_tensor_shapes().items()
        },
        root / "adapter_model.safetensors",
    )
    (root / "trainer_state.json").write_text(
        json.dumps({"global_step": global_step}), encoding="utf-8"
    )
    (root / "resume-checkpoint.json").write_text(
        json.dumps(
            {
                "schema_version": "gate-b-qlora-training-resume-v1",
                "contract_sha256": contract_sha256,
                "global_step": global_step,
            }
        ),
        encoding="utf-8",
    )
    for filename in ("optimizer.pt", "rng_state.pth", "scheduler.pt", "training_args.bin"):
        (root / filename).write_bytes(b"checkpoint-state\n")


class _ResumableTrainingRuntime(_TrainingRuntime):
    def __init__(
        self,
        *,
        fail_after_checkpoint: bool = False,
        fail_without_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.fail_after_checkpoint = fail_after_checkpoint
        self.fail_without_checkpoint = fail_without_checkpoint
        self.resume_checkpoint: Path | None = None
        self.retain_checkpoints: bool | None = None
        self.work_dir: Path | None = None

    def train(
        self,
        *,
        training_examples: tuple[object, ...],
        validation_examples: tuple[object, ...],
        work_dir: Path,
        export_dir: Path,
        plan: object,
        config: GateBConfig,
        resume_checkpoint: Path | None = None,
        retain_checkpoints: bool = False,
    ) -> RuntimeTrainingResult:
        self.resume_checkpoint = resume_checkpoint
        self.retain_checkpoints = retain_checkpoints
        self.work_dir = work_dir
        if self.fail_after_checkpoint:
            _write_complete_trainer_checkpoint(
                work_dir / "trainer" / "checkpoint-7",
                contract_sha256=runtime_module._training_resume_contract_from_root(
                    work_dir
                ),
            )
            raise RuntimeError("simulated interruption after checkpoint")
        if self.fail_without_checkpoint:
            raise RuntimeError("simulated interruption before checkpoint")
        return super().train(
            training_examples=training_examples,
            validation_examples=validation_examples,
            work_dir=work_dir,
            export_dir=export_dir,
            plan=plan,
            config=config,
        )


def test_module_import_is_cpu_safe() -> None:
    script = """
import sys
import deep_challenge.gate_b_runtime
for name in ('torch', 'transformers', 'peft', 'bitsandbytes'):
    assert name not in sys.modules, name
"""
    environment = dict(os.environ)
    environment["PYTHONNOUSERSITE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr


def test_fold_plan_derives_exact_train_and_validation_without_holdout() -> None:
    manifest = _split_manifest()
    excluded = (manifest.training_ids(0)[0],)
    expected_training = eligible_training_ids(manifest, 0, excluded)
    expected_validation = eligible_validation_ids(manifest, 0, excluded)

    plan = build_fold_sft_plan(
        _records(expected_training),
        _records(expected_validation),
        split_manifest=manifest,
        fold=0,
        excluded_ids=excluded,
    )

    assert plan.training_ids == expected_training
    assert plan.validation_ids == expected_validation
    assert set(plan.training_ids).isdisjoint(manifest.final_holdout_ids())
    assert set(plan.validation_ids).isdisjoint(manifest.final_holdout_ids())
    assert set(plan.training_ids).isdisjoint(plan.validation_ids)
    assert all(example.partition == "fold_training" for example in plan.training_examples)
    assert all(example.partition == "fold_validation" for example in plan.validation_examples)
    assert plan.config_sha256 == DEFAULT_GATE_B_CONFIG.sha256


def test_fold_plan_rejects_full_train_holdout_and_leaderboard_like_records() -> None:
    manifest = _split_manifest()
    training_ids = eligible_training_ids(manifest, 0, ())
    validation_ids = eligible_validation_ids(manifest, 0, ())
    full_records = _records(tuple(assignment.record_id for assignment in manifest.assignments))

    with pytest.raises(GateBValidationError, match="exactly match"):
        build_fold_sft_plan(
            full_records,
            _records(validation_ids),
            split_manifest=manifest,
            fold=0,
            excluded_ids=(),
        )

    invalid = replace(_records(training_ids)[0], id="leaderboard-000001")
    with pytest.raises(GateBValidationError, match="train-XXXXXX"):
        build_fold_sft_plan(
            (invalid, *_records(training_ids)[1:]),
            _records(validation_ids),
            split_manifest=manifest,
            fold=0,
            excluded_ids=(),
        )

    with pytest.raises(GateBValidationError, match="train-XXXXXX"):
        build_fold_sft_plan(
            _records(training_ids),
            _records(validation_ids),
            split_manifest=manifest,
            fold=0,
            excluded_ids=("test-000001",),
        )


def test_runtime_rejects_any_config_drift() -> None:
    manifest = _split_manifest()
    training_ids = eligible_training_ids(manifest, 0, ())
    validation_ids = eligible_validation_ids(manifest, 0, ())
    with pytest.raises(GateBValidationError, match="DEFAULT_GATE_B_CONFIG"):
        build_fold_sft_plan(
            _records(training_ids),
            _records(validation_ids),
            split_manifest=manifest,
            fold=0,
            excluded_ids=(),
            config=GateBConfig(seed=1),
        )

    with pytest.raises(GateBValidationError, match="global_step must be positive"):
        RuntimeTrainingResult(
            global_step=0,
            metrics={},
            package_versions={"torch": "2.7.0"},
        )


def test_runtime_gate_validates_canonical_smoke_and_exact_preflight_sha(
    tmp_path: Path,
) -> None:
    preflight, smoke = _gate_artifacts(tmp_path)
    evidence = validate_runtime_gate(
        preflight_artifact=preflight,
        gpu_smoke_artifact=smoke,
    )
    assert evidence.model_id == OFFICIAL_MODEL_ID
    assert evidence.revision == PINNED_MODEL_REVISION
    assert evidence.preflight_sha256 == hashlib.sha256(preflight.read_bytes()).hexdigest()
    assert evidence.smoke_sha256 == hashlib.sha256(smoke.read_bytes()).hexdigest()

    preflight.write_text(json.dumps(_ready_preflight(), indent=2), encoding="utf-8")
    with pytest.raises(GateBPreflightRequiredError, match="different preflight"):
        validate_runtime_gate(
            preflight_artifact=preflight,
            gpu_smoke_artifact=smoke,
        )


def test_runtime_gate_recomputes_canonical_smoke_payload_hash(tmp_path: Path) -> None:
    preflight, smoke = _gate_artifacts(tmp_path)
    payload = json.loads(smoke.read_text(encoding="utf-8"))
    payload["runtime"]["raw_generation"] = "Final answer: 6"
    smoke.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(GateBPreflightRequiredError, match="payload_sha256"):
        validate_runtime_gate(
            preflight_artifact=preflight,
            gpu_smoke_artifact=smoke,
        )


def test_runtime_gate_rejects_forged_competition_input_even_with_rehashed_payload(
    tmp_path: Path,
) -> None:
    preflight, smoke = _gate_artifacts(tmp_path)
    payload = json.loads(smoke.read_text(encoding="utf-8"))
    payload["input_provenance"]["competition_data_used"] = True
    payload_without_hash = dict(payload)
    payload_without_hash.pop("payload_sha256")
    payload["payload_sha256"] = hashlib.sha256(
        canonical_json_bytes(payload_without_hash)
    ).hexdigest()
    smoke.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(GateBPreflightRequiredError, match="organizer or caller"):
        validate_runtime_gate(
            preflight_artifact=preflight,
            gpu_smoke_artifact=smoke,
        )


def test_qlora_fold_training_publishes_complete_atomic_no_overwrite_bundle(
    tmp_path: Path,
) -> None:
    preflight, smoke = _gate_artifacts(tmp_path)
    manifest = _split_manifest()
    excluded = (manifest.training_ids(0)[0],)
    training_ids = eligible_training_ids(manifest, 0, excluded)
    validation_ids = eligible_validation_ids(manifest, 0, excluded)
    fake = _TrainingRuntime()
    output = tmp_path / "adapter-fold-0"

    artifact = train_qlora_fold(
        _records(training_ids),
        _records(validation_ids),
        split_manifest=manifest,
        fold=0,
        excluded_ids=excluded,
        **_data_provenance_args(),
        source_manifest=_source_manifest_evidence(tmp_path),
        preflight_artifact=preflight,
        gpu_smoke_artifact=smoke,
        output_dir=output,
        gpu_acknowledgement=GPU_EXECUTION_ACKNOWLEDGEMENT,
        runtime_factory=lambda _evidence: fake,
    )

    assert fake.closed is True
    assert fake.training_ids == training_ids
    assert fake.validation_ids == validation_ids
    assert fake.config == DEFAULT_GATE_B_CONFIG
    assert artifact.path == str(output.resolve())
    validated = validate_adapter_artifact(output)
    assert validated.artifact_sha256 == artifact.artifact_sha256
    assert validated.train_file_sha256 == "1" * 64
    assert validated.exclusions_file_sha256 == "2" * 64
    assert validated.split_artifact_sha256 == "3" * 64
    assert validated.development_shard_sha256 == "4" * 64
    assert validated.training_examples_sha256
    assert validated.validation_examples_sha256
    assert validated.preflight_sha256 == hashlib.sha256(preflight.read_bytes()).hexdigest()
    assert validated.gpu_smoke_sha256 == hashlib.sha256(smoke.read_bytes()).hexdigest()
    assert validated.source_manifest_sha256
    assert validated.source_tree_sha256
    assert validated.source_file_count == 1
    saved_adapter_config = json.loads(
        (output / "adapter_config.json").read_text(encoding="utf-8")
    )
    assert saved_adapter_config["target_modules"] == list(
        PINNED_QWEN_ALL_LINEAR_TARGET_MODULES
    )
    assert (output / "manifest.json").is_file()
    assert (output / "CHECKSUMS.sha256").is_file()
    assert not tuple(tmp_path.glob(".adapter-fold-0.training-*"))

    adapter_link = tmp_path / "adapter-link"
    adapter_link.symlink_to(output, target_is_directory=True)
    with pytest.raises(GateBValidationError, match="symbolic link"):
        validate_adapter_artifact(adapter_link)

    factory_called = False

    def unexpected_factory(_evidence: object) -> _TrainingRuntime:
        nonlocal factory_called
        factory_called = True
        return _TrainingRuntime()

    with pytest.raises(GateBArtifactExistsError, match="overwrite"):
        train_qlora_fold(
            _records(training_ids),
            _records(validation_ids),
            split_manifest=manifest,
            fold=0,
            excluded_ids=excluded,
            **_data_provenance_args(),
            source_manifest=_source_manifest_evidence(tmp_path),
            preflight_artifact=preflight,
            gpu_smoke_artifact=smoke,
            output_dir=output,
            gpu_acknowledgement=GPU_EXECUTION_ACKNOWLEDGEMENT,
            runtime_factory=unexpected_factory,
        )
    assert factory_called is False


def test_qlora_training_resume_keeps_checkpoint_and_reuses_exact_contract(
    tmp_path: Path,
) -> None:
    preflight, smoke = _gate_artifacts(tmp_path)
    manifest = _split_manifest()
    training_ids = eligible_training_ids(manifest, 0, ())
    validation_ids = eligible_validation_ids(manifest, 0, ())
    source_manifest = _source_manifest_evidence(tmp_path)
    output = tmp_path / "adapter-fold-0"
    resume_dir = tmp_path / "resume-fold-0"
    interrupted = _ResumableTrainingRuntime(fail_after_checkpoint=True)

    with pytest.raises(RuntimeError, match="simulated interruption"):
        train_qlora_fold(
            _records(training_ids),
            _records(validation_ids),
            split_manifest=manifest,
            fold=0,
            excluded_ids=(),
            **_data_provenance_args(),
            source_manifest=source_manifest,
            preflight_artifact=preflight,
            gpu_smoke_artifact=smoke,
            output_dir=output,
            gpu_acknowledgement=GPU_EXECUTION_ACKNOWLEDGEMENT,
            resume_dir=resume_dir,
            runtime_factory=lambda _evidence: interrupted,
        )

    checkpoint = resume_dir / "trainer" / "checkpoint-7"
    contract = json.loads((resume_dir / "resume-contract.json").read_text(encoding="utf-8"))
    assert interrupted.closed is True
    assert checkpoint.is_dir()
    assert contract["runtime_gate"]["preflight_sha256"] == hashlib.sha256(
        preflight.read_bytes()
    ).hexdigest()
    assert contract["runtime_gate"]["gpu_smoke_sha256"] == hashlib.sha256(
        smoke.read_bytes()
    ).hexdigest()
    assert contract["source_manifest"]["sha256"] == source_manifest.sha256
    assert contract["tokenizer_evidence"]["tokenizer_json_sha256"]
    assert contract["checkpoint_retention"] == "retain_all"

    resumed = _ResumableTrainingRuntime()
    artifact = train_qlora_fold(
        _records(training_ids),
        _records(validation_ids),
        split_manifest=manifest,
        fold=0,
        excluded_ids=(),
        **_data_provenance_args(),
        source_manifest=source_manifest,
        preflight_artifact=preflight,
        gpu_smoke_artifact=smoke,
        output_dir=output,
        gpu_acknowledgement=GPU_EXECUTION_ACKNOWLEDGEMENT,
        resume_dir=resume_dir,
        runtime_factory=lambda _evidence: resumed,
    )

    assert artifact.path == str(output.resolve())
    assert resumed.work_dir == resume_dir.resolve()
    assert resumed.resume_checkpoint == checkpoint.resolve()
    assert resumed.retain_checkpoints is True
    assert checkpoint.is_dir()
    assert not tuple(tmp_path.glob(".adapter-fold-0.training-*"))


def test_qlora_training_resume_rejects_contract_mismatch_and_quarantines_corruption(
    tmp_path: Path,
) -> None:
    preflight, smoke = _gate_artifacts(tmp_path)
    manifest = _split_manifest()
    training_ids = eligible_training_ids(manifest, 0, ())
    validation_ids = eligible_validation_ids(manifest, 0, ())
    source_manifest = _source_manifest_evidence(tmp_path)
    resume_dir = tmp_path / "resume-fold-0"

    with pytest.raises(RuntimeError, match="simulated interruption"):
        train_qlora_fold(
            _records(training_ids),
            _records(validation_ids),
            split_manifest=manifest,
            fold=0,
            excluded_ids=(),
            **_data_provenance_args(),
            source_manifest=source_manifest,
            preflight_artifact=preflight,
            gpu_smoke_artifact=smoke,
            output_dir=tmp_path / "first-output",
            gpu_acknowledgement=GPU_EXECUTION_ACKNOWLEDGEMENT,
            resume_dir=resume_dir,
            runtime_factory=lambda _evidence: _ResumableTrainingRuntime(
                fail_after_checkpoint=True
            ),
        )

    called = False

    def unexpected_factory(_evidence: object) -> _TrainingRuntime:
        nonlocal called
        called = True
        return _TrainingRuntime()

    changed_data = _data_provenance_args()
    changed_data["train_file_sha256"] = "f" * 64
    with pytest.raises(GateBValidationError, match="contract mismatches"):
        train_qlora_fold(
            _records(training_ids),
            _records(validation_ids),
            split_manifest=manifest,
            fold=0,
            excluded_ids=(),
            **changed_data,
            source_manifest=source_manifest,
            preflight_artifact=preflight,
            gpu_smoke_artifact=smoke,
            output_dir=tmp_path / "mismatched-output",
            gpu_acknowledgement=GPU_EXECUTION_ACKNOWLEDGEMENT,
            resume_dir=resume_dir,
            runtime_factory=unexpected_factory,
        )
    assert called is False

    checkpoint_contract = resume_dir / "trainer" / "checkpoint-7" / "resume-checkpoint.json"
    checkpoint_payload = json.loads(checkpoint_contract.read_text(encoding="utf-8"))
    checkpoint_payload["contract_sha256"] = "f" * 64
    checkpoint_contract.write_text(json.dumps(checkpoint_payload), encoding="utf-8")
    with pytest.raises(GateBValidationError, match="checkpoint resume contract"):
        train_qlora_fold(
            _records(training_ids),
            _records(validation_ids),
            split_manifest=manifest,
            fold=0,
            excluded_ids=(),
            **_data_provenance_args(),
            source_manifest=source_manifest,
            preflight_artifact=preflight,
            gpu_smoke_artifact=smoke,
            output_dir=tmp_path / "checkpoint-mismatch-output",
            gpu_acknowledgement=GPU_EXECUTION_ACKNOWLEDGEMENT,
            resume_dir=resume_dir,
            runtime_factory=unexpected_factory,
        )
    assert called is False
    checkpoint_payload["contract_sha256"] = json.loads(
        (resume_dir / "resume-contract.json").read_text(encoding="utf-8")
    )["contract_sha256"]
    checkpoint_contract.write_text(json.dumps(checkpoint_payload), encoding="utf-8")

    (resume_dir / "trainer" / "checkpoint-7" / "optimizer.pt").unlink()
    recovered = _ResumableTrainingRuntime()
    artifact = train_qlora_fold(
        _records(training_ids),
        _records(validation_ids),
        split_manifest=manifest,
        fold=0,
        excluded_ids=(),
        **_data_provenance_args(),
        source_manifest=source_manifest,
        preflight_artifact=preflight,
        gpu_smoke_artifact=smoke,
        output_dir=tmp_path / "corrupt-output",
        gpu_acknowledgement=GPU_EXECUTION_ACKNOWLEDGEMENT,
        resume_dir=resume_dir,
        runtime_factory=lambda _evidence: recovered,
    )
    forensic_root = resume_dir / "forensics" / "attempt-000001"
    forensic_manifest = json.loads(
        (forensic_root / "manifest.json").read_text(encoding="utf-8")
    )
    assert artifact.path == str((tmp_path / "corrupt-output").resolve())
    assert recovered.resume_checkpoint is None
    assert not (resume_dir / "trainer" / "checkpoint-7").exists()
    assert (forensic_root / "checkpoint-7" / "adapter_config.json").is_file()
    assert not (forensic_root / "checkpoint-7" / "optimizer.pt").exists()
    assert forensic_manifest["contract_sha256"] == json.loads(
        (resume_dir / "resume-contract.json").read_text(encoding="utf-8")
    )["contract_sha256"]
    assert forensic_manifest["entries"] == [
        {
            "checkpoint": "checkpoint-7",
            "global_step": 7,
            "reason": "corrupt_after_contract_binding",
        }
    ]


def test_qlora_training_resume_uses_previous_complete_checkpoint_after_partial_save(
    tmp_path: Path,
) -> None:
    preflight, smoke = _gate_artifacts(tmp_path)
    manifest = _split_manifest()
    training_ids = eligible_training_ids(manifest, 0, ())
    validation_ids = eligible_validation_ids(manifest, 0, ())
    source_manifest = _source_manifest_evidence(tmp_path)
    output = tmp_path / "adapter-fold-0"
    resume_dir = tmp_path / "resume-fold-0"

    with pytest.raises(RuntimeError, match="simulated interruption"):
        train_qlora_fold(
            _records(training_ids),
            _records(validation_ids),
            split_manifest=manifest,
            fold=0,
            excluded_ids=(),
            **_data_provenance_args(),
            source_manifest=source_manifest,
            preflight_artifact=preflight,
            gpu_smoke_artifact=smoke,
            output_dir=output,
            gpu_acknowledgement=GPU_EXECUTION_ACKNOWLEDGEMENT,
            resume_dir=resume_dir,
            runtime_factory=lambda _evidence: _ResumableTrainingRuntime(
                fail_after_checkpoint=True
            ),
        )

    complete = resume_dir / "trainer" / "checkpoint-7"
    partial = resume_dir / "trainer" / "checkpoint-8"
    partial.mkdir()
    (partial / "partial-state.bin").write_bytes(b"interrupted checkpoint\n")

    resumed = _ResumableTrainingRuntime()
    artifact = train_qlora_fold(
        _records(training_ids),
        _records(validation_ids),
        split_manifest=manifest,
        fold=0,
        excluded_ids=(),
        **_data_provenance_args(),
        source_manifest=source_manifest,
        preflight_artifact=preflight,
        gpu_smoke_artifact=smoke,
        output_dir=output,
        gpu_acknowledgement=GPU_EXECUTION_ACKNOWLEDGEMENT,
        resume_dir=resume_dir,
        runtime_factory=lambda _evidence: resumed,
    )

    forensic_root = resume_dir / "forensics" / "attempt-000001"
    assert artifact.path == str(output.resolve())
    assert resumed.resume_checkpoint == complete.resolve()
    assert complete.is_dir()
    assert not partial.exists()
    assert (forensic_root / "checkpoint-8" / "partial-state.bin").read_bytes() == (
        b"interrupted checkpoint\n"
    )
    assert json.loads((forensic_root / "manifest.json").read_text(encoding="utf-8"))[
        "entries"
    ] == [
        {
            "checkpoint": "checkpoint-8",
            "global_step": 8,
            "reason": "missing_resume_contract",
        }
    ]


def test_qlora_training_resume_rejects_missing_checkpoint_after_interruption(
    tmp_path: Path,
) -> None:
    preflight, smoke = _gate_artifacts(tmp_path)
    manifest = _split_manifest()
    training_ids = eligible_training_ids(manifest, 0, ())
    validation_ids = eligible_validation_ids(manifest, 0, ())
    source_manifest = _source_manifest_evidence(tmp_path)
    resume_dir = tmp_path / "resume-fold-0"

    with pytest.raises(RuntimeError, match="before checkpoint"):
        train_qlora_fold(
            _records(training_ids),
            _records(validation_ids),
            split_manifest=manifest,
            fold=0,
            excluded_ids=(),
            **_data_provenance_args(),
            source_manifest=source_manifest,
            preflight_artifact=preflight,
            gpu_smoke_artifact=smoke,
            output_dir=tmp_path / "first-output",
            gpu_acknowledgement=GPU_EXECUTION_ACKNOWLEDGEMENT,
            resume_dir=resume_dir,
            runtime_factory=lambda _evidence: _ResumableTrainingRuntime(
                fail_without_checkpoint=True
            ),
        )

    factory_called = False

    def unexpected_factory(_evidence: object) -> _TrainingRuntime:
        nonlocal factory_called
        factory_called = True
        return _TrainingRuntime()

    with pytest.raises(GateBValidationError, match="no checkpoint"):
        train_qlora_fold(
            _records(training_ids),
            _records(validation_ids),
            split_manifest=manifest,
            fold=0,
            excluded_ids=(),
            **_data_provenance_args(),
            source_manifest=source_manifest,
            preflight_artifact=preflight,
            gpu_smoke_artifact=smoke,
            output_dir=tmp_path / "second-output",
            gpu_acknowledgement=GPU_EXECUTION_ACKNOWLEDGEMENT,
            resume_dir=resume_dir,
            runtime_factory=unexpected_factory,
        )
    assert factory_called is False


def test_qlora_training_resume_requires_runtime_resume_keywords(tmp_path: Path) -> None:
    preflight, smoke = _gate_artifacts(tmp_path)
    manifest = _split_manifest()
    training_ids = eligible_training_ids(manifest, 0, ())
    validation_ids = eligible_validation_ids(manifest, 0, ())
    legacy = _TrainingRuntime()

    with pytest.raises(GateBValidationError, match="requires runtime.train"):
        train_qlora_fold(
            _records(training_ids),
            _records(validation_ids),
            split_manifest=manifest,
            fold=0,
            excluded_ids=(),
            **_data_provenance_args(),
            source_manifest=_source_manifest_evidence(tmp_path),
            preflight_artifact=preflight,
            gpu_smoke_artifact=smoke,
            output_dir=tmp_path / "adapter-fold-0",
            gpu_acknowledgement=GPU_EXECUTION_ACKNOWLEDGEMENT,
            resume_dir=tmp_path / "resume-fold-0",
            runtime_factory=lambda _evidence: legacy,
        )
    assert legacy.closed is True


def test_rationale_qlora_training_publishes_v4_target_provenance(
    tmp_path: Path,
) -> None:
    preflight, smoke = _gate_artifacts(tmp_path)
    manifest = _split_manifest()
    excluded = (manifest.training_ids(0)[0],)
    training_ids = eligible_training_ids(manifest, 0, excluded)
    validation_ids = eligible_validation_ids(manifest, 0, excluded)
    training_records = _records(training_ids)
    rationale_corpus, rationale_config = _rationale_corpus_evidence(
        tmp_path,
        manifest,
        training_records,
        fold=0,
        excluded_ids=excluded,
    )
    fake = _TrainingRuntime()
    output = tmp_path / "adapter-rationale-fold-0"

    artifact = train_qlora_fold(
        training_records,
        _records(validation_ids),
        split_manifest=manifest,
        fold=0,
        excluded_ids=excluded,
        **_data_provenance_args(),
        source_manifest=_source_manifest_evidence(tmp_path),
        preflight_artifact=preflight,
        gpu_smoke_artifact=smoke,
        output_dir=output,
        gpu_acknowledgement=GPU_EXECUTION_ACKNOWLEDGEMENT,
        rationale_corpus=rationale_corpus,
        rationale_config=rationale_config,
        runtime_factory=lambda _evidence: fake,
    )

    validated = validate_adapter_artifact(output)
    private_manifest = json.loads(
        (output / "manifest.json").read_text(encoding="utf-8")
    )
    assert artifact.training_target_kind == "verified_concise_rationale"
    assert validated.training_target_kind == "verified_concise_rationale"
    assert validated.rationale_candidate_config_sha256 == (
        DEFAULT_CONCISE_RATIONALE_CONFIG.sha256
    )
    assert validated.rationale_candidate_config_file_sha256 == (
        rationale_corpus.candidate_config_file_sha256
    )
    assert validated.rationale_corpus_records_sha256 == rationale_corpus.records_sha256
    assert validated.rationale_corpus_manifest_sha256 == rationale_corpus.manifest_sha256
    assert validated.rationale_corpus_audit_sha256 == rationale_corpus.audit_sha256
    assert private_manifest["schema_version"] == "gate-b-qlora-adapter-v4"
    assert private_manifest["training_target"]["kind"] == (
        "verified_concise_rationale"
    )

    private_manifest["training_target"]["candidate_config_sha256"] = "f" * 64
    manifest_bytes = (
        json.dumps(private_manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    (output / "manifest.json").write_bytes(manifest_bytes)
    checksums = (output / "CHECKSUMS.sha256").read_text(encoding="utf-8").splitlines()
    replacement = f"{hashlib.sha256(manifest_bytes).hexdigest()}  manifest.json"
    (output / "CHECKSUMS.sha256").write_text(
        "\n".join(
            replacement if line.endswith("  manifest.json") else line
            for line in checksums
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(GateBValidationError, match="locked v1 policy"):
        validate_adapter_artifact(output)


def test_pinned_tokenizer_export_copies_exact_cache_bytes(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    output = tmp_path / "output"
    cache.mkdir()
    output.mkdir()
    tokenizer_bytes = b"{}\n"
    (cache / "tokenizer.json").write_bytes(tokenizer_bytes)
    (cache / "tokenizer_config.json").write_bytes(_FAKE_TOKENIZER_CONFIG_BYTES)
    calls: list[tuple[str, str, str, bool]] = []

    def cached_file(
        model_id: str,
        filename: str,
        *,
        revision: str,
        local_files_only: bool,
    ) -> str:
        calls.append((model_id, filename, revision, local_files_only))
        return str(cache / filename)

    runtime_module._save_pinned_tokenizer_snapshot(
        output,
        cached_file=cached_file,
    )

    assert (output / "tokenizer.json").read_bytes() == tokenizer_bytes
    assert (
        output / "tokenizer_config.json"
    ).read_bytes() == _FAKE_TOKENIZER_CONFIG_BYTES
    assert calls == [
        (OFFICIAL_MODEL_ID, "tokenizer.json", PINNED_MODEL_REVISION, True),
        (OFFICIAL_MODEL_ID, "tokenizer_config.json", PINNED_MODEL_REVISION, True),
    ]
    runtime_module._validate_saved_tokenizer(output)


def test_pinned_tokenizer_export_rejects_cache_checksum_drift(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    output = tmp_path / "output"
    cache.mkdir()
    output.mkdir()
    (cache / "tokenizer.json").write_bytes(b"changed\n")
    (cache / "tokenizer_config.json").write_bytes(_FAKE_TOKENIZER_CONFIG_BYTES)

    with pytest.raises(GateBValidationError, match="cache checksum mismatch"):
        runtime_module._save_pinned_tokenizer_snapshot(
            output,
            cached_file=lambda _model, filename, **_kwargs: str(cache / filename),
        )

    assert not tuple(output.iterdir())


def test_adapter_structural_validator_rejects_fake_safetensors_and_risky_config(
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "fake-adapter"
    fake_root.mkdir()
    (fake_root / "adapter_model.safetensors").write_bytes(b"not-safetensors")
    with pytest.raises(GateBValidationError, match="safetensors file is invalid"):
        runtime_module._validate_adapter_weight_files(
            fake_root, {"adapter_model.safetensors"}
        )

    config_root = tmp_path / "risky-config"
    config_root.mkdir()
    payload = {
        "base_model_name_or_path": OFFICIAL_MODEL_ID,
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "bias": "none",
        "target_modules": list(PINNED_QWEN_ALL_LINEAR_TARGET_MODULES),
        "modules_to_save": ["lm_head"],
        "use_dora": False,
        "use_rslora": False,
        "trainable_token_indices": None,
        "target_parameters": None,
        "rank_pattern": {},
        "alpha_pattern": {},
        "inference_mode": True,
        "fan_in_fan_out": False,
    }
    (config_root / "adapter_config.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    with pytest.raises(GateBValidationError, match="modules_to_save"):
        runtime_module._validate_adapter_config(config_root)


@pytest.mark.parametrize("corruption", ["extra", "shape", "dtype", "index"])
def test_adapter_safetensors_contract_rejects_structural_corruption(
    tmp_path: Path, corruption: str
) -> None:
    root = tmp_path / corruption
    root.mkdir()
    expected = runtime_module._expected_lora_tensor_shapes()
    tensors = {
        name: np.zeros(shape, dtype=np.float32) for name, shape in expected.items()
    }
    first_name = next(iter(tensors))
    if corruption == "extra":
        tensors["base_model.model.unexpected.lora_A.weight"] = np.zeros(
            (16, 16), dtype=np.float32
        )
        save_file(tensors, root / "adapter_model.safetensors")
        names = {"adapter_model.safetensors"}
        expected_message = "unexpected tensor"
    elif corruption == "shape":
        tensors[first_name] = np.zeros((1, 1), dtype=np.float32)
        save_file(tensors, root / "adapter_model.safetensors")
        names = {"adapter_model.safetensors"}
        expected_message = "unexpected shape"
    elif corruption == "dtype":
        tensors[first_name] = np.zeros(expected[first_name], dtype=np.int32)
        save_file(tensors, root / "adapter_model.safetensors")
        names = {"adapter_model.safetensors"}
        expected_message = "unsafe dtype"
    else:
        shard_name = "adapter_model-00001-of-00001.safetensors"
        tensors.pop(first_name)
        save_file(tensors, root / shard_name)
        (root / "adapter_model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {name: shard_name for name in tensors}}),
            encoding="utf-8",
        )
        names = {"adapter_model.safetensors.index.json", shard_name}
        expected_message = "tensor inventory is not exact"

    with pytest.raises(GateBValidationError, match=expected_message):
        runtime_module._validate_adapter_weight_files(root, names)


@pytest.mark.parametrize("incomplete", ["tokenizer", "shard"])
def test_incomplete_training_export_is_rejected_and_never_published(
    tmp_path: Path, incomplete: str
) -> None:
    preflight, smoke = _gate_artifacts(tmp_path)
    manifest = _split_manifest()
    training_ids = eligible_training_ids(manifest, 0, ())
    validation_ids = eligible_validation_ids(manifest, 0, ())
    fake = _TrainingRuntime(incomplete=incomplete)
    output = tmp_path / f"bad-{incomplete}"

    with pytest.raises(GateBValidationError, match="tokenizer|shard"):
        train_qlora_fold(
            _records(training_ids),
            _records(validation_ids),
            split_manifest=manifest,
            fold=0,
            excluded_ids=(),
            **_data_provenance_args(),
            source_manifest=_source_manifest_evidence(tmp_path),
            preflight_artifact=preflight,
            gpu_smoke_artifact=smoke,
            output_dir=output,
            gpu_acknowledgement=GPU_EXECUTION_ACKNOWLEDGEMENT,
            runtime_factory=lambda _evidence: fake,
        )
    assert fake.closed is True
    assert not output.exists()
    assert not tuple(tmp_path.glob(f".{output.name}.training-*"))


def test_training_requires_explicit_gpu_ack_before_runtime_factory(tmp_path: Path) -> None:
    preflight, smoke = _gate_artifacts(tmp_path)
    manifest = _split_manifest()
    training_ids = eligible_training_ids(manifest, 0, ())
    validation_ids = eligible_validation_ids(manifest, 0, ())
    called = False

    def factory(_evidence: object) -> _TrainingRuntime:
        nonlocal called
        called = True
        return _TrainingRuntime()

    with pytest.raises(GateBPreflightRequiredError, match="explicit GPU acknowledgement"):
        train_qlora_fold(
            _records(training_ids),
            _records(validation_ids),
            split_manifest=manifest,
            fold=0,
            excluded_ids=(),
            **_data_provenance_args(),
            source_manifest=_source_manifest_evidence(tmp_path),
            preflight_artifact=preflight,
            gpu_smoke_artifact=smoke,
            output_dir=tmp_path / "adapter",
            gpu_acknowledgement="no",
            runtime_factory=factory,
        )
    assert called is False


def test_peft_020_serializes_all_linear_as_exact_qwen_module_list(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    model = transformers.Qwen2ForCausalLM(
        transformers.Qwen2Config(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=36,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
    )
    adapted = peft.get_peft_model(
        model,
        peft.LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules="all-linear",
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    adapted.save_pretrained(tmp_path, safe_serialization=True)
    payload = json.loads((tmp_path / "adapter_config.json").read_text(encoding="utf-8"))

    assert isinstance(payload["target_modules"], list)
    assert tuple(sorted(payload["target_modules"])) == PINNED_QWEN_ALL_LINEAR_TARGET_MODULES


class _Tensor:
    def __init__(self, values: list[int], *, matrix: bool = True) -> None:
        self.values = values
        self.matrix = matrix

    @property
    def shape(self) -> tuple[int, ...]:
        return (1, len(self.values)) if self.matrix else (len(self.values),)

    def to(self, _device: object) -> _Tensor:
        return self

    def __getitem__(self, key: tuple[int, slice]) -> _Tensor:
        row, selected = key
        assert row == 0
        return _Tensor(self.values[selected], matrix=False)


class _GenerationTokenizer:
    eos_token_id = 0

    def apply_chat_template(self, *_args: object, **kwargs: object) -> _Tensor:
        assert kwargs == {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_tensors": "pt",
        }
        return _Tensor([1, 2, 3])

    def decode(self, values: _Tensor, *, skip_special_tokens: bool) -> str:
        assert values.values == [8, 9]
        assert skip_special_tokens is True
        return "Final answer: 7"


class _GenerationModel:
    device = "cuda:0"

    def __init__(self) -> None:
        self.kwargs: dict[str, object] = {}
        self.evaluated = False

    def eval(self) -> None:
        self.evaluated = True

    def generate(
        self, *, input_ids: _Tensor, attention_mask: _Tensor, **kwargs: object
    ) -> _Tensor:
        assert input_ids.values == [1, 2, 3]
        assert attention_mask.values == [1, 1, 1]
        self.kwargs = kwargs
        return _Tensor([1, 2, 3, 8, 9])


class _Cuda:
    def __init__(self) -> None:
        self.seed: int | None = None
        self.emptied = False

    def is_available(self) -> bool:
        return True

    def is_bf16_supported(self) -> bool:
        return True

    def device_count(self) -> int:
        return 1

    def set_device(self, device: int) -> None:
        assert device == 0

    def get_device_name(self, device: int) -> str:
        assert device == 0
        return "NVIDIA Test GPU"

    def manual_seed_all(self, seed: int) -> None:
        self.seed = seed

    def reset_peak_memory_stats(self, _device: object) -> None:
        return None

    def synchronize(self, _device: object) -> None:
        return None

    def max_memory_allocated(self, _device: object) -> int:
        return 123_456

    def empty_cache(self) -> None:
        self.emptied = True


class _Torch:
    def __init__(self) -> None:
        self.cuda = _Cuda()
        self.seed: int | None = None

    def manual_seed(self, seed: int) -> None:
        self.seed = seed

    def ones_like(self, values: _Tensor) -> _Tensor:
        return _Tensor([1] * len(values.values), matrix=values.matrix)

    def inference_mode(self) -> nullcontext[None]:
        return nullcontext()


def _generation_request() -> GenerationRequest:
    messages = (
        ChatMessage("system", DEFAULT_GATE_B_CONFIG.system_prompt),
        ChatMessage("user", "What is 3+4?"),
    )
    prompt_sha256 = hashlib.sha256(
        canonical_json_bytes(
            {
                "messages": [message.as_dict() for message in messages],
                "add_generation_prompt": True,
            }
        )
    ).hexdigest()
    return GenerationRequest(
        problem_id="train-000001",
        messages=messages,
        seed=123,
        sample_index=0,
        model_id=OFFICIAL_MODEL_ID,
        revision=PINNED_MODEL_REVISION,
        route=DEFAULT_GATE_B_CONFIG.route,
        prompt_sha256=prompt_sha256,
        config_sha256=DEFAULT_GATE_B_CONFIG.sha256,
        decoding_policy=DEFAULT_GATE_B_CONFIG.decoding_policy,
    )


def test_base_generation_backend_is_lazy_deterministic_and_reusable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preflight, smoke = _gate_artifacts(tmp_path)
    torch = _Torch()
    tokenizer = _GenerationTokenizer()
    model = _GenerationModel()
    load_calls = 0

    def fake_modules() -> dict[str, object]:
        return {"torch": torch}

    def fake_load(*_args: object, **_kwargs: object) -> tuple[object, object]:
        nonlocal load_calls
        load_calls += 1
        return tokenizer, model

    monkeypatch.setattr(runtime_module, "_lazy_gpu_modules", fake_modules)
    monkeypatch.setattr(runtime_module, "_load_pinned_nf4_base", fake_load)
    backend = create_base_development_backend(
        preflight_artifact=preflight,
        gpu_smoke_artifact=smoke,
        gpu_acknowledgement=GPU_EXECUTION_ACKNOWLEDGEMENT,
    )
    assert load_calls == 0
    assert backend.checkpoint_sha256 == runtime_module.BASE_MODEL_CHECKPOINT_SHA256

    request = _generation_request()
    first = backend.generate(request)
    second = backend.generate(request)

    assert load_calls == 1
    assert first == second
    assert first.text == "Final answer: 7"
    assert first.finish_reason == "stop"
    assert (first.input_token_count, first.output_token_count) == (3, 2)
    assert first.peak_vram_allocated_bytes == 123_456
    assert torch.seed == request.seed and torch.cuda.seed == request.seed
    assert model.kwargs["do_sample"] is False
    assert model.kwargs["num_beams"] == 1
    assert model.kwargs["max_new_tokens"] == DEFAULT_GATE_B_CONFIG.max_new_tokens
    assert model.kwargs["eos_token_id"] == tokenizer.eos_token_id
    assert model.kwargs["use_cache"] is True

    with pytest.raises(GateBValidationError, match="prompt_sha256"):
        backend.generate(replace(request, prompt_sha256="0" * 64))
    backend.close()
    assert torch.cuda.emptied is True
    with pytest.raises(GateBValidationError, match="closed"):
        backend.generate(request)


def test_base_and_adapted_backend_factories_require_gate_and_do_not_load_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preflight, smoke = _gate_artifacts(tmp_path)

    def forbidden_load() -> dict[str, object]:
        raise AssertionError("factory must remain lazy")

    monkeypatch.setattr(runtime_module, "_lazy_gpu_modules", forbidden_load)
    with pytest.raises(GateBPreflightRequiredError, match="explicit GPU acknowledgement"):
        create_base_development_backend(
            preflight_artifact=preflight,
            gpu_smoke_artifact=smoke,
            gpu_acknowledgement="not-yet",
        )

    manifest = _split_manifest()
    training_ids = eligible_training_ids(manifest, 0, ())
    validation_ids = eligible_validation_ids(manifest, 0, ())
    adapter = train_qlora_fold(
        _records(training_ids),
        _records(validation_ids),
        split_manifest=manifest,
        fold=0,
        excluded_ids=(),
        **_data_provenance_args(),
        source_manifest=_source_manifest_evidence(tmp_path),
        preflight_artifact=preflight,
        gpu_smoke_artifact=smoke,
        output_dir=tmp_path / "adapter",
        gpu_acknowledgement=GPU_EXECUTION_ACKNOWLEDGEMENT,
        runtime_factory=lambda _evidence: _TrainingRuntime(),
    )
    adapted = create_adapted_development_backend(
        preflight_artifact=preflight,
        gpu_smoke_artifact=smoke,
        adapter_path=adapter.path,
        split_manifest=manifest,
        fold=0,
        excluded_ids=(),
        **_data_provenance_args(),
        gpu_acknowledgement=GPU_EXECUTION_ACKNOWLEDGEMENT,
    )
    assert adapted.checkpoint_sha256 == adapter.artifact_sha256
    adapted.close()

    with pytest.raises(GateBValidationError, match="requested split/fold"):
        create_adapted_development_backend(
            preflight_artifact=preflight,
            gpu_smoke_artifact=smoke,
            adapter_path=adapter.path,
            split_manifest=manifest,
            fold=1,
            excluded_ids=(),
            **_data_provenance_args(),
            gpu_acknowledgement=GPU_EXECUTION_ACKNOWLEDGEMENT,
        )

    changed_provenance = _data_provenance_args()
    changed_provenance["train_file_sha256"] = "f" * 64
    with pytest.raises(GateBValidationError, match="requested split/fold"):
        create_adapted_development_backend(
            preflight_artifact=preflight,
            gpu_smoke_artifact=smoke,
            adapter_path=adapter.path,
            split_manifest=manifest,
            fold=0,
            excluded_ids=(),
            **changed_provenance,
            gpu_acknowledgement=GPU_EXECUTION_ACKNOWLEDGEMENT,
        )


def test_training_arguments_follow_every_locked_optimizer_schedule_policy(tmp_path: Path) -> None:
    class FakeArguments:
        def __init__(
            self,
            output_dir: str,
            *,
            eval_strategy: str | None = None,
            save_strategy: str | None = None,
            **kwargs: object,
        ) -> None:
            self.output_dir = output_dir
            self.eval_strategy = eval_strategy
            self.save_strategy = save_strategy
            self.kwargs = kwargs

    arguments = runtime_module._training_arguments(
        FakeArguments,
        output_dir=tmp_path / "trainer",
        config=DEFAULT_GATE_B_CONFIG,
    )
    assert arguments.eval_strategy == "steps"
    assert arguments.save_strategy == "steps"
    assert arguments.kwargs["per_device_train_batch_size"] == 1
    assert arguments.kwargs["gradient_accumulation_steps"] == 16
    assert arguments.kwargs["optim"] == "paged_adamw_8bit"
    assert arguments.kwargs["learning_rate"] == DEFAULT_GATE_B_CONFIG.learning_rate
    assert arguments.kwargs["lr_scheduler_type"] == "cosine"
    assert arguments.kwargs["warmup_ratio"] == 0.03
    assert arguments.kwargs["num_train_epochs"] == 1.0
    assert arguments.kwargs["max_grad_norm"] == 1.0
    assert arguments.kwargs["eval_steps"] == 100
    assert arguments.kwargs["save_steps"] == 100
    assert arguments.kwargs["save_total_limit"] == 2
    assert arguments.kwargs["bf16"] is True
    assert arguments.kwargs["gradient_checkpointing"] is True

    retained = runtime_module._training_arguments(
        FakeArguments,
        output_dir=tmp_path / "persistent-trainer",
        config=DEFAULT_GATE_B_CONFIG,
        retain_checkpoints=True,
    )
    assert retained.kwargs["save_total_limit"] is None

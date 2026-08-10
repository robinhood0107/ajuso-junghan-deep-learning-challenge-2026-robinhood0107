from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from deep_challenge.data import MathRecord
from deep_challenge.gate_b import (
    DEFAULT_GATE_B_CONFIG,
    PINNED_MODEL_REVISION,
    DecodingPolicy,
    DevelopmentExecutionEvidence,
    GateBArtifactExistsError,
    GateBConfig,
    GateBPreflightRequiredError,
    GateBValidationError,
    GenerationRequest,
    GenerationResult,
    TransformersGenerationBackend,
    build_direct_answer_sft_examples,
    create_development_execution_evidence,
    encode_response_only_example,
    run_development_baseline,
    write_development_artifacts,
)
from deep_challenge.model_preflight import OFFICIAL_MODEL_ID
from deep_challenge.provenance import (
    build_source_tree_manifest,
    validate_source_tree_manifest_artifact,
    write_json_atomic,
)
from deep_challenge.splits import (
    SplitManifest,
    eligible_training_ids,
    eligible_validation_ids,
    make_grouped_split_manifest,
)
from deep_challenge.tokenizer_profile import DEFAULT_SYSTEM_PROMPT


def _record(identifier: str, answer: int | None = None, question: str | None = None) -> MathRecord:
    value = int(identifier.removeprefix("train-")) if answer is None else answer
    text = question or f"What integer is associated with {identifier}?"
    return MathRecord(
        id=identifier,
        question_raw=text,
        question_normalized=text,
        answer_raw=None if value is None else str(value),
        answer=value,
        row_number=2,
    )


def _split_manifest() -> SplitManifest:
    ids = tuple(f"train-{index:06d}" for index in range(1, 13))
    return make_grouped_split_manifest(
        ids,
        dict(zip(ids, ids, strict=True)),
        n_folds=2,
        holdout_fraction=0.25,
        seed=20_260_731,
        version="splits-v4-test",
    )


def _records_for_ids(ids: tuple[str, ...]) -> list[MathRecord]:
    return [_record(identifier) for identifier in ids]


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


class _PrefixTokenizer:
    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        assert tokenize is True
        assert [message["role"] for message in conversation[:2]] == ["system", "user"]
        prefix = [11, 12, 13]
        if add_generation_prompt:
            assert len(conversation) == 2
            return prefix
        assert [message["role"] for message in conversation] == [
            "system",
            "user",
            "assistant",
        ]
        return [*prefix, 41, 42]


class _OverflowTokenizer(_PrefixTokenizer):
    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        del conversation, tokenize
        if add_generation_prompt:
            return [0]
        return list(range(2_049))


class _MismatchedTokenizer(_PrefixTokenizer):
    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        del conversation, tokenize
        return [1, 2] if add_generation_prompt else [1, 3, 4]


class _FakeBackend:
    def __init__(self, outputs: dict[tuple[str, int], str]) -> None:
        self.outputs = outputs
        self.requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        return GenerationResult(
            text=self.outputs[(request.problem_id, request.sample_index)],
            finish_reason="stop",
            input_token_count=17,
            output_token_count=5,
            peak_vram_allocated_bytes=123_456,
        )


def _development_records(
    *, samples_per_problem: int = 1
) -> tuple[SplitManifest, tuple[object, ...], _FakeBackend]:
    manifest = _split_manifest()
    validation_ids = eligible_validation_ids(manifest, 0, ())
    outputs: dict[tuple[str, int], str] = {}
    for problem_index, problem_id in enumerate(validation_ids):
        answer = int(problem_id.removeprefix("train-"))
        for sample_index in range(samples_per_problem):
            if problem_index == 0:
                text = f"Work.\nFinal answer: {answer}"
            elif problem_index == 1:
                text = f"Final answer: {answer}\nFinal answer: {answer + 1}"
            elif problem_index == 2:
                text = "Final answer: 5.5"
            else:
                text = f"Final answer: {answer}"
            outputs[(problem_id, sample_index)] = text
    backend = _FakeBackend(outputs)
    clock_values = iter([1_000_000, 2_500_000] * len(validation_ids) * samples_per_problem)
    result = run_development_baseline(
        _records_for_ids(validation_ids),
        split_manifest=manifest,
        fold=0,
        excluded_ids=(),
        backend=backend,
        checkpoint_sha256="A" * 64,
        samples_per_problem=samples_per_problem,
        clock_ns=lambda: next(clock_values),
    )
    return manifest, result, backend


def test_gate_b_config_is_complete_content_addressed_and_12gb_safe() -> None:
    config = GateBConfig()

    assert config.model_id == OFFICIAL_MODEL_ID
    assert config.revision == PINNED_MODEL_REVISION
    assert config.system_prompt == DEFAULT_SYSTEM_PROMPT
    assert config.hardware_profile == "nvidia-rtx-4070-super-12gb"
    assert config.max_sequence_length == 2_048
    assert (config.micro_batch_size, config.gradient_accumulation_steps) == (1, 16)
    assert (config.quantization, config.load_in_4bit) == ("nf4", True)
    assert config.double_quantization is True
    assert (config.compute_dtype, config.bf16_training) == ("bfloat16", True)
    assert (config.lora_rank, config.lora_alpha, config.lora_dropout) == (16, 32, 0.05)
    assert config.lora_target_modules == "all-linear"
    assert (config.response_only_loss, config.packing) == (True, False)
    assert config.optimizer == "paged_adamw_8bit"
    assert config.learning_rate == 0.0001
    assert (config.lr_scheduler_type, config.warmup_ratio) == ("cosine", 0.03)
    assert (config.num_train_epochs, config.max_grad_norm) == (1.0, 1.0)
    assert (config.eval_steps, config.save_steps) == (100, 100)
    assert (config.gradient_checkpointing, config.use_cache) == (True, False)
    assert config.max_new_tokens == 512
    assert config.decoding_policy == DecodingPolicy(
        do_sample=False,
        num_beams=1,
        max_new_tokens=512,
        temperature=None,
        top_p=None,
        top_k=None,
        repetition_penalty=1.0,
    )
    assert len(config.sha256) == 64
    assert config.sha256 == GateBConfig().sha256
    assert GateBConfig(seed=7).sha256 != config.sha256

    with pytest.raises(FrozenInstanceError):
        config.seed = 1  # type: ignore[misc]


def test_versioned_gate_b_config_file_matches_locked_runtime_profile() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "gate_b"
        / "rtx4070-super-12gb-direct-answer-v1.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    stored_sha256 = payload.pop("config_sha256")
    config = GateBConfig(**payload)

    assert config == GateBConfig()
    assert stored_sha256 == config.sha256


@pytest.mark.parametrize(
    "kwargs,fragment",
    [
        ({"model_id": "Qwen/Qwen2.5-Math-7B"}, "model_id"),
        ({"revision": "b" * 40}, "revision"),
        ({"route": "concise_cot"}, "route"),
        ({"max_sequence_length": 4_096}, "max_sequence_length"),
        ({"micro_batch_size": 2}, "micro_batch_size"),
        ({"quantization": "none"}, "quantization"),
        ({"double_quantization": False}, "double_quantization"),
        ({"compute_dtype": "float16"}, "compute_dtype"),
        ({"lora_rank": 32}, "lora_rank"),
        ({"lora_dropout": 0.0}, "lora_dropout"),
        ({"lora_target_modules": "q_proj"}, "lora_target_modules"),
        ({"response_only_loss": False}, "response_only_loss"),
        ({"optimizer": "adamw_torch"}, "optimizer"),
        ({"learning_rate": 0.0002}, "learning_rate"),
        ({"warmup_ratio": 0.05}, "warmup_ratio"),
        ({"num_train_epochs": 2.0}, "num_train_epochs"),
        ({"eval_steps": 50}, "eval_steps"),
        ({"gradient_checkpointing": False}, "gradient_checkpointing"),
        ({"use_cache": True}, "use_cache"),
        ({"generation_do_sample": True}, "generation_do_sample"),
        ({"max_new_tokens": 64}, "max_new_tokens"),
    ],
)
def test_gate_b_config_rejects_contract_drift(kwargs: dict[str, object], fragment: str) -> None:
    with pytest.raises(GateBValidationError, match=fragment):
        GateBConfig(**kwargs)  # type: ignore[arg-type]


def test_direct_answer_builder_derives_training_partition_and_records_provenance() -> None:
    manifest = _split_manifest()
    excluded_id = manifest.training_ids(0)[0]
    training_ids = eligible_training_ids(manifest, 0, (excluded_id,))

    examples = build_direct_answer_sft_examples(
        _records_for_ids(training_ids),
        split_manifest=manifest,
        fold=0,
        excluded_ids=(excluded_id,),
    )

    assert [example.problem_id for example in examples] == list(training_ids)
    assert all(example.target_text.startswith("Final answer: ") for example in examples)
    assert all("reason" not in example.target_text.casefold() for example in examples)
    assert all(example.split_version == manifest.version for example in examples)
    assert all(example.split_sha256 == manifest.sha256 for example in examples)
    assert all(
        example.source_groups_sha256 == manifest.source_groups_sha256
        for example in examples
    )
    assert all(example.fold == 0 and example.partition == "fold_training" for example in examples)
    assert all(example.split_partition == "cross_validation" for example in examples)
    assignments = manifest.assignment_by_id()
    assert all(example.group_id == assignments[example.problem_id].group_id for example in examples)
    assert len({example.eligibility_ids_sha256 for example in examples}) == 1


@pytest.mark.parametrize("view", ["missing", "all_train", "holdout", "leaderboard"])
def test_builder_rejects_any_view_other_than_derived_training_partition(view: str) -> None:
    manifest = _split_manifest()
    training_ids = eligible_training_ids(manifest, 0, ())
    if view == "missing":
        records = _records_for_ids(training_ids[1:])
        fragment = "missing"
    elif view == "all_train":
        records = _records_for_ids(tuple(item.record_id for item in manifest.assignments))
        fragment = "extra"
    elif view == "holdout":
        records = _records_for_ids(manifest.final_holdout_ids())
        fragment = "extra"
    else:
        records = [_record("leaderboard-000001", 1)]
        fragment = "leaderboard/test-like"

    with pytest.raises(GateBValidationError, match=fragment):
        build_direct_answer_sft_examples(
            records,
            split_manifest=manifest,
            fold=0,
            excluded_ids=(),
        )


def test_builder_rejects_unknown_exclusion_instead_of_weakening_split() -> None:
    manifest = _split_manifest()
    with pytest.raises(GateBValidationError, match="absent from split manifest"):
        build_direct_answer_sft_examples(
            _records_for_ids(eligible_training_ids(manifest, 0, ())),
            split_manifest=manifest,
            fold=0,
            excluded_ids=("train-999999",),
        )


def test_response_only_encoding_masks_prompt_and_carries_split_provenance() -> None:
    manifest = _split_manifest()
    training_ids = eligible_training_ids(manifest, 0, ())
    example = build_direct_answer_sft_examples(
        _records_for_ids(training_ids),
        split_manifest=manifest,
        fold=0,
        excluded_ids=(),
    )[0]

    encoded = encode_response_only_example(example, _PrefixTokenizer())

    assert encoded.input_ids == (11, 12, 13, 41, 42)
    assert encoded.attention_mask == (1, 1, 1, 1, 1)
    assert encoded.labels == (-100, -100, -100, 41, 42)
    assert encoded.prompt_token_count == 3
    assert encoded.sequence_token_count == 5
    assert encoded.config_sha256 == GateBConfig().sha256
    assert encoded.split_sha256 == manifest.sha256
    assert encoded.group_id == example.group_id
    assert encoded.eligibility_ids_sha256 == example.eligibility_ids_sha256


@pytest.mark.parametrize(
    "tokenizer,fragment",
    [
        (_OverflowTokenizer(), "truncation is forbidden"),
        (_MismatchedTokenizer(), "not a prefix"),
    ],
)
def test_response_only_encoding_fails_closed(tokenizer: object, fragment: str) -> None:
    manifest = _split_manifest()
    training_ids = eligible_training_ids(manifest, 0, ())
    example = build_direct_answer_sft_examples(
        _records_for_ids(training_ids),
        split_manifest=manifest,
        fold=0,
        excluded_ids=(),
    )[0]

    with pytest.raises(GateBValidationError, match=fragment):
        encode_response_only_example(example, tokenizer)  # type: ignore[arg-type]


def test_development_baseline_is_split_bound_structured_and_preserves_parser() -> None:
    manifest, result, backend = _development_records()

    assert [item.parse.status for item in result[:3]] == ["ok", "conflict", "invalid"]
    assert [item.exact_match for item in result[:3]] == [True, False, False]
    assert all(item.checkpoint_sha256 == "a" * 64 for item in result)
    assert all(item.config_sha256 == GateBConfig().sha256 for item in result)
    assert all(item.latency_ms == 1.5 for item in result)
    assert all(item.peak_vram_allocated_bytes == 123_456 for item in result)
    assert all((item.input_token_count, item.output_token_count) == (17, 5) for item in result)
    assert all(item.finish_reason == "stop" for item in result)
    assert all(item.split_sha256 == manifest.sha256 for item in result)
    assert all(item.partition == "fold_validation" and item.fold == 0 for item in result)
    assert all(request.decoding_policy.max_new_tokens == 512 for request in backend.requests)
    assert all(request.config_sha256 == GateBConfig().sha256 for request in backend.requests)
    assert all(request.model_id == OFFICIAL_MODEL_ID for request in backend.requests)
    assert all(request.revision == PINNED_MODEL_REVISION for request in backend.requests)

    parsed_json = json.loads(result[1].to_json_line())
    assert parsed_json["parse"]["status"] == "conflict"
    assert parsed_json["raw_completion"].count("Final answer:") == 2
    assert parsed_json["decoding_policy"]["do_sample"] is False
    assert parsed_json["split_sha256"] == manifest.sha256

    validation_ids = eligible_validation_ids(manifest, 0, ())
    outputs = {
        (problem_id, 0): f"Final answer: {int(problem_id.removeprefix('train-'))}"
        for problem_id in validation_ids
    }
    repeat = run_development_baseline(
        _records_for_ids(validation_ids),
        split_manifest=manifest,
        fold=0,
        excluded_ids=(),
        backend=_FakeBackend(outputs),
        checkpoint_sha256="a" * 64,
        clock_ns=lambda: 0,
    )
    assert [item.seed for item in repeat] == [item.seed for item in result]
    assert [item.prompt_sha256 for item in repeat] == [item.prompt_sha256 for item in result]


def test_development_baseline_rejects_full_train_and_locked_holdout() -> None:
    manifest = _split_manifest()
    backend = _FakeBackend({})
    all_train = _records_for_ids(tuple(item.record_id for item in manifest.assignments))
    holdout = _records_for_ids(manifest.final_holdout_ids())

    for records in (all_train, holdout):
        with pytest.raises(GateBValidationError, match="split-derived eligible partition"):
            run_development_baseline(
                records,
                split_manifest=manifest,
                fold=0,
                excluded_ids=(),
                backend=backend,
                checkpoint_sha256="a" * 64,
            )


def test_development_baseline_reports_data_free_progress_counts() -> None:
    manifest = _split_manifest()
    validation_ids = eligible_validation_ids(manifest, 0, ())
    outputs = {
        (problem_id, sample_index): f"Final answer: {int(problem_id.removeprefix('train-'))}"
        for problem_id in validation_ids
        for sample_index in range(2)
    }
    observed: list[tuple[int, int]] = []

    result = run_development_baseline(
        _records_for_ids(validation_ids),
        split_manifest=manifest,
        fold=0,
        excluded_ids=(),
        backend=_FakeBackend(outputs),
        checkpoint_sha256="a" * 64,
        samples_per_problem=2,
        clock_ns=lambda: 0,
        progress_callback=lambda completed, total: observed.append((completed, total)),
    )

    assert observed == [(index, len(result)) for index in range(1, len(result) + 1)]
    assert all(total == len(result) for _, total in observed)


def test_generation_result_and_backend_contract_fail_closed() -> None:
    with pytest.raises(GateBValidationError, match="output_token_count"):
        GenerationResult("text", "stop", 1, -1)
    with pytest.raises(GateBValidationError, match="peak_vram"):
        GenerationResult("text", "stop", 1, 1, -1)

    manifest = _split_manifest()
    validation_ids = eligible_validation_ids(manifest, 0, ())

    class _StringBackend:
        def generate(self, request: GenerationRequest) -> str:
            del request
            return "Final answer: 1"

    with pytest.raises(TypeError, match="GenerationResult"):
        run_development_baseline(
            _records_for_ids(validation_ids),
            split_manifest=manifest,
            fold=0,
            excluded_ids=(),
            backend=_StringBackend(),  # type: ignore[arg-type]
            checkpoint_sha256="a" * 64,
            clock_ns=lambda: 0,
        )


def test_atomic_artifact_writer_roundtrips_checksum_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    _, records, _ = _development_records(samples_per_problem=2)
    jsonl_path = tmp_path / "development.jsonl"
    manifest_path = tmp_path / "development.manifest.json"
    evidence = _execution_evidence(tmp_path)

    written = write_development_artifacts(
        records,
        jsonl_path=jsonl_path,
        manifest_path=manifest_path,
        execution_evidence=evidence,
    )

    payload = jsonl_path.read_bytes()
    lines = payload.decode("utf-8").splitlines()
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(lines) == len(records) == written.record_count
    assert all(json.loads(line)["schema_version"].endswith("v2") for line in lines)
    assert hashlib.sha256(payload).hexdigest() == written.records_sha256
    assert manifest_payload["records_sha256"] == written.records_sha256
    assert manifest_payload["record_count"] == len(records)
    assert manifest_payload["samples_per_problem"] == 2
    assert manifest_payload["parser_status_counts"]["conflict"] >= 1
    assert manifest_payload["schema_version"] == "gate-b1-development-run-v2"
    assert manifest_payload["execution_evidence"]["config_file"]["config_sha256"] == (
        DEFAULT_GATE_B_CONFIG.sha256
    )
    assert manifest_payload["generation_evidence"]["latency_ms"]["count"] == len(records)
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == written.manifest_sha256
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob(".gate-b-*.lock"))

    with pytest.raises(GateBArtifactExistsError, match="overwrite"):
        write_development_artifacts(
            records,
            jsonl_path=jsonl_path,
            manifest_path=manifest_path,
            execution_evidence=evidence,
        )


def test_execution_evidence_rejects_config_file_semantic_mismatch(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
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
    config.write_text(
        json.dumps(
            {
                **DEFAULT_GATE_B_CONFIG.as_dict(),
                "seed": DEFAULT_GATE_B_CONFIG.seed + 1,
                "config_sha256": DEFAULT_GATE_B_CONFIG.sha256,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    preflight = tmp_path / "preflight.json"
    smoke = tmp_path / "smoke.json"
    preflight.write_text("{}\n", encoding="utf-8")
    smoke.write_text("{}\n", encoding="utf-8")

    with pytest.raises(GateBValidationError, match="semantic SHA"):
        create_development_execution_evidence(
            source_manifest=source_manifest,
            config_path=config,
            config_sha256=DEFAULT_GATE_B_CONFIG.sha256,
            preflight_report_path=preflight,
            preflight_report_sha256=hashlib.sha256(preflight.read_bytes()).hexdigest(),
            gpu_smoke_report_path=smoke,
            gpu_smoke_report_sha256=hashlib.sha256(smoke.read_bytes()).hexdigest(),
            gpu_device_name="NVIDIA Test GPU",
        )


def test_atomic_artifact_writer_has_one_winner_under_race(tmp_path: Path) -> None:
    _, records, _ = _development_records()
    jsonl_path = tmp_path / "race.jsonl"
    manifest_path = tmp_path / "race.manifest.json"
    evidence = _execution_evidence(tmp_path)
    barrier = threading.Barrier(2)

    def write_once() -> str:
        barrier.wait()
        try:
            write_development_artifacts(
                records,
                jsonl_path=jsonl_path,
                manifest_path=manifest_path,
                execution_evidence=evidence,
            )
        except GateBArtifactExistsError:
            return "exists"
        return "written"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _: write_once(), range(2)))

    assert sorted(outcomes) == ["exists", "written"]
    assert jsonl_path.is_file() and manifest_path.is_file()
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert hashlib.sha256(jsonl_path.read_bytes()).hexdigest() == manifest_payload["records_sha256"]
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob(".gate-b-*.lock"))


def test_artifact_writer_rejects_missing_generation_without_publishing(tmp_path: Path) -> None:
    _, records, _ = _development_records()
    jsonl_path = tmp_path / "incomplete.jsonl"
    manifest_path = tmp_path / "incomplete.manifest.json"
    evidence = _execution_evidence(tmp_path)

    with pytest.raises(GateBValidationError, match="eligibility_ids_sha256"):
        write_development_artifacts(
            records[:-1],
            jsonl_path=jsonl_path,
            manifest_path=manifest_path,
            execution_evidence=evidence,
        )
    assert not jsonl_path.exists()
    assert not manifest_path.exists()


def test_real_transformers_adapter_is_an_explicit_preflight_gate() -> None:
    config = GateBConfig()
    request = GenerationRequest(
        problem_id="train-000001",
        messages=(),
        seed=0,
        sample_index=0,
        model_id=OFFICIAL_MODEL_ID,
        revision=PINNED_MODEL_REVISION,
        route="direct_answer",
        prompt_sha256="a" * 64,
        config_sha256=config.sha256,
        decoding_policy=config.decoding_policy,
    )

    with pytest.raises(GateBPreflightRequiredError, match="training_ready=true"):
        TransformersGenerationBackend().generate(request)

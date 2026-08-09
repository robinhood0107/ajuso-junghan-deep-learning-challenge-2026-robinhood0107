from __future__ import annotations

from pathlib import Path

import pytest

from deep_challenge.data import MathRecord
from deep_challenge.model_preflight import OFFICIAL_REVISION
from deep_challenge.tokenizer_profile import (
    DEFAULT_SYSTEM_PROMPT,
    _snapshot_commit_from_path,
    load_and_profile_datasets,
    profile_records,
)

REVISION = OFFICIAL_REVISION


class FakeTokenizer:
    name_or_path = "fake"
    init_kwargs: dict[str, object] = {}

    def __call__(self, text: str, *, add_special_tokens: bool):
        assert add_special_tokens is False
        return {"input_ids": text.split()}

    def apply_chat_template(
        self, conversation, *, tokenize: bool, add_generation_prompt: bool
    ):
        assert tokenize is True
        assert add_generation_prompt is True
        return ["system", *conversation[0]["content"].split(), *conversation[1]["content"].split()]


def _record(identifier: str, question: str) -> MathRecord:
    return MathRecord(
        id=identifier,
        question_raw=question,
        question_normalized=question,
        answer_raw="1",
        answer=1,
        row_number=2,
    )


def test_profile_records_persists_exact_prompt_and_lengths() -> None:
    report = profile_records(
        [_record("short", "one two"), _record("long", "one two three four")],
        FakeTokenizer(),
    )
    assert report["system_prompt"] == DEFAULT_SYSTEM_PROMPT
    assert report["record_count"] == 2
    assert report["raw_tokens"]["min"] == 2
    assert report["raw_tokens"]["max"] == 4
    assert report["chat_input_tokens"]["max"] > report["raw_tokens"]["max"]
    assert report["longest_chat_inputs"][0]["id"] == "long"


def test_snapshot_commit_is_inferred_from_huggingface_cache_path() -> None:
    path = Path(f"/cache/models--x/snapshots/{REVISION}/tokenizer.json")
    assert _snapshot_commit_from_path(path) == REVISION
    assert _snapshot_commit_from_path(Path("/cache/models--x/snapshots/abc123/x")) is None
    assert _snapshot_commit_from_path(Path("/cache/tokenizer.json")) is None


def _cached_tokenizer_files(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "models--Qwen--Qwen2.5-3B-Instruct" / "snapshots" / REVISION
    root.mkdir(parents=True)
    files: dict[str, Path] = {}
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        path = root / name
        path.write_text("{}", encoding="utf-8")
        files[name] = path
    return files


def test_multiple_datasets_share_one_tokenizer_and_identical_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from transformers import AutoTokenizer
    from transformers.utils import hub

    files = _cached_tokenizer_files(tmp_path)
    tokenizer = FakeTokenizer()
    tokenizer.init_kwargs = {"_commit_hash": REVISION}
    load_calls: list[dict[str, object]] = []

    def fake_from_pretrained(_model_id: str, **kwargs: object) -> FakeTokenizer:
        load_calls.append(kwargs)
        return tokenizer

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(
        hub,
        "cached_file",
        lambda _model, filename, **_kwargs: str(files[filename]) if filename in files else None,
    )

    reports = load_and_profile_datasets(
        {
            "train": [_record("train", "one two")],
            "leaderboard": [_record("leaderboard", "three four")],
        },
        revision=REVISION,
    )

    assert len(load_calls) == 1
    assert reports["train"]["provenance"] == reports["leaderboard"]["provenance"]
    assert reports["train"]["provenance"]["resolved_commit"] == REVISION


def test_mixed_tokenizer_snapshot_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from transformers import AutoTokenizer
    from transformers.utils import hub

    files = _cached_tokenizer_files(tmp_path)
    other_root = tmp_path / "models--Qwen--Qwen2.5-3B-Instruct" / "snapshots" / ("d" * 40)
    other_root.mkdir(parents=True)
    other_config = other_root / "config.json"
    other_config.write_text("{}", encoding="utf-8")
    files["config.json"] = other_config
    tokenizer = FakeTokenizer()
    tokenizer.init_kwargs = {"_commit_hash": REVISION}
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *_args, **_kwargs: tokenizer)
    monkeypatch.setattr(
        hub,
        "cached_file",
        lambda _model, filename, **_kwargs: str(files[filename]) if filename in files else None,
    )

    with pytest.raises(RuntimeError, match="not from exactly the requested snapshot"):
        load_and_profile_datasets(
            {"train": [_record("train", "one two")]}, revision=REVISION
        )

from __future__ import annotations

import hashlib
import json

import pytest

from deep_challenge.data import (
    DatasetKind,
    DatasetValidationError,
    FileManifest,
    load_leaderboard_csv,
    load_train_csv,
    load_train_exclusion_csv,
    normalize_question,
)


def test_train_loader_preserves_raw_fields_and_supports_rfc4180(tmp_path) -> None:
    source = tmp_path / "train.csv"
    payload = (
        b'id,question,answer\r\n'
        b'train-1,"Compute, carefully:\r\n2\xc2\xa0+\xef\xbc\x92",4\r\n'
        b'train-2,"He said ""five"".",-5\r\n'
    )
    source.write_bytes(payload)

    dataset = load_train_csv(source)

    assert dataset.kind is DatasetKind.TRAIN
    assert dataset.raw_header == ("id", "question", "answer")
    assert len(dataset) == 2
    assert dataset.records[0].question_raw == "Compute, carefully:\r\n2\u00a0+２"
    assert dataset.records[0].question_normalized == "Compute, carefully:\n2 +2"
    assert dataset.records[0].answer_raw == "4"
    assert dataset.records[0].answer == 4
    assert dataset.records[1].question == 'He said "five".'
    assert dataset.manifest.sha256 == hashlib.sha256(payload).hexdigest()
    assert dataset.manifest.size_bytes == len(payload)


def test_normalize_question_is_conservative() -> None:
    assert normalize_question("  A\u00a0Ｂ\r\n C\rD  ") == "  A B\n C\nD  "


def test_file_manifest_is_content_addressed(tmp_path) -> None:
    source = tmp_path / "bytes.bin"
    source.write_bytes(b"abc\x00def")
    manifest = FileManifest.from_path(source)
    assert manifest.path == str(source.resolve())
    assert manifest.size_bytes == 7
    assert manifest.sha256 == hashlib.sha256(b"abc\x00def").hexdigest()


def test_leaderboard_allows_raw_header_bug_and_only_trailing_blank_padding(tmp_path) -> None:
    source = tmp_path / "leaderboard.csv"
    source.write_text(
        'id,question, answer\r\nval-1,"What is 1, plus 1?"\r\nval-2,What is 3+3?,\r\n',
        encoding="utf-8",
        newline="",
    )

    dataset = load_leaderboard_csv(source)

    assert dataset.raw_header == ("id", "question", " answer")
    assert [record.id for record in dataset] == ["val-1", "val-2"]
    assert all(record.answer is None and record.answer_raw is None for record in dataset)


def test_leaderboard_accepts_current_filtered_two_column_contract(tmp_path) -> None:
    source = tmp_path / "leaderboard-filtered.csv"
    source.write_text(
        'id,question\r\nval-1,"What is 1, plus 1?"\r\nval-2,What is 3+3?\r\n',
        encoding="utf-8",
        newline="",
    )

    dataset = load_leaderboard_csv(source)

    assert dataset.raw_header == ("id", "question")
    assert [record.id for record in dataset] == ["val-1", "val-2"]
    assert [record.question_raw for record in dataset] == [
        "What is 1, plus 1?",
        "What is 3+3?",
    ]
    assert all(record.answer is None and record.answer_raw is None for record in dataset)


@pytest.mark.parametrize(
    "payload,loader,error_fragment",
    [
        ("id,question,answer\na,q\n", load_train_csv, "expected exactly 3 fields"),
        ("id,question,answer\na,q,1,extra\n", load_train_csv, "got 4"),
        ("id,question,answer\na,q,1\na,q2,2\n", load_train_csv, "duplicate id"),
        ("id,question,answer\n,q,1\n", load_train_csv, "missing id"),
        ("id,question,answer\na,,1\n", load_train_csv, "missing question"),
        ("id,question,answer\na,q,\n", load_train_csv, "missing train answer"),
        ("id,question,answer\na,q,1.5\n", load_train_csv, "must be an integer"),
        ("id,question,answer\na,q,+1\n", load_train_csv, "must be an integer"),
        ("id,question,answer\na,q,01\n", load_train_csv, "must be an integer"),
        ("id,question,answer\na,q,-0\n", load_train_csv, "must be an integer"),
        ("question,id,answer\nq,a,1\n", load_train_csv, "train header"),
        ("id,question,answer\na,q,42\n", load_leaderboard_csv, "must be empty"),
        ("id,question,answer\na\n", load_leaderboard_csv, "got 1"),
        ("id,question,answer\na,q,,extra\n", load_leaderboard_csv, "got 4"),
        ("id,question\na\n", load_leaderboard_csv, "expected exactly 2 fields"),
        ("id,question\na,q,\n", load_leaderboard_csv, "expected exactly 2 fields"),
        ("id,answer\na,\n", load_leaderboard_csv, "leaderboard header"),
    ],
)
def test_loader_rejects_contract_violations(tmp_path, payload, loader, error_fragment) -> None:
    source = tmp_path / "bad.csv"
    source.write_text(payload, encoding="utf-8", newline="")
    with pytest.raises(DatasetValidationError, match=error_fragment):
        loader(source)


def test_loader_rejects_malformed_csv_and_empty_dataset(tmp_path) -> None:
    malformed = tmp_path / "malformed.csv"
    malformed.write_text('id,question,answer\r\na,"unterminated,1\r\n', encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="invalid CSV"):
        load_train_csv(malformed)

    header_only = tmp_path / "header-only.csv"
    header_only.write_text("id,question,answer\r\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="contains no records"):
        load_train_csv(header_only)


def test_utf8_bom_is_accepted_without_changing_content_hash(tmp_path) -> None:
    source = tmp_path / "bom.csv"
    payload = b"\xef\xbb\xbfid,question,answer\r\na,q,0\r\n"
    source.write_bytes(payload)
    dataset = load_train_csv(source)
    assert dataset.records[0].answer == 0
    assert dataset.manifest.sha256 == hashlib.sha256(payload).hexdigest()


def test_train_exclusion_loader_accepts_official_and_minimal_headers(tmp_path) -> None:
    official = tmp_path / "train_filtered_ids.csv"
    official_payload = (
        b'id,answer,question\r\n'
        b'train-2,not-authoritative,"opaque, quoted field"\r\n'
        b'train-1,,\r\n'
    )
    official.write_bytes(official_payload)
    minimal = tmp_path / "train_filtered_ids-minimal.csv"
    minimal.write_bytes(b"id\r\ntrain-1\r\ntrain-2\r\n")

    current = load_train_exclusion_csv(
        official,
        train_ids=["train-1", "train-2", "train-3"],
    )
    future = load_train_exclusion_csv(
        minimal,
        train_ids=["train-3", "train-2", "train-1"],
    )

    assert current.raw_header == ("id", "answer", "question")
    assert future.raw_header == ("id",)
    assert current.ids == future.ids == ("train-1", "train-2")
    expected_logical_hash = hashlib.sha256(
        json.dumps(
            ("train-1", "train-2"),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert current.ids_sha256 == future.ids_sha256 == expected_logical_hash
    assert current.manifest.sha256 == hashlib.sha256(official_payload).hexdigest()
    assert current.manifest.size_bytes == len(official_payload)


@pytest.mark.parametrize(
    "payload,error_fragment",
    [
        (b"ID\r\ntrain-1\r\n", "exclusion header"),
        (b"id,question,answer\r\ntrain-1,q,1\r\n", "exclusion header"),
        (b"id\r\n\r\n", "expected exactly 1 fields"),
        (b"id\r\n train-1\r\n", "surrounding whitespace"),
        (b"id\r\ntrain-1\r\ntrain-1\r\n", "duplicate exclusion id"),
        (b"id,answer,question\r\ntrain-1,\r\n", "expected exactly 3 fields"),
    ],
)
def test_train_exclusion_loader_rejects_noncanonical_csv(
    tmp_path,
    payload,
    error_fragment,
) -> None:
    source = tmp_path / "bad-exclusions.csv"
    source.write_bytes(payload)

    with pytest.raises(DatasetValidationError, match=error_fragment):
        load_train_exclusion_csv(source, train_ids=["train-1"])


def test_train_exclusion_loader_rejects_unknown_ids_and_invalid_utf8(tmp_path) -> None:
    unknown = tmp_path / "unknown.csv"
    unknown.write_bytes(b"id\r\ntrain-missing\r\n")
    with pytest.raises(DatasetValidationError, match="unknown train IDs"):
        load_train_exclusion_csv(unknown, train_ids=["train-1"])

    invalid_utf8 = tmp_path / "invalid-utf8.csv"
    invalid_utf8.write_bytes(b"id\r\ntrain-\xff\r\n")
    with pytest.raises(DatasetValidationError, match="invalid train exclusion CSV"):
        load_train_exclusion_csv(invalid_utf8, train_ids=["train-1"])


def test_empty_train_exclusion_set_is_valid_and_deterministic(tmp_path) -> None:
    source = tmp_path / "empty-exclusions.csv"
    source.write_bytes(b"id\r\n")

    exclusions = load_train_exclusion_csv(source, train_ids=["train-1"])

    assert exclusions.ids == ()
    assert exclusions.ids_sha256 == hashlib.sha256(b"[]").hexdigest()

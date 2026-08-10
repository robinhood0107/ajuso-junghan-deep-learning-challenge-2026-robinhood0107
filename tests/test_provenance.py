from __future__ import annotations

import json
from pathlib import Path

import pytest

from deep_challenge.provenance import (
    build_source_tree_manifest,
    canonical_json_bytes,
    sha256_file,
    validate_source_tree_manifest_artifact,
    write_json_atomic,
)


def test_sha256_file_known_value(tmp_path: Path) -> None:
    path = tmp_path / "value.txt"
    path.write_bytes(b"abc")
    assert sha256_file(path) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_sha256_rejects_invalid_chunk_size(tmp_path: Path) -> None:
    path = tmp_path / "value.txt"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="positive"):
        sha256_file(path, chunk_size=0)


def test_source_tree_manifest_is_order_independent_and_excludes_cache(tmp_path: Path) -> None:
    (tmp_path / "b.py").write_text("b = 2\n", encoding="utf-8")
    (tmp_path / "a.py").write_text("a = 1\n", encoding="utf-8")
    (tmp_path / ".python-version").write_text("3.12\n", encoding="utf-8")
    cache = tmp_path / ".venv"
    cache.mkdir()
    (cache / "ignored.py").write_text("secret = 3\n", encoding="utf-8")
    gstack_state = tmp_path / ".gstack"
    gstack_state.mkdir()
    (gstack_state / "browse.json").write_text('{"runtime": true}\n', encoding="utf-8")

    first = build_source_tree_manifest(tmp_path)
    second = build_source_tree_manifest(tmp_path)

    assert first == second
    assert [entry.path for entry in first.files] == [".python-version", "a.py", "b.py"]
    assert len(first.tree_sha256) == 64


def test_source_tree_digest_changes_with_content(tmp_path: Path) -> None:
    path = tmp_path / "a.py"
    path.write_text("a = 1\n", encoding="utf-8")
    first = build_source_tree_manifest(tmp_path)
    path.write_text("a = 2\n", encoding="utf-8")
    second = build_source_tree_manifest(tmp_path)
    assert first.tree_sha256 != second.tree_sha256


def test_source_tree_manifest_can_exclude_its_own_output(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("a = 1\n", encoding="utf-8")
    output = tmp_path / "manifest.json"
    first = build_source_tree_manifest(tmp_path, excluded_paths=(output,))
    write_json_atomic(output, first.as_dict())
    second = build_source_tree_manifest(tmp_path, excluded_paths=(output,))
    assert first == second
    assert [entry.path for entry in second.files] == ["a.py"]


def test_runtime_source_manifest_must_match_the_current_tree(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    module = source_root / "a.py"
    module.write_text("a = 1\n", encoding="utf-8")
    output = tmp_path / "source-manifest.json"
    write_json_atomic(
        output,
        build_source_tree_manifest(source_root, excluded_paths=(output,)).as_dict(),
    )

    evidence = validate_source_tree_manifest_artifact(output, root=source_root)
    assert evidence.path == str(output.resolve())
    assert evidence.file_count == 1
    assert evidence.tree_sha256

    module.write_text("a = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        validate_source_tree_manifest_artifact(output, root=source_root)


def test_source_tree_manifest_rejects_symbolic_links(tmp_path: Path) -> None:
    target = tmp_path / "real.py"
    target.write_text("secret = 1\n", encoding="utf-8")
    (tmp_path / "linked.py").symlink_to(target)
    with pytest.raises(ValueError, match="refuses symbolic links"):
        build_source_tree_manifest(tmp_path)


def test_canonical_json_rejects_nan() -> None:
    with pytest.raises(ValueError):
        canonical_json_bytes({"bad": float("nan")})


def test_atomic_json_roundtrip(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "manifest.json"
    write_json_atomic(target, {"한글": [2, 1], "ok": True})
    assert json.loads(target.read_text(encoding="utf-8")) == {
        "ok": True,
        "한글": [2, 1],
    }
    assert not list(target.parent.glob("*.tmp"))

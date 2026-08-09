from __future__ import annotations

from deep_challenge.public_repo_guard import find_forbidden_paths, find_secret_markers


def test_forbidden_paths_cover_competition_data_credentials_and_model_outputs() -> None:
    violations = find_forbidden_paths(
        (
            "artifacts/run.json",
            "archive/artifacts/run.json",
            "datasets/external.parquet",
            "deep-learning-challenge-2026/deep_chal_math_train.csv",
            "env",
            ".env.production",
            ".kaggle/kaggle.json",
            "NU_",
            "id_ed25519",
            "models/adapter_model.safetensors",
            "notes/private.pem",
            "notes/secrets.kdbx",
            "reports/generated/model.bin",
            "notes/result.jsonl.gz",
            "notes/result.tar",
            "notes/result.zip",
            "submission.csv",
        )
    )

    assert violations == (
        ".env.production",
        ".kaggle/kaggle.json",
        "NU_",
        "archive/artifacts/run.json",
        "artifacts/run.json",
        "datasets/external.parquet",
        "deep-learning-challenge-2026/deep_chal_math_train.csv",
        "env",
        "id_ed25519",
        "models/adapter_model.safetensors",
        "notes/private.pem",
        "notes/result.jsonl.gz",
        "notes/result.tar",
        "notes/result.zip",
        "notes/secrets.kdbx",
        "reports/generated/model.bin",
        "submission.csv",
    )


def test_forbidden_path_guard_allows_public_source_and_documentation() -> None:
    assert find_forbidden_paths(
        (
            "README.md",
            "LICENSE",
            "docs/11_EXECUTION_CONTINUATION_PLAN.md",
            "src/deep_challenge/gate_b_runtime.py",
            "tests/test_gate_b_runtime.py",
            "configs/gate_b/rtx4070-super-12gb-direct-answer-v1.json",
        )
    ) == ()


def test_secret_markers_report_categories_without_token_values() -> None:
    named_fixture = b"KAGGLE_API_TOKEN=" + b"abcdefghijklmnop"
    github_fixture = b"token = ghp_" + b"abcdefghijklmnopqrstuvwxyz123456"
    assert find_secret_markers(named_fixture) == (
        "named-access-token",
    )
    assert find_secret_markers(github_fixture) == (
        "github-token",
    )
    private_key_fixture = b"-----BEGIN " + b"PRIVATE KEY-----"
    assert find_secret_markers(private_key_fixture) == ("private-key",)
    assert find_secret_markers(b"KAGGLE_API_TOKEN is documented but redacted") == ()

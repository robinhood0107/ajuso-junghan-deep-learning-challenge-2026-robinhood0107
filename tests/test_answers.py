"""Boundary and adversarial tests for conservative answer extraction."""

from __future__ import annotations

import builtins

import pytest

from deep_challenge.answers import parse_answer


@pytest.mark.parametrize(
    ("completion", "expected", "source"),
    [
        ("Final answer: 42", 42, "final_answer"),
        ("**Final answer:** -17", -17, "final_answer"),
        ("Therefore, the final answer is +9.", 9, "final_answer"),
        ("The result is \\boxed{1,234}.", 1234, "boxed"),
        ("\\boxed{\\frac{84}{2}}", 42, "boxed"),
        ("reasoning\n#### −12", -12, "hashes"),
        ("Answer: -0", 0, "answer"),
        ("After checking, the result is 81", 81, "fallback"),
        ("Final answer: 42.000", 42, "final_answer"),
        ("Final answer: -84 / 2", -42, "final_answer"),
        ("Final answer: 84 / -2", -42, "final_answer"),
        ("Final answer:\n\n\\boxed{7}", 7, "final_answer"),
        ("The final answer is: 19.", 19, "final_answer"),
        ("The final answer is: `Final answer: 15552`", 15552, "final_answer"),
        ("**Final answer: 676**", 676, "final_answer"),
        ("**Final answer:** **2**", 2, "final_answer"),
        (r"Therefore, the final answer is: \[ \text{Final answer: } 160 \]", 160, "final_answer"),
        ("The solution is \\boxed{x=42}.", 42, "boxed"),
    ],
)
def test_supported_answer_forms(completion: str, expected: int, source: str) -> None:
    result = parse_answer(completion)

    assert result.ok
    assert result.status == "ok"
    assert result.value == expected
    assert result.source == source
    assert result.reason == "parsed_unambiguous_integer"


def test_highest_priority_marker_wins_over_lower_priority_marker() -> None:
    result = parse_answer("A discarded candidate was \\boxed{99}.\nFinal answer: 5")

    assert result.ok
    assert result.value == 5
    assert result.source == "final_answer"


@pytest.mark.parametrize(
    "completion",
    [
        "Answering carefully, I obtain 42.",
        "Final answering should not be treated as a marker: 42.",
    ],
)
def test_answer_prefix_inside_longer_word_is_not_a_marker(completion: str) -> None:
    result = parse_answer(completion)

    assert result.ok
    assert result.value == 42
    assert result.source == "fallback"


def test_repeated_selected_markers_must_agree() -> None:
    agreeing = parse_answer("Final answer: 8\nFinal answer: +8")
    conflict = parse_answer("Final answer: 8\nFinal answer: 9")

    assert agreeing.ok and agreeing.value == 8
    assert conflict.status == "conflict"
    assert conflict.value is None
    assert conflict.source == "final_answer"
    assert "conflicting_marker_values" in conflict.reason


def test_nested_selected_markers_do_not_hide_conflicts() -> None:
    conflict = parse_answer(
        "The final answer is: Final answer: 8\nFinal answer: 9"
    )

    assert conflict.status == "conflict"
    assert conflict.value is None
    assert conflict.source == "final_answer"


@pytest.mark.parametrize(
    ("completion", "reason_fragment"),
    [
        ("Final answer: 2 or 3", "multiple_values"),
        ("Final answer: 4.5", "non_integral_decimal"),
        ("Final answer: 5/2", "non_integral_fraction"),
        ("Final answer: 4/0", "zero_denominator"),
        ("Final answer: 1e3", "scientific_notation"),
        ("Final answer: NaN", "non_finite"),
        ("Final answer: Infinity", "non_finite"),
        ("Final answer: 12,34", "unsupported_numeric_payload"),
        ("Final answer: 2 + 2 = 4", "multiple_values"),
        ("Final answer: 2**3", "multiple_values"),
        ("\\boxed{42", "unbalanced_braces"),
        ("\\boxed 42", "missing_open_brace"),
        ("Final answer: __import__('os').system('echo no')", "unsupported_numeric_payload"),
    ],
)
def test_ambiguous_noninteger_and_malicious_payloads_are_invalid(
    completion: str, reason_fragment: str
) -> None:
    result = parse_answer(completion)

    assert result.status == "invalid"
    assert result.value is None
    assert reason_fragment in result.reason


def test_boxed_parser_balances_nested_braces() -> None:
    result = parse_answer(r"The final object is \boxed{\text{42}}.")

    assert result.ok
    assert result.value == 42
    assert result.source == "boxed"


def test_conflicting_boxes_are_not_silently_resolved() -> None:
    result = parse_answer(r"\boxed{10} and later \boxed{11}")

    assert result.status == "conflict"
    assert result.value is None
    assert result.source == "boxed"


def test_fallback_uses_last_standalone_numeric_token() -> None:
    result = parse_answer("We tried 20, corrected the computation, and obtained 21.")

    assert result.ok
    assert result.value == 21
    assert result.source == "fallback"


def test_fallback_does_not_extract_from_identifier() -> None:
    result = parse_answer("The opaque identifier is problem42x.")

    assert result.status == "invalid"
    assert result.value is None
    assert result.reason == "no_numeric_candidate"


@pytest.mark.parametrize("completion", ["result 1e3", "result NaN", "result 4.25"])
def test_invalid_last_numeric_token_does_not_fall_back_to_its_digits(completion: str) -> None:
    result = parse_answer(completion)

    assert result.status == "invalid"
    assert result.value is None


def test_parser_never_calls_eval(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden_eval(*args: object, **kwargs: object) -> object:
        raise AssertionError("eval must never be called")

    monkeypatch.setattr(builtins, "eval", forbidden_eval)

    result = parse_answer("Final answer: (40 + 2)")

    assert result.status == "invalid"
    assert result.value is None


def test_empty_and_non_string_completion_contract() -> None:
    empty = parse_answer(" \n\t")

    assert empty.status == "invalid"
    assert empty.value is None
    assert empty.source is None
    assert empty.reason == "empty_completion"
    with pytest.raises(TypeError):
        parse_answer(42)  # type: ignore[arg-type]

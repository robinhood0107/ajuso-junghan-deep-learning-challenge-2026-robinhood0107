"""Conservative extraction of one integer from an assistant completion.

The parser deliberately accepts only the assistant's completion text.  It does
not inspect prompts, reference answers, or external state, and it never
evaluates model-produced code or arithmetic expressions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

ParseStatus = Literal["ok", "invalid", "conflict"]
ParseSource = Literal["final_answer", "boxed", "hashes", "answer", "fallback"]


@dataclass(frozen=True, slots=True)
class AnswerParseResult:
    """Result of parsing exactly one integer answer.

    Attributes:
        status: ``"ok"``, ``"invalid"``, or ``"conflict"``.
        value: Parsed integer when ``status`` is ``"ok"``; otherwise ``None``.
        source: Marker class selected by the documented precedence, if any.
        reason: Stable, human-readable explanation suitable for audit logs.
    """

    status: ParseStatus
    value: int | None
    source: ParseSource | None
    reason: str

    @property
    def ok(self) -> bool:
        """Return whether the completion produced one unambiguous integer."""

        return self.status == "ok"


@dataclass(frozen=True, slots=True)
class _Candidate:
    payload: str
    error: str | None = None


_MINUS_TRANSLATION = str.maketrans(
    {
        "−": "-",
        "﹣": "-",
        "－": "-",
    }
)
_UNSIGNED_INTEGER = r"(?:\d+|\d{1,3}(?:,\d{3})+)"
_SIGNED_INTEGER_RE = re.compile(rf"(?P<sign>[+-]?)(?P<body>{_UNSIGNED_INTEGER})\Z")
_DECIMAL_RE = re.compile(
    rf"(?P<sign>[+-]?)(?P<body>{_UNSIGNED_INTEGER})\.(?P<fraction>\d+)\Z"
)
_FORBIDDEN_NONFINITE_RE = re.compile(r"(?i)(?:nan|[+-]?(?:inf|infinity))")
_SCIENTIFIC_RE = re.compile(r"(?i)[+-]?(?:\d+(?:\.\d*)?|\.\d+)[eE][+-]?\d+")
_NUMERIC_LIKE_RE = re.compile(
    r"(?<![\w.])"
    r"(?P<value>"
    r"(?:NaN|[+-]?(?:Inf|Infinity))"
    r"|[+\-−﹣－]?"
    r"(?:\d[\d,]*(?:\.\d+)?|\.\d+)"
    r"(?:[eE][+-]?\d+)?"
    r"(?:\s*/\s*[+\-−﹣－]?\d[\d,]*)?"
    r")"
    r"(?!\w|\.\d)",
    re.IGNORECASE,
)

_FINAL_ANSWER_RE = re.compile(
    r"(?im)^[ \t]*(?:[-*][ \t]+)?(?:\*\*|__)?"
    r"(?:(?:therefore|thus|hence)[,:]?[ \t]+)?(?:the[ \t]+)?"
    r"final[ \t]+answer\b(?:\*\*|__)?[ \t]*(?:is\b|[:=])?"
    r"[ \t]*(?:\*\*|__)?[ \t]*"
    r"(?P<payload>[^\r\n]*)"
)
_HASHES_RE = re.compile(r"(?m)^[ \t]*####[ \t]*(?P<payload>[^\r\n]*)")
_ANSWER_RE = re.compile(
    r"(?im)^[ \t]*(?:[-*][ \t]+)?(?:\*\*|__)?"
    r"answer\b(?:\*\*|__)?[ \t]*(?:is\b|[:=])?[ \t]*(?:\*\*|__)?"
    r"[ \t]*(?P<payload>[^\r\n]*)"
)
_BOX_COMMAND_RE = re.compile(r"\\(?:boxed|fbox)\s*")


def parse_answer(completion: str) -> AnswerParseResult:
    """Parse one signed integer from an assistant completion.

    Marker precedence is ``Final answer`` > balanced ``\\boxed{...}`` >
    ``####`` > ``Answer`` > the last standalone numeric token.  Repeated
    markers in the selected class must all be valid and agree.  Lower-priority
    markers are intentionally ignored once a higher-priority class is present.

    Accepted answer forms are signed integers, correctly grouped commas,
    Unicode minus signs, integral decimals, and a single exact integer
    fraction.  Arbitrary expressions, scientific notation, non-finite values,
    non-integral numbers, and multi-valued payloads are rejected.

    Args:
        completion: Assistant-generated completion only.

    Returns:
        An :class:`AnswerParseResult`; invalid results never contain a fallback
        value.

    Raises:
        TypeError: If ``completion`` is not a string.
    """

    if not isinstance(completion, str):
        raise TypeError("completion must be a string containing assistant output only")
    if not completion.strip():
        return AnswerParseResult("invalid", None, None, "empty_completion")

    source_candidates: tuple[tuple[ParseSource, list[_Candidate]], ...] = (
        ("final_answer", _line_marker_candidates(completion, _FINAL_ANSWER_RE)),
        ("boxed", _boxed_candidates(completion)),
        ("hashes", _line_marker_candidates(completion, _HASHES_RE)),
        ("answer", _line_marker_candidates(completion, _ANSWER_RE)),
    )

    for source, candidates in source_candidates:
        if candidates:
            return _resolve_candidates(candidates, source)

    fallback = _last_numeric_candidate(completion)
    if fallback is None:
        return AnswerParseResult("invalid", None, "fallback", "no_numeric_candidate")
    return _resolve_candidates([fallback], "fallback")


def _line_marker_candidates(text: str, pattern: re.Pattern[str]) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    for match in pattern.finditer(text):
        payload = match.group("payload").strip()
        if not payload:
            payload = _next_nonempty_line(text, match.end())
        candidates.append(_Candidate(payload, None if payload else "empty_marker_payload"))
    return candidates


def _next_nonempty_line(text: str, offset: int) -> str:
    for line in text[offset:].splitlines():
        if line.strip():
            return line.strip()
    return ""


def _boxed_candidates(text: str) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    for match in _BOX_COMMAND_RE.finditer(text):
        open_index = match.end()
        if open_index >= len(text) or text[open_index] != "{":
            candidates.append(_Candidate("", "boxed_missing_open_brace"))
            continue
        balanced = _extract_balanced_group(text, open_index)
        if balanced is None:
            candidates.append(_Candidate("", "boxed_unbalanced_braces"))
            continue
        payload, _ = balanced
        candidates.append(_Candidate(payload))
    return candidates


def _extract_balanced_group(text: str, open_index: int) -> tuple[str, int] | None:
    if open_index >= len(text) or text[open_index] != "{":
        return None
    depth = 0
    for index in range(open_index, len(text)):
        char = text[index]
        if char not in "{}" or _is_escaped(text, index):
            continue
        if char == "{":
            depth += 1
        else:
            depth -= 1
            if depth == 0:
                return text[open_index + 1 : index], index + 1
            if depth < 0:
                return None
    return None


def _is_escaped(text: str, index: int) -> bool:
    backslashes = 0
    index -= 1
    while index >= 0 and text[index] == "\\":
        backslashes += 1
        index -= 1
    return backslashes % 2 == 1


def _last_numeric_candidate(text: str) -> _Candidate | None:
    matches = list(_NUMERIC_LIKE_RE.finditer(text))
    if not matches:
        return None
    return _Candidate(matches[-1].group("value"))


def _resolve_candidates(
    candidates: list[_Candidate], source: ParseSource
) -> AnswerParseResult:
    values: list[int] = []
    for index, candidate in enumerate(candidates, start=1):
        if candidate.error is not None:
            return AnswerParseResult(
                "invalid", None, source, f"marker_{index}:{candidate.error}"
            )
        value, error = _parse_numeric_payload(candidate.payload)
        if error is not None:
            return AnswerParseResult("invalid", None, source, f"marker_{index}:{error}")
        assert value is not None
        values.append(value)

    distinct = set(values)
    if len(distinct) > 1:
        rendered = ",".join(str(value) for value in sorted(distinct))
        return AnswerParseResult(
            "conflict", None, source, f"conflicting_marker_values:{rendered}"
        )
    return AnswerParseResult("ok", values[0], source, "parsed_unambiguous_integer")


def _parse_numeric_payload(payload: str) -> tuple[int | None, str | None]:
    text = _normalize_presentation(payload)
    if not text:
        return None, "empty_numeric_payload"
    assignment = re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*\s*=\s*(.+)", text)
    if assignment is not None:
        text = assignment.group(1).strip()
    if _FORBIDDEN_NONFINITE_RE.fullmatch(text):
        return None, "non_finite_value"
    if _SCIENTIFIC_RE.fullmatch(text):
        return None, "scientific_notation_not_allowed"

    latex_fraction = _unwrap_latex_fraction(text)
    if latex_fraction is not None:
        text = latex_fraction

    if "/" in text:
        if text.count("/") != 1:
            return None, "multiple_fraction_operators"
        numerator_text, denominator_text = (part.strip() for part in text.split("/", 1))
        numerator, numerator_error = _parse_plain_integer(numerator_text)
        denominator, denominator_error = _parse_plain_integer(denominator_text)
        if numerator_error is not None or denominator_error is not None:
            return None, "fraction_must_contain_two_plain_integers"
        assert numerator is not None and denominator is not None
        if denominator == 0:
            return None, "zero_denominator"
        quotient, remainder = divmod(numerator, denominator)
        if remainder != 0:
            return None, "non_integral_fraction"
        return quotient, None

    integer, integer_error = _parse_plain_integer(text)
    if integer_error is None:
        return integer, None

    decimal = _DECIMAL_RE.fullmatch(text)
    if decimal is not None:
        if set(decimal.group("fraction")) != {"0"}:
            return None, "non_integral_decimal"
        integer_text = f"{decimal.group('sign')}{decimal.group('body')}"
        return _parse_plain_integer(integer_text)

    if _contains_multiple_numeric_tokens(text):
        return None, "multiple_values_in_payload"
    return None, "unsupported_numeric_payload"


def _parse_plain_integer(text: str) -> tuple[int | None, str | None]:
    normalized = text.translate(_MINUS_TRANSLATION)
    match = _SIGNED_INTEGER_RE.fullmatch(normalized)
    if match is None:
        return None, "not_plain_integer"
    digits = match.group("body")
    if "," in digits and re.fullmatch(r"\d{1,3}(?:,\d{3})+", digits) is None:
        return None, "invalid_comma_grouping"
    canonical_digits = digits.replace(",", "")
    try:
        value = int(f"{match.group('sign')}{canonical_digits}")
    except ValueError:
        return None, "integer_outside_runtime_safety_limit"
    return value, None


def _contains_multiple_numeric_tokens(text: str) -> bool:
    return len(list(_NUMERIC_LIKE_RE.finditer(text))) > 1


def _normalize_presentation(payload: str) -> str:
    text = payload.strip().translate(_MINUS_TRANSLATION)
    text = _strip_sentence_punctuation(text)

    changed = True
    while changed and text:
        changed = False
        for left, right in (("**", "**"), ("__", "__"), ("`", "`"), ("$", "$")):
            wrapped = (
                text.startswith(left)
                and text.endswith(right)
                and len(text) >= len(left) + len(right)
            )
            if wrapped:
                text = text[len(left) : -len(right)].strip()
                changed = True
        for left, right in (("\\(", "\\)"), ("\\[", "\\]"), ("(", ")"), ("[", "]")):
            if text.startswith(left) and text.endswith(right):
                text = text[len(left) : -len(right)].strip()
                changed = True

        unwrapped = _unwrap_full_braced_command(text, ("\\boxed", "\\fbox", "\\text"))
        if unwrapped is not None:
            text = unwrapped.strip()
            changed = True
        text = _strip_sentence_punctuation(text)
    return text.strip()


def _strip_sentence_punctuation(text: str) -> str:
    stripped = text.strip()
    while stripped.endswith(("!", ";")):
        stripped = stripped[:-1].rstrip()
    if stripped.endswith("."):
        stripped = stripped[:-1].rstrip()
    return stripped


def _unwrap_full_braced_command(text: str, commands: tuple[str, ...]) -> str | None:
    for command in commands:
        if not text.startswith(command):
            continue
        open_index = len(command)
        while open_index < len(text) and text[open_index].isspace():
            open_index += 1
        balanced = _extract_balanced_group(text, open_index)
        if balanced is None:
            return None
        content, end = balanced
        if text[end:].strip():
            return None
        return content
    return None


def _unwrap_latex_fraction(text: str) -> str | None:
    if not text.startswith("\\frac"):
        return None
    index = len("\\frac")
    while index < len(text) and text[index].isspace():
        index += 1
    numerator = _extract_balanced_group(text, index)
    if numerator is None:
        return None
    numerator_text, index = numerator
    while index < len(text) and text[index].isspace():
        index += 1
    denominator = _extract_balanced_group(text, index)
    if denominator is None:
        return None
    denominator_text, index = denominator
    if text[index:].strip():
        return None
    return f"{numerator_text}/{denominator_text}"


__all__ = [
    "AnswerParseResult",
    "ParseSource",
    "ParseStatus",
    "parse_answer",
]

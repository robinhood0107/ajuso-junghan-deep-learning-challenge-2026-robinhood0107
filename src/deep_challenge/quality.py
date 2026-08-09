"""Non-destructive data-quality signals and leakage-oriented fingerprints.

Every detector in this module is a triage hint, not a deletion rule.  Math
corpora contain terse valid questions, non-ASCII symbols, multi-part reasoning,
and embedded drawing code, so a flag must always remain reviewable.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from .data import MathRecord, normalize_question


class QualityFlag(StrEnum):
    """Auditable warning categories; none implies automatic exclusion."""

    FRAGMENT = "fragment"
    ANSWER_OR_SOLUTION_CUE = "answer_or_solution_cue"
    NUMERIC_BOXED = "numeric_boxed"
    URL_OR_IMAGE = "url_or_image"
    MISSING_VISUAL = "missing_visual"
    ASYMPTOTE = "asymptote"
    MULTI_PART = "multi_part"
    MULTILINGUAL = "multilingual"
    NON_ASCII = "non_ascii"
    LATEX_PROXY = "latex_proxy"


@dataclass(frozen=True, slots=True)
class QualityAssessment:
    """Flags for one question with an explicit non-deletion contract."""

    flags: frozenset[QualityFlag]
    reasons: tuple[str, ...]
    auto_drop: bool = False

    def has(self, flag: QualityFlag) -> bool:
        return flag in self.flags


_WORD_RE: Final[re.Pattern[str]] = re.compile(r"[A-Za-z]+|\d+")
_ANSWER_CUE_RE: Final[re.Pattern[str]] = re.compile(
    r"(?im)(?:^|[\n.!?]\s*)(?:final\s+)?(?:answer|solution)\s*(?:is\b|[:=])|####\s*[+-]?\d+"
)
_NUMERIC_BOXED_RE: Final[re.Pattern[str]] = re.compile(
    r"\\(?:boxed|fbox)\s*\{\s*[+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?\s*\}"
)
_URL_OR_IMAGE_RE: Final[re.Pattern[str]] = re.compile(
    r"(?i)(?:https?://|www\.|<img\b|!\[[^]]*\]\([^)]*\)|"
    r"\b[^\s]+\.(?:png|jpe?g|gif|svg|webp)\b)"
)
_VISUAL_REFERENCE_RE: Final[re.Pattern[str]] = re.compile(
    r"(?i)\b(?:shown|pictured|illustrated|displayed)\s+(?:above|below)\b|"
    r"\b(?:above|below|following|given)\s+(?:figure|diagram|graph|image|picture)\b|"
    r"\b(?:figure|diagram|graph|image|picture)\s+(?:above|below)\b|"
    r"\baccording\s+to\s+(?:the\s+)?(?:figure|diagram|graph|image|picture)\b|"
    r"\bfrom\s+(?:the\s+)?(?:figure|diagram|graph|image|picture)\b"
)
_ASYMPTOTE_RE: Final[re.Pattern[str]] = re.compile(
    r"(?i)(?:\[asy\]|\[/asy\]|\bimport\s+(?:graph|geometry|three)\s*;|"
    r"\b(?:draw|dot|label|fill|size)\s*\([^\n;]*\)\s*;)"
)
_PART_MARKER_RE: Final[re.Pattern[str]] = re.compile(
    r"(?im)(?:^|[\s;])\((?:[a-z]|[ivx]{1,5}|\d+)\)\s*|"
    r"(?:^|\n)\s*(?:part\s+)?(?:[a-z]|[ivx]{1,5}|\d+)[.):]\s+"
)
_MULTIPART_LANGUAGE_RE: Final[re.Pattern[str]] = re.compile(
    r"(?i)\b(?:answer|solve|find)\s+(?:both|all)\b|\bparts?\s+\(?[a-z]\)?\s+(?:and|through)\s+\(?[a-z]\)?"
)
_FOREIGN_SCRIPT_RE: Final[re.Pattern[str]] = re.compile(
    "[\u0400-\u052f\u0600-\u06ff\u0900-\u097f\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]"
)
_LATEX_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:\$[^$]+\$|\\\(|\\\)|\\\[|\\\]|"
    r"\\(?:frac|dfrac|tfrac|sqrt|sum|prod|int|begin|end|overline|underline|"
    r"mathbf|mathbb|mathrm|text|boxed|left|right)\b)"
)
_NUMBER_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<![A-Za-z0-9_])(?P<sign>[+-]?)(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
    r"(?:[eE][+-]?\d+)?(?![A-Za-z0-9_])"
)
# TeX control words contain ASCII letters only.  ``\b`` is not appropriate
# here because a digit immediately following a command (``\tfrac1``) is a word
# character to the regex engine but a valid command boundary to TeX.
_LATEX_FRACTION_VARIANT_RE: Final[re.Pattern[str]] = re.compile(
    r"\\(?:dfrac|tfrac)(?![A-Za-z])"
)
_LATEX_SIZING_RE: Final[re.Pattern[str]] = re.compile(r"\\(?:left|right)(?![A-Za-z])")
_LATEX_SPACING_RE: Final[re.Pattern[str]] = re.compile(
    r"\\(?:,|;|:|!|quad\b|qquad\b|enspace\b|thinspace\b)"
)
_COLLAPSIBLE_SPACE_RE: Final[re.Pattern[str]] = re.compile(r"\s+")
_MATH_EDGE_SPACE_RE: Final[re.Pattern[str]] = re.compile(
    r"\s*([{}()[\]^_=+\-*/<>|,;:])\s*"
)
_LEADING_SOURCE_NUMBER_RE: Final[re.Pattern[str]] = re.compile(
    r"^\d+(?:\.\d+)*(?:[.)])?\s+"
    r"(?=(?:what|which|find|if|how|determine|compute|solve)\b)",
    re.IGNORECASE,
)
_TRAILING_STANDALONE_HASH_RE: Final[re.Pattern[str]] = re.compile(r"\s*#\s*$")
_SPACE_BEFORE_SENTENCE_PUNCTUATION_RE: Final[re.Pattern[str]] = re.compile(
    r"\s+([?.!])"
)


def assess_question(question: str) -> QualityAssessment:
    """Return conservative review flags for ``question``.

    The function never recommends automatic deletion.  Callers should combine
    the flags with manual adjudication, model behavior, and source provenance.
    """

    text = normalize_question(question)
    stripped = text.strip()
    flags: set[QualityFlag] = set()
    reasons: list[str] = []
    has_asymptote = _ASYMPTOTE_RE.search(stripped) is not None

    if _looks_like_fragment(stripped):
        flags.add(QualityFlag.FRAGMENT)
        reasons.append("question is unusually short or syntactically fragmentary")
    if _ANSWER_CUE_RE.search(stripped):
        flags.add(QualityFlag.ANSWER_OR_SOLUTION_CUE)
        reasons.append("text contains an answer/solution marker that may leak a target")
    if _NUMERIC_BOXED_RE.search(stripped):
        flags.add(QualityFlag.NUMERIC_BOXED)
        reasons.append("text contains an already boxed numeric value")
    if _URL_OR_IMAGE_RE.search(stripped):
        flags.add(QualityFlag.URL_OR_IMAGE)
        reasons.append("text contains an external URL or image reference")
    if _VISUAL_REFERENCE_RE.search(stripped) and not has_asymptote:
        flags.add(QualityFlag.MISSING_VISUAL)
        reasons.append("text refers to a visual whose availability must be verified")
    if has_asymptote:
        flags.add(QualityFlag.ASYMPTOTE)
        reasons.append("text contains embedded Asymptote-like drawing code")
    if _is_multi_part(stripped):
        flags.add(QualityFlag.MULTI_PART)
        reasons.append("text appears to contain multiple requested parts")
    if any(ord(character) > 127 for character in stripped):
        flags.add(QualityFlag.NON_ASCII)
        reasons.append("text contains non-ASCII characters")
    if _FOREIGN_SCRIPT_RE.search(stripped):
        flags.add(QualityFlag.MULTILINGUAL)
        reasons.append("text contains a non-Latin script")
    if _LATEX_RE.search(stripped):
        flags.add(QualityFlag.LATEX_PROXY)
        reasons.append("text contains LaTeX-style math markup")

    return QualityAssessment(flags=frozenset(flags), reasons=tuple(reasons), auto_drop=False)


def assess_records(records: Iterable[MathRecord]) -> dict[str, QualityAssessment]:
    """Assess records without changing their content or membership."""

    return {record.id: assess_question(record.question_raw) for record in records}


def canonical_math_text(question: str) -> str:
    """Create a stable, math-aware comparison string.

    This view removes representational LaTeX differences (delimiter and spacing
    commands) while retaining words, operators, braces, and numeric values.  It
    is intended for candidate duplicate grouping, not proof of equivalence.
    """

    text = normalize_question(question).casefold()
    text = text.translate(
        str.maketrans(
            {
                "−": "-",
                "–": "-",
                "—": "-",
                "×": "\\times",
                "÷": "/",
            }
        )
    )
    # Command boundaries are significant: a plain string replacement would
    # corrupt unrelated commands such as ``\leftarrow`` into ``arrow``.
    text = _LATEX_FRACTION_VARIANT_RE.sub(r"\\frac", text)
    text = _LATEX_SIZING_RE.sub("", text)
    for delimiter in ("$$", "$", "\\(", "\\)", "\\[", "\\]"):
        text = text.replace(delimiter, "")
    text = _LATEX_SPACING_RE.sub("", text)
    text = _COLLAPSIBLE_SPACE_RE.sub(" ", text).strip()
    return _MATH_EDGE_SPACE_RE.sub(r"\1", text)


def math_aware_fingerprint(question: str) -> str:
    """SHA-256 fingerprint of :func:`canonical_math_text`."""

    return _sha256_text(canonical_math_text(question))


def source_format_text(question: str) -> str:
    """Remove narrowly scoped source-format artifacts from canonical math text.

    Some source collections prefix the same problem with an exercise number or
    append a standalone Markdown heading marker.  Those are removed only when
    the leading token is followed by common question language; every numeric
    literal inside the mathematical problem remains intact.  Whitespace before
    sentence punctuation is also representational.  This remains a duplicate
    *candidate* signal rather than a proof that two problems are equivalent.
    """

    text = canonical_math_text(question)
    text = _LEADING_SOURCE_NUMBER_RE.sub("", text, count=1)
    text = _TRAILING_STANDALONE_HASH_RE.sub("", text)
    text = _SPACE_BEFORE_SENTENCE_PUNCTUATION_RE.sub(r"\1", text)
    return text.strip()


def source_format_fingerprint(question: str) -> str:
    """SHA-256 fingerprint of :func:`source_format_text`."""

    return _sha256_text(source_format_text(question))


def number_masked_template_text(question: str) -> str:
    """Canonical form with numeric magnitudes masked and leading signs retained.

    Keeping the leading sign prevents structurally different constraints such
    as ``x=-1`` and ``x=1`` from becoming the same template.
    """

    return _NUMBER_RE.sub(
        lambda match: f"{match.group('sign')}<num>", canonical_math_text(question)
    )


def number_masked_template_fingerprint(question: str) -> str:
    """SHA-256 fingerprint for finding shared templates with changed numbers."""

    return _sha256_text(number_masked_template_text(question))


def _looks_like_fragment(text: str) -> bool:
    if not text:
        return True
    tokens = _WORD_RE.findall(text)
    if re.fullmatch(r"\d+\s*[.)]?", text):
        return True
    if len(text) <= 20 and len(tokens) <= 3 and not re.search(r"[?=]", text):
        return True
    if len(text) <= 48 and re.search(r"(?:[=?]|[:#])\s*$", text):
        question_language = re.search(
            r"(?i)\b(?:what|which|find|solve|compute|determine|how)\b", text
        )
        return question_language is None
    return False


def _is_multi_part(text: str) -> bool:
    markers = _PART_MARKER_RE.findall(text)
    return len(markers) >= 2 or _MULTIPART_LANGUAGE_RE.search(text) is not None


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


__all__ = [
    "QualityAssessment",
    "QualityFlag",
    "assess_question",
    "assess_records",
    "canonical_math_text",
    "math_aware_fingerprint",
    "number_masked_template_fingerprint",
    "number_masked_template_text",
    "source_format_fingerprint",
    "source_format_text",
]

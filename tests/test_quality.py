from __future__ import annotations

from deep_challenge.data import MathRecord
from deep_challenge.quality import (
    QualityFlag,
    assess_question,
    assess_records,
    canonical_math_text,
    math_aware_fingerprint,
    number_masked_template_fingerprint,
    number_masked_template_text,
    source_format_fingerprint,
    source_format_text,
)


def test_quality_flags_cover_known_risk_signals_without_auto_drop() -> None:
    question = r"""
    (a) According to the graph shown above, find $x$.
    (b) See https://example.com/plot.png and compute the result.
    Solution: \boxed{42}
    [asy] size(100); draw((0,0)--(1,1)); [/asy]
    설명을 확인하세요.
    """

    assessment = assess_question(question)

    expected = {
        QualityFlag.ANSWER_OR_SOLUTION_CUE,
        QualityFlag.NUMERIC_BOXED,
        QualityFlag.URL_OR_IMAGE,
        QualityFlag.ASYMPTOTE,
        QualityFlag.MULTI_PART,
        QualityFlag.MULTILINGUAL,
        QualityFlag.NON_ASCII,
        QualityFlag.LATEX_PROXY,
    }
    assert expected <= assessment.flags
    assert assessment.auto_drop is False
    assert len(assessment.reasons) == len(assessment.flags)


def test_missing_visual_is_distinguished_from_embedded_asymptote() -> None:
    missing = assess_question("According to the graph shown above, find x.")
    embedded = assess_question(
        "According to the following graph, find x. [asy] size(100); draw((0,0)--(1,1)); [/asy]"
    )
    assert missing.has(QualityFlag.MISSING_VISUAL)
    assert not embedded.has(QualityFlag.MISSING_VISUAL)
    assert embedded.has(QualityFlag.ASYMPTOTE)


def test_fragment_detection_is_conservative() -> None:
    assert assess_question("8.").has(QualityFlag.FRAGMENT)
    assert assess_question('3. "?" =').has(QualityFlag.FRAGMENT)
    assert not assess_question("What is 2+2?").has(QualityFlag.FRAGMENT)


def test_non_ascii_does_not_necessarily_mean_multilingual() -> None:
    assessment = assess_question("Compute 3 × 4.")
    assert assessment.has(QualityFlag.NON_ASCII)
    assert not assessment.has(QualityFlag.MULTILINGUAL)


def test_math_fingerprint_canonicalizes_representational_latex_differences() -> None:
    left = r"Find $\left( \dfrac{ 1 }{ 2 } + x \right)$."
    right = r"find \( (\frac{1}{2}+x) \)."
    assert canonical_math_text(left) == canonical_math_text(right)
    assert math_aware_fingerprint(left) == math_aware_fingerprint(right)


def test_tex_command_normalization_respects_command_boundaries() -> None:
    assert r"\leftarrow" in canonical_math_text(r"A \leftarrow B")
    assert "arrow" not in canonical_math_text(r"A \leftarrow B").replace(r"\leftarrow", "")
    assert canonical_math_text(r"\left( x \right)") == canonical_math_text("(x)")
    assert r"\dfracx" in canonical_math_text(r"\dfracx")
    assert canonical_math_text(r"\tfrac12") == canonical_math_text(r"\frac12")


def test_number_masked_template_groups_numeric_variants_but_not_exact_fingerprint() -> None:
    first = "Alice has 1,200 marbles and gives away 25. How many remain?"
    second = "Alice has 4,500 marbles and gives away 30. How many remain?"

    assert math_aware_fingerprint(first) != math_aware_fingerprint(second)
    assert number_masked_template_fingerprint(first) == number_masked_template_fingerprint(second)
    assert number_masked_template_text(first).count("<num>") == 2


def test_number_masking_preserves_leading_sign() -> None:
    negative = number_masked_template_text("Solve x=-1.")
    positive = number_masked_template_text("Solve x=1.")
    explicit_positive = number_masked_template_text("Solve x=+1.")
    assert negative == "solve x=-<num>."
    assert positive == "solve x=<num>."
    assert explicit_positive == "solve x=+<num>."
    assert len({negative, positive, explicit_positive}) == 3


def test_source_format_fingerprint_ignores_numbering_and_trailing_hash() -> None:
    leaderboard = (
        "6. What is the maximum number of natural numbers not exceeding 2016 "
        "that can be marked?"
    )
    train = (
        "What is the maximum number of natural numbers not exceeding 2016 "
        "that can be marked?\n\n#"
    )
    assert source_format_text(leaderboard) == source_format_text(train)
    assert source_format_fingerprint(leaderboard) == source_format_fingerprint(train)


def test_source_format_fingerprint_handles_decimal_numbering_and_question_spacing() -> None:
    assert source_format_fingerprint("10.3. What is $2+3$ ?") == source_format_fingerprint(
        "9.4. What is $2 + 3$?"
    )
    assert source_format_fingerprint("7. If $a>0$, find $a$ ?") == source_format_fingerprint(
        "If $a>0$, find $a$?"
    )


def test_source_format_normalization_preserves_problem_numbers() -> None:
    assert source_format_text("What is 2016 + 7?") == "what is 2016+7?"
    assert source_format_fingerprint("What is 2+3?") != source_format_fingerprint(
        "What is 4+5?"
    )


def test_assess_records_preserves_membership() -> None:
    records = [
        MathRecord("a", "8.", "8.", "8", 8, 2),
        MathRecord("b", "What is 1+1?", "What is 1+1?", "2", 2, 3),
    ]
    report = assess_records(records)
    assert set(report) == {"a", "b"}
    assert report["a"].auto_drop is False
    assert report["b"].auto_drop is False

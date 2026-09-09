"""Assertion framework: exact, contains, predicate, semantic, LLM-judge."""

from __future__ import annotations

from pytracer_sdk import SpanStatus

from pytracer_replay import (
    Contains,
    ExactMatch,
    LLMJudge,
    NoErrors,
    Predicate,
    ReplayMode,
    ReplayResult,
    SemanticSimilarity,
    all_passed,
    run_assertions,
    token_jaccard,
)


def _ok(rv: object) -> ReplayResult:
    return ReplayResult(return_value=rv, status=SpanStatus.OK, mode=ReplayMode.FULL)


def _err() -> ReplayResult:
    return ReplayResult(
        return_value=None, status=SpanStatus.ERROR, mode=ReplayMode.FULL, error_type="ValueError"
    )


def test_exact_match() -> None:
    assert ExactMatch(42).check(_ok(42)).passed
    assert not ExactMatch(42).check(_ok(41)).passed


def test_contains() -> None:
    assert Contains("cat").check(_ok("the cat sat")).passed
    assert not Contains("dog").check(_ok("the cat sat")).passed


def test_no_errors() -> None:
    assert NoErrors().check(_ok("x")).passed
    assert not NoErrors().check(_err()).passed


def test_predicate_and_raising_predicate() -> None:
    assert Predicate(lambda r: r.return_value == "y").check(_ok("y")).passed

    def boom(_: ReplayResult) -> bool:
        raise RuntimeError("nope")

    res = Predicate(boom).check(_ok("y"))
    assert not res.passed
    assert "RuntimeError" in res.detail


def test_token_jaccard_and_semantic_similarity() -> None:
    assert token_jaccard("the cat sat", "the cat sat") == 1.0
    assert token_jaccard("the cat sat", "the cat sat on the mat") == 0.6
    assert SemanticSimilarity("the cat sat on the mat", threshold=0.5).check(_ok("the cat sat")).passed
    assert not SemanticSimilarity("the cat sat on the mat", threshold=0.7).check(_ok("the cat sat")).passed


def test_llm_judge_with_injected_judge() -> None:
    good = LLMJudge(rubric="is polite", judge=lambda out, rubric: 0.9)
    bad = LLMJudge(rubric="is polite", judge=lambda out, rubric: 0.1)
    assert good.check(_ok("hello there")).passed
    assert not bad.check(_ok("hello there")).passed


def test_run_assertions_and_all_passed() -> None:
    results = run_assertions(_ok("the cat"), [NoErrors(), Contains("cat"), ExactMatch("the cat")])
    assert all_passed(results)
    results2 = run_assertions(_ok("the cat"), [Contains("dog")])
    assert not all_passed(results2)

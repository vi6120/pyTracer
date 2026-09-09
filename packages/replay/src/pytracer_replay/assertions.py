"""Assertion framework for replay results.

Pluggable scorers judge a :class:`ReplayResult`. The built-ins cover the four
kinds the guide calls for - exact match, semantic similarity, LLM-as-judge, and
custom Python - plus a couple of conveniences. Semantic similarity and the LLM
judge take injectable backends so the core stays dependency-free; the semantic
default is a no-dependency token-overlap score.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from .engine import ReplayResult


@dataclass
class AssertionResult:
    """One assertion's verdict: its name, whether it passed, and why not."""

    name: str
    passed: bool
    detail: str = ""


class Assertion(Protocol):
    """A named scorer that judges a replay result via ``check``."""

    name: str

    def check(self, result: ReplayResult) -> AssertionResult: ...


def run_assertions(result: ReplayResult, assertions: list[Assertion]) -> list[AssertionResult]:
    """Evaluate every assertion against a replay result."""
    return [a.check(result) for a in assertions]


def all_passed(results: list[AssertionResult]) -> bool:
    """True if every assertion result passed."""
    return all(r.passed for r in results)


@dataclass
class NoErrors:
    """Passes when the replay completed without any error."""

    name: str = "no_errors"

    def check(self, result: ReplayResult) -> AssertionResult:
        return AssertionResult(self.name, not result.failed, result.error_type or "")


@dataclass
class ExactMatch:
    """Passes when the agent's return value equals ``expected``."""

    expected: object
    name: str = "exact_match"

    def check(self, result: ReplayResult) -> AssertionResult:
        ok = result.return_value == self.expected
        return AssertionResult(self.name, ok, "" if ok else f"got {result.return_value!r}")


@dataclass
class Contains:
    """Passes when ``needle`` is a substring of the return value's text form."""

    needle: str
    name: str = "contains"

    def check(self, result: ReplayResult) -> AssertionResult:
        ok = self.needle in str(result.return_value)
        return AssertionResult(self.name, ok, "" if ok else f"{self.needle!r} not found")


@dataclass
class Predicate:
    """Custom Python assertion: passes when ``fn(result)`` is truthy."""

    fn: Callable[[ReplayResult], bool]
    name: str = "predicate"

    def check(self, result: ReplayResult) -> AssertionResult:
        try:
            ok = bool(self.fn(result))
            return AssertionResult(self.name, ok, "" if ok else "predicate returned false")
        except Exception as exc:  # noqa: BLE001 - a raising predicate is a failed assertion
            return AssertionResult(self.name, False, f"predicate raised {type(exc).__name__}: {exc}")


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def token_jaccard(a: str, b: str) -> float:
    """Dependency-free similarity: Jaccard overlap of word tokens (0..1)."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


@dataclass
class SemanticSimilarity:
    """Passes when similarity(return_value, expected) >= threshold.

    ``similarity`` is injectable (e.g. an embedding cosine); the default is a
    no-dependency token-overlap score.
    """

    expected: str
    threshold: float = 0.7
    similarity: Callable[[str, str], float] = token_jaccard
    name: str = "semantic_similarity"

    def check(self, result: ReplayResult) -> AssertionResult:
        score = self.similarity(str(result.return_value), self.expected)
        ok = score >= self.threshold
        return AssertionResult(self.name, ok, f"score={score:.3f} threshold={self.threshold}")


@dataclass
class LLMJudge:
    """LLM-as-judge scoring via an injected ``judge`` callable.

    ``judge(output, rubric)`` returns a score in 0..1. No judge is bundled (that
    needs a model); provide one to use this assertion.
    """

    rubric: str
    judge: Callable[[str, str], float]
    threshold: float = 0.5
    name: str = "llm_judge"

    def check(self, result: ReplayResult) -> AssertionResult:
        score = self.judge(str(result.return_value), self.rubric)
        ok = score >= self.threshold
        return AssertionResult(self.name, ok, f"score={score:.3f} threshold={self.threshold}")

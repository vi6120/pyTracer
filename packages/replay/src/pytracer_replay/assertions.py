"""Assertion framework for replay results.

Pluggable scorers judge a :class:`ReplayResult`. The built-ins cover the four
kinds the guide calls for - exact match, semantic similarity, LLM-as-judge, and
custom Python - plus a couple of conveniences. Semantic similarity and the LLM
judge take injectable backends so the core stays dependency-free; the semantic
default is a no-dependency token-overlap score.
"""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from pytracer_sdk.schema import Span, SpanKind, SpanStatus

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
    """Passes when the replay completed without an error.

    ``scope="root"`` (the default) judges the run itself, so a tool error the agent
    caught and recovered from still passes, matching the run verdict. ``scope="any"``
    is stricter: it also fails on a handled error in any child span.
    """

    scope: str = "root"
    name: str = "no_errors"

    def check(self, result: ReplayResult) -> AssertionResult:
        errored = result.failed
        detail = result.error_type or ""
        if self.scope == "any":
            child = next((s for s in result.spans if s.status is SpanStatus.ERROR), None)
            if child is not None:
                errored = True
                detail = child.error_type or "error"
        return AssertionResult(self.name, not errored, detail)


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


# --- structural assertions (how the agent got there, not just the output) ------
# These judge the recorded span tree, so a regression that keeps the final answer
# but changes the path - an extra step, a skipped tool, a different route between
# agents - is caught. Output-only assertions miss exactly these.


def _agents_in_order(result: ReplayResult) -> list[str]:
    """Agent names in execution order (spans arrive sorted by start time)."""
    return [s.agent_name or s.name for s in result.spans if s.kind is SpanKind.AGENT]


def _tools_called(result: ReplayResult) -> list[str]:
    """Tool names in execution order."""
    return [s.tool_name or s.name for s in result.spans if s.kind is SpanKind.TOOL]


@dataclass
class Route:
    """Passes when the ordered sequence of agents visited equals ``expected``.

    Catches routing regressions, e.g. a run that loops back through a verifier a
    second time, even when the final answer is unchanged.
    """

    expected: list[str]
    name: str = "route"

    def check(self, result: ReplayResult) -> AssertionResult:
        actual = _agents_in_order(result)
        ok = actual == list(self.expected)
        return AssertionResult(
            self.name, ok, "" if ok else f"route was {actual}, expected {list(self.expected)}"
        )


@dataclass
class ToolsUsed:
    """Passes when the tools called include ``expected`` (or equal it if ``exact``)."""

    expected: list[str]
    exact: bool = False
    name: str = "tools_used"

    def check(self, result: ReplayResult) -> AssertionResult:
        actual = set(_tools_called(result))
        want = set(self.expected)
        if self.exact:
            ok = actual == want
            detail = "" if ok else f"tools were {sorted(actual)}, expected exactly {sorted(want)}"
        else:
            missing = want - actual
            ok = not missing
            detail = "" if ok else f"missing tools {sorted(missing)} (called {sorted(actual)})"
        return AssertionResult(self.name, ok, detail)


@dataclass
class StepCount:
    """Passes when the number of spans (optionally of ``kind``) equals ``expected``.

    Catches a change that adds or drops a step, e.g. an extra retry round.
    """

    expected: int
    kind: SpanKind | None = None
    name: str = "step_count"

    def check(self, result: ReplayResult) -> AssertionResult:
        spans = (
            result.spans if self.kind is None else [s for s in result.spans if s.kind is self.kind]
        )
        actual = len(spans)
        ok = actual == self.expected
        label = f"{self.kind.value} " if self.kind is not None else ""
        return AssertionResult(
            self.name, ok, "" if ok else f"{label}step count was {actual}, expected {self.expected}"
        )


# --- side-effect assertions (what the agent wrote, and how often) -------------
# These judge the write tool calls in the span tree, frozen or live, so a full
# replay checks them too: a fix that stops a duplicate refund is `fixed` only
# when the refund now happens once, not merely when no error is visible.


def _call_arguments(span: Span) -> dict[str, Any]:
    """A tool call's arguments by name, whichever adapter recorded it.

    ``@op`` records ``{"args": [...], "kwargs": {...}}`` (only keyword arguments
    have names); adapters record ``{"arguments": {...}}``, ``{"args": {...}}`` or
    ``{"input": ...}``.
    """
    data = span.input
    if not isinstance(data, dict):
        return {}
    for key in ("arguments", "args", "input"):
        value = data.get(key)
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                continue
        if isinstance(value, dict):
            return value
    kwargs = data.get("kwargs")
    return dict(kwargs) if isinstance(kwargs, dict) else {}


def _matches(actual: Any, expected: Any) -> bool:
    """``expected`` is contained in ``actual``: equal, or for mappings every
    expected key present with a matching value (recursively)."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        return all(k in actual and _matches(actual[k], v) for k, v in expected.items())
    return bool(actual == expected)


def _calls_of(result: ReplayResult, tool: str) -> list[Span]:
    return [s for s in result.spans if s.kind is SpanKind.TOOL and (s.tool_name or s.name) == tool]


def _describe(calls: list[Span]) -> str:
    """Each attempt and how it ended, the retry evidence behind a repeated write."""
    parts = []
    for number, span in enumerate(calls, 1):
        outcome = f"failed ({span.error_type or 'error'})" if span.status is SpanStatus.ERROR else "ok"
        retry = " retry" if "pytracer.retry_of" in span.attributes else ""
        parts.append(f"#{number}{retry} {outcome}")
    return ", ".join(parts)


def _label(tool: str, args: dict[str, Any] | None) -> str:
    if not args:
        return tool
    shown = ", ".join(f"{k}={v!r}" for k, v in args.items())
    return f"{tool}({shown})"


@dataclass
class WriteCall:
    """One expected write: the tool, optionally how many times, and a subset of its
    arguments (calls whose arguments do not contain it are not counted)."""

    tool: str
    times: int | None = None  # exact number of calls; None means at least once
    args: dict[str, Any] | None = None


@dataclass
class Writes:
    """Passes when each expected write call happened, exactly ``times`` when given.

    Every call counts, failed attempts too: a timed-out write that is retried may
    still have gone through. With ``only``, a declared write tool (``write_tools``,
    the scenario's ``writes``) that is not expected must not be called at all.
    """

    expected: list[WriteCall]
    write_tools: list[str] = field(default_factory=list)
    only: bool = False
    name: str = "writes"

    def check(self, result: ReplayResult) -> AssertionResult:
        problems = []
        for want in self.expected:
            calls = [s for s in _calls_of(result, want.tool) if _matches(_call_arguments(s), want.args or {})]
            label = _label(want.tool, want.args)
            if want.times is None and not calls:
                problems.append(f"{label} was not called")
            elif want.times is not None and len(calls) != want.times:
                detail = f": {_describe(calls)}" if calls else ""
                problems.append(f"{label} was called {len(calls)} times, expected {want.times}{detail}")
        if self.only:
            expected_tools = {w.tool for w in self.expected}
            for tool in sorted(set(self.write_tools) - expected_tools):
                calls = _calls_of(result, tool)
                if calls:
                    problems.append(f"{tool} was called {len(calls)} times, expected none")
        return AssertionResult(self.name, not problems, "; ".join(problems))


@dataclass
class NoDuplicateWrites:
    """Fails when a write tool was called more than once with the same arguments.

    The classic case is a retry after a timeout whose write had already gone
    through: the run looks fine, and the customer is refunded twice. ``tools`` are
    the write tools to watch (the scenario's ``writes``).
    """

    tools: list[str]
    name: str = "no_duplicate_writes"

    def check(self, result: ReplayResult) -> AssertionResult:
        problems = []
        for tool in self.tools:
            groups: dict[str, list[Span]] = {}
            for span in _calls_of(result, tool):
                key = json.dumps(_call_arguments(span), sort_keys=True, default=repr)
                groups.setdefault(key, []).append(span)
            for calls in groups.values():
                if len(calls) > 1:
                    label = _label(tool, _call_arguments(calls[0]))
                    problems.append(f"{label} was called {len(calls)} times: {_describe(calls)}")
        return AssertionResult(self.name, not problems, "; ".join(problems))


def _plain(value: Any) -> Any:
    """A return value as plain data, so its fields can be matched."""
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return dump(mode="json")
        except Exception:  # noqa: BLE001 - fall back to the value itself
            return value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    return value


@dataclass
class Returns:
    """Passes when the agent's return value contains ``expected``.

    For a mapping (or a pydantic model or dataclass, by its fields) every expected
    key must be present with a matching value, recursively; anything else must be
    equal. Use it to check the state the agent reports, such as ``{"status":
    "refunded"}``, without pinning every other field.
    """

    expected: Any
    name: str = "returns"

    def check(self, result: ReplayResult) -> AssertionResult:
        actual = _plain(result.return_value)
        ok = _matches(actual, self.expected)
        return AssertionResult(self.name, ok, "" if ok else f"returned {actual!r}")

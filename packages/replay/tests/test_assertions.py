"""Assertion framework: exact, contains, predicate, semantic, LLM-judge."""

from __future__ import annotations

from pytracer_sdk import SpanKind, SpanStatus, new_span_id, new_trace_id
from pytracer_sdk.schema import Span

from pytracer_replay import (
    Contains,
    ExactMatch,
    LLMJudge,
    NoErrors,
    Predicate,
    ReplayMode,
    ReplayResult,
    Route,
    SemanticSimilarity,
    StepCount,
    ToolsUsed,
    all_passed,
    run_assertions,
    token_jaccard,
)

_TRACE = new_trace_id()


def _span(
    kind: SpanKind,
    *,
    agent_name: str | None = None,
    tool_name: str | None = None,
    name: str = "op",
    t: int = 1,
) -> Span:
    return Span(
        trace_id=_TRACE,
        span_id=new_span_id(),
        name=name,
        kind=kind,
        agent_name=agent_name,
        tool_name=tool_name,
        start_time_ns=t,
        end_time_ns=t + 1,
    )


def _with_spans(spans: list[Span]) -> ReplayResult:
    # The engine hands spans in execution order; mirror that here.
    return ReplayResult(return_value="ok", status=SpanStatus.OK, mode=ReplayMode.FULL, spans=spans)


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


def test_no_errors_scope_root_passes_a_handled_child_error_but_any_flags_it() -> None:
    # The run is OK (agent recovered), but a child span carries a handled error (V3).
    tool = Span(
        trace_id=_TRACE,
        span_id=new_span_id(),
        name="query",
        kind=SpanKind.TOOL,
        status=SpanStatus.ERROR,
        error_type="ServiceUnavailable",
        start_time_ns=1,
        end_time_ns=2,
    )
    result = ReplayResult(
        return_value="fallback", status=SpanStatus.OK, mode=ReplayMode.FULL, spans=[tool]
    )
    assert NoErrors().check(result).passed  # scope="root" (default)
    flagged = NoErrors(scope="any").check(result)
    assert not flagged.passed
    assert flagged.detail == "ServiceUnavailable"


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


def test_route_checks_ordered_agents() -> None:
    r = _with_spans(
        [
            _span(SpanKind.AGENT, agent_name="orchestrator", t=1),
            _span(SpanKind.AGENT, agent_name="library", t=2),
            _span(SpanKind.AGENT, agent_name="verifier", t=3),
        ]
    )
    assert Route(["orchestrator", "library", "verifier"]).check(r).passed
    # A run that looped back through the verifier a second time would not match a
    # two-step expected route; output-only assertions miss this.
    bad = Route(["orchestrator", "library"]).check(r)
    assert not bad.passed
    assert "verifier" in bad.detail


def test_tools_used_subset_and_exact() -> None:
    r = _with_spans(
        [
            _span(SpanKind.TOOL, tool_name="search", t=1),
            _span(SpanKind.TOOL, tool_name="fetch", t=2),
        ]
    )
    assert ToolsUsed(["search"]).check(r).passed  # subset: the listed tool was used
    assert not ToolsUsed(["search", "web"]).check(r).passed  # a required tool is missing
    assert ToolsUsed(["search", "fetch"], exact=True).check(r).passed
    assert not ToolsUsed(["search"], exact=True).check(r).passed  # exact rejects the extra tool


def test_step_count_total_and_by_kind() -> None:
    r = _with_spans(
        [
            _span(SpanKind.AGENT, agent_name="a", t=1),
            _span(SpanKind.TOOL, tool_name="t", t=2),
            _span(SpanKind.LLM, name="m", t=3),
        ]
    )
    assert StepCount(3).check(r).passed
    assert StepCount(1, kind=SpanKind.TOOL).check(r).passed
    bad = StepCount(2, kind=SpanKind.AGENT).check(r)
    assert not bad.passed
    assert "agent step count was 1" in bad.detail

"""Replay engine: modes, recorded-response injection, determinism, strictness."""

from __future__ import annotations

import pytracer_sdk as pytracer
from pytracer_sdk import SpanKind, SpanStatus

from pytracer_replay import MockSet, ReplayEngine, ReplayMode


def _record(agent: object) -> object:
    """Run an agent live and capture its trace as a recording."""
    return ReplayEngine(ReplayMode.LIVE).replay(agent, MockSet())  # type: ignore[arg-type]


def test_full_replay_injects_recorded_and_skips_live_body() -> None:
    calls = {"n": 0}

    @pytracer.op(kind=SpanKind.TOOL, tool_name="flaky")
    def flaky(x: int) -> dict[str, int]:
        calls["n"] += 1
        return {"val": x, "call": calls["n"]}

    @pytracer.op(kind=SpanKind.AGENT, agent_name="a")
    def agent() -> dict[str, int]:
        return flaky(7)

    rec = _record(agent)
    assert calls["n"] == 1
    mocks = MockSet.from_trace(rec.spans)  # type: ignore[attr-defined]

    res = ReplayEngine(ReplayMode.FULL).replay(agent, mocks)
    assert calls["n"] == 1  # live tool body was NOT executed again
    assert res.return_value == rec.return_value  # type: ignore[attr-defined]
    assert res.status is SpanStatus.OK

    # Determinism: replaying again yields the same result and still no live call.
    res2 = ReplayEngine(ReplayMode.FULL).replay(agent, mocks)
    assert res2.return_value == rec.return_value  # type: ignore[attr-defined]
    assert calls["n"] == 1


def test_partial_mode_mocks_tools_but_runs_model() -> None:
    tool_calls = {"n": 0}
    model_calls = {"n": 0}

    @pytracer.op(kind=SpanKind.TOOL, tool_name="fetch")
    def fetch() -> str:
        tool_calls["n"] += 1
        return "data"

    @pytracer.op(kind=SpanKind.LLM)
    def model(x: str) -> str:
        model_calls["n"] += 1
        return "resp:" + x

    @pytracer.op(kind=SpanKind.AGENT)
    def agent() -> str:
        return model(fetch())

    rec = _record(agent)
    assert tool_calls["n"] == 1 and model_calls["n"] == 1
    mocks = MockSet.from_trace(rec.spans)  # type: ignore[attr-defined]

    res = ReplayEngine(ReplayMode.PARTIAL).replay(agent, mocks)
    assert tool_calls["n"] == 1  # tool mocked, not run again
    assert model_calls["n"] == 2  # model ran live
    assert res.return_value == "resp:data"


def test_live_mode_runs_everything() -> None:
    calls = {"n": 0}

    @pytracer.op(kind=SpanKind.TOOL, tool_name="t")
    def t() -> int:
        calls["n"] += 1
        return calls["n"]

    @pytracer.op(kind=SpanKind.AGENT)
    def agent() -> int:
        return t()

    r1 = ReplayEngine(ReplayMode.LIVE).replay(agent, MockSet())
    r2 = ReplayEngine(ReplayMode.LIVE).replay(agent, MockSet())
    assert r1.return_value == 1
    assert r2.return_value == 2


def test_strict_missing_mock_becomes_error_result() -> None:
    @pytracer.op(kind=SpanKind.TOOL, tool_name="x")
    def x() -> str:
        return "live"

    @pytracer.op(kind=SpanKind.AGENT)
    def agent() -> str:
        return x()

    res = ReplayEngine(ReplayMode.FULL, strict=True).replay(agent, MockSet())
    assert res.failed
    assert res.error_type == "MissingMockError"


def test_agent_exception_becomes_error_result() -> None:
    @pytracer.op(kind=SpanKind.AGENT)
    def agent() -> None:
        raise ValueError("boom")

    res = ReplayEngine(ReplayMode.FULL).replay(agent, MockSet())
    assert res.failed
    assert res.error_type == "ValueError"


def test_replayed_spans_are_marked() -> None:
    @pytracer.op(kind=SpanKind.TOOL, tool_name="s")
    def s() -> str:
        return "real"

    @pytracer.op(kind=SpanKind.AGENT)
    def agent() -> str:
        return s()

    rec = _record(agent)
    mocks = MockSet.from_trace(rec.spans)  # type: ignore[attr-defined]
    res = ReplayEngine(ReplayMode.FULL).replay(agent, mocks)
    tool_span = next(sp for sp in res.spans if sp.tool_name == "s")
    assert tool_span.attributes.get("pytracer.replayed") is True

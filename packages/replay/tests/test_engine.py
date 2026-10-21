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


def test_fuzzy_match_injects_recorded_output_on_drifted_input() -> None:
    calls = {"n": 0}
    state = {"x": 1}

    @pytracer.op(kind=SpanKind.TOOL, tool_name="t")
    def t(x: int) -> str:
        calls["n"] += 1
        return f"out-{x}"

    @pytracer.op(kind=SpanKind.AGENT, agent_name="a")
    def agent() -> str:
        return t(state["x"])

    rec = _record(agent)  # records t(1) -> "out-1"
    assert calls["n"] == 1
    mocks = MockSet.from_trace(rec.spans)  # type: ignore[attr-defined]

    state["x"] = 2  # the op input drifts between capture and replay

    # Exact match misses the drifted input, so the tool body runs live.
    exact = ReplayEngine(ReplayMode.FULL).replay(agent, mocks)
    assert calls["n"] == 2
    assert exact.return_value == "out-2"

    # Fuzzy falls back to the identifier's sole recording: no live call, recorded output.
    fuzzy = ReplayEngine(ReplayMode.FULL, fuzzy_match=True).replay(agent, mocks)
    assert calls["n"] == 2  # tool body was NOT executed again
    assert fuzzy.return_value == "out-1"


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


def _two_op_agent(tool_calls: dict[str, int], model_calls: dict[str, int]) -> object:
    @pytracer.op(kind=SpanKind.TOOL, tool_name="search")
    def search(q: str) -> str:
        tool_calls["n"] += 1
        return f"hits-{tool_calls['n']}"

    @pytracer.op(kind=SpanKind.LLM)
    def model(text: str) -> str:
        model_calls["n"] += 1
        return f"answer-{model_calls['n']}"

    @pytracer.op(kind=SpanKind.AGENT, agent_name="a")
    def agent() -> str:
        return model(search("q"))

    return agent


def test_live_ops_runs_named_op_live_while_freezing_the_rest() -> None:
    tool_calls, model_calls = {"n": 0}, {"n": 0}
    agent = _two_op_agent(tool_calls, model_calls)
    rec = _record(agent)
    assert (tool_calls["n"], model_calls["n"]) == (1, 1)
    mocks = MockSet.from_trace(rec.spans)  # type: ignore[attr-defined]

    # Full mode freezes everything; live_ops punches a hole for `model` only.
    res = ReplayEngine(ReplayMode.FULL, live_ops=["model"]).replay(agent, mocks)
    assert tool_calls["n"] == 1  # search stayed frozen (served from the recording)
    assert model_calls["n"] == 2  # model ran live
    assert res.return_value == "answer-2"


def test_frozen_ops_serves_named_op_from_recording_in_live_mode() -> None:
    tool_calls, model_calls = {"n": 0}, {"n": 0}
    agent = _two_op_agent(tool_calls, model_calls)
    rec = _record(agent)
    mocks = MockSet.from_trace(rec.spans)  # type: ignore[attr-defined]

    # Live mode runs everything; frozen_ops freezes just `search`.
    res = ReplayEngine(ReplayMode.LIVE, frozen_ops=["search"]).replay(agent, mocks)
    assert tool_calls["n"] == 1  # search frozen (served from the recording, no new call)
    assert model_calls["n"] == 2  # model ran live
    assert res.return_value == "answer-2"


def test_recorded_failure_replays_as_the_same_failure() -> None:
    @pytracer.op(kind=SpanKind.TOOL, tool_name="boom")
    def boom(x: int) -> int:
        raise ValueError("kaboom")

    @pytracer.op(kind=SpanKind.AGENT, agent_name="a")
    def agent() -> int:
        return boom(1)

    rec = _record(agent)  # LIVE: boom raises, captured as an ERROR span
    assert rec.status is SpanStatus.ERROR  # type: ignore[attr-defined]
    assert rec.error_type == "ValueError"  # type: ignore[attr-defined]
    mocks = MockSet.from_trace(rec.spans)  # type: ignore[attr-defined]

    # Full replay must re-raise the recorded failure, not serve it as a success.
    res = ReplayEngine(ReplayMode.FULL).replay(agent, mocks)
    assert res.status is SpanStatus.ERROR
    assert res.error_type == "ValueError"


def test_recorded_failure_is_catchable_by_its_builtin_type() -> None:
    @pytracer.op(kind=SpanKind.TOOL, tool_name="boom")
    def boom(x: int) -> str:
        raise KeyError("missing")

    @pytracer.op(kind=SpanKind.AGENT, agent_name="a")
    def agent() -> str:
        try:
            return str(boom(1))
        except KeyError:
            return "handled"

    rec = _record(agent)  # boom raises KeyError; the agent catches it -> "handled"
    assert rec.return_value == "handled"  # type: ignore[attr-defined]
    mocks = MockSet.from_trace(rec.spans)  # type: ignore[attr-defined]

    # The recorded KeyError is re-raised as a real KeyError, so the agent's
    # `except KeyError` path replays identically (not str(None) == "None"), and the
    # replay matches the recording.
    res = ReplayEngine(ReplayMode.FULL).replay(agent, mocks)
    assert res.return_value == "handled"
    assert res.status is rec.status  # type: ignore[attr-defined]
    assert res.error_type == rec.error_type  # type: ignore[attr-defined]

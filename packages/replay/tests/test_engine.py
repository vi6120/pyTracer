"""Replay engine: modes, recorded-response injection, determinism, strictness."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytracer_sdk as pytracer
from pytracer_sdk import SpanKind, SpanStatus
from pytracer_sdk.schema import Span

from pytracer_replay import MockSet, ReplayEngine, ReplayMode


def _record(agent: object) -> object:
    """Run an agent live and capture its trace as a recording."""
    return ReplayEngine(ReplayMode.LIVE).replay(agent, MockSet())  # type: ignore[arg-type]


class DbUnavailable(Exception):
    """A non-builtin exception (module-level so it is importable), to test that
    replay reconstructs the real type and the agent's own except fires (R4)."""


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
    state = {"q": "summarize the quarterly earnings report"}

    @pytracer.op(kind=SpanKind.TOOL, tool_name="t")
    def t(q: str) -> str:
        calls["n"] += 1
        return "recorded-output"

    @pytracer.op(kind=SpanKind.AGENT, agent_name="a")
    def agent() -> str:
        return t(state["q"])

    rec = _record(agent)
    assert calls["n"] == 1
    mocks = MockSet.from_trace(rec.spans)  # type: ignore[attr-defined]

    # Small drift: one word changes, so the input stays highly similar.
    state["q"] = "summarize the quarterly earnings summary"

    # Exact match misses the drifted input, so the tool body runs live.
    ReplayEngine(ReplayMode.FULL).replay(agent, mocks)
    assert calls["n"] == 2
    # Fuzzy falls back to the near recording: no live call, recorded output.
    fuzzy = ReplayEngine(ReplayMode.FULL, fuzzy_match=True).replay(agent, mocks)
    assert calls["n"] == 2  # tool body was NOT executed again
    assert fuzzy.return_value == "recorded-output"


def test_partial_mode_positional_matches_drifted_tool_input() -> None:
    # In partial mode the model runs live and words a tool's input differently;
    # the frozen tool still matches by call order, and the match is marked.
    tool_calls = {"n": 0}

    @pytracer.op(kind=SpanKind.TOOL, tool_name="search")
    def search(q: str) -> str:
        tool_calls["n"] += 1
        return f"hits for {q}"

    @pytracer.op(kind=SpanKind.AGENT, agent_name="a")
    def agent(q: str) -> str:
        return search(q)

    rec = ReplayEngine(ReplayMode.LIVE).replay(lambda: agent("recorded wording"), MockSet())
    assert tool_calls["n"] == 1
    mocks = MockSet.from_trace(rec.spans)  # type: ignore[attr-defined]

    # Tool input drifts, but positional matching serves the recording (no live call).
    res = ReplayEngine(ReplayMode.PARTIAL).replay(lambda: agent("different wording now"), mocks)
    assert tool_calls["n"] == 1
    assert res.return_value == "hits for recorded wording"
    tool_span = next(s for s in res.spans if s.tool_name == "search")
    assert tool_span.attributes.get("pytracer.mock_match") == "positional"


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


def test_recorded_non_builtin_failure_is_catchable_by_its_real_type() -> None:
    # A non-builtin exception (e.g. a driver's error) is reconstructed from its
    # module on replay, so the agent's `except ThatError` path fires (R4).
    @pytracer.op(kind=SpanKind.TOOL, tool_name="query")
    def query(x: int) -> str:
        raise DbUnavailable("db down")

    @pytracer.op(kind=SpanKind.AGENT, agent_name="a")
    def agent() -> str:
        try:
            return str(query(1))
        except DbUnavailable:
            return "handled"

    rec = _record(agent)
    assert rec.return_value == "handled"  # type: ignore[attr-defined]
    mocks = MockSet.from_trace(rec.spans)  # type: ignore[attr-defined]

    res = ReplayEngine(ReplayMode.FULL).replay(agent, mocks)
    assert res.return_value == "handled"  # the real DbUnavailable was re-raised


def test_a_handled_tool_error_leaves_the_run_passing() -> None:
    # V3: the agent catches a tool failure and returns a fallback, so the run's
    # verdict is OK (the app survived the outage). The child error is still on its
    # span, left to a scope="any" assertion if a test wants to flag it.
    @pytracer.op(kind=SpanKind.TOOL, tool_name="db")
    def db() -> str:
        raise DbUnavailable("db down")

    @pytracer.op(kind=SpanKind.AGENT, agent_name="a")
    def agent() -> str:
        try:
            return db()
        except DbUnavailable:
            return "fallback"

    rec = _record(agent)
    assert rec.return_value == "fallback"  # type: ignore[attr-defined]
    assert not rec.failed  # type: ignore[attr-defined]  # the handled error does not fail the run
    assert any(  # ...but the child error is still recorded, not hidden
        s.status is SpanStatus.ERROR and s.tool_name == "db"
        for s in rec.spans  # type: ignore[attr-defined]
    )

    res = ReplayEngine(ReplayMode.FULL).replay(agent, MockSet.from_trace(rec.spans))  # type: ignore[attr-defined]
    assert res.return_value == "fallback"
    assert not res.failed


def test_decimal_output_replays_as_decimal_from_a_saved_recording() -> None:
    # A tool returning a Decimal, written to a recording and loaded back, must hand
    # the agent a Decimal on replay: served as the string "19.99", price * qty would
    # silently repeat the text instead of multiplying.
    @pytracer.op(kind=SpanKind.TOOL, tool_name="get_item")
    def get_item(item_id: str) -> dict[str, Any]:
        return {"price": Decimal("19.99"), "qty": 2}

    @pytracer.op(kind=SpanKind.AGENT, agent_name="a")
    def agent() -> object:
        item = get_item("x")
        return item["price"] * item["qty"]

    rec = _record(agent)
    reloaded = [Span.from_json(s.to_json()) for s in rec.spans]  # type: ignore[attr-defined]
    res = ReplayEngine(ReplayMode.FULL).replay(agent, MockSet.from_trace(reloaded))
    assert res.return_value == Decimal("39.98")


def test_uninstrumented_replay_is_flagged() -> None:
    # Recordings exist, but the agent makes no @op calls (stands in for an
    # adapter-captured run): nothing is injected and the result is flagged.
    @pytracer.op(kind=SpanKind.TOOL, tool_name="t")
    def t(x: int) -> int:
        return x

    @pytracer.op(kind=SpanKind.AGENT, agent_name="a")
    def agent() -> int:
        return t(1)

    rec = _record(agent)
    mocks = MockSet.from_trace(rec.spans)  # type: ignore[attr-defined]
    assert len(mocks) > 0

    flagged = ReplayEngine(ReplayMode.FULL).replay(lambda: 42, mocks)
    assert flagged.uninstrumented is True

    ok = ReplayEngine(ReplayMode.FULL).replay(agent, mocks)
    assert ok.uninstrumented is False


# --- X4: a changed model input (a prompt edit) ---------------------------------

def _changed_prompt_agent() -> object:
    @pytracer.op(kind=SpanKind.LLM)
    def call_model(prompt: str) -> str:
        return f"live:{prompt}"

    @pytracer.op(kind=SpanKind.AGENT, agent_name="a")
    def agent() -> str:
        return call_model("a NEW prompt not in the recording")

    return agent


def _mocks_with_old_prompt() -> MockSet:
    mocks = MockSet()
    mocks.add("call_model", "the OLD recorded prompt", "recorded answer")
    return mocks


def test_full_mode_strict_fails_on_a_changed_model_input_and_records_it() -> None:
    # A frozen model whose input changed can't be judged: strict fails, and the op
    # is recorded as a changed model input so the verdict can be explained (X4).
    result = ReplayEngine(ReplayMode.FULL, strict=True).replay(
        _changed_prompt_agent(), _mocks_with_old_prompt()
    )
    assert result.failed
    assert result.error_type == "MissingMockError"
    assert "call_model" in result.changed_models


def test_evaluate_changed_prompts_runs_the_model_live_instead_of_failing() -> None:
    # With the opt-in flag, the changed model op runs live so the edit is evaluated.
    result = ReplayEngine(
        ReplayMode.FULL, strict=True, evaluate_changed_prompts=True
    ).replay(_changed_prompt_agent(), _mocks_with_old_prompt())
    assert result.return_value == "live:a NEW prompt not in the recording"
    assert "call_model" in result.changed_models
    assert not result.failed

# --- R7: concurrent same-input ordering ---------------------------------------

def _differing_recordings() -> tuple[Any, MockSet]:
    """A fetch op and mocks where fetch(1) has two DIFFERENT recorded responses."""
    n = {"i": 0}

    @pytracer.op(kind=SpanKind.TOOL, tool_name="fetch")
    def fetch(x: int) -> str:
        n["i"] += 1
        return f"r{n['i']}"

    @pytracer.op(kind=SpanKind.AGENT, agent_name="cap")
    def capture() -> str:
        return fetch(1) + fetch(1)  # same input, two differing recordings

    mocks = MockSet.from_trace(_record(capture).spans)  # type: ignore[attr-defined]
    return fetch, mocks


def test_sequential_same_input_calls_are_not_flagged() -> None:
    # Sequential (and single-threaded async) calls arrive in issue order, so they
    # replay in order; no ambiguity (A).
    fetch, mocks = _differing_recordings()

    @pytracer.op(kind=SpanKind.AGENT, agent_name="a")
    def agent() -> str:
        return fetch(1) + fetch(1)

    result = ReplayEngine(ReplayMode.FULL).replay(agent, mocks)
    assert result.return_value == "r1r2"  # served in order, not live
    assert result.concurrency_ambiguities == []


def test_concurrent_same_input_differing_recordings_are_flagged() -> None:
    # Same input on two OS threads with differing recordings cannot be ordered from
    # arrival; the ambiguity is surfaced, not silently swapped (B).
    import threading

    fetch, mocks = _differing_recordings()

    @pytracer.op(kind=SpanKind.AGENT, agent_name="a")
    def agent() -> str:
        out: list[str] = []
        threads = [threading.Thread(target=lambda: out.append(fetch(1))) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return ",".join(sorted(out))

    result = ReplayEngine(ReplayMode.FULL).replay(agent, mocks)
    assert result.return_value == "r1,r2"  # both served from the recording (not live)
    assert "fetch" in result.concurrency_ambiguities


def test_concurrent_same_input_identical_recordings_not_flagged() -> None:
    # Identical recordings make order irrelevant, so concurrency is not flagged (A).
    import threading

    @pytracer.op(kind=SpanKind.TOOL, tool_name="fetch")
    def fetch(x: int) -> str:
        return "same"

    @pytracer.op(kind=SpanKind.AGENT, agent_name="cap")
    def capture() -> str:
        return fetch(1) + fetch(1)

    mocks = MockSet.from_trace(_record(capture).spans)  # type: ignore[attr-defined]

    @pytracer.op(kind=SpanKind.AGENT, agent_name="a")
    def agent() -> str:
        out: list[str] = []
        threads = [threading.Thread(target=lambda: out.append(fetch(1))) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return ",".join(out)

    result = ReplayEngine(ReplayMode.FULL).replay(agent, mocks)
    assert result.concurrency_ambiguities == []


def _per_worker_recordings() -> tuple[Any, Any, MockSet]:
    """worker(order) calls fetch(1); each worker's fetch(1) recorded a different
    response (r1 under worker "a", r2 under worker "b")."""
    import threading

    n = {"i": 0}

    @pytracer.op(kind=SpanKind.TOOL, tool_name="fetch")
    def fetch(x: int) -> str:
        n["i"] += 1
        return f"r{n['i']}"

    @pytracer.op(kind=SpanKind.AGENT, agent_name="worker")
    def worker(order: str) -> str:
        return f"{order}:{fetch(1)}"

    @pytracer.op(kind=SpanKind.AGENT, agent_name="cap")
    def capture() -> str:
        out: list[str] = []
        for order in ("a", "b"):  # a records first, then b
            t = threading.Thread(target=lambda o=order: out.append(worker(o)))
            t.start()
            t.join()
        return ",".join(out)

    mocks = MockSet.from_trace(_record(capture).spans)  # type: ignore[attr-defined]
    return fetch, worker, mocks


def test_concurrent_same_input_calls_resolve_by_their_caller() -> None:
    # R7: the same input on different threads, but from different callers
    # (worker("a") and worker("b")). Even when the threads arrive in the opposite
    # order to the recording, each call gets the response recorded under its own
    # caller, and nothing is flagged.
    import threading

    _, worker, mocks = _per_worker_recordings()

    @pytracer.op(kind=SpanKind.AGENT, agent_name="a")
    def agent() -> str:
        out: list[str] = []
        for order in ("b", "a"):  # reversed arrival order
            t = threading.Thread(target=lambda o=order: out.append(worker(o)))
            t.start()
            t.join()
        return ",".join(sorted(out))

    result = ReplayEngine(ReplayMode.FULL).replay(agent, mocks)
    assert result.return_value == "a:r1,b:r2"  # not swapped to a:r2,b:r1
    assert result.concurrency_ambiguities == []


def test_concurrent_same_input_calls_under_the_same_caller_are_still_flagged() -> None:
    # An identical code path (same caller with the same input on two threads)
    # cannot be told apart, so differing recordings are still surfaced.
    import threading

    n = {"i": 0}

    @pytracer.op(kind=SpanKind.TOOL, tool_name="fetch")
    def fetch(x: int) -> str:
        n["i"] += 1
        return f"r{n['i']}"

    @pytracer.op(kind=SpanKind.AGENT, agent_name="worker")
    def worker(order: str) -> str:
        return fetch(1)

    @pytracer.op(kind=SpanKind.AGENT, agent_name="cap")
    def capture() -> str:
        return worker("a") + worker("a")

    mocks = MockSet.from_trace(_record(capture).spans)  # type: ignore[attr-defined]

    @pytracer.op(kind=SpanKind.AGENT, agent_name="a")
    def agent() -> str:
        out: list[str] = []
        threads = [threading.Thread(target=lambda: out.append(worker("a"))) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return ",".join(sorted(out))

    result = ReplayEngine(ReplayMode.FULL).replay(agent, mocks)
    assert result.return_value == "r1,r2"  # both served from the recording
    assert "fetch" in result.concurrency_ambiguities


def test_caller_scoped_mocks_fall_back_without_a_caller_op() -> None:
    # A replay whose fetch has no calling op (or a caller the recording never saw)
    # still resolves by input and order, as before.
    fetch, _, mocks = _per_worker_recordings()

    @pytracer.op(kind=SpanKind.AGENT, agent_name="other")
    def agent() -> str:
        return fetch(1) + fetch(1)

    result = ReplayEngine(ReplayMode.FULL).replay(agent, mocks)
    assert result.return_value == "r1r2"


# --- C0: framework calls routed through a seam replay without @op --------------


def test_a_seam_routed_framework_call_replays_without_running_live() -> None:
    from pytracer_sdk.seam import call_through

    live = {"model": 0, "tool": 0}

    # Stand-ins for a framework's model and tool, which the user never decorates.
    def framework_model(messages: list[str], request_id: str) -> str:
        live["model"] += 1
        return "call search"

    def framework_tool(query: str) -> str:
        live["tool"] += 1
        return f"results for {query}"

    def agent() -> str:
        plan = call_through(
            framework_model, (["find cats"], f"req-{live['model']}"),
            kind=SpanKind.LLM, name="fake-model", input={"messages": ["find cats"]},
        )
        found = call_through(
            framework_tool, ("cats",), kind=SpanKind.TOOL, name="search", tool_name="search",
            input={"query": "cats"},
        )
        return f"{plan} -> {found}"

    recorded = ReplayEngine(ReplayMode.LIVE).replay(agent, MockSet())
    assert recorded.return_value == "call search -> results for cats"
    assert live == {"model": 1, "tool": 1}

    replayed = ReplayEngine(ReplayMode.FULL).replay(agent, MockSet.from_trace(recorded.spans))
    assert replayed.return_value == recorded.return_value
    assert live == {"model": 1, "tool": 1}  # nothing ran live: both came from the recording
    assert replayed.uninstrumented is False


def test_recorded_exception_from_an_unimported_package_is_not_imported() -> None:
    # A recording names the exception's module; replay only looks in packages the
    # process already imported, so it falls back instead of importing a new one.
    import sys

    from pytracer_replay.engine import ReplayedError, _rebuild_exception
    from pytracer_replay.mocks import RecordedError

    sys.modules.pop("this", None)
    exc = _rebuild_exception(RecordedError("Boom", "failed", error_module="this"))
    assert type(exc) is ReplayedError
    assert "this" not in sys.modules
    # A class from an imported module is still rebuilt as its real type.
    exc = _rebuild_exception(
        RecordedError("ImportedModuleError", "failed", error_module=__name__)
    )
    assert type(exc) is ImportedModuleError


class ImportedModuleError(Exception):
    """A non-builtin exception in a module the process has imported (this one)."""


def test_strict_miss_fails_the_replay_even_when_the_agent_catches_it() -> None:
    # A framework that swallows a handler's exception (as AutoGen's runtime does)
    # must not turn a strict miss into a passing replay with a truncated result.
    @pytracer.op(kind=SpanKind.TOOL, tool_name="lookup")
    def lookup(q: str) -> str:
        return f"found {q}"

    def make_agent(query: str) -> Any:
        @pytracer.op(kind=SpanKind.AGENT, agent_name="a")
        def agent() -> str:
            try:
                return lookup(query)
            except Exception:  # noqa: BLE001 - the framework's catch-all
                return "gave up"

        return agent

    rec = _record(make_agent("x"))
    mocks = MockSet.from_trace(rec.spans)  # type: ignore[attr-defined]
    res = ReplayEngine(ReplayMode.FULL, strict=True).replay(make_agent("changed"), mocks)
    assert res.return_value == "gave up"
    assert res.failed
    assert res.error_type == "MissingMockError" and "no recording" in (res.error_message or "")

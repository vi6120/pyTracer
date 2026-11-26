"""Side-effect assertions: write calls, duplicate writes on retry, and return values.

The end-to-end cases capture an agent that retries a refund after a timeout (the
timed-out refund had gone through, so the customer is refunded twice), replay it
in full mode with every tool frozen, and check that the write assertions see the
duplicate, with the failed first attempt as evidence. A fixed agent that refunds
once passes against the same recording.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pytracer_sdk import SpanKind, SpanStatus, Tracer, new_span_id, new_trace_id
from pytracer_sdk.schema import Resource, Span
from pytracer_sdk.transport import InMemoryTransport

from pytracer_replay import (
    MockSet,
    NoDuplicateWrites,
    ReplayEngine,
    ReplayMode,
    ReplayResult,
    Returns,
    WriteCall,
    Writes,
)

LIVE = {"refunds": 0}


def _agents(tracer: Tracer) -> tuple[Any, Any]:
    timeouts = [TimeoutError("payment gateway timed out")]

    @tracer.op(kind=SpanKind.TOOL, tool_name="refund_payment")
    def refund_payment(order_id: str, amount: int) -> dict[str, str]:
        LIVE["refunds"] += 1  # the refund goes through, even when the reply times out
        if timeouts:
            raise timeouts.pop()
        return {"status": "refunded"}

    @tracer.op(kind=SpanKind.TOOL, tool_name="lookup_order")
    def lookup_order(order_id: str) -> dict[str, Any]:
        return {"order_id": order_id, "amount": 30}

    def retrying_agent() -> dict[str, Any]:
        order = lookup_order(order_id="A1")
        for _ in range(3):
            try:
                refund_payment(order_id="A1", amount=order["amount"])
                return {"order_id": "A1", "status": "refunded"}
            except TimeoutError:
                continue
        return {"order_id": "A1", "status": "failed"}

    def careful_agent() -> dict[str, Any]:
        order = lookup_order(order_id="A1")
        try:
            refund_payment(order_id="A1", amount=order["amount"])
            return {"order_id": "A1", "status": "refunded"}
        except TimeoutError:
            return {"order_id": "A1", "status": "pending"}  # check before refunding again

    return retrying_agent, careful_agent


def _recording() -> tuple[MockSet, Any, Any]:
    sink = InMemoryTransport()
    tracer = Tracer(sink, resource=Resource(service_name="refunds", branch="main", commit="c0"))
    retrying, careful = _agents(tracer)

    @tracer.op(kind=SpanKind.AGENT, name="support")
    def run() -> dict[str, Any]:
        return retrying()

    run()
    tracer.flush()
    spans = [Span.from_json(s.to_json()) for s in sink.spans]

    # The same agents again, for the replays (each replay runs them afresh, under
    # an agent span as in the capture: attempts are linked within one parent).
    def replay_retrying() -> dict[str, Any]:
        tracer = _active_tracer()
        return tracer.op(kind=SpanKind.AGENT, name="support")(_agents(tracer)[0])()  # type: ignore[no-any-return]

    def replay_careful() -> dict[str, Any]:
        tracer = _active_tracer()
        return tracer.op(kind=SpanKind.AGENT, name="support")(_agents(tracer)[1])()  # type: ignore[no-any-return]

    return MockSet.from_trace(spans), replay_retrying, replay_careful


def _active_tracer() -> Tracer:
    import pytracer_sdk

    return pytracer_sdk.get_tracer()


def _replay(agent: Any, mocks: MockSet) -> ReplayResult:
    return ReplayEngine(ReplayMode.FULL, strict=True).replay(agent, mocks)


def test_a_full_replay_shows_the_duplicate_refund_and_its_failed_first_attempt() -> None:
    mocks, retrying, _ = _recording()
    before = LIVE["refunds"]
    result = _replay(retrying, mocks)
    assert result.error_message is None
    assert LIVE["refunds"] == before  # frozen: the replay refunded nobody

    once = Writes([WriteCall("refund_payment", times=1, args={"order_id": "A1"})]).check(result)
    assert not once.passed
    assert "refund_payment(order_id='A1') was called 2 times, expected 1" in once.detail
    assert "#1 failed (TimeoutError), #2 retry ok" in once.detail

    duplicates = NoDuplicateWrites(["refund_payment"]).check(result)
    assert not duplicates.passed and "called 2 times" in duplicates.detail


def test_a_fixed_agent_passes_against_the_same_recording() -> None:
    mocks, _, careful = _recording()
    result = _replay(careful, mocks)
    assert result.error_message is None
    assert Writes([WriteCall("refund_payment", times=1)]).check(result).passed
    assert NoDuplicateWrites(["refund_payment"]).check(result).passed
    assert Returns({"status": "pending"}).check(result).passed


def test_the_replayed_spans_link_the_retry() -> None:
    mocks, retrying, _ = _recording()
    result = _replay(retrying, mocks)
    refunds = [s for s in result.spans if s.tool_name == "refund_payment"]
    assert refunds[1].attributes["pytracer.retry_of"] == refunds[0].span_id


# --- unit cases: argument shapes, `only`, and return values ---------------------


def _tool(tool: str, data: Any, status: SpanStatus = SpanStatus.OK) -> Span:
    return Span(
        trace_id=new_trace_id(), span_id=new_span_id(), name=tool, kind=SpanKind.TOOL,
        tool_name=tool, input=data, status=status, start_time_ns=1, end_time_ns=2,
    )


def _result(spans: list[Span], value: Any = None) -> ReplayResult:
    return ReplayResult(return_value=value, status=SpanStatus.OK, mode=ReplayMode.FULL, spans=spans)


def test_arguments_are_read_from_every_adapters_input_shape() -> None:
    shapes = [
        {"args": [], "kwargs": {"order_id": "A1"}},  # @op
        {"arguments": {"order_id": "A1"}},  # OpenAI Agents, Agent Framework, Claude Agent SDK
        {"args": {"order_id": "A1"}},  # CrewAI, AutoGen
        {"input": {"order_id": "A1"}},  # LangGraph
        {"input": '{"order_id": "A1"}'},  # LangGraph, a JSON string tool input
    ]
    for shape in shapes:
        result = _result([_tool("refund", shape)])
        assert Writes([WriteCall("refund", times=1, args={"order_id": "A1"})]).check(result).passed, shape


def test_args_filter_which_calls_count() -> None:
    result = _result([_tool("refund", {"arguments": {"order_id": "A1"}}),
                      _tool("refund", {"arguments": {"order_id": "B2"}})])
    assert Writes([WriteCall("refund", times=1, args={"order_id": "B2"})]).check(result).passed
    assert NoDuplicateWrites(["refund"]).check(result).passed  # different orders


def test_only_rejects_an_unexpected_write() -> None:
    result = _result([_tool("refund", {}), _tool("delete_account", {})])
    check = Writes([WriteCall("refund")], write_tools=["refund", "delete_account"], only=True)
    outcome = check.check(result)
    assert not outcome.passed and "delete_account was called 1 times, expected none" in outcome.detail


def test_an_expected_write_that_never_happened_fails() -> None:
    outcome = Writes([WriteCall("refund")]).check(_result([]))
    assert not outcome.passed and outcome.detail == "refund was not called"


@dataclass
class Ticket:
    status: str
    order_id: str


def test_returns_matches_a_subset_of_fields() -> None:
    assert Returns({"status": "refunded"}).check(_result([], {"status": "refunded", "id": 7})).passed
    assert Returns({"status": "refunded"}).check(_result([], Ticket("refunded", "A1"))).passed
    failed = Returns({"status": "refunded"}).check(_result([], {"status": "failed"}))
    assert not failed.passed and "failed" in failed.detail
    assert Returns("done").check(_result([], "done")).passed

"""Repeated op calls are numbered, and a retry is linked to the attempt that failed."""

from __future__ import annotations

import asyncio

import pytest

from pytracer_sdk import Resource, SpanKind, SpanStatus, Tracer
from pytracer_sdk.transport import InMemoryTransport


def _tracer() -> tuple[Tracer, InMemoryTransport]:
    sink = InMemoryTransport()
    return Tracer(sink, resource=Resource(service_name="attempts", branch="main")), sink


def _tools(tracer: Tracer, sink: InMemoryTransport) -> list:  # type: ignore[type-arg]
    tracer.flush()
    return [s for s in sink.spans if s.kind is SpanKind.TOOL]


def test_a_retry_after_a_failure_is_linked_to_the_failed_attempt() -> None:
    tracer, sink = _tracer()
    failures = [TimeoutError("gateway timeout")]

    @tracer.op(kind=SpanKind.TOOL, tool_name="refund")
    def refund(order_id: str) -> str:
        if failures:
            raise failures.pop()
        return "refunded"

    @tracer.op(kind=SpanKind.AGENT)
    def agent() -> str:
        for _ in range(3):
            try:
                return refund(order_id="A1")
            except TimeoutError:
                continue
        return "gave up"

    assert agent() == "refunded"
    first, second = _tools(tracer, sink)
    assert first.status is SpanStatus.ERROR and "pytracer.attempt" not in first.attributes
    assert second.attributes["pytracer.attempt"] == 2
    assert second.attributes["pytracer.retry_of"] == first.span_id


def test_a_repeat_after_success_is_numbered_but_not_a_retry() -> None:
    tracer, sink = _tracer()

    @tracer.op(kind=SpanKind.TOOL, tool_name="refund")
    def refund(order_id: str) -> str:
        return "refunded"

    @tracer.op(kind=SpanKind.AGENT)
    def agent() -> None:
        refund(order_id="A1")
        refund(order_id="A1")

    agent()
    first, second = _tools(tracer, sink)
    assert second.attributes["pytracer.attempt"] == 2
    assert "pytracer.retry_of" not in second.attributes


def test_calls_with_other_input_or_under_another_parent_are_not_linked() -> None:
    tracer, sink = _tracer()

    @tracer.op(kind=SpanKind.TOOL, tool_name="refund")
    def refund(order_id: str) -> str:
        return "refunded"

    @tracer.op(kind=SpanKind.AGENT)
    def agent() -> None:
        refund(order_id="A1")
        refund(order_id="B2")

    agent()
    agent()  # a second run: a new parent span
    assert all("pytracer.attempt" not in s.attributes for s in _tools(tracer, sink))


def test_async_retries_are_linked_too() -> None:
    tracer, sink = _tracer()
    failures = [ConnectionError("reset")]

    @tracer.op(kind=SpanKind.TOOL, tool_name="charge")
    async def charge(amount: int) -> str:
        if failures:
            raise failures.pop()
        return "charged"

    @tracer.op(kind=SpanKind.AGENT)
    async def agent() -> str:
        with pytest.raises(ConnectionError):
            await charge(amount=5)
        return await charge(amount=5)

    assert asyncio.run(agent()) == "charged"
    first, second = _tools(tracer, sink)
    assert second.attributes["pytracer.retry_of"] == first.span_id

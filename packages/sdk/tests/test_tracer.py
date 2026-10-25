"""Tracer: capture, parent/child context propagation, errors, and redaction."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator

import pytest

from pytracer_sdk import Intercept, Resource, SpanKind, SpanStatus, Tracer
from pytracer_sdk.tracer import Interceptor
from pytracer_sdk.transport import InMemoryTransport


def _tracer(interceptor: Interceptor | None = None) -> tuple[Tracer, InMemoryTransport]:
    sink = InMemoryTransport()
    tracer = Tracer(
        sink, resource=Resource(service_name="test-agent", branch="main"), interceptor=interceptor
    )
    return tracer, sink


def test_op_decorator_captures_a_span() -> None:
    tracer, sink = _tracer()

    @tracer.op(kind=SpanKind.TOOL, tool_name="add")
    def add(a: int, b: int) -> int:
        return a + b

    assert add(2, 3) == 5
    tracer.flush()
    (span,) = sink.spans
    assert span.name == "add"
    assert span.kind is SpanKind.TOOL
    assert span.tool_name == "add"
    assert span.status is SpanStatus.OK
    assert span.output == 5
    assert span.resource.service_name == "test-agent"
    assert span.latency_ms is not None


def test_nested_spans_share_trace_and_link_parent() -> None:
    tracer, sink = _tracer()

    @tracer.op(kind=SpanKind.TOOL)
    def child() -> str:
        return "ok"

    @tracer.op(kind=SpanKind.AGENT)
    def parent() -> str:
        return child()

    parent()
    tracer.flush()
    spans = {s.name: s for s in sink.spans}
    assert set(spans) == {"parent", "child"}
    # Same trace, and child points at parent.
    assert spans["child"].trace_id == spans["parent"].trace_id
    assert spans["child"].parent_span_id == spans["parent"].span_id
    assert spans["parent"].parent_span_id is None


def test_exceptions_are_recorded_and_reraised() -> None:
    tracer, sink = _tracer()

    @tracer.op()
    def boom() -> None:
        raise ValueError("nope")

    with pytest.raises(ValueError):
        boom()
    tracer.flush()
    (span,) = sink.spans
    assert span.status is SpanStatus.ERROR
    assert span.error_type == "ValueError"


def test_input_is_redacted_at_capture() -> None:
    tracer, sink = _tracer()

    @tracer.op(kind=SpanKind.LLM)
    def call(prompt: str) -> str:
        return "response for alice@example.com"

    call("my key sk-abc123DEF456ghi789JKL012mno")
    tracer.flush()
    (span,) = sink.spans
    assert "sk-abc123" not in str(span.input)
    assert "alice@example.com" not in str(span.output)


def test_async_functions_are_supported() -> None:
    tracer, sink = _tracer()

    @tracer.op(kind=SpanKind.LLM)
    async def acall(x: int) -> int:
        await asyncio.sleep(0)
        return x * 2

    assert asyncio.run(acall(21)) == 42
    tracer.flush()
    (span,) = sink.spans
    assert span.output == 42
    assert span.name == "acall"


def test_intercept_stream_delivers_a_sync_iterator() -> None:
    def interceptor(kind: SpanKind, name: str, tool_name: str | None, inp: object) -> Intercept:
        return Intercept(["a", "b", "c"], stream=True)

    tracer, _ = _tracer(interceptor)

    @tracer.op(kind=SpanKind.LLM)
    def model(prompt: str) -> object:
        raise AssertionError("live body must not run when intercepted")

    assert list(model("hi")) == ["a", "b", "c"]


def test_intercept_stream_delivers_an_async_iterator() -> None:
    def interceptor(kind: SpanKind, name: str, tool_name: str | None, inp: object) -> Intercept:
        return Intercept(["x", "y"], stream=True)

    tracer, _ = _tracer(interceptor)

    @tracer.op(kind=SpanKind.LLM)
    async def model(prompt: str) -> object:
        raise AssertionError("live body must not run when intercepted")

    async def consume() -> list[str]:
        return [chunk async for chunk in await model("hi")]

    assert asyncio.run(consume()) == ["x", "y"]


def test_sync_generator_output_is_captured_over_iteration() -> None:
    tracer, sink = _tracer()

    @tracer.op(kind=SpanKind.LLM)
    def stream_reply(prompt: str) -> Iterator[str]:
        yield "Hel"
        yield "lo"
        yield "!"

    gen = stream_reply("hi")
    # The span must stay open until the stream is consumed, not close early.
    tracer.flush()
    assert sink.spans == []
    assert list(gen) == ["Hel", "lo", "!"]  # behavior unchanged: same items, same order
    tracer.flush()
    (span,) = sink.spans
    assert span.name == "stream_reply"
    assert span.kind is SpanKind.LLM
    assert span.status is SpanStatus.OK
    # The assembled stream is captured, not the bare generator object.
    assert span.output == ["Hel", "lo", "!"]


def test_async_generator_output_is_captured_over_iteration() -> None:
    tracer, sink = _tracer()

    @tracer.op(kind=SpanKind.LLM)
    async def stream_reply(prompt: str) -> AsyncIterator[str]:
        for chunk in ("Hel", "lo", "!"):
            yield chunk

    async def consume() -> list[str]:
        return [c async for c in stream_reply("hi")]

    assert asyncio.run(consume()) == ["Hel", "lo", "!"]
    tracer.flush()
    (span,) = sink.spans
    assert span.name == "stream_reply"
    assert span.status is SpanStatus.OK
    assert span.output == ["Hel", "lo", "!"]


def test_module_level_op_captures_generator_output() -> None:
    import pytracer_sdk as pytracer

    sink = InMemoryTransport()
    pytracer.set_tracer(Tracer(sink))
    try:

        @pytracer.op(kind=SpanKind.LLM)
        def stream_reply(prompt: str) -> Iterator[str]:
            yield "a"
            yield "b"

        assert list(stream_reply("hi")) == ["a", "b"]
        pytracer.flush()
        (span,) = sink.spans
        assert span.output == ["a", "b"]
    finally:
        pytracer.set_tracer(None)


def test_generator_error_is_recorded_and_propagates() -> None:
    tracer, sink = _tracer()

    @tracer.op(kind=SpanKind.LLM)
    def stream_reply(prompt: str) -> Iterator[str]:
        yield "Hel"
        raise ValueError("stream broke")

    with pytest.raises(ValueError, match="stream broke"):
        list(stream_reply("hi"))
    tracer.flush()
    (span,) = sink.spans
    assert span.status is SpanStatus.ERROR
    assert span.error_type == "ValueError"


def test_object_outputs_keep_their_type_through_capture() -> None:
    # A pydantic model or dataclass returned by an op must be recorded as that
    # object, not flattened to a dict, so a replay hands the agent the same type.
    import dataclasses

    from pydantic import BaseModel

    from pytracer_sdk import Span
    from pytracer_sdk.transport import InMemoryTransport

    class Price(BaseModel):
        amount: float

    @dataclasses.dataclass
    class Order:
        id: int

    sink = InMemoryTransport()
    tracer = Tracer(sink)

    @tracer.op(kind=SpanKind.TOOL)
    def quote() -> Price:
        return Price(amount=1.5)

    @tracer.op(kind=SpanKind.TOOL)
    def order() -> Order:
        return Order(id=7)

    quote()
    order()
    tracer.flush()
    outputs = {s.name: s.output for s in sink.spans}
    assert outputs["quote"] == Price(amount=1.5)
    assert outputs["order"] == Order(id=7)
    assert all(s.latency_ms is not None for s in sink.spans)
    # Through JSON, a local class falls back to plain data (it cannot be imported
    # back); the type survives the capture itself, which is what replay serves.
    assert all(Span.from_json(s.to_json()).output is not None for s in sink.spans)

"""Tracer: capture, parent/child context propagation, errors, and redaction."""

from __future__ import annotations

import asyncio

import pytest

from pytracer_sdk import Resource, SpanKind, SpanStatus, Tracer
from pytracer_sdk.transport import InMemoryTransport


def _tracer() -> tuple[Tracer, InMemoryTransport]:
    sink = InMemoryTransport()
    tracer = Tracer(sink, resource=Resource(service_name="test-agent", branch="main"))
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

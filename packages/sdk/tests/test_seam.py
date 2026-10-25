"""Seams route a framework's own calls through the op path (adapter replay, C0)."""

from __future__ import annotations

import asyncio
from typing import Any

from pytracer_sdk import Resource, SpanKind, SpanRecorder, Tracer
from pytracer_sdk.seam import (
    Patcher,
    acall_through,
    active_seam,
    astream_through,
    call_through,
    stream_through,
)
from pytracer_sdk.tracer import SpanContext
from pytracer_sdk.transport import InMemoryTransport


def _tracer() -> tuple[Tracer, InMemoryTransport]:
    sink = InMemoryTransport()
    return Tracer(sink, resource=Resource(service_name="seam", branch="main", commit="c0")), sink


def test_call_through_records_the_call_with_the_normalized_input() -> None:
    tracer, sink = _tracer()

    def model(prompt: str, run_id: str) -> str:
        return f"reply to {prompt}"

    out = call_through(
        model, ("hi", "run-123"), kind=SpanKind.LLM, name="fake-model",
        input={"prompt": "hi"}, tracer=tracer,
    )
    tracer.flush()
    assert out == "reply to hi"
    (span,) = sink.spans
    assert span.kind is SpanKind.LLM and span.name == "fake-model"
    assert span.input == {"prompt": "hi"}  # the volatile run id is not part of the key
    assert span.output == "reply to hi"


def test_explicit_parent_places_the_span() -> None:
    tracer, sink = _tracer()
    with tracer.start_span("agent") as agent:
        pass
    parent = SpanContext(agent.span.trace_id, agent.span.span_id)
    call_through(lambda: "ok", kind=SpanKind.TOOL, name="t", tool_name="t", parent=parent, tracer=tracer)
    tracer.flush()
    tool = next(s for s in sink.spans if s.name == "t")
    assert tool.parent_span_id == agent.span.span_id
    assert tool.trace_id == agent.span.trace_id


def test_a_nested_call_of_the_same_kind_is_recorded_once() -> None:
    tracer, sink = _tracer()

    def stream() -> str:
        return "chunked"

    def invoke() -> str:  # a framework's invoke that calls its own stream
        return str(call_through(stream, kind=SpanKind.LLM, name="m", tracer=tracer))

    call_through(invoke, kind=SpanKind.LLM, name="m", tracer=tracer)
    tracer.flush()
    assert len(sink.spans) == 1


def test_active_seam_is_reported_only_during_the_call() -> None:
    tracer, _ = _tracer()
    seen: list[Any] = []
    call_through(lambda: seen.append(active_seam()), kind=SpanKind.TOOL, name="t", tracer=tracer)
    assert seen[0] is not None and seen[0].kind is SpanKind.TOOL
    assert active_seam() is None


def test_recorder_adopts_the_seam_call_instead_of_recording_it_twice() -> None:
    tracer, sink = _tracer()
    recorder = SpanRecorder(tracer)

    def framework_tool() -> str:
        # The framework's own callback fires inside the call.
        assert recorder.adopt("tool-run")
        recorder.start("inner-run", name="inner", parent_run_id="tool-run")
        recorder.end("inner-run", output="x")
        recorder.end("tool-run", output="ignored")  # the seam closes its own span
        return "done"

    call_through(framework_tool, kind=SpanKind.TOOL, name="search", tool_name="search", tracer=tracer)
    tracer.flush()
    by_name = {s.name: s for s in sink.spans}
    assert set(by_name) == {"search", "inner"}  # no duplicate span for "tool-run"
    assert by_name["inner"].parent_span_id == by_name["search"].span_id
    assert by_name["search"].output == "done"
    assert not recorder.adopt("outside")  # no seam call active


def test_async_and_stream_variants_record() -> None:
    tracer, sink = _tracer()

    async def amodel() -> str:
        return "async reply"

    def tokens() -> Any:
        yield from ("a", "b")

    async def atokens() -> Any:
        for token in ("c", "d"):
            yield token

    async def run() -> list[str]:
        out = [await acall_through(amodel, kind=SpanKind.LLM, name="am", tracer=tracer)]
        out += list(stream_through(tokens, kind=SpanKind.LLM, name="sm", tracer=tracer))
        out += [t async for t in astream_through(atokens, kind=SpanKind.LLM, name="asm", tracer=tracer)]
        return out

    assert asyncio.run(run()) == ["async reply", "a", "b", "c", "d"]
    tracer.flush()
    outputs = {s.name: s.output for s in sink.spans}
    assert outputs == {"am": "async reply", "sm": ["a", "b"], "asm": ["c", "d"]}


def test_patcher_wraps_once_and_restores() -> None:
    class Model:
        def invoke(self, x: int) -> int:
            return x

    calls: list[int] = []

    def make(original: Any) -> Any:
        def wrapper(self: Any, x: int) -> int:
            calls.append(x)
            return int(original(self, x))

        return wrapper

    patcher = Patcher()
    assert patcher.patch(Model, "invoke", make)
    assert not patcher.patch(Model, "invoke", make)  # already wrapped
    assert Model().invoke(3) == 3 and calls == [3]
    patcher.unpatch_all()
    assert Model().invoke(4) == 4 and calls == [3]
    assert len(patcher) == 0

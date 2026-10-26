"""Seams route a framework's own calls through the op path (adapter replay, C0)."""

from __future__ import annotations

import asyncio
from typing import Any

from pytracer_sdk import Resource, SpanKind, SpanRecorder, Tracer
from pytracer_sdk.seam import (
    ClassSeams,
    acall_through,
    active_seam,
    astream_through,
    call_through,
    key_data,
    stream_through,
    tool_names,
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


def _counting(calls: list[str]) -> Any:
    def make(name: str, original: Any) -> Any:
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            calls.append(f"{type(self).__name__}.{name}")
            return original(self, *args, **kwargs)

        return wrapper

    return make


def test_class_seams_wrap_a_family_and_restore_it() -> None:
    import abc

    class Base(abc.ABC):
        @abc.abstractmethod
        def run(self) -> str: ...

    class Existing(Base):
        def run(self) -> str:
            return "existing"

    calls: list[str] = []
    seams = ClassSeams({Base: [("run", _counting(calls))]})
    seams.install()
    seams.install()  # a second install does not wrap twice

    class Later(Base):  # defined after install
        def run(self) -> str:
            return "later"

    assert Existing().run() == "existing" and Later().run() == "later"
    assert calls == ["Existing.run", "Later.run"]  # the abstract Base.run is left alone
    seams.uninstall()
    seams.uninstall()
    calls.clear()

    class AfterUninstall(Base):
        def run(self) -> str:
            return "plain"

    assert Existing().run() == "existing" and AfterUninstall().run() == "plain"
    assert calls == []
    assert "__init_subclass__" not in Base.__dict__


def test_class_seams_hook_pydantic_subclasses() -> None:
    from pydantic import BaseModel

    class Model(BaseModel):
        def call(self) -> str:
            return "base"

    calls: list[str] = []
    seams = ClassSeams({Model: [("call", _counting(calls))]})
    seams.install()
    try:

        class Provider(Model):  # a lazily imported provider class
            x: int = 1

            def call(self) -> str:
                return f"provider {self.x}"

        assert Provider(x=2).call() == "provider 2"  # pydantic setup still works
        assert calls == ["Provider.call"]
    finally:
        seams.uninstall()
    assert "__pydantic_init_subclass__" not in Model.__dict__


def test_key_data_and_tool_names() -> None:
    import dataclasses

    from pydantic import BaseModel

    @dataclasses.dataclass
    class Settings:
        temperature: float

    class Message(BaseModel):
        role: str
        content: str | None = None

    assert key_data([Message(role="user"), Settings(0.5), int]) == [
        {"role": "user"}, {"temperature": 0.5}, "int",
    ]
    tools = [{"type": "function", "function": {"name": "b"}}, {"name": "a"}, Message(role="c")]
    assert tool_names(tools)[:2] == ["a", "b"]


def test_a_call_made_inside_a_stream_is_not_recorded_again() -> None:
    # A client whose stream calls its own create must be recorded once, or the
    # replayed stream could be served the inner call's recording.
    tracer, sink = _tracer()

    async def create() -> str:
        return "full reply"

    async def stream() -> Any:
        reply = await acall_through(create, kind=SpanKind.LLM, name="m", tracer=tracer)
        for word in str(reply).split(" "):
            yield word

    async def run() -> list[str]:
        return [w async for w in astream_through(stream, kind=SpanKind.LLM, name="m", tracer=tracer)]

    assert asyncio.run(run()) == ["full", "reply"]
    assert active_seam() is None
    tracer.flush()
    (span,) = sink.spans
    assert span.output == ["full", "reply"]


def test_a_run_adopted_inside_a_stream_nests_under_the_stream_span() -> None:
    tracer, sink = _tracer()
    recorder = SpanRecorder(tracer)

    def framework_stream() -> Any:
        assert recorder.adopt("stream-run", SpanKind.LLM)
        recorder.start("child-run", name="child", parent_run_id="stream-run")
        recorder.end("child-run")
        yield "token"

    assert list(stream_through(framework_stream, kind=SpanKind.LLM, name="m", tracer=tracer)) == ["token"]
    tracer.flush()
    by_name = {s.name: s for s in sink.spans}
    assert by_name["child"].parent_span_id == by_name["m"].span_id


def test_a_seam_call_records_its_agent_name_and_the_tokens_its_output_reports() -> None:
    tracer, sink = _tracer()

    async def turn(task: str) -> dict[str, Any]:
        return {"reply": "done", "usage": 7}

    async def stream(task: str) -> Any:
        yield {"chunk": "do"}
        yield {"reply": "done", "usage": 9}

    def usage(output: Any) -> int | None:
        last = output[-1] if isinstance(output, list) else output
        return last.get("usage")

    asyncio.run(acall_through(
        turn, ("t",), kind=SpanKind.AGENT, name="invoke_agent writer", agent_name="writer",
        tokens=usage, tracer=tracer,
    ))

    async def consume() -> None:
        async for _ in astream_through(
            stream, ("t",), kind=SpanKind.LLM, name="model", tokens=usage, tracer=tracer
        ):
            pass

    asyncio.run(consume())
    tracer.flush()
    agent, model = sink.spans
    assert agent.agent_name == "writer" and agent.token_count == 7
    assert model.token_count == 9  # from the stream's last item


def test_a_token_reader_that_fails_never_fails_the_call() -> None:
    tracer, sink = _tracer()

    def broken(output: Any) -> int | None:
        raise KeyError("usage")

    assert call_through(lambda: "ok", kind=SpanKind.LLM, name="m", tokens=broken, tracer=tracer) == "ok"
    tracer.flush()
    (span,) = sink.spans
    assert span.token_count is None

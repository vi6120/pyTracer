"""SpanRecorder: framework-event to span mapping, parent linkage, redaction."""

from __future__ import annotations

from pytracer_sdk import Resource, SpanKind, SpanRecorder, SpanStatus, Tracer
from pytracer_sdk.tracer import SpanContext, _current
from pytracer_sdk.transport import InMemoryTransport


def _recorder() -> tuple[SpanRecorder, InMemoryTransport, Tracer]:
    sink = InMemoryTransport()
    tracer = Tracer(sink, resource=Resource(service_name="fw", branch="main", commit="c0"))
    return SpanRecorder(tracer), sink, tracer


def test_records_a_run_tree_with_parent_linkage() -> None:
    rec, sink, tracer = _recorder()
    # Simulate a framework: agent -> (llm, tool), reported by run id.
    rec.start("root", name="agent", kind=SpanKind.AGENT, agent_name="planner")
    rec.start("llm1", name="chat", kind=SpanKind.LLM, parent_run_id="root", input={"prompt": "hi"})
    rec.end("llm1", output="hello", token_count=12, cost_usd=0.001)
    rec.start("t1", name="search", kind=SpanKind.TOOL, parent_run_id="root", tool_name="search")
    rec.end("t1", output=["r1"])
    rec.end("root", output="done")
    tracer.flush()

    spans = {s.name: s for s in sink.spans}
    assert set(spans) == {"agent", "chat", "search"}
    root = spans["agent"]
    assert root.parent_span_id is None
    # Children share the root's trace and point at it.
    assert spans["chat"].trace_id == root.trace_id
    assert spans["chat"].parent_span_id == root.span_id
    assert spans["search"].parent_span_id == root.span_id
    assert spans["chat"].token_count == 12
    assert spans["chat"].latency_ms is not None
    assert root.resource.branch == "main"


def test_unknown_end_is_ignored() -> None:
    rec, sink, tracer = _recorder()
    rec.end("never-started", output="x")  # must not raise
    tracer.flush()
    assert sink.spans == []


def test_framework_root_nests_under_an_active_pytracer_span() -> None:
    # P5: when a pyTracer span is active in this context (e.g. one framework handing
    # off to another, or an @op around the call), a framework root nests under it
    # instead of starting its own trace.
    rec, sink, tracer = _recorder()
    ambient = SpanContext(trace_id="a" * 32, span_id="b" * 16)
    token = _current.set(ambient)
    try:
        rec.start("fw_root", name="agent", kind=SpanKind.AGENT)
        rec.end("fw_root", output="done")
    finally:
        _current.reset(token)
    tracer.flush()
    (span,) = sink.spans
    assert span.trace_id == "a" * 32  # joined the ambient trace
    assert span.parent_span_id == "b" * 16  # nested under the ambient span


def test_two_framework_roots_under_one_ambient_span_share_a_trace() -> None:
    # P5: one request through two frameworks is one trace, not two.
    rec, sink, tracer = _recorder()
    ambient = SpanContext(trace_id="c" * 32, span_id="d" * 16)
    token = _current.set(ambient)
    try:
        rec.start("fw1", name="langgraph", kind=SpanKind.AGENT)
        rec.end("fw1")
        rec.start("fw2", name="crewai", kind=SpanKind.AGENT)
        rec.end("fw2")
    finally:
        _current.reset(token)
    tracer.flush()
    assert {s.trace_id for s in sink.spans} == {"c" * 32}  # one trace
    assert all(s.parent_span_id == "d" * 16 for s in sink.spans)


def test_error_marks_span_failed() -> None:
    rec, sink, tracer = _recorder()
    rec.start("r", name="tool", kind=SpanKind.TOOL)
    rec.end("r", error=ValueError("boom"))
    tracer.flush()
    (span,) = sink.spans
    assert span.status is SpanStatus.ERROR
    assert span.error_type == "ValueError"


def test_string_error_is_supported() -> None:
    rec, sink, tracer = _recorder()
    rec.start("r", name="tool", kind=SpanKind.TOOL)
    rec.end("r", error="upstream 500")
    tracer.flush()
    (span,) = sink.spans
    assert span.status is SpanStatus.ERROR
    assert span.error_type == "Error"


def test_input_and_output_are_redacted() -> None:
    rec, sink, tracer = _recorder()
    rec.start("r", name="chat", kind=SpanKind.LLM, input={"key": "sk-abc123DEF456ghi789JKL012mno"})
    rec.end("r", output="reply to bob@example.com")
    tracer.flush()
    (span,) = sink.spans
    assert "sk-abc123" not in str(span.input)
    assert "bob@example.com" not in str(span.output)


def test_open_count_tracks_in_flight_spans() -> None:
    rec, _, _ = _recorder()
    assert rec.open_count() == 0
    rec.start("a", name="x", kind=SpanKind.TOOL)
    rec.start("b", name="y", kind=SpanKind.TOOL)
    assert rec.open_count() == 2
    rec.end("a")
    assert rec.open_count() == 1


def test_input_can_be_supplied_at_end() -> None:
    # Some frameworks only know a call's input by the time it finishes.
    rec, sink, tracer = _recorder()
    rec.start("r", name="chat", kind=SpanKind.LLM)
    rec.end("r", input={"prompt": "hi"}, output="ok")
    tracer.flush()
    (span,) = sink.spans
    assert span.input == {"prompt": "hi"}
    assert span.output == "ok"


def test_orphan_parent_starts_new_trace() -> None:
    rec, sink, tracer = _recorder()
    # Parent id refers to a run we never saw: treat as a root, do not crash.
    rec.start("child", name="tool", kind=SpanKind.TOOL, parent_run_id="missing")
    rec.end("child", output="ok")
    tracer.flush()
    (span,) = sink.spans
    assert span.parent_span_id is None


def test_recorder_keeps_object_outputs() -> None:
    import dataclasses

    from pytracer_sdk import Tracer
    from pytracer_sdk.transport import InMemoryTransport

    @dataclasses.dataclass
    class Reply:
        text: str

    sink = InMemoryTransport()
    recorder = SpanRecorder(Tracer(sink))
    recorder.start("r1", name="model")
    recorder.end("r1", output=Reply("hi"))
    recorder._tracer.flush()  # type: ignore[attr-defined]
    (span,) = sink.spans
    assert span.output == Reply("hi")
    assert span.latency_ms is not None

"""SpanRecorder: framework-event to span mapping, parent linkage, redaction."""

from __future__ import annotations

from pytracer_sdk import Resource, SpanKind, SpanRecorder, SpanStatus, Tracer
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

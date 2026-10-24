"""Hermetic tests: the processor maps SDK span data to pyTracer spans.

These use the SDK's real span-data classes but drive the processor with small
fake Span/Trace wrappers, so they run without the tracing runtime.
"""

from __future__ import annotations

from typing import Any

from agents.tracing import (
    AgentSpanData,
    FunctionSpanData,
    GenerationSpanData,
    HandoffSpanData,
)
from pytracer_sdk import Resource, SpanStatus, Tracer
from pytracer_sdk import SpanKind as K
from pytracer_sdk.transport import InMemoryTransport

from pytracer_openai_agents import PyTracerTracingProcessor


class _FakeTrace:
    def __init__(self, trace_id: str, name: str) -> None:
        self.trace_id = trace_id
        self.name = name


class _FakeSpan:
    def __init__(
        self, span_id: str, trace_id: str, span_data: Any, parent_id: str | None = None,
        error: Any = None,
    ) -> None:
        self.span_id = span_id
        self.trace_id = trace_id
        self.parent_id = parent_id
        self.span_data = span_data
        self.error = error


def _setup() -> tuple[PyTracerTracingProcessor, InMemoryTransport, Tracer]:
    sink = InMemoryTransport()
    tracer = Tracer(sink, resource=Resource(service_name="oa", branch="main", commit="c0"))
    return PyTracerTracingProcessor(tracer), sink, tracer


def _run(proc: PyTracerTracingProcessor, spans: list[_FakeSpan]) -> None:
    trace = _FakeTrace("trace_1", "run")
    proc.on_trace_start(trace)  # type: ignore[arg-type]
    for s in spans:
        proc.on_span_start(s)  # type: ignore[arg-type]
        proc.on_span_end(s)  # type: ignore[arg-type]
    proc.on_trace_end(trace)  # type: ignore[arg-type]


def test_agent_and_function_mapping_and_linkage() -> None:
    proc, sink, tracer = _setup()
    trace = _FakeTrace("trace_1", "run")
    agent = _FakeSpan("s_agent", "trace_1", AgentSpanData(name="planner"))
    fn = _FakeSpan(
        "s_fn", "trace_1", FunctionSpanData(name="search", input='{"q":"x"}', output="results"),
        parent_id="s_agent",
    )
    # Real nesting: the parent stays open while the child runs.
    proc.on_trace_start(trace)  # type: ignore[arg-type]
    proc.on_span_start(agent)  # type: ignore[arg-type]
    proc.on_span_start(fn)  # type: ignore[arg-type]
    proc.on_span_end(fn)  # type: ignore[arg-type]
    proc.on_span_end(agent)  # type: ignore[arg-type]
    proc.on_trace_end(trace)  # type: ignore[arg-type]
    tracer.flush()
    spans = {s.name: s for s in sink.spans}
    assert spans["run"].kind is K.AGENT and spans["run"].parent_span_id is None
    assert spans["planner"].kind is K.AGENT
    assert spans["planner"].parent_span_id == spans["run"].span_id
    assert spans["search"].kind is K.TOOL and spans["search"].tool_name == "search"
    assert spans["search"].parent_span_id == spans["planner"].span_id
    assert spans["search"].output == "results"


def test_structural_spans_are_folded_into_their_parent() -> None:
    # The SDK's structural "task"/"turn" spans are folded, so no "unknown" span is
    # emitted and the real tool call nests under the run instead (P4).
    proc, sink, tracer = _setup()
    trace = _FakeTrace("trace_1", "run")

    class _Structural:
        type = "turn"

    turn = _FakeSpan("s_turn", "trace_1", _Structural())
    fn = _FakeSpan(
        "s_fn", "trace_1", FunctionSpanData(name="search", input=None, output="r"),
        parent_id="s_turn",
    )
    proc.on_trace_start(trace)  # type: ignore[arg-type]
    proc.on_span_start(turn)  # type: ignore[arg-type]
    proc.on_span_start(fn)  # type: ignore[arg-type]
    proc.on_span_end(fn)  # type: ignore[arg-type]
    proc.on_span_end(turn)  # type: ignore[arg-type]
    proc.on_trace_end(trace)  # type: ignore[arg-type]
    tracer.flush()
    by_name = {s.name: s for s in sink.spans}
    assert "turn" not in by_name  # the structural span was folded, not emitted
    assert all(s.kind is not K.UNKNOWN for s in sink.spans)  # no unknown spans remain
    assert by_name["search"].parent_span_id == by_name["run"].span_id  # re-parented to the run


def test_generation_span_captures_model_and_tokens() -> None:
    proc, sink, tracer = _setup()
    gen = _FakeSpan(
        "s_gen", "trace_1",
        GenerationSpanData(
            input=[{"role": "user", "content": "hi"}],
            output=[{"role": "assistant", "content": "yo"}],
            model="gpt-4o",
            usage={"total_tokens": 15},
        ),
    )
    _run(proc, [gen])
    tracer.flush()
    llm = next(s for s in sink.spans if s.kind is K.LLM)
    assert llm.name == "gpt-4o"
    assert llm.token_count == 15


def test_handoff_span_maps_to_handoff_kind() -> None:
    proc, sink, tracer = _setup()
    ho = _FakeSpan("s_ho", "trace_1", HandoffSpanData(from_agent="planner", to_agent="writer"))
    _run(proc, [ho])
    tracer.flush()
    (span,) = [s for s in sink.spans if s.kind is K.HANDOFF]
    assert "planner -> writer" == span.name


def test_error_marks_span_failed() -> None:
    proc, sink, tracer = _setup()
    fn = _FakeSpan(
        "s_fn", "trace_1", FunctionSpanData(name="t", input=None, output=None),
        error={"message": "boom", "data": {}},
    )
    _run(proc, [fn])
    tracer.flush()
    (span,) = [s for s in sink.spans if s.tool_name == "t"]
    assert span.status is SpanStatus.ERROR


def test_install_replaces_processors_rather_than_appending() -> None:
    # install() must REPLACE the processor list (set_trace_processors), not append,
    # or the SDK's default exporter keeps uploading every trace to OpenAI (A6).
    import pytracer_openai_agents.processor as mod

    captured: dict[str, Any] = {}
    original = mod.set_trace_processors
    mod.set_trace_processors = lambda procs: captured.update(procs=procs)  # type: ignore[assignment]
    try:
        returned = mod.install()
    finally:
        mod.set_trace_processors = original  # type: ignore[assignment]
    assert captured["procs"] == [returned]
    assert isinstance(returned, PyTracerTracingProcessor)


def test_input_is_redacted() -> None:
    proc, sink, tracer = _setup()
    secret = "sk-abc123DEF456ghi789JKL012mno"
    fn = _FakeSpan(
        "s_fn", "trace_1",
        FunctionSpanData(name="call", input=f'{{"key":"{secret}"}}', output="done"),
    )
    _run(proc, [fn])
    tracer.flush()
    (span,) = [s for s in sink.spans if s.tool_name == "call"]
    assert secret not in str(span.input)

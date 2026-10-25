"""Exporter tests against the real OpenTelemetry SDK.

These emit GenAI-convention OTel spans through the pyTracer exporter (no AutoGen
needed, only opentelemetry-sdk, a runtime dependency) and assert the mapping.
"""

from __future__ import annotations

import pytest
import pytracer_sdk as pytracer
from opentelemetry.trace import Status, StatusCode
from pytracer_sdk import SpanKind as K
from pytracer_sdk import SpanStatus
from pytracer_sdk.transport import InMemoryTransport

from pytracer_autogen import install, pytracer_tracer_provider


def _setup() -> tuple[object, InMemoryTransport, pytracer.Tracer]:
    sink = InMemoryTransport()
    tracer = pytracer.Tracer(sink, resource=pytracer.Resource(service_name="ag", branch="main", commit="c0"))
    return pytracer_tracer_provider(tracer).get_tracer("test"), sink, tracer


def test_kinds_and_parent_linkage() -> None:
    ot, sink, tracer = _setup()
    with ot.start_as_current_span("invoke_agent planner") as agent:  # type: ignore[attr-defined]
        agent.set_attribute("gen_ai.operation.name", "invoke_agent")
        agent.set_attribute("gen_ai.agent.name", "planner")
        with ot.start_as_current_span("execute_tool search") as tool:  # type: ignore[attr-defined]
            tool.set_attribute("gen_ai.operation.name", "execute_tool")
            tool.set_attribute("gen_ai.tool.name", "search")
    tracer.flush()
    agent_span = next(s for s in sink.spans if s.kind is K.AGENT)
    tool_span = next(s for s in sink.spans if s.kind is K.TOOL)
    assert agent_span.agent_name == "planner"
    assert tool_span.tool_name == "search"
    assert tool_span.parent_span_id == agent_span.span_id
    assert tool_span.trace_id == agent_span.trace_id


def test_llm_kind_and_tokens() -> None:
    ot, sink, tracer = _setup()
    with ot.start_as_current_span("chat gpt-4o") as span:  # type: ignore[attr-defined]
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.usage.total_tokens", 42)
    tracer.flush()
    (llm,) = [s for s in sink.spans if s.kind is K.LLM]
    assert llm.token_count == 42


def test_error_status_is_mapped() -> None:
    ot, sink, tracer = _setup()
    with ot.start_as_current_span("chat") as span:  # type: ignore[attr-defined]
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("error.type", "ValueError")
        span.set_status(Status(StatusCode.ERROR))
    tracer.flush()
    (span,) = sink.spans
    assert span.status is SpanStatus.ERROR
    assert span.error_type == "ValueError"


def test_runtime_messaging_span_without_genai_attrs_is_skipped() -> None:
    # AutoGen's internal messaging spans carry no GenAI attributes; they are noise
    # and must not be recorded (A5: 44 "unknown" id-named spans per run).
    ot, sink, tracer = _setup()
    with ot.start_as_current_span("some.internal.op"):  # type: ignore[attr-defined]
        pass
    tracer.flush()
    assert sink.spans == []


def test_model_span_recognized_from_request_model_attr() -> None:
    # A model call that arrives with only gen_ai.request.model (no operation name)
    # is still classified as an LLM span, not dropped as unknown.
    ot, sink, tracer = _setup()
    with ot.start_as_current_span("ChatCompletion") as span:  # type: ignore[attr-defined]
        span.set_attribute("gen_ai.request.model", "gpt-5-mini")
    tracer.flush()
    (llm,) = [s for s in sink.spans if s.kind is K.LLM]
    assert llm.attributes["gen_ai.request.model"] == "gpt-5-mini"


def test_model_output_read_from_span_events() -> None:
    # Newer GenAI instrumentation records prompt/response as span events, not
    # attributes; the exporter must still capture them (A5: empty outputs).
    ot, sink, tracer = _setup()
    with ot.start_as_current_span("chat gpt-4o") as span:  # type: ignore[attr-defined]
        span.set_attribute("gen_ai.operation.name", "chat")
        span.add_event("gen_ai.user.message", {"content": "how many words?"})
        span.add_event("gen_ai.choice", {"content": "34876"})
    tracer.flush()
    (llm,) = [s for s in sink.spans if s.kind is K.LLM]
    assert "how many words?" in str(llm.input)
    assert "34876" in str(llm.output)


def test_attributes_are_redacted() -> None:
    ot, sink, tracer = _setup()
    secret = "sk-abc123DEF456ghi789JKL012mno"
    with ot.start_as_current_span("chat") as span:  # type: ignore[attr-defined]
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.input", f"my key {secret}")
    tracer.flush()
    (span,) = sink.spans
    assert secret not in str(span.input)
    assert secret not in str(span.attributes)


def test_exporter_built_before_configure_uses_the_configured_tracer() -> None:
    # P8: the tracer is resolved at export time, so an exporter built before
    # pytracer.configure(...) still records into the configured transport.
    ot = pytracer_tracer_provider().get_tracer("test")  # built with no explicit tracer
    sink = InMemoryTransport()
    previous = pytracer.get_tracer()
    pytracer.configure(sink, service_name="ag", branch="main", commit="c0")
    try:
        with ot.start_as_current_span("chat gpt-4o") as span:  # type: ignore[attr-defined]
            span.set_attribute("gen_ai.operation.name", "chat")
        pytracer.get_tracer().flush()
    finally:
        pytracer.set_tracer(previous)
    assert any(s.kind is K.LLM for s in sink.spans)


def test_install_registers_the_provider_globally(monkeypatch: pytest.MonkeyPatch) -> None:
    # P1/P2: install() registers pyTracer on the global OTel provider, where
    # AutoGen's GenAI spans go (passing a provider only to the runtime misses them).
    from opentelemetry import trace as otel_trace

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        otel_trace, "set_tracer_provider", lambda provider: captured.__setitem__("provider", provider)
    )
    provider = install()
    assert captured["provider"] is provider


# --- P2: one trace per run ---------------------------------------------------
# AutoGen starts a new OTel trace for each message and links it to the sender's
# trace (a span link, not a parent). These build that shape with plain OTel spans.


def _genai(span: object, operation: str, **attrs: str) -> None:
    span.set_attribute("gen_ai.operation.name", operation)  # type: ignore[attr-defined]
    for key, value in attrs.items():
        span.set_attribute(f"gen_ai.{key}", value)  # type: ignore[attr-defined]


def _message_hop(ot: object, sender_span: object) -> object:
    """A runtime 'publish' root in a new trace, linked to the sender's span."""
    from opentelemetry.trace import Link

    return ot.start_as_current_span(  # type: ignore[attr-defined]
        "autogen publish", context=None, links=[Link(sender_span.get_span_context())]  # type: ignore[attr-defined]
    )


def test_linked_message_traces_are_one_trace() -> None:
    from opentelemetry import context as otel_context

    ot, sink, tracer = _setup()
    token = otel_context.attach(otel_context.Context())  # start from no active span
    try:
        with ot.start_as_current_span("invoke_agent writer") as writer:  # type: ignore[attr-defined]
            _genai(writer, "invoke_agent", **{"agent.name": "writer"})
            with ot.start_as_current_span("autogen create group_topic") as create:  # type: ignore[attr-defined]
                pass
        # The next message is delivered in a fresh trace, linked to `create`.
        with _message_hop(ot, create):  # type: ignore[attr-defined]
            with ot.start_as_current_span("autogen process critic"):  # type: ignore[attr-defined]
                with ot.start_as_current_span("invoke_agent critic") as critic:  # type: ignore[attr-defined]
                    _genai(critic, "invoke_agent", **{"agent.name": "critic"})
    finally:
        otel_context.detach(token)
    tracer.flush()
    spans = {s.agent_name: s for s in sink.spans}
    assert spans["writer"].trace_id == spans["critic"].trace_id  # one trace for the run
    assert spans["critic"].parent_span_id is None  # top of the run, not a dropped noise span


def test_unlinked_runs_stay_separate_traces() -> None:
    from opentelemetry import context as otel_context

    ot, sink, tracer = _setup()
    for name in ("first", "second"):
        token = otel_context.attach(otel_context.Context())
        try:
            with ot.start_as_current_span(f"invoke_agent {name}") as span:  # type: ignore[attr-defined]
                _genai(span, "invoke_agent", **{"agent.name": name})
        finally:
            otel_context.detach(token)
    tracer.flush()
    assert len({s.trace_id for s in sink.spans}) == 2


def test_tool_nests_under_its_agent_across_dropped_messaging_spans() -> None:
    ot, sink, tracer = _setup()
    with ot.start_as_current_span("invoke_agent planner") as agent:  # type: ignore[attr-defined]
        _genai(agent, "invoke_agent", **{"agent.name": "planner"})
        with ot.start_as_current_span("autogen create output_topic"):  # type: ignore[attr-defined]
            with ot.start_as_current_span("execute_tool search") as tool:  # type: ignore[attr-defined]
                _genai(tool, "execute_tool", **{"tool.name": "search"})
    tracer.flush()
    agent_span = next(s for s in sink.spans if s.kind is K.AGENT)
    tool_span = next(s for s in sink.spans if s.kind is K.TOOL)
    assert tool_span.parent_span_id == agent_span.span_id  # not the dropped noise span


def test_run_nests_under_an_enclosing_pytracer_span() -> None:
    ot, sink, tracer = _setup()
    with tracer.start_span("request") as request:
        with ot.start_as_current_span("invoke_agent writer") as writer:  # type: ignore[attr-defined]
            _genai(writer, "invoke_agent", **{"agent.name": "writer"})
    tracer.flush()
    request_span = next(s for s in sink.spans if s.name == "request")
    writer_span = next(s for s in sink.spans if s.agent_name == "writer")
    assert writer_span.trace_id == request_span.trace_id
    assert writer_span.parent_span_id == request.span.span_id


# --- tool results (AutoGen sends them in a later runtime message) ---------------


def _tool_call(ot: object, call_id: str) -> None:
    with ot.start_as_current_span("execute_tool search") as tool:  # type: ignore[attr-defined]
        _genai(tool, "execute_tool", **{"tool.name": "search", "tool.call.id": call_id})


def _result_message(ot: object, call_id: str, content: str, *, is_error: bool = False) -> None:
    import json

    message = {
        "message": {
            "type": "ToolCallExecutionEvent",
            "source": "planner",
            "content": [{"content": content, "name": "search", "call_id": call_id, "is_error": is_error}],
        }
    }
    with ot.start_as_current_span("autogen process manager") as span:  # type: ignore[attr-defined]
        span.set_attribute("message", json.dumps(message))


def test_tool_output_is_joined_from_the_result_message() -> None:
    ot, sink, tracer = _setup()
    with ot.start_as_current_span("invoke_agent planner") as agent:  # type: ignore[attr-defined]
        _genai(agent, "invoke_agent", **{"agent.name": "planner"})
        _tool_call(ot, "c1")
        _result_message(ot, "c1", "3 results")
    tracer.flush()
    (tool,) = [s for s in sink.spans if s.kind is K.TOOL]
    assert tool.output == "3 results"
    assert tool.status is SpanStatus.OK


def test_tool_error_result_marks_the_span_failed() -> None:
    ot, sink, tracer = _setup()
    with ot.start_as_current_span("invoke_agent planner") as agent:  # type: ignore[attr-defined]
        _genai(agent, "invoke_agent", **{"agent.name": "planner"})
        _tool_call(ot, "c2")
        _result_message(ot, "c2", "Error: index offline", is_error=True)
    tracer.flush()
    (tool,) = [s for s in sink.spans if s.kind is K.TOOL]
    assert tool.status is SpanStatus.ERROR
    assert tool.error_message == "Error: index offline"


def test_tool_span_without_a_result_is_still_recorded_on_flush() -> None:
    sink = InMemoryTransport()
    tracer = pytracer.Tracer(sink, resource=pytracer.Resource(service_name="ag", branch="main", commit="c0"))
    provider = pytracer_tracer_provider(tracer)
    ot = provider.get_tracer("test")
    _tool_call(ot, "never-answered")
    provider.force_flush()
    tracer.flush()
    (tool,) = [s for s in sink.spans if s.kind is K.TOOL]
    assert tool.output is None

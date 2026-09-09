"""Exporter tests against the real OpenTelemetry SDK.

These emit GenAI-convention OTel spans through the pyTracer exporter (no AutoGen
needed, only opentelemetry-sdk, a runtime dependency) and assert the mapping.
"""

from __future__ import annotations

import pytracer_sdk as pytracer
from opentelemetry.trace import Status, StatusCode
from pytracer_sdk import SpanKind as K
from pytracer_sdk import SpanStatus
from pytracer_sdk.transport import InMemoryTransport

from pytracer_autogen import pytracer_tracer_provider


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


def test_span_without_genai_attrs_is_unknown() -> None:
    ot, sink, tracer = _setup()
    with ot.start_as_current_span("some.internal.op"):  # type: ignore[attr-defined]
        pass
    tracer.flush()
    (span,) = sink.spans
    assert span.kind is K.UNKNOWN


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

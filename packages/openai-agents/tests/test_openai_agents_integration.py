"""Integration test against the real OpenAI Agents SDK tracing runtime.

Uses the SDK's own trace/span context managers with our processor registered
(replacing the default exporter, so nothing leaves the machine) and confirms a
run is captured as one connected trace with the right span kinds. Skipped when
the SDK is not installed.
"""

from __future__ import annotations

import importlib.util

import navik_sdk as navik
import pytest
from navik_sdk import SpanKind
from navik_sdk.transport import InMemoryTransport

from navik_openai_agents import NavikTracingProcessor

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("agents") is None, reason="openai-agents not installed"
)


def test_real_tracing_run_is_captured() -> None:
    from agents.tracing import (
        agent_span,
        function_span,
        generation_span,
        set_trace_processors,
        trace,
    )

    sink = InMemoryTransport()
    tracer = navik.Tracer(sink, resource=navik.Resource(service_name="oa", branch="main", commit="c0"))
    set_trace_processors([NavikTracingProcessor(tracer)])
    try:
        with trace("weather-run"), agent_span(name="planner"):
            with generation_span(model="gpt-4o", usage={"total_tokens": 12}) as gen:
                gen.span_data.output = [{"role": "assistant", "content": "sunny"}]
            with function_span(name="get_weather", input='{"city":"Paris"}') as fn:
                fn.span_data.output = "sunny in Paris"
    finally:
        set_trace_processors([])  # restore global state for other tests

    tracer.flush()
    spans = sink.spans
    assert len({s.trace_id for s in spans}) == 1
    kinds = {s.kind for s in spans}
    assert SpanKind.AGENT in kinds
    assert SpanKind.LLM in kinds
    assert SpanKind.TOOL in kinds
    tool = next(s for s in spans if s.kind is SpanKind.TOOL)
    assert tool.tool_name == "get_weather"
    assert "sunny in Paris" in str(tool.output)
    llm = next(s for s in spans if s.kind is SpanKind.LLM)
    assert llm.token_count == 12
    assert len([s for s in spans if s.parent_span_id is None]) == 1

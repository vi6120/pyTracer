"""Integration test using AutoGen's own OpenTelemetry telemetry.

Drives AutoGen's ``trace_tool_span`` with our tracer provider and confirms the
resulting span is captured as a pyTracer tool span. Skipped when autogen-core is
not installed.
"""

from __future__ import annotations

import importlib.util

import pytest
import pytracer_sdk as pytracer
from pytracer_sdk import SpanKind
from pytracer_sdk.transport import InMemoryTransport

from pytracer_autogen import pytracer_tracer_provider

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("autogen_core") is None, reason="autogen-core not installed"
)


def test_autogen_tool_span_is_captured() -> None:
    from autogen_core._telemetry import trace_tool_span

    sink = InMemoryTransport()
    tracer = pytracer.Tracer(sink, resource=pytracer.Resource(service_name="ag", branch="main", commit="c0"))
    ot = pytracer_tracer_provider(tracer).get_tracer("pytracer-test")

    with trace_tool_span("get_weather", tracer=ot):
        pass
    tracer.flush()

    tool_spans = [s for s in sink.spans if s.kind is SpanKind.TOOL]
    assert len(tool_spans) == 1
    assert tool_spans[0].tool_name == "get_weather"
    assert tool_spans[0].attributes.get("gen_ai.operation.name") == "execute_tool"

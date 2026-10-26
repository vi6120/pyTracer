"""Integration test using AutoGen's own OpenTelemetry telemetry.

Drives AutoGen's ``trace_tool_span`` with our tracer provider and confirms the
resulting span is captured as a pyTracer tool span. AutoGen before 0.6.2 has no
tool span; there a real tool call is captured through the seam ``install()`` sets
up. Skipped when autogen-core is not installed.
"""

from __future__ import annotations

import asyncio
import importlib.util

import pytest
import pytracer_sdk as pytracer
from pytracer_sdk import SpanKind
from pytracer_sdk.transport import InMemoryTransport

from pytracer_autogen import exporter_for, install_seams, pytracer_tracer_provider, uninstall_seams

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("autogen_core") is None, reason="autogen-core not installed"
)


def get_weather(city: str) -> str:
    """Report the weather."""
    return f"sunny in {city}"


def test_autogen_tool_call_is_captured() -> None:
    import autogen_core._telemetry as telemetry

    sink = InMemoryTransport()
    tracer = pytracer.Tracer(sink, resource=pytracer.Resource(service_name="ag", branch="main", commit="c0"))
    provider = pytracer_tracer_provider(tracer)

    if hasattr(telemetry, "trace_tool_span"):
        with telemetry.trace_tool_span("get_weather", tracer=provider.get_tracer("pytracer-test")):
            pass
        tracer.flush()
        (tool,) = [s for s in sink.spans if s.kind is SpanKind.TOOL]
        assert tool.tool_name == "get_weather"
        assert tool.attributes.get("gen_ai.operation.name") == "execute_tool"
        return

    # AutoGen before 0.6.2 emits no tool span: the seam records the real call.
    from autogen_core import CancellationToken
    from autogen_core.tools import FunctionTool

    exporter = exporter_for(provider)
    assert exporter is not None
    previous = pytracer.get_tracer()
    pytracer.set_tracer(tracer)
    install_seams(exporter)
    try:
        tool_impl = FunctionTool(get_weather, description="Report the weather.")
        asyncio.run(tool_impl.run_json({"city": "Paris"}, CancellationToken()))
    finally:
        uninstall_seams()
        pytracer.set_tracer(previous)
    tracer.flush()
    (tool,) = [s for s in sink.spans if s.kind is SpanKind.TOOL]
    assert tool.tool_name == "get_weather"
    assert tool.output == "sunny in Paris"

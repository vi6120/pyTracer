"""Integration test against the real CrewAI event bus.

Registers the listener on the real bus and emits real tool-usage events through
it, confirming the bus's own id correlation drives a captured span. Emits are
dispatched on a thread pool, so we wait on each emit's future. Skipped when
CrewAI is not installed.
"""

from __future__ import annotations

import datetime
import importlib.util
from typing import Any

import pytest
import pytracer_sdk as pytracer
from pytracer_sdk import SpanKind
from pytracer_sdk.transport import InMemoryTransport

from pytracer_crewai import PyTracerEventListener

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("crewai") is None, reason="crewai not installed"
)


def _emit(bus: Any, source: Any, event: Any) -> None:
    future = bus.emit(source, event)
    if future is not None:
        future.result(timeout=10)  # handlers run on a thread pool


def test_real_bus_tool_usage_is_captured() -> None:
    from crewai.events.event_bus import crewai_event_bus
    from crewai.events.types.tool_usage_events import ToolUsageFinishedEvent, ToolUsageStartedEvent

    sink = InMemoryTransport()
    tracer = pytracer.Tracer(sink, resource=pytracer.Resource(service_name="cw", branch="main", commit="c0"))

    with crewai_event_bus.scoped_handlers():
        PyTracerEventListener(tracer)  # registers on the scoped bus
        _emit(crewai_event_bus, "src", ToolUsageStartedEvent(tool_name="search", tool_args={"q": "cats"}))
        now = datetime.datetime.now(datetime.timezone.utc)
        _emit(
            crewai_event_bus, "src",
            ToolUsageFinishedEvent(
                tool_name="search", tool_args={"q": "cats"}, output="results for cats",
                started_at=now, finished_at=now,
            ),
        )

    tracer.flush()
    tool_spans = [s for s in sink.spans if s.kind is SpanKind.TOOL]
    assert len(tool_spans) == 1
    assert tool_spans[0].tool_name == "search"
    assert "results for cats" in str(tool_spans[0].output)

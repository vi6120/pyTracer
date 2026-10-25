"""A run captured through the CrewAI adapter replays without @op (C4).

A real Crew (one agent, one task, a tool) runs with a scripted ReAct-style LLM
(no network), is captured with the event listener and ``install_seams()``,
written to JSONL and reloaded, then replayed in full mode with the network
sandbox on. Skipped when CrewAI is not installed.
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Iterator
from typing import Any

import pytest
import pytracer_sdk as pytracer
from pytracer_sdk import Span, SpanKind
from pytracer_sdk.transport import InMemoryTransport

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("crewai") is None, reason="crewai not installed"
)

LIVE = {"model": 0, "tool": 0}

REPLIES = [
    'Thought: I should look it up.\nAction: lookup\nAction Input: {"city": "Paris"}',
    "Thought: I now know the final answer\nFinal Answer: It is sunny in Paris.",
]


@pytest.fixture
def capture() -> Iterator[tuple[InMemoryTransport, Any]]:
    os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
    from pytracer_crewai import PyTracerEventListener, install_seams, uninstall_seams

    sink = InMemoryTransport()
    previous = pytracer.get_tracer()
    tracer = pytracer.configure(sink, service_name="cw", branch="main", commit="c0")
    listener = PyTracerEventListener(tracer)
    install_seams(listener)
    LIVE.update(model=0, tool=0)
    try:
        yield sink, _crew_fn()
    finally:
        uninstall_seams()
        pytracer.set_tracer(previous)


def _crew_fn() -> Any:
    # Defined after install_seams, so the LLM subclass is patched by the hook for
    # classes CrewAI (or the user) defines later.
    from crewai import Agent, Crew, Task
    from crewai.events import crewai_event_bus
    from crewai.llms.base_llm import BaseLLM
    from crewai.tools import tool

    class ScriptedLLM(BaseLLM):
        """Answers from a ReAct script, one reply per call."""

        def call(self, messages: Any, *args: Any, **kwargs: Any) -> str:
            LIVE["model"] += 1
            return REPLIES[self._turn()]

        def _turn(self) -> int:
            turn = getattr(self, "_n", 0)
            object.__setattr__(self, "_n", turn + 1)
            return turn

        def supports_function_calling(self) -> bool:
            return False

    @tool("lookup")
    def lookup(city: str) -> str:
        """Weather for a city."""
        LIVE["tool"] += 1
        return f"sunny in {city}"

    def run() -> str:
        agent = Agent(
            role="forecaster", goal="report the weather", backstory="a careful forecaster",
            llm=ScriptedLLM(model="scripted"), tools=[lookup], verbose=False,
        )
        task = Task(description="weather in Paris?", expected_output="one sentence", agent=agent)
        output = Crew(agents=[agent], tasks=[task], verbose=False).kickoff()
        crewai_event_bus.flush()
        return str(output.raw)

    return run


def _recorded(sink: InMemoryTransport, run: Any) -> tuple[str, list[Span]]:
    output = run()
    pytracer.get_tracer().flush()
    return output, [Span.from_json(s.to_json()) for s in sink.spans]


def test_capture_records_each_call_once_under_the_agent(capture: tuple[InMemoryTransport, Any]) -> None:
    sink, run = capture
    output, spans = _recorded(sink, run)
    assert output == "It is sunny in Paris."
    llm = [s for s in spans if s.kind is SpanKind.LLM]
    tools = [s for s in spans if s.kind is SpanKind.TOOL]
    assert len(llm) == 2 and len(tools) == 1  # no duplicate event spans
    agent = next(s for s in spans if s.kind is SpanKind.AGENT and s.name == "forecaster")
    assert all(s.parent_span_id == agent.span_id for s in llm + tools)
    assert tools[0].output == "sunny in Paris"


def test_full_replay_runs_nothing_live(capture: tuple[InMemoryTransport, Any]) -> None:
    from pytracer_replay import MockSet, ReplayEngine, ReplayMode

    sink, run = capture
    output, spans = _recorded(sink, run)
    before = dict(LIVE)
    result = ReplayEngine(ReplayMode.FULL).replay(run, MockSet.from_trace(spans))
    assert result.error_message is None
    assert result.return_value == output
    assert LIVE == before  # neither the model nor the tool ran again
    assert result.uninstrumented is False
    assert not result.side_effects.effects


def test_a_changed_recording_changes_the_replay(capture: tuple[InMemoryTransport, Any]) -> None:
    from pytracer_replay import MockSet, ReplayEngine, ReplayMode

    sink, run = capture
    _, spans = _recorded(sink, run)
    for span in spans:
        if span.kind is SpanKind.TOOL:
            span.output = "storming in Paris"
    strict = ReplayEngine(ReplayMode.FULL, strict=True).replay(run, MockSet.from_trace(spans))
    assert strict.failed  # the second model call was recorded after "sunny"
    assert "no recording" in (strict.error_message or "")


def test_partial_replay_freezes_the_tool_and_runs_the_model(capture: tuple[InMemoryTransport, Any]) -> None:
    from pytracer_replay import MockSet, ReplayEngine, ReplayMode

    sink, run = capture
    output, spans = _recorded(sink, run)
    before = dict(LIVE)
    result = ReplayEngine(ReplayMode.PARTIAL).replay(run, MockSet.from_trace(spans))
    assert result.return_value == output
    assert LIVE["tool"] == before["tool"]
    assert LIVE["model"] == before["model"] + 2


def test_native_tool_calls_go_through_the_seam(capture: tuple[InMemoryTransport, Any]) -> None:
    # Native function calling runs a tool through BaseTool.run (the ReAct crew
    # above uses the structured-tool path), so check that path records and replays.
    from crewai.tools import tool
    from pytracer_replay import MockSet, ReplayEngine, ReplayMode

    sink, _ = capture

    @tool("convert")
    def convert(celsius: float) -> str:
        """Celsius to Fahrenheit."""
        LIVE["tool"] += 1
        return f"{celsius * 9 / 5 + 32:.0f}F"

    def run() -> str:
        return str(convert.run(celsius=20))

    output, spans = _recorded(sink, run)
    assert output == "68F"
    assert [s.tool_name for s in spans if s.kind is SpanKind.TOOL] == ["convert"]
    before = dict(LIVE)
    result = ReplayEngine(ReplayMode.FULL).replay(run, MockSet.from_trace(spans))
    assert result.return_value == output
    assert LIVE == before

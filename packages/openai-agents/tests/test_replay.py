"""A run captured through the OpenAI Agents adapter replays without @op (C1).

A real Runner run with a scripted model (no network) and a function tool is
captured with ``install(replayable=True)``, written to JSONL and reloaded, then
replayed in full mode with the network sandbox on. Skipped when the SDK is absent.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Iterator
from typing import Any

import pytest
import pytracer_sdk as pytracer
from pytracer_sdk import Span, SpanKind
from pytracer_sdk.transport import InMemoryTransport

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("agents") is None, reason="openai-agents not installed"
)

CALLS = {"model": 0, "tool": 0}

if importlib.util.find_spec("agents") is not None:
    from agents import Agent, Runner, function_tool
    from agents.items import ModelResponse
    from agents.models.interface import Model
    from agents.usage import Usage
    from openai.types.responses import (
        Response,
        ResponseCompletedEvent,
        ResponseFunctionToolCall,
        ResponseOutputMessage,
        ResponseOutputText,
    )

    @function_tool
    def weather(city: str) -> str:
        """Weather for a city."""
        CALLS["tool"] += 1
        return f"sunny in {city}"

    def _message(text: str) -> Any:
        return ResponseOutputMessage(
            id="msg_1", role="assistant", status="completed", type="message",
            content=[ResponseOutputText(text=text, type="output_text", annotations=[])],
        )

    def _script() -> list[Any]:
        call = ResponseFunctionToolCall(
            id="fc_1", call_id="call_1", name="weather", arguments='{"city": "Paris"}',
            type="function_call",
        )
        return [call, _message("It is sunny in Paris.")]

    class ScriptedModel(Model):
        """Replies from a script; each reply is one model call."""

        def __init__(self) -> None:
            self.replies = _script()

        async def get_response(self, *args: Any, **kwargs: Any) -> ModelResponse:
            CALLS["model"] += 1
            return ModelResponse(
                output=[self.replies.pop(0)],
                usage=Usage(requests=1, input_tokens=3, output_tokens=2, total_tokens=5),
                response_id=None,
            )

        async def stream_response(self, *args: Any, **kwargs: Any) -> Any:
            CALLS["model"] += 1
            response = Response(
                id="resp_1", created_at=0, model="scripted", object="response",
                output=[self.replies.pop(0)], tool_choice="auto", tools=[],
                parallel_tool_calls=False,
            )
            yield ResponseCompletedEvent(type="response.completed", response=response, sequence_number=0)


@pytest.fixture
def seams() -> Iterator[InMemoryTransport]:
    from pytracer_openai_agents import install, uninstall_seams

    sink = InMemoryTransport()
    previous = pytracer.get_tracer()
    tracer = pytracer.configure(sink, service_name="oa", branch="main", commit="c0")
    install(tracer, replayable=True)
    CALLS.update(model=0, tool=0)
    try:
        yield sink
    finally:
        uninstall_seams()
        pytracer.set_tracer(previous)


def _agent_fn(*, stream: bool = False) -> Any:
    def run() -> str:
        agent = Agent(name="assistant", instructions="help", tools=[weather], model=ScriptedModel())
        if not stream:
            return str(Runner.run_sync(agent, "weather in Paris?").final_output)

        import asyncio

        async def streamed() -> str:
            result = Runner.run_streamed(agent, "weather in Paris?")
            async for _ in result.stream_events():
                pass
            return str(result.final_output)

        return asyncio.run(streamed())

    return run


def _capture(sink: InMemoryTransport, run: Any) -> tuple[str, list[Span]]:
    output = run()
    pytracer.get_tracer().flush()
    # Through the JSONL format and the codec, as a saved recording is read back.
    spans = [Span.from_json(s.to_json()) for s in sink.spans]
    return output, spans


def test_capture_records_model_and_tool_calls_once_under_the_agent(seams: InMemoryTransport) -> None:
    output, spans = _capture(seams, _agent_fn())
    assert output == "It is sunny in Paris."
    llm = [s for s in spans if s.kind is SpanKind.LLM]
    tools = [s for s in spans if s.kind is SpanKind.TOOL]
    assert len(llm) == 2 and len(tools) == 1  # no duplicate SDK generation/function spans
    agent = next(s for s in spans if s.kind is SpanKind.AGENT and s.name == "assistant")
    assert all(s.parent_span_id == agent.span_id for s in llm + tools)
    assert tools[0].output == "sunny in Paris"
    assert type(llm[0].output).__name__ == "ModelResponse"  # real type after the codec


def test_full_replay_runs_nothing_live(seams: InMemoryTransport) -> None:
    from pytracer_replay import MockSet, ReplayEngine, ReplayMode

    output, spans = _capture(seams, _agent_fn())
    before = dict(CALLS)
    result = ReplayEngine(ReplayMode.FULL).replay(_agent_fn(), MockSet.from_trace(spans))
    assert result.error_message is None
    assert result.return_value == output
    assert CALLS == before  # neither the model nor the tool ran again
    assert result.uninstrumented is False
    assert not result.side_effects.effects


def test_a_changed_recording_changes_the_replay(seams: InMemoryTransport) -> None:
    from pytracer_replay import MockSet, ReplayEngine, ReplayMode

    output, spans = _capture(seams, _agent_fn())
    for span in spans:
        if span.kind is SpanKind.TOOL:
            span.output = "storming in Paris"
    # The second model call was recorded after "sunny"; with "storming" its input no
    # longer matches, so the recording is genuinely what drives the replay.
    lenient = ReplayEngine(ReplayMode.FULL).replay(_agent_fn(), MockSet.from_trace(spans))
    assert "ScriptedModel" in lenient.model_input_changes  # the change is reported
    strict = ReplayEngine(ReplayMode.FULL, strict=True).replay(_agent_fn(), MockSet.from_trace(spans))
    assert strict.failed
    assert "no recording" in (strict.error_message or "")


def test_partial_replay_freezes_tools_and_runs_the_model(seams: InMemoryTransport) -> None:
    from pytracer_replay import MockSet, ReplayEngine, ReplayMode

    output, spans = _capture(seams, _agent_fn())
    before = dict(CALLS)
    result = ReplayEngine(ReplayMode.PARTIAL).replay(_agent_fn(), MockSet.from_trace(spans))
    assert result.return_value == output
    assert CALLS["tool"] == before["tool"]  # frozen
    assert CALLS["model"] == before["model"] + 2  # live


def test_streamed_run_replays(seams: InMemoryTransport) -> None:
    from pytracer_replay import MockSet, ReplayEngine, ReplayMode

    output, spans = _capture(seams, _agent_fn(stream=True))
    assert output == "It is sunny in Paris."
    before = dict(CALLS)
    result = ReplayEngine(ReplayMode.FULL).replay(_agent_fn(stream=True), MockSet.from_trace(spans))
    assert result.error_message is None
    assert result.return_value == output
    assert CALLS == before


def test_uninstall_restores_the_sdk(seams: InMemoryTransport) -> None:
    from agents.tool import FunctionTool

    from pytracer_openai_agents import uninstall_seams

    uninstall_seams()
    assert "on_invoke_tool" not in FunctionTool.__dict__
    assert not getattr(ScriptedModel.get_response, "__pytracer_seam__", False)
    assert callable(weather.on_invoke_tool)


def test_tools_stay_equal_with_seams_installed(seams: InMemoryTransport) -> None:
    import dataclasses

    from pytracer_openai_agents.replay import _SeamedInvoke

    assert weather == weather  # broken before: every read built a new wrapper
    assert weather.on_invoke_tool is weather.on_invoke_tool  # one wrapper per tool
    # A copy behaves as it does without seams (the SDK binds each copy its own
    # callable, so a copy is not equal to the original) and stores the real
    # callable, never a seam wrapper.
    copy = dataclasses.replace(weather)
    assert copy == copy
    assert not isinstance(copy.__dict__["on_invoke_tool"], _SeamedInvoke)

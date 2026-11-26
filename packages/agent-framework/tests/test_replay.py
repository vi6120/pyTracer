"""A run captured through the Agent Framework adapter replays without @op.

A real Agent with a scripted chat client (no network) and a function tool is
captured with ``install()``, written to JSONL and reloaded, then replayed with
the network sandbox on. Skipped when Agent Framework is absent.
"""

from __future__ import annotations

import asyncio
import importlib.util
from collections.abc import Iterator
from typing import Any

import pytest
import pytracer_sdk as pytracer
from pytracer_sdk import Span, SpanKind
from pytracer_sdk.transport import InMemoryTransport

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("agent_framework") is None, reason="agent-framework not installed"
)

CALLS = {"model": 0, "tool": 0}
SEEN: dict[str, Any] = {}

if importlib.util.find_spec("agent_framework") is not None:
    from agent_framework import (
        Agent,
        BaseChatClient,
        ChatMiddlewareLayer,
        ChatResponse,
        ChatResponseUpdate,
        Content,
        FunctionInvocationLayer,
        Message,
        tool,
    )

    @tool
    def weather(city: str) -> str:
        """Weather for a city."""
        CALLS["tool"] += 1
        return f"sunny in {city}"

    def _script() -> list[list[Any]]:
        return [
            [Content.from_function_call(call_id="call_1", name="weather", arguments='{"city": "Paris"}')],
            [Content.from_text("It is sunny in Paris.")],
        ]

    USAGE = {"input_token_count": 3, "output_token_count": 2, "total_token_count": 5}

    class ScriptedClient(FunctionInvocationLayer, ChatMiddlewareLayer, BaseChatClient):  # type: ignore[misc]
        """Replies from a script; each reply is one model call."""

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.replies = _script()

        def _inner_get_response(self, *, messages: Any, stream: bool, options: Any, **kwargs: Any) -> Any:
            CALLS["model"] += 1
            contents = self.replies.pop(0)
            if stream:

                async def updates() -> Any:
                    yield ChatResponseUpdate(role="assistant", contents=contents, model="scripted")
                    yield ChatResponseUpdate(
                        role="assistant", contents=[Content.from_usage(USAGE)], model="scripted"
                    )

                return self._build_response_stream(updates())

            async def respond() -> Any:
                response = ChatResponse(
                    messages=[Message(role="assistant", contents=contents)],
                    model="scripted", usage_details=USAGE, raw_representation={"live": True},
                )
                SEEN.setdefault("responses", []).append(response)
                return response

            return respond()


@pytest.fixture
def seams() -> Iterator[InMemoryTransport]:
    from pytracer_agent_framework import install, uninstall

    sink = InMemoryTransport()
    previous = pytracer.get_tracer()
    pytracer.configure(sink, service_name="af", branch="main", commit="c0")
    install()
    CALLS.update(model=0, tool=0)
    SEEN.clear()
    try:
        yield sink
    finally:
        uninstall()
        pytracer.set_tracer(previous)


def _agent_fn(*, stream: bool = False) -> Any:
    def run() -> str:
        agent = Agent(client=ScriptedClient(), name="assistant", instructions="help", tools=[weather])

        async def go() -> str:
            if not stream:
                response = await agent.run("weather in Paris?")
                SEEN["final"] = response
                return str(response.text)
            updates = agent.run("weather in Paris?", stream=True)
            async for _ in updates:
                pass
            return str((await updates.get_final_response()).text)

        return asyncio.run(go())

    return run


def _capture(sink: InMemoryTransport, run: Any) -> tuple[str, list[Span]]:
    output = run()
    pytracer.get_tracer().flush()
    # Through the JSONL format and the codec, as a saved recording is read back.
    spans = [Span.from_json(s.to_json()) for s in sink.spans]
    return output, spans


@pytest.mark.parametrize("stream", [False, True])
def test_capture_records_model_and_tool_calls_once_under_the_agent(
    seams: InMemoryTransport, stream: bool
) -> None:
    output, spans = _capture(seams, _agent_fn(stream=stream))
    assert output == "It is sunny in Paris."
    llm = [s for s in spans if s.kind is SpanKind.LLM]
    tools = [s for s in spans if s.kind is SpanKind.TOOL]
    agents = [s for s in spans if s.kind is SpanKind.AGENT]
    assert len(llm) == 2 and len(tools) == 1
    assert len(agents) == 1 and agents[0].name == "assistant"  # Agent.run and RawAgent.run: one span
    assert all(s.parent_span_id == agents[0].span_id for s in llm + tools)
    assert tools[0].tool_name == "weather"
    assert agents[0].output == "It is sunny in Paris."


@pytest.mark.parametrize("stream", [False, True])
def test_captured_model_calls_carry_their_token_counts(seams: InMemoryTransport, stream: bool) -> None:
    _, spans = _capture(seams, _agent_fn(stream=stream))
    llm = [s for s in spans if s.kind is SpanKind.LLM]
    assert len(llm) == 2 and all(s.token_count == 5 for s in llm)


def test_capture_returns_the_frameworks_own_objects(seams: InMemoryTransport) -> None:
    # Recording stores plain data, but a live call must hand the agent the exact
    # object the client returned (raw_representation and all), not a rebuilt copy.
    _capture(seams, _agent_fn())
    assert all(r.raw_representation == {"live": True} for r in SEEN["responses"])


@pytest.mark.parametrize("stream", [False, True])
def test_full_replay_runs_nothing_live(seams: InMemoryTransport, stream: bool) -> None:
    from pytracer_replay import MockSet, ReplayEngine, ReplayMode

    output, spans = _capture(seams, _agent_fn(stream=stream))
    before = dict(CALLS)
    result = ReplayEngine(ReplayMode.FULL, strict=True).replay(
        _agent_fn(stream=stream), MockSet.from_trace(spans)
    )
    assert result.error_message is None
    assert result.return_value == output
    assert CALLS == before  # neither the model nor the tool ran again
    assert result.uninstrumented is False
    assert not result.side_effects.effects


def test_replayed_responses_keep_their_types(seams: InMemoryTransport) -> None:
    from pytracer_replay import MockSet, ReplayEngine, ReplayMode

    _, spans = _capture(seams, _agent_fn())
    ReplayEngine(ReplayMode.FULL).replay(_agent_fn(), MockSet.from_trace(spans))
    final = SEEN["final"]
    assert type(final).__name__ == "AgentResponse"
    assert final.usage_details and final.usage_details.get("total_token_count") == 10


def test_a_changed_recording_changes_the_replay(seams: InMemoryTransport) -> None:
    from pytracer_replay import MockSet, ReplayEngine, ReplayMode

    _, spans = _capture(seams, _agent_fn())
    for span in spans:
        if span.kind is SpanKind.TOOL:
            span.output = {"contents": [Content.from_text("storming in Paris").to_dict()]}
    # The second model call was recorded after "sunny"; with "storming" its input no
    # longer matches, so the recording is genuinely what drives the replay.
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


def test_a_client_defined_after_install_is_captured(seams: InMemoryTransport) -> None:
    class LateClient(ScriptedClient):
        def _inner_get_response(self, **kwargs: Any) -> Any:
            return super()._inner_get_response(**kwargs)

    def run() -> str:
        agent = Agent(client=LateClient(), name="late", tools=[weather])
        return str(asyncio.run(agent.run("weather in Paris?")).text)

    _, spans = _capture(seams, run)
    # The subclass and its parent are both wrapped; the nested call is not recorded twice.
    assert len([s for s in spans if s.kind is SpanKind.LLM]) == 2


def test_uninstall_restores_the_framework(seams: InMemoryTransport) -> None:
    from agent_framework import FunctionTool, RawAgent

    from pytracer_agent_framework import uninstall

    uninstall()
    assert not getattr(ScriptedClient._inner_get_response, "__pytracer_seam__", False)
    assert not getattr(FunctionTool.invoke, "__pytracer_seam__", False)
    assert not getattr(RawAgent.run, "__pytracer_seam__", False)
    assert asyncio.run(Agent(client=ScriptedClient(), tools=[weather]).run("hi")).text

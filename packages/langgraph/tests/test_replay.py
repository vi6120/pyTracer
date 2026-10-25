"""A run captured through the LangGraph adapter replays without @op (C3).

A real StateGraph (model node with a tool-calling chat model, a ToolNode, and
tools_condition) runs with a scripted chat model (no network), is captured with
the callback handler and ``install_seams()``, written to JSONL and reloaded, then
replayed in full mode with the network sandbox on. Skipped without LangGraph.
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
    importlib.util.find_spec("langgraph") is None, reason="langgraph not installed"
)

LIVE = {"model": 0, "tool": 0}

if importlib.util.find_spec("langgraph") is not None:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, AIMessageChunk
    from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
    from langchain_core.tools import tool
    from langgraph.graph import START, MessagesState, StateGraph
    from langgraph.prebuilt import ToolNode, tools_condition

    @tool
    def lookup(city: str) -> str:
        """Weather for a city."""
        LIVE["tool"] += 1
        return f"sunny in {city}"

    def _script() -> list[AIMessage]:
        return [
            AIMessage(content="", tool_calls=[{"name": "lookup", "args": {"city": "Paris"}, "id": "call_1"}]),
            AIMessage(content="It is sunny in Paris."),
        ]

    class ScriptedChat(BaseChatModel):
        """A chat model that answers from a script, one reply per call."""

        replies: list[Any] = []

        @property
        def _llm_type(self) -> str:
            return "scripted"

        def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
            return self.bind(tools=[t.name for t in tools], **kwargs)

        def _generate(self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
            LIVE["model"] += 1
            return ChatResult(generations=[ChatGeneration(message=self.replies.pop(0))])

        def _stream(self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any) -> Any:
            LIVE["model"] += 1
            reply = self.replies.pop(0)
            if reply.tool_calls:
                yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=[
                    {"name": c["name"], "args": '{"city": "Paris"}', "id": c["id"], "index": 0}
                    for c in reply.tool_calls
                ]))
                return
            for word in str(reply.content).split(" "):
                yield ChatGenerationChunk(message=AIMessageChunk(content=word + " "))


def _graph(*, stream: bool = False) -> Any:
    model = ScriptedChat(replies=_script()).bind_tools([lookup])

    def agent(state: Any) -> dict[str, Any]:
        if stream:
            reply = None
            for chunk in model.stream(state["messages"]):
                reply = chunk if reply is None else reply + chunk
            return {"messages": [reply]}
        return {"messages": [model.invoke(state["messages"])]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode([lookup]))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    return graph.compile()


@pytest.fixture
def capture() -> Iterator[tuple[InMemoryTransport, Any]]:
    from pytracer_langgraph import PyTracerCallbackHandler, install_seams, uninstall_seams

    sink = InMemoryTransport()
    previous = pytracer.get_tracer()
    tracer = pytracer.configure(sink, service_name="lg", branch="main", commit="c0")
    install_seams()
    handler = PyTracerCallbackHandler(tracer)
    LIVE.update(model=0, tool=0)
    try:
        yield sink, handler
    finally:
        uninstall_seams()
        pytracer.set_tracer(previous)


def _run(handler: Any, *, stream: bool = False) -> Any:
    def run() -> str:
        state = _graph(stream=stream).invoke(
            {"messages": [("user", "weather in Paris?")]}, config={"callbacks": [handler]}
        )
        return str(state["messages"][-1].content).strip()

    return run


def _recorded(sink: InMemoryTransport, run: Any) -> tuple[str, list[Span]]:
    output = run()
    pytracer.get_tracer().flush()
    return output, [Span.from_json(s.to_json()) for s in sink.spans]


def test_capture_records_each_call_once_under_its_node(capture: tuple[InMemoryTransport, Any]) -> None:
    sink, handler = capture
    output, spans = _recorded(sink, _run(handler))
    assert output == "It is sunny in Paris."
    llm = [s for s in spans if s.kind is SpanKind.LLM]
    tools = [s for s in spans if s.kind is SpanKind.TOOL]
    assert len(llm) == 2 and len(tools) == 1  # no duplicate callback spans
    nodes = {s.span_id: s.name for s in spans}
    assert [nodes.get(s.parent_span_id or "") for s in llm] == ["agent", "agent"]
    assert nodes.get(tools[0].parent_span_id or "") == "tools"
    assert type(llm[0].output).__name__ == "AIMessage"  # real type after the codec


def test_full_replay_runs_nothing_live(capture: tuple[InMemoryTransport, Any]) -> None:
    from pytracer_replay import MockSet, ReplayEngine, ReplayMode

    sink, handler = capture
    output, spans = _recorded(sink, _run(handler))
    before = dict(LIVE)
    result = ReplayEngine(ReplayMode.FULL).replay(_run(handler), MockSet.from_trace(spans))
    assert result.error_message is None
    assert result.return_value == output
    assert LIVE == before
    assert result.uninstrumented is False
    assert not result.side_effects.effects


def test_a_changed_recording_changes_the_replay(capture: tuple[InMemoryTransport, Any]) -> None:
    from pytracer_replay import MockSet, ReplayEngine, ReplayMode

    sink, handler = capture
    _, spans = _recorded(sink, _run(handler))
    for span in spans:
        if span.kind is SpanKind.TOOL:
            span.output = span.output.model_copy(update={"content": "storming in Paris"})
    strict = ReplayEngine(ReplayMode.FULL, strict=True).replay(_run(handler), MockSet.from_trace(spans))
    assert strict.failed  # the second model call was recorded after "sunny"
    assert "no recording" in (strict.error_message or "")


def test_partial_replay_freezes_the_tool_and_runs_the_model(capture: tuple[InMemoryTransport, Any]) -> None:
    from pytracer_replay import MockSet, ReplayEngine, ReplayMode

    sink, handler = capture
    output, spans = _recorded(sink, _run(handler))
    before = dict(LIVE)
    result = ReplayEngine(ReplayMode.PARTIAL).replay(_run(handler), MockSet.from_trace(spans))
    assert result.return_value == output
    assert LIVE["tool"] == before["tool"]
    assert LIVE["model"] == before["model"] + 2


def test_streamed_run_replays(capture: tuple[InMemoryTransport, Any]) -> None:
    from pytracer_replay import MockSet, ReplayEngine, ReplayMode

    sink, handler = capture
    output, spans = _recorded(sink, _run(handler, stream=True))
    assert output == "It is sunny in Paris."
    before = dict(LIVE)
    result = ReplayEngine(ReplayMode.FULL).replay(_run(handler, stream=True), MockSet.from_trace(spans))
    assert result.error_message is None
    assert result.return_value == output
    assert LIVE == before


def test_async_run_nests_under_start_span_and_replays(capture: tuple[InMemoryTransport, Any]) -> None:
    # The ainvoke path: the graph and its model run async, inside an enclosing
    # pytracer.start_span, so the whole request is one trace and still replays.
    import asyncio

    from pytracer_replay import MockSet, ReplayEngine, ReplayMode

    sink, handler = capture

    def run() -> str:
        model = ScriptedChat(replies=_script()).bind_tools([lookup])

        async def agent(state: Any) -> dict[str, Any]:
            return {"messages": [await model.ainvoke(state["messages"])]}

        graph = StateGraph(MessagesState)
        graph.add_node("agent", agent)
        graph.add_node("tools", ToolNode([lookup]))
        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", tools_condition)
        graph.add_edge("tools", "agent")
        app = graph.compile()

        async def request() -> str:
            with pytracer.start_span("request"):
                state = await app.ainvoke(
                    {"messages": [("user", "weather in Paris?")]}, config={"callbacks": [handler]}
                )
            return str(state["messages"][-1].content)

        return asyncio.run(request())

    output, spans = _recorded(sink, run)
    root = next(s for s in spans if s.name == "request")
    assert all(s.trace_id == root.trace_id for s in spans)  # one trace for the request
    assert len([s for s in spans if s.kind is SpanKind.LLM]) == 2
    before = dict(LIVE)
    result = ReplayEngine(ReplayMode.FULL).replay(run, MockSet.from_trace(spans))
    assert result.error_message is None
    assert result.return_value == output
    assert LIVE == before

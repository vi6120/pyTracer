"""Adapter replay through the production path: gateway -> store -> record -> replay.

For each framework adapter, a real run (scripted model, no network) is captured
with replay switched on, sent to the ingestion gateway exactly as HTTPTransport
sends it, written to ClickHouse by the gateway's own pipeline, read back, recorded
to JSONL as ``pytracer record`` does, reloaded, and replayed in full mode with the
network sandbox on. The replay must return the same answer with neither the model
nor the tools running again. Needs ClickHouse (docker compose); each adapter case
skips when its framework is not installed.
"""

from __future__ import annotations

import asyncio
import importlib.util
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import pytracer_sdk as pytracer
from pytracer_sdk import Span, SpanKind
from pytracer_sdk.schema import Resource
from pytracer_sdk.transport import InMemoryTransport

from pytracer_stores import TraceStore
from pytracer_stores.config import ClickHouseConfig

RES = Resource(service_name="adapter-replay", branch="main", commit="abc123")
LIVE = {"model": 0, "tool": 0}


def _installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _through_gateway_and_store(
    spans: list[Span], trace_store: TraceStore, tmp_path: Path
) -> list[Span]:
    """Send spans through the gateway into the store, read them back, record them to
    JSONL, and reload them the way replay reads a recording."""
    from fastapi.testclient import TestClient
    from pytracer_cli.traces import record_trace
    from pytracer_gateway import create_app
    from pytracer_gateway.ingest import TraceStoreWriter

    stamped = [s.model_copy(update={"resource": RES}) for s in spans]  # the store needs branch/commit
    body = {"spans": [s.model_dump(mode="json", exclude_none=True) for s in stamped]}  # as HTTPTransport
    app = create_app(writer=TraceStoreWriter(trace_store), api_keys={"key": "adapters"})
    # Read back through a separate client: the gateway's writer thread owns the
    # first one, and a ClickHouse session takes one query at a time.
    reader = TraceStore(ClickHouseConfig(), table=trace_store.table)
    with TestClient(app) as client:
        response = client.post("/v1/spans", json=body, headers={"X-API-Key": "key"})
        assert response.status_code == 202, response.text
        assert response.json()["accepted"] == len(stamped)
        trace_ids = sorted({s.trace_id for s in stamped})
        deadline = time.monotonic() + 15
        read_back: list[Span] = []
        while time.monotonic() < deadline:
            read_back = [s for tid in trace_ids for s in reader.get_trace(tid, project="adapters")]
            if len(read_back) == len(stamped):
                break
            time.sleep(0.1)
    reader.close()
    assert len(read_back) == len(stamped), "the gateway did not write every span"
    record_trace(read_back, agent="tests:agent", entry={}, out_dir=tmp_path)
    jsonl = next((tmp_path / "traces").glob("*.jsonl"))
    return [Span.from_json(line) for line in jsonl.read_text().splitlines() if line.strip()]


def _check_replay(run: Callable[[], str], spans: list[Span], output: str) -> None:
    from pytracer_replay import MockSet, ReplayEngine, ReplayMode

    assert any(s.kind is SpanKind.LLM for s in spans) and any(s.kind is SpanKind.TOOL for s in spans)
    before = dict(LIVE)
    result = ReplayEngine(ReplayMode.FULL).replay(run, MockSet.from_trace(spans))
    assert result.error_message is None
    assert result.return_value == output
    assert LIVE == before  # nothing ran live: every answer came from the store's recording
    assert result.uninstrumented is False


def _capture(run: Callable[[], str]) -> tuple[str, list[Span]]:
    sink = InMemoryTransport()
    tracer = pytracer.configure(sink, service_name="adapter-replay", branch="main", commit="abc123")
    LIVE.update(model=0, tool=0)
    output = run()
    tracer.flush()
    return output, list(sink.spans)


@pytest.fixture(autouse=True)
def _restore_tracer() -> Any:
    previous = pytracer.get_tracer()
    yield
    pytracer.set_tracer(previous)


# --- OpenAI Agents --------------------------------------------------------------


@pytest.mark.skipif(not _installed("agents"), reason="openai-agents not installed")
def test_openai_agents_run_replays_from_the_store(trace_store: TraceStore, tmp_path: Path) -> None:
    from agents import Agent, Runner, function_tool
    from agents.items import ModelResponse
    from agents.models.interface import Model
    from agents.usage import Usage
    from openai.types.responses import (
        ResponseFunctionToolCall,
        ResponseOutputMessage,
        ResponseOutputText,
    )

    from pytracer_openai_agents import install, uninstall_seams

    @function_tool
    def weather(city: str) -> str:
        """Weather for a city."""
        LIVE["tool"] += 1
        return f"sunny in {city}"

    class Scripted(Model):
        def __init__(self) -> None:
            self.replies: list[Any] = [
                ResponseFunctionToolCall(
                    id="fc_1", call_id="call_1", name="weather", arguments='{"city": "Paris"}',
                    type="function_call",
                ),
                ResponseOutputMessage(
                    id="msg_1", role="assistant", status="completed", type="message",
                    content=[ResponseOutputText(text="It is sunny in Paris.", type="output_text", annotations=[])],
                ),
            ]

        async def get_response(self, *args: Any, **kwargs: Any) -> ModelResponse:
            LIVE["model"] += 1
            return ModelResponse(output=[self.replies.pop(0)], usage=Usage(requests=1), response_id=None)

        async def stream_response(self, *args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError
            yield  # pragma: no cover

    def run() -> str:
        agent = Agent(name="assistant", instructions="help", tools=[weather], model=Scripted())
        return str(Runner.run_sync(agent, "weather in Paris?").final_output)

    install(pytracer.get_tracer(), replayable=True)
    try:
        output, spans = _capture(run)
        _check_replay(run, _through_gateway_and_store(spans, trace_store, tmp_path), output)
    finally:
        uninstall_seams()


# --- CrewAI ---------------------------------------------------------------------


@pytest.mark.skipif(not _installed("crewai"), reason="crewai not installed")
def test_crewai_run_replays_from_the_store(trace_store: TraceStore, tmp_path: Path) -> None:
    from pytracer_crewai import PyTracerEventListener, install_seams, uninstall_seams

    sink = InMemoryTransport()
    listener = PyTracerEventListener(
        pytracer.configure(sink, service_name="adapter-replay", branch="main", commit="abc123")
    )
    install_seams(listener)
    try:
        from crewai import Agent, Crew, Task
        from crewai.events import crewai_event_bus
        from crewai.llms.base_llm import BaseLLM
        from crewai.tools import tool

        replies = [
            'Thought: I should look it up.\nAction: lookup\nAction Input: {"city": "Paris"}',
            "Thought: I now know the final answer\nFinal Answer: It is sunny in Paris.",
        ]

        class Scripted(BaseLLM):
            def call(self, messages: Any, *args: Any, **kwargs: Any) -> str:
                turn = getattr(self, "_n", 0)
                object.__setattr__(self, "_n", turn + 1)
                LIVE["model"] += 1
                return replies[turn]

            def supports_function_calling(self) -> bool:
                return False

        @tool("lookup")
        def lookup(city: str) -> str:
            """Weather for a city."""
            LIVE["tool"] += 1
            return f"sunny in {city}"

        def run() -> str:
            agent = Agent(
                role="forecaster", goal="report the weather", backstory="careful",
                llm=Scripted(model="scripted"), tools=[lookup], verbose=False,
            )
            task = Task(description="weather in Paris?", expected_output="one sentence", agent=agent)
            output = Crew(agents=[agent], tasks=[task], verbose=False).kickoff()
            crewai_event_bus.flush()
            return str(output.raw)

        LIVE.update(model=0, tool=0)
        output = run()
        pytracer.get_tracer().flush()
        spans = list(sink.spans)
        _check_replay(run, _through_gateway_and_store(spans, trace_store, tmp_path), output)
    finally:
        uninstall_seams()


# --- LangGraph ------------------------------------------------------------------


@pytest.mark.skipif(not _installed("langgraph"), reason="langgraph not installed")
def test_langgraph_run_replays_from_the_store(trace_store: TraceStore, tmp_path: Path) -> None:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.tools import tool
    from langgraph.graph import START, MessagesState, StateGraph
    from langgraph.prebuilt import ToolNode, tools_condition

    from pytracer_langgraph import PyTracerCallbackHandler, install_seams, uninstall_seams

    @tool
    def lookup(city: str) -> str:
        """Weather for a city."""
        LIVE["tool"] += 1
        return f"sunny in {city}"

    class Scripted(BaseChatModel):
        replies: list[Any] = []

        @property
        def _llm_type(self) -> str:
            return "scripted"

        def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
            return self.bind(tools=[t.name for t in tools], **kwargs)

        def _generate(self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
            LIVE["model"] += 1
            return ChatResult(generations=[ChatGeneration(message=self.replies.pop(0))])

    install_seams()
    try:
        sink = InMemoryTransport()
        tracer = pytracer.configure(sink, service_name="adapter-replay", branch="main", commit="abc123")
        handler = PyTracerCallbackHandler(tracer)

        def run() -> str:
            model = Scripted(replies=[
                AIMessage(content="", tool_calls=[{"name": "lookup", "args": {"city": "Paris"}, "id": "call_1"}]),
                AIMessage(content="It is sunny in Paris."),
            ]).bind_tools([lookup])
            graph = StateGraph(MessagesState)
            graph.add_node("agent", lambda state: {"messages": [model.invoke(state["messages"])]})
            graph.add_node("tools", ToolNode([lookup]))
            graph.add_edge(START, "agent")
            graph.add_conditional_edges("agent", tools_condition)
            graph.add_edge("tools", "agent")
            state = graph.compile().invoke(
                {"messages": [("user", "weather in Paris?")]}, config={"callbacks": [handler]}
            )
            return str(state["messages"][-1].content)

        LIVE.update(model=0, tool=0)
        output = run()
        tracer.flush()
        spans = list(sink.spans)
        _check_replay(run, _through_gateway_and_store(spans, trace_store, tmp_path), output)
    finally:
        uninstall_seams()


# --- AutoGen --------------------------------------------------------------------


@pytest.mark.skipif(not _installed("autogen_agentchat"), reason="autogen-agentchat not installed")
def test_autogen_run_replays_from_the_store(trace_store: TraceStore, tmp_path: Path) -> None:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.conditions import MaxMessageTermination
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_core import FunctionCall
    from autogen_core.models import ChatCompletionClient, CreateResult, ModelInfo, RequestUsage
    from opentelemetry import trace as otel_trace

    from pytracer_autogen import exporter_for, install_seams, pytracer_tracer_provider, uninstall_seams

    # pyTracer as the global OpenTelemetry provider, as install() does (the global
    # can be set once per process, so it is reused if already ours).
    exporter = exporter_for(otel_trace.get_tracer_provider())
    if exporter is None:
        provider = pytracer_tracer_provider()
        otel_trace.set_tracer_provider(provider)
        if otel_trace.get_tracer_provider() is not provider:
            pytest.skip("another global OpenTelemetry provider is already registered")
        exporter = exporter_for(provider)
    assert exporter is not None

    def lookup(q: str) -> str:
        """Look something up."""
        LIVE["tool"] += 1
        return f"found {q}"

    install_seams(exporter)
    try:

        class Scripted(ChatCompletionClient):
            def __init__(self, replies: list[Any]) -> None:
                self._replies = list(replies)

            async def create(self, messages: Any, **kwargs: Any) -> CreateResult:
                LIVE["model"] += 1
                reply = self._replies.pop(0)
                return CreateResult(
                    finish_reason="function_calls" if isinstance(reply, list) else "stop",
                    content=reply, usage=RequestUsage(prompt_tokens=3, completion_tokens=2), cached=False,
                )

            def create_stream(self, *args: Any, **kwargs: Any) -> Any:
                raise NotImplementedError

            async def close(self) -> None:
                pass

            def actual_usage(self) -> RequestUsage:
                return RequestUsage(prompt_tokens=0, completion_tokens=0)

            def total_usage(self) -> RequestUsage:
                return RequestUsage(prompt_tokens=0, completion_tokens=0)

            def count_tokens(self, *args: Any, **kwargs: Any) -> int:
                return 0

            def remaining_tokens(self, *args: Any, **kwargs: Any) -> int:
                return 1000

            @property
            def capabilities(self) -> Any:
                return self.model_info

            @property
            def model_info(self) -> ModelInfo:
                return ModelInfo(
                    vision=False, function_calling=True, json_output=False,
                    family="unknown", structured_output=False,
                )

        def run() -> str:
            call = FunctionCall(id="1", name="lookup", arguments='{"q": "x"}')
            writer = AssistantAgent("writer", model_client=Scripted([[call], "draft"]), tools=[lookup])
            critic = AssistantAgent("critic", model_client=Scripted(["ok"]))
            team = RoundRobinGroupChat([writer, critic], termination_condition=MaxMessageTermination(4))
            result = asyncio.run(team.run(task="write"))
            return " | ".join(f"{m.source}: {m.to_text()}" for m in result.messages)

        output, spans = _capture(run)
        _check_replay(run, _through_gateway_and_store(spans, trace_store, tmp_path), output)
    finally:
        uninstall_seams()

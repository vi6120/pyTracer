"""A real AgentChat team run is recorded as one trace (P2).

Runs a two-agent RoundRobinGroupChat with a scripted model client (no network),
with pyTracer registered as the global OpenTelemetry provider as the README says.
Skipped when autogen-agentchat is not installed.
"""

from __future__ import annotations

import asyncio
import importlib.util
from collections.abc import Iterator
from typing import Any

import pytest
import pytracer_sdk as pytracer
from opentelemetry import trace as otel_trace
from pytracer_sdk import SpanKind as K
from pytracer_sdk import SpanStatus
from pytracer_sdk.transport import InMemoryTransport

from pytracer_autogen import exporter_for, pytracer_tracer_provider

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("autogen_agentchat") is None, reason="autogen-agentchat not installed"
)


LIVE = {"model": 0, "tool": 0}


def _lookup(q: str) -> str:
    """Look something up."""
    LIVE["tool"] += 1
    return f"found {q}"


def _team(tool: Any = _lookup, *, log_calls: bool = False, stream: bool = False) -> Any:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.conditions import MaxMessageTermination
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_core import FunctionCall
    from autogen_core.models import ChatCompletionClient, CreateResult, ModelInfo, RequestUsage

    class Scripted(ChatCompletionClient):
        """Returns canned replies in order, so the team runs without a model."""

        def __init__(self, replies: list[Any], *, log_calls: bool) -> None:
            self._replies = list(replies)
            self._log_calls = log_calls

        async def create(self, messages: Any, **kwargs: Any) -> CreateResult:
            LIVE["model"] += 1
            reply = self._replies.pop(0)
            if self._log_calls:
                # As the autogen_ext clients (OpenAI, Anthropic, Ollama, ...) do.
                import logging

                from autogen_core.logging import LLMCallEvent

                logging.getLogger("autogen_core.events").info(
                    LLMCallEvent(
                        messages=[{"role": "user", "content": "task"}],
                        response={"model": "scripted", "choices": [{"message": {"content": str(reply)}}]},
                        prompt_tokens=3,
                        completion_tokens=2,
                    )
                )
            return CreateResult(
                finish_reason="function_calls" if isinstance(reply, list) else "stop",
                content=reply,
                usage=RequestUsage(prompt_tokens=3, completion_tokens=2),
                cached=False,
            )

        async def create_stream(self, messages: Any, **kwargs: Any) -> Any:
            result = await self.create(messages, **kwargs)
            if isinstance(result.content, str):
                for word in result.content.split(" "):
                    yield word
            yield result

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

    call = FunctionCall(id="1", name=tool.__name__, arguments='{"q": "x"}')
    writer = AssistantAgent(
        "writer", model_client=Scripted([[call], "draft"], log_calls=log_calls), tools=[tool],
        model_client_stream=stream,
    )
    critic = AssistantAgent(
        "critic", model_client=Scripted(["ok"], log_calls=log_calls), model_client_stream=stream
    )
    return RoundRobinGroupChat([writer, critic], termination_condition=MaxMessageTermination(4))


_PROVIDER = None
_EXPORTER = None


@pytest.fixture
def recorded() -> Iterator[tuple[InMemoryTransport, pytracer.Tracer]]:
    # As the README says: pyTracer is the global OpenTelemetry provider. OTel lets
    # the global be set once per process, so register a provider that resolves the
    # tracer lazily once, and point the default tracer at each test's sink.
    global _PROVIDER, _EXPORTER
    if _PROVIDER is None:
        _PROVIDER = pytracer_tracer_provider()
        _EXPORTER = exporter_for(_PROVIDER)
        otel_trace.set_tracer_provider(_PROVIDER)
    if otel_trace.get_tracer_provider() is not _PROVIDER:
        pytest.skip("another global OpenTelemetry provider is already registered")
    assert _EXPORTER is not None
    sink = InMemoryTransport()
    tracer = pytracer.Tracer(sink, resource=pytracer.Resource(service_name="ag", branch="main", commit="c0"))
    previous = pytracer.get_tracer()
    pytracer.set_tracer(tracer)
    try:
        yield sink, tracer
    finally:
        pytracer.set_tracer(previous)


def test_team_run_is_one_trace(recorded: tuple[InMemoryTransport, pytracer.Tracer]) -> None:
    sink, tracer = recorded
    team = _team()
    asyncio.run(team.run(task="write"))
    tracer.flush()
    run = [s for s in sink.spans if not s.name.startswith("create_agent")]
    assert {s.agent_name for s in run if s.kind is K.AGENT} == {"writer", "critic"}
    assert len({s.trace_id for s in run}) == 1
    tool = next(s for s in run if s.kind is K.TOOL)
    assert tool.parent_span_id in {s.span_id for s in run if s.agent_name == "writer"}
    assert tool.output == "found x"  # joined from AutoGen's result message


def _broken(q: str) -> str:
    """A tool that fails."""
    raise RuntimeError("index offline")


def test_failed_tool_is_recorded_as_an_error(recorded: tuple[InMemoryTransport, pytracer.Tracer]) -> None:
    sink, tracer = recorded
    asyncio.run(_team(tool=_broken).run(task="write"))
    tracer.flush()
    tool = next(s for s in sink.spans if s.kind is K.TOOL)
    assert tool.status is SpanStatus.ERROR
    assert "index offline" in (tool.error_message or "")


def test_team_run_nests_under_an_enclosing_span(
    recorded: tuple[InMemoryTransport, pytracer.Tracer],
) -> None:
    sink, tracer = recorded

    async def request() -> None:
        with tracer.start_span("request"):
            await _team().run(task="write")

    asyncio.run(request())
    tracer.flush()
    root = next(s for s in sink.spans if s.name == "request")
    assert all(s.trace_id == root.trace_id for s in sink.spans)  # agents built inside too
    agents = [s for s in sink.spans if s.name.startswith("invoke_agent")]
    assert agents and all(s.parent_span_id == root.span_id for s in agents)


def _model_calls_by_agent(sink: InMemoryTransport) -> dict[str, int]:
    agents = {s.span_id: s.agent_name for s in sink.spans if s.kind is K.AGENT}
    counts: dict[str, int] = {}
    for s in sink.spans:
        if s.kind is K.LLM:
            name = agents.get(s.parent_span_id or "", "?") or "?"
            counts[name] = counts.get(name, 0) + 1
    return counts


def test_logged_model_calls_are_recorded_under_their_agent_turns(
    recorded: tuple[InMemoryTransport, pytracer.Tracer],
) -> None:
    sink, tracer = recorded
    asyncio.run(_team(log_calls=True).run(task="write"))
    tracer.flush()
    llm = [s for s in sink.spans if s.kind is K.LLM]
    assert _model_calls_by_agent(sink) == {"writer": 2, "critic": 1}
    assert all(s.name == "chat scripted" and s.token_count == 5 for s in llm)
    assert not any(s.attributes.get("pytracer.derived") for s in llm)  # no duplicates


def test_model_calls_are_derived_when_the_client_logs_nothing(
    recorded: tuple[InMemoryTransport, pytracer.Tracer],
) -> None:
    sink, tracer = recorded
    asyncio.run(_team(log_calls=False).run(task="write"))
    tracer.flush()
    llm = [s for s in sink.spans if s.kind is K.LLM]
    assert _model_calls_by_agent(sink) == {"writer": 2, "critic": 1}
    assert all(s.attributes.get("pytracer.derived") is True for s in llm)
    assert {str(s.output) for s in llm} >= {"draft", "ok"}


# --- C2: replaying a captured AutoGen run without @op -----------------------------


@pytest.fixture
def seams(recorded: tuple[InMemoryTransport, pytracer.Tracer]) -> Iterator[InMemoryTransport]:
    from pytracer_autogen import install_seams, uninstall_seams

    sink, _ = recorded
    assert _EXPORTER is not None
    install_seams(_EXPORTER)
    try:
        yield sink
    finally:
        uninstall_seams()


def _run_team(*, stream: bool = False) -> Any:
    def run() -> str:
        result = asyncio.run(_team(stream=stream).run(task="write"))
        return " | ".join(f"{m.source}: {m.to_text()}" for m in result.messages)

    return run


def _capture(sink: InMemoryTransport, run: Any) -> tuple[str, list[pytracer.Span]]:
    LIVE.update(model=0, tool=0)
    output = run()
    pytracer.get_tracer().flush()
    return output, [pytracer.Span.from_json(s.to_json()) for s in sink.spans]


def test_capture_records_each_call_once_under_its_agent_turn(seams: InMemoryTransport) -> None:
    _, spans = _capture(seams, _run_team())
    llm = [s for s in spans if s.kind is K.LLM]
    tools = [s for s in spans if s.kind is K.TOOL]
    assert len(llm) == 3 and len(tools) == 1  # no derived or duplicate AutoGen spans
    assert not any(s.attributes.get("pytracer.derived") for s in llm)
    agents = {s.span_id: s.agent_name for s in spans if s.kind is K.AGENT}
    assert sorted(agents[s.parent_span_id] for s in llm) == ["critic", "writer", "writer"]
    assert agents[tools[0].parent_span_id] == "writer"
    assert tools[0].output == "found x"
    assert type(llm[0].output).__name__ == "CreateResult"  # real type after the codec


def test_full_replay_runs_nothing_live(seams: InMemoryTransport) -> None:
    from pytracer_replay import MockSet, ReplayEngine, ReplayMode

    output, spans = _capture(seams, _run_team())
    before = dict(LIVE)
    result = ReplayEngine(ReplayMode.FULL).replay(_run_team(), MockSet.from_trace(spans))
    assert result.error_message is None
    assert result.return_value == output
    assert LIVE == before  # neither a model nor the tool ran again
    assert result.uninstrumented is False
    assert not result.side_effects.effects


def test_a_changed_recording_changes_the_replay(seams: InMemoryTransport) -> None:
    from pytracer_replay import MockSet, ReplayEngine, ReplayMode

    _, spans = _capture(seams, _run_team())
    for span in spans:
        if span.kind is K.TOOL:
            span.output = "nothing found"
    strict = ReplayEngine(ReplayMode.FULL, strict=True).replay(_run_team(), MockSet.from_trace(spans))
    assert strict.failed  # the writer's next model call was recorded after "found x"
    assert "no recording" in (strict.error_message or "")


def test_partial_replay_freezes_the_tool_and_runs_the_models(seams: InMemoryTransport) -> None:
    from pytracer_replay import MockSet, ReplayEngine, ReplayMode

    output, spans = _capture(seams, _run_team())
    before = dict(LIVE)
    result = ReplayEngine(ReplayMode.PARTIAL).replay(_run_team(), MockSet.from_trace(spans))
    assert result.return_value == output
    assert LIVE["tool"] == before["tool"]
    assert LIVE["model"] == before["model"] + 3


def test_streamed_team_run_replays(seams: InMemoryTransport) -> None:
    from pytracer_replay import MockSet, ReplayEngine, ReplayMode

    output, spans = _capture(seams, _run_team(stream=True))
    before = dict(LIVE)
    result = ReplayEngine(ReplayMode.FULL).replay(_run_team(stream=True), MockSet.from_trace(spans))
    assert result.error_message is None
    assert result.return_value == output
    assert LIVE == before

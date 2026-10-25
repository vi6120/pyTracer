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

from pytracer_autogen import pytracer_tracer_provider

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("autogen_agentchat") is None, reason="autogen-agentchat not installed"
)


def _lookup(q: str) -> str:
    """Look something up."""
    return f"found {q}"


def _team(tool: Any = _lookup) -> Any:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.conditions import MaxMessageTermination
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_core import FunctionCall
    from autogen_core.models import ChatCompletionClient, CreateResult, ModelInfo, RequestUsage

    class Scripted(ChatCompletionClient):
        """Returns canned replies in order, so the team runs without a model."""

        def __init__(self, replies: list[Any]) -> None:
            self._replies = list(replies)

        async def create(self, messages: Any, **kwargs: Any) -> CreateResult:
            reply = self._replies.pop(0)
            return CreateResult(
                finish_reason="function_calls" if isinstance(reply, list) else "stop",
                content=reply,
                usage=RequestUsage(prompt_tokens=3, completion_tokens=2),
                cached=False,
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

    call = FunctionCall(id="1", name=tool.__name__, arguments='{"q": "x"}')
    writer = AssistantAgent("writer", model_client=Scripted([[call], "draft"]), tools=[tool])
    critic = AssistantAgent("critic", model_client=Scripted(["ok"]))
    return RoundRobinGroupChat([writer, critic], termination_condition=MaxMessageTermination(4))


_PROVIDER = None


@pytest.fixture
def recorded() -> Iterator[tuple[InMemoryTransport, pytracer.Tracer]]:
    # As the README says: pyTracer is the global OpenTelemetry provider. OTel lets
    # the global be set once per process, so register a provider that resolves the
    # tracer lazily once, and point the default tracer at each test's sink.
    global _PROVIDER
    if _PROVIDER is None:
        _PROVIDER = pytracer_tracer_provider()
        otel_trace.set_tracer_provider(_PROVIDER)
    if otel_trace.get_tracer_provider() is not _PROVIDER:
        pytest.skip("another global OpenTelemetry provider is already registered")
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

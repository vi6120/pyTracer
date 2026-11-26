"""Capture and replay through Agent Framework's own OpenAI Chat Completions client.

The scripted client in test_replay.py checks the seams; this checks a real
connector end to end: its request preparation, its streamed and non-streamed
response parsing, and its own stream finalizer. The OpenAI SDK talks to a fake
HTTP server, so there is no network. Skipped when the connector is absent.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
from collections.abc import Iterator
from typing import Any

import pytest
import pytracer_sdk as pytracer
from pytracer_sdk import Span, SpanKind
from pytracer_sdk.transport import InMemoryTransport

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("agent_framework_openai") is None,
    reason="agent-framework-openai not installed",
)

HITS = {"http": 0, "tool": 0}
USAGE = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}


def _reply(turn: int) -> tuple[dict[str, Any], str]:
    if turn % 2 == 0:
        call = {"id": "call_1", "type": "function",
                "function": {"name": "weather", "arguments": '{"city": "Paris"}'}}
        return {"role": "assistant", "content": None, "tool_calls": [call]}, "tool_calls"
    return {"role": "assistant", "content": "It is sunny in Paris."}, "stop"


def _response(request: Any) -> Any:
    http = _http_module()
    turn = HITS["http"]
    HITS["http"] += 1
    stream = bool(json.loads(request.content).get("stream"))
    message, finish = _reply(turn)
    head = {"id": "c", "created": 0, "model": "gpt-test"}
    if not stream:
        body = {**head, "object": "chat.completion", "usage": USAGE,
                "choices": [{"index": 0, "message": message, "finish_reason": finish}]}
        return http.Response(200, json=body)
    delta = dict(message)
    if "tool_calls" in delta:
        delta["tool_calls"] = [dict(delta["tool_calls"][0], index=0)]
    chunks = [
        {**head, "object": "chat.completion.chunk",
         "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]},
        {**head, "object": "chat.completion.chunk", "choices": [], "usage": USAGE},
    ]
    text = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    return http.Response(200, text=text, headers={"content-type": "text/event-stream"})


def _http_module() -> Any:
    """The HTTP package the installed OpenAI SDK is built on (httpx, or httpx2 in 3.x)."""
    import inspect

    from openai import AsyncOpenAI

    annotation = str(inspect.signature(AsyncOpenAI.__init__).parameters["http_client"].annotation)
    return importlib.import_module("httpx2" if "httpx2" in annotation else "httpx")


def _agent_fn(*, stream: bool = False) -> Any:
    from agent_framework import Agent, tool
    from agent_framework_openai import OpenAIChatCompletionClient
    from openai import AsyncOpenAI

    @tool
    def weather(city: str) -> str:
        """Weather for a city."""
        HITS["tool"] += 1
        return f"sunny in {city}"

    def run() -> str:
        http = _http_module()
        transport = http.MockTransport(_response)
        openai = AsyncOpenAI(api_key="test", http_client=http.AsyncClient(transport=transport))
        client = OpenAIChatCompletionClient(model="gpt-test", async_client=openai)
        agent = Agent(client=client, name="assistant", instructions="help", tools=[weather])

        async def go() -> str:
            if not stream:
                return str((await agent.run("weather in Paris?")).text)
            updates = agent.run("weather in Paris?", stream=True)
            async for _ in updates:
                pass
            return str((await updates.get_final_response()).text)

        return asyncio.run(go())

    return run


@pytest.fixture
def sink() -> Iterator[InMemoryTransport]:
    from pytracer_agent_framework import install, uninstall

    memory = InMemoryTransport()
    previous = pytracer.get_tracer()
    pytracer.configure(memory, service_name="af-openai", branch="main", commit="c0")
    install()
    HITS.update(http=0, tool=0)
    try:
        yield memory
    finally:
        uninstall()
        pytracer.set_tracer(previous)


@pytest.mark.parametrize("stream", [False, True])
def test_openai_connector_run_captures_and_replays(sink: InMemoryTransport, stream: bool) -> None:
    from pytracer_replay import MockSet, ReplayEngine, ReplayMode

    output = _agent_fn(stream=stream)()
    pytracer.get_tracer().flush()
    spans = [Span.from_json(s.to_json()) for s in sink.spans]
    assert output == "It is sunny in Paris."
    llm = [s for s in spans if s.kind is SpanKind.LLM]
    assert [s.name for s in llm] == ["gpt-test", "gpt-test"]
    assert all(s.token_count == 5 for s in llm)

    before = dict(HITS)
    result = ReplayEngine(ReplayMode.FULL, strict=True).replay(
        _agent_fn(stream=stream), MockSet.from_trace(spans)
    )
    assert result.error_message is None
    assert result.return_value == output
    assert HITS == before  # no request reached the server, and the tool did not run

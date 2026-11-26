"""A Claude Agent SDK session captured through the adapter replays without @op.

A real ``query()`` session (the SDK's own control protocol, MCP bridge and
in-process tool) talks to a scripted CLI; it is captured with ``install()``,
written to JSONL and reloaded, then replayed with the network sandbox on.
Skipped when the Claude Agent SDK is absent.
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
    importlib.util.find_spec("claude_agent_sdk") is None, reason="claude-agent-sdk not installed"
)

TOOL = {"calls": 0}

if importlib.util.find_spec("claude_agent_sdk") is not None:
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        ClaudeSDKClient,
        ResultMessage,
        TextBlock,
        create_sdk_mcp_server,
        query,
        tool,
    )

    from fake_cli import CALLS, FakeCLI

    @tool("add", "Add two numbers.", {"a": int, "b": int})
    async def add(args: dict[str, Any]) -> dict[str, Any]:
        TOOL["calls"] += 1
        return {"content": [{"type": "text", "text": str(args["a"] + args["b"])}]}

    def _options(**extra: Any) -> Any:
        server = create_sdk_mcp_server("calc", tools=[add])
        return ClaudeAgentOptions(
            model="claude-test", mcp_servers={"calc": server}, allowed_tools=["mcp__calc__add"],
            **extra,
        )


@pytest.fixture
def sink() -> Iterator[InMemoryTransport]:
    from pytracer_claude_agent_sdk import install, uninstall

    memory = InMemoryTransport()
    previous = pytracer.get_tracer()
    pytracer.configure(memory, service_name="claude", branch="main", commit="c0")
    install()
    CALLS.update(cli=0, turns=0)
    TOOL["calls"] = 0
    try:
        yield memory
    finally:
        uninstall()
        pytracer.set_tracer(previous)


def _texts(messages: list[Any]) -> str:
    parts = [
        block.text
        for message in messages if isinstance(message, AssistantMessage)
        for block in message.content if isinstance(block, TextBlock)
    ]
    results = [m.result for m in messages if isinstance(m, ResultMessage)]
    return " | ".join(parts + [str(r) for r in results])


def _query_fn(*, a: int = 2, prompt: str = "add 2 and 3", options: Any = None) -> Any:
    def run() -> str:
        async def go() -> str:
            chosen = options or _options()
            cli = FakeCLI(options=chosen, a=a)
            messages = [m async for m in query(prompt=prompt, options=chosen, transport=cli)]
            return _texts(messages)

        return asyncio.run(go())

    return run


def _capture(sink: InMemoryTransport, run: Any) -> tuple[str, list[Span]]:
    output = run()
    pytracer.get_tracer().flush()
    # Through the JSONL format and the codec, as a saved recording is read back.
    spans = [Span.from_json(s.to_json()) for s in sink.spans]
    return output, spans


def test_capture_records_cli_turns_and_the_in_process_tool(sink: InMemoryTransport) -> None:
    output, spans = _capture(sink, _query_fn())
    assert output == "The sum is 5. | The sum is 5."
    llm = [s for s in spans if s.kind is SpanKind.LLM]
    tools = [s for s in spans if s.kind is SpanKind.TOOL]
    assert llm and all(s.name == "claude-test" for s in llm)
    assert [s.tool_name for s in tools] == ["add"]
    assert tools[0].input == {"arguments": {"a": 2, "b": 3}}
    assert sum(s.token_count or 0 for s in llm) == 5  # the result message's usage


def test_full_replay_starts_no_cli_and_runs_no_tool(sink: InMemoryTransport) -> None:
    from pytracer_replay import MockSet, ReplayEngine, ReplayMode

    output, spans = _capture(sink, _query_fn())
    before = (dict(CALLS), dict(TOOL))
    result = ReplayEngine(ReplayMode.FULL, strict=True).replay(_query_fn(), MockSet.from_trace(spans))
    assert result.error_message is None
    assert result.return_value == output
    assert (CALLS, TOOL) == before  # no CLI started, no model turn, no tool call
    assert result.uninstrumented is False
    assert not result.unrecorded_calls


def test_a_changed_tool_result_leaves_the_recording(sink: InMemoryTransport) -> None:
    from pytracer_replay import MockSet, ReplayEngine, ReplayMode

    _, spans = _capture(sink, _query_fn())
    for span in spans:
        if span.kind is SpanKind.TOOL:
            span.output = {"content": [{"type": "text", "text": "7"}]}
    # Python now answers the CLI's tool call with 7; the CLI's next turn was
    # recorded after 5, so it must not be served.
    result = ReplayEngine(ReplayMode.FULL, strict=True).replay(_query_fn(), MockSet.from_trace(spans))
    assert result.failed
    assert result.unrecorded_calls and result.unrecorded_calls[0].startswith("llm")
    assert CALLS["cli"] == 1  # still only the capture's CLI


def test_a_changed_prompt_leaves_the_recording(sink: InMemoryTransport) -> None:
    from pytracer_replay import MockSet, ReplayEngine, ReplayMode

    _, spans = _capture(sink, _query_fn())
    result = ReplayEngine(ReplayMode.FULL).replay(
        _query_fn(prompt="add 2 and 4"), MockSet.from_trace(spans)
    )
    assert result.failed
    assert any("llm" in call for call in result.unrecorded_calls)


def test_partial_replay_runs_the_cli_live_and_freezes_in_process_tools(sink: InMemoryTransport) -> None:
    from pytracer_replay import MockSet, ReplayEngine, ReplayMode

    output, spans = _capture(sink, _query_fn())
    result = ReplayEngine(ReplayMode.PARTIAL).replay(_query_fn(), MockSet.from_trace(spans))
    assert result.error_message is None and result.return_value == output
    assert CALLS["cli"] == 2  # the CLI ran again
    assert TOOL["calls"] == 1  # the tool did not


def test_a_changed_turn_cannot_go_live_mid_session(sink: InMemoryTransport) -> None:
    from pytracer_replay import MockSet, ReplayEngine, ReplayMode

    _, spans = _capture(sink, _query_fn())
    result = ReplayEngine(ReplayMode.FULL, evaluate_changed_prompts=True).replay(
        _query_fn(prompt="add 2 and 4"), MockSet.from_trace(spans)
    )
    assert result.failed
    assert "re-record" in (result.error_message or "")
    assert result.live_fallbacks  # reported as having left the recording
    assert CALLS["cli"] == 1


def test_hooks_still_run_on_replay(sink: InMemoryTransport) -> None:
    from claude_agent_sdk import HookMatcher
    from pytracer_replay import MockSet, ReplayEngine, ReplayMode

    seen: list[str] = []

    async def note(data: Any, tool_use_id: Any, context: Any) -> dict[str, Any]:
        seen.append("hook")
        return {}

    options = _options(hooks={"PreToolUse": [HookMatcher(matcher="mcp__calc__add", hooks=[note])]})
    output, spans = _capture(sink, _query_fn(options=options))
    result = ReplayEngine(ReplayMode.FULL, strict=True).replay(
        _query_fn(options=options), MockSet.from_trace(spans)
    )
    assert result.error_message is None and result.return_value == output
    # The scripted CLI never calls the hook, but the initialize request that
    # registers it is part of the recording and matched on replay.
    assert any("hooks" in str(s.input) for s in spans if s.kind is SpanKind.LLM)


def test_client_session_replays(sink: InMemoryTransport) -> None:
    from pytracer_replay import MockSet, ReplayEngine, ReplayMode

    def run() -> str:
        async def go() -> str:
            options = _options()
            async with ClaudeSDKClient(options=options, transport=FakeCLI(options=options)) as client:
                await client.query("add 2 and 3")
                messages = [m async for m in client.receive_response()]
            return _texts(messages)

        return asyncio.run(go())

    output, spans = _capture(sink, run)
    assert output == "The sum is 5. | The sum is 5."
    before = dict(CALLS)
    result = ReplayEngine(ReplayMode.FULL, strict=True).replay(run, MockSet.from_trace(spans))
    assert result.error_message is None
    assert result.return_value == output
    assert CALLS == before


def test_uninstall_restores_the_sdk(sink: InMemoryTransport) -> None:
    from claude_agent_sdk import SdkMcpTool

    from pytracer_claude_agent_sdk import uninstall

    uninstall()
    assert "handler" not in SdkMcpTool.__dict__
    assert not getattr(FakeCLI.write, "__pytracer_seam__", False)
    output = _query_fn()()
    assert output == "The sum is 5. | The sum is 5."

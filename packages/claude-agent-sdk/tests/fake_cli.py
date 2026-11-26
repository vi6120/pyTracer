"""A scripted stand-in for the Claude Code CLI, speaking its control protocol.

It answers ``initialize``, then for each prompt plays one agent turn: the model
asks for the in-process ``add`` tool, the CLI calls it through Python (the MCP
handshake, then ``tools/call``), and the model answers with the tool's result.
Everything on the Python side (``query()``, ``ClaudeSDKClient``, the control
protocol, the MCP bridge, hooks) is the real SDK; only the process is scripted.
"""

from __future__ import annotations

import asyncio
import itertools
import json
from collections.abc import AsyncIterator
from typing import Any

from claude_agent_sdk import Transport

CALLS = {"cli": 0, "turns": 0}
USAGE = {"input_tokens": 3, "output_tokens": 2}


class FakeCLI(Transport):
    def __init__(self, *, options: Any = None, a: int = 2, b: int = 3) -> None:
        # Kept where the SDK's subprocess transport keeps its options.
        self._options = options
        self._out: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self._waiting: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._ids = itertools.count(1)
        self._tasks: list[asyncio.Task[None]] = []
        self._ready = False
        self.model = getattr(options, "model", None) or "claude-test"
        self.args = {"a": a, "b": b}

    async def connect(self) -> None:
        CALLS["cli"] += 1
        self._ready = True

    def is_ready(self) -> bool:
        return self._ready

    async def write(self, data: str) -> None:
        for line in data.splitlines():
            message = json.loads(line)
            kind = message.get("type")
            if kind == "control_request" and message["request"]["subtype"] == "initialize":
                self._out.put_nowait({
                    "type": "control_response",
                    "response": {"subtype": "success", "request_id": message["request_id"],
                                 "response": {"commands": []}},
                })
            elif kind == "control_response":
                response = message["response"]
                waiter = self._waiting.pop(response["request_id"], None)
                if waiter is not None:
                    waiter.set_result(response.get("response") or {})
            elif kind == "user":
                self._tasks.append(asyncio.ensure_future(self._turn(message["message"]["content"])))

    async def _ask(self, request: dict[str, Any]) -> dict[str, Any]:
        """Send a control request to Python and wait for its answer."""
        request_id = f"cli_{next(self._ids)}"
        waiter: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._waiting[request_id] = waiter
        self._out.put_nowait({"type": "control_request", "request_id": request_id, "request": request})
        return await waiter

    async def _mcp(self, rpc_id: int | None, method: str, params: dict[str, Any]) -> dict[str, Any]:
        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method, "params": params}
        if rpc_id is not None:
            message["id"] = rpc_id
        answer = await self._ask({"subtype": "mcp_message", "server_name": "calc", "message": message})
        return dict(answer.get("mcp_response") or {})

    async def _turn(self, prompt: str) -> None:
        CALLS["turns"] += 1
        session = {"session_id": "s1"}
        self._out.put_nowait({"type": "system", "subtype": "init", "model": self.model, **session})
        self._out.put_nowait({
            "type": "assistant", **session, "parent_tool_use_id": None,
            "message": {"role": "assistant", "model": self.model, "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "mcp__calc__add", "input": self.args},
            ]},
        })
        await self._mcp(0, "initialize", {
            "protocolVersion": "2025-06-18", "capabilities": {},
            "clientInfo": {"name": "fake-cli", "version": "0"},
        })
        await self._mcp(None, "notifications/initialized", {})
        called = await self._mcp(1, "tools/call", {"name": "add", "arguments": self.args})
        text = called["result"]["content"][0]["text"]
        self._out.put_nowait({
            "type": "user", **session, "parent_tool_use_id": None,
            "message": {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": text},
            ]},
        })
        self._out.put_nowait({
            "type": "assistant", **session, "parent_tool_use_id": None,
            "message": {"role": "assistant", "model": self.model,
                        "content": [{"type": "text", "text": f"The sum is {text}."}]},
        })
        self._out.put_nowait({
            "type": "result", "subtype": "success", **session, "is_error": False,
            "duration_ms": 1, "duration_api_ms": 1, "num_turns": 2, "result": f"The sum is {text}.",
            "total_cost_usd": 0.001, "usage": USAGE,
        })

    def read_messages(self) -> AsyncIterator[dict[str, Any]]:
        return self._read()

    async def _read(self) -> AsyncIterator[dict[str, Any]]:
        while True:
            message = await self._out.get()
            if message is None:
                return
            yield message

    async def end_input(self) -> None:
        await asyncio.gather(*self._tasks)
        self._out.put_nowait(None)

    async def close(self) -> None:
        self._ready = False
        self._out.put_nowait(None)

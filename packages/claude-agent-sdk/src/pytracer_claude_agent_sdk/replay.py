"""Seams for the Claude Agent SDK, so a session is captured and replays without ``@op``.

The Claude Agent SDK runs the agent loop in the Claude Code CLI, a subprocess the
Python side talks to through a ``Transport``: it writes JSON lines (the
initialize request, the prompt, answers to the CLI's control requests) and reads
the CLI's messages back. The model and the built-in tools (Bash, Read, Edit, web
search) run inside the CLI, so Python never sees a model call on its own; what it
sees is the conversation on that pipe. The adapter records that conversation:

- **CLI turns**: every line Python writes starts a turn, and the turn is the
  messages the CLI sends back until Python writes again. Each turn is an LLM op
  keyed on what Python wrote (with the per-run request ids and uuids left out),
  so a replay serves each turn only after Python has written exactly what it
  wrote when the session was captured. When a tool result, a hook's answer or a
  permission decision changes, the next turn's input no longer matches: the
  replay engine reports the model input as changed, as for any other adapter,
  instead of serving responses the CLI gave to different input.
- **In-process tools**: the handler of every ``@tool`` (``SdkMcpTool``), recorded
  as tool spans, so they freeze like any other tool.

A replayed session never starts the CLI. Hooks and ``can_use_tool`` callbacks are
the agent's own code and still run; their answers are checked as above.
"""

from __future__ import annotations

import asyncio
import functools
import json
from collections.abc import AsyncIterator, Callable
from typing import Any

from claude_agent_sdk import SdkMcpTool, Transport
from pytracer_sdk import SpanKind
from pytracer_sdk.seam import ClassSeams, acall_through, astream_through, key_data

# Keys of a written line that hold a value fresh to each run, or (serverInfo, in an
# in-process MCP server's handshake) the installed mcp library's version.
_VOLATILE_KEYS = frozenset({"uuid", "session_id", "serverInfo"})
# Option fields that shape the session; machine-specific ones (cwd, env, paths to
# the CLI) are left out of the match key so a recording replays on another machine.
_SESSION_OPTIONS = (
    "model", "fallback_model", "system_prompt", "append_system_prompt", "allowed_tools",
    "disallowed_tools", "permission_mode", "max_turns", "effort",
)
_SEGMENT = "pytracer_segment"  # the first item of each recorded turn
_EOF = "pytracer_eof"  # the CLI closed its output
_SESSION = "_pytracer_session"
_METHODS = ("connect", "write", "read_messages", "close", "end_input", "is_ready")


class ReplayedSessionError(RuntimeError):
    """A replayed session needed the live CLI, which a replay never starts."""


def _normalize(message: Any) -> Any:
    if isinstance(message, dict):
        return {k: _normalize(v) for k, v in message.items() if k not in _VOLATILE_KEYS}
    if isinstance(message, list):
        return [_normalize(v) for v in message]
    return message


def _outgoing_request_id(message: Any) -> str | None:
    """The id of a control request Python sends (generated fresh every run)."""
    if isinstance(message, dict) and message.get("type") == "control_request":
        value = message.get("request_id")
        return value if isinstance(value, str) else None
    return None


def _write_key(message: Any) -> Any:
    """What a written line is matched on: the line without its run-specific ids."""
    key = _normalize(message)
    if _outgoing_request_id(message) is not None:
        key = {k: v for k, v in key.items() if k != "request_id"}
    return key


def _session_key(transport: Any) -> dict[str, Any]:
    options = getattr(transport, "_options", None)
    fields = {name: getattr(options, name, None) for name in _SESSION_OPTIONS}
    return {"session": key_data({k: v for k, v in fields.items() if v is not None})}


def _model_name(transport: Any) -> str:
    model = getattr(getattr(transport, "_options", None), "model", None)
    return model if isinstance(model, str) and model else "claude-agent"


def _turn_tokens(output: Any) -> int | None:
    """Tokens the CLI reported for the turns in a recorded segment (its result
    messages carry the usage of the whole run)."""
    total = 0
    for item in output if isinstance(output, list) else ():
        if isinstance(item, dict) and item.get("type") == "result":
            usage = item.get("usage") or {}
            total += int(usage.get("input_tokens") or 0) + int(usage.get("output_tokens") or 0)
    return total or None


class _Session:
    """One transport's conversation with the CLI, recorded turn by turn."""

    def __init__(self, transport: Any) -> None:
        self._transport = transport
        # The transport's own methods, under the seam wrappers.
        cls = type(transport)
        self._originals = {
            name: getattr(getattr(cls, name), "__wrapped__", getattr(cls, name))
            for name in _METHODS
        }
        self._model = _model_name(transport)
        self.live: bool | None = None
        self._inbound: asyncio.Queue[Any] = asyncio.Queue()  # live: CLI messages and turn ends
        self._writes: asyncio.Queue[Any] = asyncio.Queue()  # written lines not yet a turn
        self._turn: Any = None
        self._ids: dict[str, str] = {}  # replay: recorded request id -> this run's
        self._pump: asyncio.Task[None] | None = None
        self._closed = False

    # --- turns ---------------------------------------------------------------

    def _open_turn(self, key: Any, request_id: str | None) -> Any:
        async def body() -> AsyncIterator[Any]:
            if self.live is False:
                raise ReplayedSessionError(
                    "this Claude Agent SDK session was replayed from its recording, so "
                    "the CLI is not running and a turn cannot run live; re-record the "
                    "session instead"
                )
            if self.live is None:
                self.live = True
                await self._originals["connect"](self._transport)
                self._pump = asyncio.ensure_future(self._read_cli())
            yield {_SEGMENT: request_id}
            while True:
                item = await self._inbound.get()
                if item is None:
                    return
                if isinstance(item, BaseException):
                    raise item
                yield item
                if isinstance(item, dict) and _EOF in item:
                    return

        return astream_through(
            body, kind=SpanKind.LLM, name=self._model, tokens=_turn_tokens, input=key
        )

    async def _read_cli(self) -> None:
        try:
            async for message in self._originals["read_messages"](self._transport):
                self._inbound.put_nowait(message)
            self._inbound.put_nowait({_EOF: True})
        except BaseException as exc:  # noqa: BLE001 - surfaced to the reader as the CLI's error
            self._inbound.put_nowait(exc)

    def _map_ids(self, message: Any) -> Any:
        """Point a recorded reply to Python's request at this run's request id."""
        if not self._ids or not isinstance(message, dict) or message.get("type") != "control_response":
            return message
        response = message.get("response")
        if isinstance(response, dict) and response.get("request_id") in self._ids:
            response = {**response, "request_id": self._ids[response["request_id"]]}
            return {**message, "response": response}
        return message

    # --- transport methods ---------------------------------------------------

    async def connect(self) -> None:
        turn = self._open_turn(_session_key(self._transport), None).__aiter__()
        self._turn = turn

        async def header() -> Any:
            return await turn.__anext__()

        # Read the turn's header in a task of its own: the seam marks its context
        # while a recorded stream is open, and the caller's context must stay
        # unmarked so later turns (opened by the reader) are recorded too.
        await asyncio.ensure_future(header())
        if self.live is None:
            self.live = False

    async def write(self, data: str) -> None:
        for line in data.splitlines():
            if not line.strip():
                continue
            message = json.loads(line)
            if self.live:
                # The CLI's messages from here on are a new turn. Marked before the
                # line is sent, so a reply to it can never land in the turn before.
                self._inbound.put_nowait(None)
            self._writes.put_nowait((_write_key(message), _outgoing_request_id(message)))
        if self.live:
            await self._originals["write"](self._transport, data)

    async def read_messages(self) -> AsyncIterator[dict[str, Any]]:
        turn = self._turn
        while turn is not None:
            async for item in turn:
                if isinstance(item, dict) and _SEGMENT in item:
                    continue
                if isinstance(item, dict) and _EOF in item:
                    return
                yield self._map_ids(item)
            pending = await self._writes.get()
            if pending is None:
                return
            key, request_id = pending
            turn = self._open_turn(key, request_id).__aiter__()
            header = await turn.__anext__()
            recorded_id = header.get(_SEGMENT) if isinstance(header, dict) else None
            if self.live is False and recorded_id and request_id:
                self._ids[recorded_id] = request_id

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._writes.put_nowait(None)
        if self.live:
            await self._originals["close"](self._transport)
        if self._pump is not None:
            await asyncio.wait([self._pump], timeout=5)

    async def end_input(self) -> None:
        if self.live:
            await self._originals["end_input"](self._transport)


def _session(transport: Any) -> _Session:
    found = transport.__dict__.get(_SESSION)
    if isinstance(found, _Session):
        return found
    session = _Session(transport)
    transport.__dict__[_SESSION] = session
    return session


def _wrap_transport(name: str, original: Callable[..., Any]) -> Callable[..., Any]:
    if name == "read_messages":

        @functools.wraps(original)
        def read_messages(transport: Any) -> AsyncIterator[dict[str, Any]]:
            return _session(transport).read_messages()

        return read_messages
    if name == "is_ready":

        @functools.wraps(original)
        def is_ready(transport: Any) -> bool:
            session = _session(transport)
            return bool(original(transport)) if session.live else session.live is False

        return is_ready

    @functools.wraps(original)
    async def method(transport: Any, *args: Any, **kwargs: Any) -> Any:
        return await getattr(_session(transport), name)(*args, **kwargs)

    return method


# --- in-process tools ----------------------------------------------------------

_HANDLER = "handler"
_SEAMED_HANDLER = "_pytracer_seamed_handler"


class _SeamedHandler:
    """An ``@tool`` handler routed through a seam; compares like the handler it wraps."""

    def __init__(self, tool: Any, original: Callable[..., Any]) -> None:
        self._tool = tool
        self.original = original

    async def __call__(self, arguments: Any) -> Any:
        name = self._tool.name
        return await acall_through(
            self.original, (arguments,), kind=SpanKind.TOOL, name=name, tool_name=name,
            input={"arguments": key_data(arguments)},
        )

    def __eq__(self, other: object) -> bool:
        other_original = other.original if isinstance(other, _SeamedHandler) else other
        return bool(self.original == other_original)

    def __hash__(self) -> int:
        return hash(self.original)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.original, name)


class _HandlerDescriptor:
    """Reads ``SdkMcpTool.handler`` through a seam, for tools made before install too."""

    def __set__(self, tool: Any, value: Any) -> None:
        tool.__dict__[_HANDLER] = value.original if isinstance(value, _SeamedHandler) else value

    def __get__(self, tool: Any, owner: type | None = None) -> Any:
        if tool is None:
            return self
        original = tool.__dict__.get(_HANDLER)
        if original is None:
            raise AttributeError(_HANDLER)
        cached = tool.__dict__.get(_SEAMED_HANDLER)
        if not isinstance(cached, _SeamedHandler) or cached.original is not original:
            cached = _SeamedHandler(tool, original)
            tool.__dict__[_SEAMED_HANDLER] = cached
        return cached


class Seams:
    """The installed Claude Agent SDK seams; :meth:`uninstall` removes them."""

    def __init__(self) -> None:
        self._seams = ClassSeams({Transport: [(name, _wrap_transport) for name in _METHODS]})

    def install(self) -> None:
        self._seams.install()
        SdkMcpTool.handler = _HandlerDescriptor()  # type: ignore[assignment,misc]

    def uninstall(self) -> None:
        self._seams.uninstall()
        if isinstance(SdkMcpTool.__dict__.get(_HANDLER), _HandlerDescriptor):
            delattr(SdkMcpTool, _HANDLER)


_installed: Seams | None = None


def install() -> Seams:
    """Capture every Claude Agent SDK session, replayably, and return the seams.

    ``query()`` and ``ClaudeSDKClient`` sessions are recorded with no change to the
    agent code, and a captured session replays without starting the CLI. Spans go
    to the tracer active when each call runs (the one ``pytracer.configure(...)``
    set, or a replay's). Installing again replaces the previous seams.
    """
    global _installed
    uninstall()
    _installed = Seams()
    _installed.install()
    return _installed


def uninstall() -> None:
    """Remove the seams installed by :func:`install`."""
    global _installed
    if _installed is not None:
        _installed.uninstall()
        _installed = None

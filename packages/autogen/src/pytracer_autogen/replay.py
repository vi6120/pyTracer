"""Seams for AutoGen, so a captured run replays without ``@op``.

The exporter only observes AutoGen's OpenTelemetry spans. To make a run
replayable, AutoGen's own call boundaries are routed through pyTracer's op path
(``pytracer_sdk.seam``):

- **Model calls**: ``create`` and ``create_stream`` on every ``ChatCompletionClient``
  class (the ``autogen_ext`` clients, and any client class defined later),
  recorded as LLM spans.
- **Tools**: ``BaseTool.run_json``, which every tool call goes through (the
  workbench, handoffs, the tool agent), recorded as tool spans.
- **Agent turns and team runs**, only on AutoGen releases that emit no agent spans
  of their own (see :mod:`pytracer_autogen.versions`): ``BaseChatAgent.on_messages``
  / ``on_messages_stream`` as agent spans, and ``BaseGroupChat.run_stream`` as the
  span a team run is recorded under, so the run is one trace.

Each model call's recorded input is what was asked: the messages, the tools
offered, the tool choice and output format, without the cancellation token.
"""

from __future__ import annotations

import functools
import inspect
from dataclasses import replace as dataclass_replace
from collections.abc import Callable
from typing import Any

from autogen_core.models import ChatCompletionClient
from autogen_core.tools import BaseTool
from pytracer_sdk import SpanKind, get_tracer
from pytracer_sdk.seam import ClassSeams, acall_through, astream_through, key_data, tool_names
from pytracer_sdk.tracer import SpanContext, _OpMeta

# Resolves the pyTracer span a seam call should nest under (set by install).
ParentResolver = Callable[[], "SpanContext | None"]

# Bind against the interface, not the subclass, so the key names the same
# arguments however a client declares its signature.
_CREATE = inspect.signature(ChatCompletionClient.create)
_CREATE_STREAM = inspect.signature(ChatCompletionClient.create_stream)


def _model_key(bound: inspect.BoundArguments) -> dict[str, Any]:
    args = bound.arguments
    tool_choice = args.get("tool_choice", "auto")
    return {
        "messages": key_data(list(args.get("messages") or ())),
        "tools": tool_names(args.get("tools")),
        "tool_choice": tool_choice if isinstance(tool_choice, str) else tool_names([tool_choice])[0],
        "json_output": key_data(args.get("json_output")),
        "extra_create_args": key_data(dict(args.get("extra_create_args") or {})),
    }


def _create_tokens(output: Any) -> int | None:
    """Prompt plus completion tokens of a ``CreateResult`` (a stream's is its last item)."""
    result = output[-1] if isinstance(output, list) and output else output
    usage = getattr(result, "usage", None)
    if usage is None:
        return None
    return int(getattr(usage, "prompt_tokens", 0) or 0) + int(
        getattr(usage, "completion_tokens", 0) or 0
    )


def _model_name(client: Any) -> str:
    for attr in ("_resolved_model", "model", "_model"):
        value = getattr(client, attr, None)
        if isinstance(value, str) and value:
            return value
    info = getattr(client, "_raw_config", None)
    if isinstance(info, dict) and isinstance(info.get("model"), str):
        return str(info["model"])
    return type(client).__name__


def _wrap_create(original: Callable[..., Any], parent: ParentResolver) -> Callable[..., Any]:
    @functools.wraps(original)
    async def create(self: Any, *args: Any, **kwargs: Any) -> Any:
        bound = _CREATE.bind(self, *args, **kwargs)
        return await acall_through(
            original, (self, *args), kwargs, kind=SpanKind.LLM, name=_model_name(self),
            tokens=_create_tokens, input=_model_key(bound), parent=parent(),
        )

    return create


def _wrap_create_stream(original: Callable[..., Any], parent: ParentResolver) -> Callable[..., Any]:
    @functools.wraps(original)
    def create_stream(self: Any, *args: Any, **kwargs: Any) -> Any:
        bound = _CREATE_STREAM.bind(self, *args, **kwargs)
        return astream_through(
            original, (self, *args), kwargs, kind=SpanKind.LLM, name=_model_name(self),
            tokens=_create_tokens, input=_model_key(bound), parent=parent(),
        )

    return create_stream


def _wrap_run_json(original: Callable[..., Any], parent: ParentResolver) -> Callable[..., Any]:
    @functools.wraps(original)
    async def run_json(self: Any, args: Any, *rest: Any, **kwargs: Any) -> Any:
        return await acall_through(
            original, (self, args, *rest), kwargs, kind=SpanKind.TOOL, name=self.name,
            tool_name=self.name, input={"args": key_data(dict(args))}, parent=parent(),
        )

    return run_json


def _wrap_on_messages(original: Callable[..., Any], parent: ParentResolver) -> Callable[..., Any]:
    @functools.wraps(original)
    async def on_messages(self: Any, messages: Any, *rest: Any, **kwargs: Any) -> Any:
        return await acall_through(
            original, (self, messages, *rest), kwargs, kind=SpanKind.AGENT,
            name=f"invoke_agent {self.name}", agent_name=self.name,
            input={"messages": key_data(list(messages))}, parent=parent(),
        )

    return on_messages


def _wrap_on_messages_stream(
    original: Callable[..., Any], parent: ParentResolver
) -> Callable[..., Any]:
    @functools.wraps(original)
    def on_messages_stream(self: Any, messages: Any, *rest: Any, **kwargs: Any) -> Any:
        return astream_through(
            original, (self, messages, *rest), kwargs, kind=SpanKind.AGENT,
            name=f"invoke_agent {self.name}", agent_name=self.name,
            input={"messages": key_data(list(messages))}, parent=parent(),
        )

    return on_messages_stream


def _wrap_team_run_stream(original: Callable[..., Any]) -> Callable[..., Any]:
    # A plain recorded span, not a seam call: the agent turns inside it are seam
    # calls of the same kind and must still be recorded. The runtime's tasks start
    # inside it, so the whole run shares its trace.
    meta = _OpMeta(name="", kind=SpanKind.AGENT, agent_name=None, tool_name=None, capture_io=False)

    @functools.wraps(original)
    def run_stream(self: Any, *args: Any, **kwargs: Any) -> Any:
        team = dataclass_replace(meta, name=f"run_team {type(self).__name__}")
        return get_tracer().call_async_gen_op(original, (self, *args), kwargs, team)

    return run_stream


class Seams:
    """The installed AutoGen seams: every ChatCompletionClient and BaseTool class,
    including ones defined later (a custom client or tool), and with
    ``agent_turns`` every AgentChat agent and team class."""

    def __init__(self, parent: ParentResolver, *, agent_turns: bool = False) -> None:
        wrappers: dict[type, list[tuple[str, Any]]] = {
            ChatCompletionClient: [
                ("create", lambda _, original: _wrap_create(original, parent)),
                ("create_stream", lambda _, original: _wrap_create_stream(original, parent)),
            ],
            BaseTool: [("run_json", lambda _, original: _wrap_run_json(original, parent))],
        }
        if agent_turns:
            wrappers.update(_agent_turn_wrappers(parent))
        self._classes = ClassSeams(wrappers)

    def install(self) -> None:
        self._classes.install()

    def uninstall(self) -> None:
        self._classes.uninstall()


def _agent_turn_wrappers(parent: ParentResolver) -> dict[type, list[tuple[str, Any]]]:
    try:
        from autogen_agentchat.agents import BaseChatAgent
        from autogen_agentchat.teams import BaseGroupChat
    except ImportError:  # autogen-core alone: no AgentChat agents or teams to wrap
        return {}
    return {
        BaseChatAgent: [
            ("on_messages", lambda _, original: _wrap_on_messages(original, parent)),
            ("on_messages_stream", lambda _, original: _wrap_on_messages_stream(original, parent)),
        ],
        BaseGroupChat: [("run_stream", lambda _, original: _wrap_team_run_stream(original))],
    }

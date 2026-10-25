"""Seams for AutoGen, so a captured run replays without ``@op``.

The exporter only observes AutoGen's OpenTelemetry spans. To make a run
replayable, AutoGen's own call boundaries are routed through pyTracer's op path
(``pytracer_sdk.seam``):

- **Model calls**: ``create`` and ``create_stream`` on every ``ChatCompletionClient``
  class (the ``autogen_ext`` clients, and any client class defined later),
  recorded as LLM spans.
- **Tools**: ``BaseTool.run_json``, which every tool call goes through (the
  workbench, handoffs, the tool agent), recorded as tool spans.

Each model call's recorded input is what was asked: the messages, the tools
offered, the tool choice and output format, without the cancellation token.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import Any

from autogen_core.models import ChatCompletionClient
from autogen_core.tools import BaseTool
from pytracer_sdk import SpanKind
from pytracer_sdk.seam import ClassSeams, acall_through, astream_through, key_data, tool_names
from pytracer_sdk.tracer import SpanContext

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
            input=_model_key(bound), parent=parent(),
        )

    return create


def _wrap_create_stream(original: Callable[..., Any], parent: ParentResolver) -> Callable[..., Any]:
    @functools.wraps(original)
    def create_stream(self: Any, *args: Any, **kwargs: Any) -> Any:
        bound = _CREATE_STREAM.bind(self, *args, **kwargs)
        return astream_through(
            original, (self, *args), kwargs, kind=SpanKind.LLM, name=_model_name(self),
            input=_model_key(bound), parent=parent(),
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


class Seams:
    """The installed AutoGen seams: every ChatCompletionClient and BaseTool class,
    including ones defined later (a custom client or tool)."""

    def __init__(self, parent: ParentResolver) -> None:
        self._classes = ClassSeams({
            ChatCompletionClient: [
                ("create", lambda _, original: _wrap_create(original, parent)),
                ("create_stream", lambda _, original: _wrap_create_stream(original, parent)),
            ],
            BaseTool: [("run_json", lambda _, original: _wrap_run_json(original, parent))],
        })

    def install(self) -> None:
        self._classes.install()

    def uninstall(self) -> None:
        self._classes.uninstall()

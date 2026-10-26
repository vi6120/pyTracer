"""Seams for CrewAI, so a captured run replays without ``@op``.

The event listener only observes. To make a run replayable, CrewAI's own call
boundaries are routed through pyTracer's op path (``pytracer_sdk.seam``):

- **Model calls**: ``call`` / ``acall`` on every ``BaseLLM`` class (``LLM``, the
  native provider classes CrewAI imports lazily, and custom LLMs), recorded as
  LLM spans.
- **Tools**: ``BaseTool.run`` / ``arun`` (native function calling) and
  ``CrewStructuredTool.invoke`` / ``ainvoke`` (the ReAct tool path), recorded as
  tool spans. The two paths never call each other, so a call is recorded once.

A model call's recorded input is what was asked: the messages, the tools
offered, and the response model, without CrewAI's callbacks, agent and task
objects, or the functions it passes for native tool calls.

A model call's token count comes from CrewAI's own usage tally: each provider LLM
adds a call's usage through ``BaseLLM._track_token_usage_internal`` while the call
runs, and that increment is credited to the call's seam span.
"""

from __future__ import annotations

import functools
import inspect
import json
import threading
from collections.abc import Callable
from typing import Any

from crewai.llms.base_llm import BaseLLM
from crewai.tools import BaseTool
from crewai.tools.structured_tool import CrewStructuredTool
from pytracer_sdk import SpanKind
from pytracer_sdk.seam import (
    ClassSeams,
    acall_through,
    add_tokens,
    call_through,
    key_data,
    tool_names,
)
from pytracer_sdk.tracer import SpanContext

# Resolves the pyTracer span a seam call should nest under (set by install).
ParentResolver = Callable[[], "SpanContext | None"]

_CALL = inspect.signature(BaseLLM.call)


def _model_key(bound: inspect.BoundArguments) -> dict[str, Any]:
    args = bound.arguments
    return {
        "messages": key_data(args.get("messages")),
        "tools": tool_names(args.get("tools")),
        "response_model": key_data(args.get("response_model")),
    }


def _model_name(llm: Any) -> str:
    value = getattr(llm, "model", None)
    return str(value) if isinstance(value, str) and value else type(llm).__name__


def _tool_input(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """A tool call's arguments, the same shape on both tool paths."""
    if kwargs:
        return {"args": key_data(kwargs)}
    if len(args) == 1 and isinstance(args[0], dict):
        return {"args": key_data(args[0])}
    if len(args) == 1 and isinstance(args[0], str):
        try:
            parsed = json.loads(args[0])
            return {"args": key_data(parsed)}
        except ValueError:
            return {"args": args[0]}
    return {"args": key_data(list(args))}


def _wrap_call(name: str, original: Callable[..., Any], parent: ParentResolver) -> Callable[..., Any]:
    def route_args(self: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
        bound = _CALL.bind(self, *args, **kwargs)
        return {
            "kind": SpanKind.LLM, "name": _model_name(self), "input": _model_key(bound),
            "parent": parent(),
        }

    if name == "acall":

        @functools.wraps(original)
        async def acall(self: Any, *args: Any, **kwargs: Any) -> Any:
            return await acall_through(original, (self, *args), kwargs, **route_args(self, args, kwargs))

        wrapper: Callable[..., Any] = acall
    else:

        @functools.wraps(original)
        def call(self: Any, *args: Any, **kwargs: Any) -> Any:
            return call_through(original, (self, *args), kwargs, **route_args(self, args, kwargs))

        wrapper = call
    return wrapper


def _wrap_tool(name: str, original: Callable[..., Any], parent: ParentResolver) -> Callable[..., Any]:
    is_async = name in ("arun", "ainvoke")
    structured = name in ("invoke", "ainvoke")

    def route_args(self: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
        if structured:
            # invoke(input, config=None, **kwargs): the arguments are the input.
            given = args[0] if args else kwargs.get("input", {})
            tool_input = _tool_input((given,), {})
        else:
            tool_input = _tool_input(args, kwargs)
        return {
            "kind": SpanKind.TOOL, "name": self.name, "tool_name": self.name,
            "input": tool_input, "parent": parent(),
        }

    if is_async:

        @functools.wraps(original)
        async def awrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            return await acall_through(original, (self, *args), kwargs, **route_args(self, args, kwargs))

        wrapper: Callable[..., Any] = awrapper
    else:

        @functools.wraps(original)
        def swrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            return call_through(original, (self, *args), kwargs, **route_args(self, args, kwargs))

        wrapper = swrapper
    return wrapper


# Held only around CrewAI's own tally update, so concurrent calls on one LLM each
# credit exactly their own increment.
_TALLY = threading.Lock()


def _wrap_track_usage(original: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original)
    def track(self: Any, *args: Any, **kwargs: Any) -> Any:
        with _TALLY:
            before = int(self._token_usage.get("total_tokens", 0))
            result = original(self, *args, **kwargs)
            added = int(self._token_usage.get("total_tokens", 0)) - before
        add_tokens(added)  # to the model call's seam span running here
        return result

    return track


class Seams:
    """The installed CrewAI seams: every BaseLLM, BaseTool and CrewStructuredTool
    class, including the native provider LLM classes CrewAI imports lazily (when
    an LLM for that provider is first built) and custom ones."""

    def __init__(self, parent: ParentResolver) -> None:
        def call(name: str, original: Callable[..., Any]) -> Callable[..., Any]:
            return _wrap_call(name, original, parent)

        def tool(name: str, original: Callable[..., Any]) -> Callable[..., Any]:
            return _wrap_tool(name, original, parent)

        self._classes = ClassSeams({
            BaseLLM: [
                ("call", call),
                ("acall", call),
                ("_track_token_usage_internal", lambda _, original: _wrap_track_usage(original)),
            ],
            BaseTool: [("run", tool), ("arun", tool)],
            CrewStructuredTool: [("invoke", tool), ("ainvoke", tool)],
        })

    def install(self) -> None:
        self._classes.install()

    def uninstall(self) -> None:
        self._classes.uninstall()

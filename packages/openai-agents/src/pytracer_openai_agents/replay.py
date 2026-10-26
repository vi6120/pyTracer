"""Seams for the OpenAI Agents SDK, so a captured run replays without ``@op``.

The trace processor only observes. To make a run replayable, the SDK's own call
boundaries are routed through pyTracer's op path (``pytracer_sdk.seam``):

- **Model calls**: ``get_response`` and ``stream_response`` on every ``Model``
  class (the SDK's OpenAI models, and any subclass defined later), recorded as LLM
  spans named after the model.
- **Function tools**: ``FunctionTool.on_invoke_tool``, recorded as tool spans. A
  descriptor on the class wraps the callable when it is read, so tools created at
  import time (``@function_tool`` at module level, before ``install``) are covered.

Hosted tools (web search, file search, code interpreter) run inside the model
call, so recording the model response covers them.

Each call's recorded input is normalized: the SDK's tracing handle is left out,
and so are the ids the SDK passes for server-side conversation state, so a replay
of the same conversation matches the recording.
"""

from __future__ import annotations

import functools
import inspect
import json
from collections.abc import Callable
from typing import Any

from agents.models.interface import Model
from agents.tool import FunctionTool
from agents.tracing import get_current_span, get_current_trace
from pytracer_sdk import SpanKind
from pytracer_sdk.seam import ClassSeams, acall_through, astream_through, key_data
from pytracer_sdk.tracer import SpanContext

# Resolves the pyTracer span a seam call should nest under (set by install).
ParentResolver = Callable[[], "SpanContext | None"]

_TOOL_INVOKE = "on_invoke_tool"
# Where a tool keeps its seam wrapper, so every read returns the same object.
_SEAMED_INVOKE = "_pytracer_seamed_invoke"


def _model_key(bound: inspect.BoundArguments) -> dict[str, Any]:
    """The model call's input as recorded and matched: what was asked, not the
    per-run plumbing (the tracing handle, server-side conversation ids)."""
    args = bound.arguments
    output_schema = args.get("output_schema")
    schema_name = getattr(output_schema, "name", None)
    return {
        "system": args.get("system_instructions"),
        "input": key_data(args.get("input")),
        "tools": sorted(str(getattr(t, "name", t)) for t in args.get("tools") or ()),
        "handoffs": sorted(str(getattr(h, "tool_name", h)) for h in args.get("handoffs") or ()),
        "output_schema": schema_name() if callable(schema_name) else None,
        "settings": key_data(args.get("model_settings")),
    }


def _model_name(model: Any) -> str:
    name = getattr(model, "model", None)
    return str(name) if isinstance(name, str) and name else type(model).__name__


def _usage_tokens(usage: Any) -> int | None:
    """Total tokens of an Agents SDK ``Usage`` or an OpenAI ``ResponseUsage``."""
    if usage is None:
        return None
    total = getattr(usage, "total_tokens", None)
    if total:
        return int(total)
    count = int(getattr(usage, "input_tokens", 0) or 0) + int(getattr(usage, "output_tokens", 0) or 0)
    return count or None


def _response_tokens(output: Any) -> int | None:
    """A model call's tokens: ``ModelResponse.usage``, or for a stream the usage on
    its ``response.completed`` event."""
    if isinstance(output, list):
        for event in reversed(output):
            response = getattr(event, "response", None)
            found = _usage_tokens(getattr(response, "usage", None))
            if found:
                return found
        return None
    return _usage_tokens(getattr(output, "usage", None))


# Bind against the Model interface, not the subclass: a subclass may declare
# ``(*args, **kwargs)``, and the key must name the same arguments either way.
_GET_RESPONSE = inspect.signature(Model.get_response)
_STREAM_RESPONSE = inspect.signature(Model.stream_response)


def _wrap_get_response(original: Callable[..., Any], parent: ParentResolver) -> Callable[..., Any]:
    signature = _GET_RESPONSE

    @functools.wraps(original)
    async def get_response(self: Any, *args: Any, **kwargs: Any) -> Any:
        bound = signature.bind(self, *args, **kwargs)
        return await acall_through(
            original, (self, *args), kwargs, kind=SpanKind.LLM, name=_model_name(self),
            tokens=_response_tokens, input=_model_key(bound), parent=parent(),
        )

    return get_response


def _wrap_stream_response(original: Callable[..., Any], parent: ParentResolver) -> Callable[..., Any]:
    signature = _STREAM_RESPONSE

    @functools.wraps(original)
    def stream_response(self: Any, *args: Any, **kwargs: Any) -> Any:
        bound = signature.bind(self, *args, **kwargs)
        return astream_through(
            original, (self, *args), kwargs, kind=SpanKind.LLM, name=_model_name(self),
            tokens=_response_tokens, input=_model_key(bound), parent=parent(),
        )

    return stream_response


def _tool_arguments(arguments: Any) -> Any:
    if isinstance(arguments, str):
        try:
            return json.loads(arguments) if arguments else {}
        except ValueError:
            return arguments
    return arguments


class _SeamedInvoke:
    """A tool's ``on_invoke_tool`` routed through a seam.

    It compares and hashes like the callable it wraps, so tools stay equal to
    themselves and to copies (dataclass equality compares this field), and other
    attributes are read from the wrapped callable.
    """

    def __init__(self, tool: Any, original: Callable[..., Any], parent: ParentResolver) -> None:
        self._tool = tool
        self.original = original
        self._parent = parent

    async def __call__(self, context: Any, arguments: str) -> Any:
        name = self._tool.name
        return await acall_through(
            self.original, (context, arguments), kind=SpanKind.TOOL, name=name, tool_name=name,
            input={"arguments": _tool_arguments(arguments)}, parent=self._parent(),
        )

    def __eq__(self, other: object) -> bool:
        other_original = other.original if isinstance(other, _SeamedInvoke) else other
        return bool(self.original == other_original)

    def __hash__(self) -> int:
        return hash(self.original)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.original, name)


class _ToolInvokeDescriptor:
    """Reads ``FunctionTool.on_invoke_tool`` through a seam.

    A data descriptor on the class takes precedence over the instance attribute,
    so every tool (including ones built before ``install``) is wrapped when the
    SDK reads the callable to invoke it. The real callable stays in the instance's
    own ``on_invoke_tool`` slot, so removing the descriptor restores plain access;
    the wrapper is built once per tool and reused, so reads return the same object.
    """

    def __init__(self, parent: ParentResolver) -> None:
        self._parent = parent

    def __set__(self, tool: Any, value: Any) -> None:
        # Store the real callable, never a seam wrapper read back from another tool
        # (e.g. by dataclasses.replace), so a call is never wrapped twice.
        tool.__dict__[_TOOL_INVOKE] = value.original if isinstance(value, _SeamedInvoke) else value

    def __get__(self, tool: Any, owner: type | None = None) -> Any:
        if tool is None:
            return self
        original = tool.__dict__.get(_TOOL_INVOKE)
        if original is None:
            raise AttributeError(_TOOL_INVOKE)
        cached = tool.__dict__.get(_SEAMED_INVOKE)
        if not isinstance(cached, _SeamedInvoke) or cached.original is not original:
            cached = _SeamedInvoke(tool, original, self._parent)
            tool.__dict__[_SEAMED_INVOKE] = cached
        return cached


class Seams:
    """The installed OpenAI Agents seams; :meth:`uninstall` removes them."""

    def __init__(self, parent: ParentResolver) -> None:
        self._parent = parent
        # Every Model class, including ones defined later (e.g. a custom model).
        self._models = ClassSeams({
            Model: [
                ("get_response", lambda _, original: _wrap_get_response(original, parent)),
                ("stream_response", lambda _, original: _wrap_stream_response(original, parent)),
            ],
        })

    def install(self) -> None:
        self._models.install()
        # Tools: keep each existing instance's callable where the descriptor reads it.
        FunctionTool.on_invoke_tool = _ToolInvokeDescriptor(self._parent)  # type: ignore[assignment,misc]

    def uninstall(self) -> None:
        self._models.uninstall()
        if isinstance(FunctionTool.__dict__.get(_TOOL_INVOKE), _ToolInvokeDescriptor):
            delattr(FunctionTool, _TOOL_INVOKE)


def current_sdk_run_id() -> str | None:
    """The id of the Agents SDK span (or trace) active in this context."""
    span = get_current_span()
    if span is not None:
        return str(span.span_id)
    trace = get_current_trace()
    return str(trace.trace_id) if trace is not None else None

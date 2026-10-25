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

import dataclasses
import functools
import inspect
import json
from collections.abc import Callable
from typing import Any

from agents.models.interface import Model
from agents.tool import FunctionTool
from agents.tracing import get_current_span, get_current_trace
from pytracer_sdk import SpanKind
from pytracer_sdk.seam import acall_through, astream_through
from pytracer_sdk.tracer import SpanContext

# Resolves the pyTracer span a seam call should nest under (set by install).
ParentResolver = Callable[[], "SpanContext | None"]

_SEAM_ATTR = "__pytracer_seam__"
_TOOL_INVOKE = "on_invoke_tool"
# Where a tool keeps its seam wrapper, so every read returns the same object.
_SEAMED_INVOKE = "_pytracer_seamed_invoke"


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return dump(mode="json", exclude_none=True)
        except Exception:  # noqa: BLE001 - fall back to the plain value
            return str(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def _model_key(bound: inspect.BoundArguments) -> dict[str, Any]:
    """The model call's input as recorded and matched: what was asked, not the
    per-run plumbing (the tracing handle, server-side conversation ids)."""
    args = bound.arguments
    output_schema = args.get("output_schema")
    schema_name = getattr(output_schema, "name", None)
    return {
        "system": args.get("system_instructions"),
        "input": _jsonable(args.get("input")),
        "tools": sorted(str(getattr(t, "name", t)) for t in args.get("tools") or ()),
        "handoffs": sorted(str(getattr(h, "tool_name", h)) for h in args.get("handoffs") or ()),
        "output_schema": schema_name() if callable(schema_name) else None,
        "settings": _jsonable(args.get("model_settings")),
    }


def _model_name(model: Any) -> str:
    name = getattr(model, "model", None)
    return str(name) if isinstance(name, str) and name else type(model).__name__


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
            input=_model_key(bound), parent=parent(),
        )

    setattr(get_response, _SEAM_ATTR, True)
    return get_response


def _wrap_stream_response(original: Callable[..., Any], parent: ParentResolver) -> Callable[..., Any]:
    signature = _STREAM_RESPONSE

    @functools.wraps(original)
    def stream_response(self: Any, *args: Any, **kwargs: Any) -> Any:
        bound = signature.bind(self, *args, **kwargs)
        return astream_through(
            original, (self, *args), kwargs, kind=SpanKind.LLM, name=_model_name(self),
            input=_model_key(bound), parent=parent(),
        )

    setattr(stream_response, _SEAM_ATTR, True)
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
        self._patched: list[tuple[type, str, Any]] = []
        self._previous_init_subclass: Any = None

    def install(self) -> None:
        for cls in _model_classes():
            self._patch_model(cls)
        # Model classes defined after install (e.g. a custom model) are patched too.
        self._previous_init_subclass = Model.__dict__.get("__init_subclass__")
        seams = self

        def __init_subclass__(cls: type, **kwargs: Any) -> None:  # noqa: N807
            super(Model, cls).__init_subclass__(**kwargs)  # type: ignore[misc]
            seams._patch_model(cls)

        Model.__init_subclass__ = classmethod(__init_subclass__)  # type: ignore[assignment]
        # Tools: keep each existing instance's callable where the descriptor reads it.
        FunctionTool.on_invoke_tool = _ToolInvokeDescriptor(self._parent)  # type: ignore[assignment,misc]

    def _patch_model(self, cls: type) -> None:
        for attr, wrap in (("get_response", _wrap_get_response), ("stream_response", _wrap_stream_response)):
            original = cls.__dict__.get(attr)
            if original is None or getattr(original, _SEAM_ATTR, False) or getattr(original, "__isabstractmethod__", False):
                continue
            setattr(cls, attr, wrap(original, self._parent))
            self._patched.append((cls, attr, original))

    def uninstall(self) -> None:
        while self._patched:
            cls, attr, original = self._patched.pop()
            setattr(cls, attr, original)
        if self._previous_init_subclass is None:
            if "__init_subclass__" in Model.__dict__:
                del Model.__init_subclass__
        else:
            Model.__init_subclass__ = self._previous_init_subclass  # type: ignore[method-assign]
        if isinstance(FunctionTool.__dict__.get(_TOOL_INVOKE), _ToolInvokeDescriptor):
            delattr(FunctionTool, _TOOL_INVOKE)


def _model_classes() -> list[type]:
    found: list[type] = []
    pending = list(Model.__subclasses__())
    while pending:
        cls = pending.pop()
        found.append(cls)
        pending.extend(cls.__subclasses__())
    return found


def current_sdk_run_id() -> str | None:
    """The id of the Agents SDK span (or trace) active in this context."""
    span = get_current_span()
    if span is not None:
        return str(span.span_id)
    trace = get_current_trace()
    return str(trace.trace_id) if trace is not None else None

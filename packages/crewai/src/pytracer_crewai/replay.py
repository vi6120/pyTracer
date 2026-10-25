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
"""

from __future__ import annotations

import functools
import inspect
import json
from collections.abc import Callable
from typing import Any

from crewai.llms.base_llm import BaseLLM
from crewai.tools import BaseTool
from crewai.tools.structured_tool import CrewStructuredTool
from pytracer_sdk import SpanKind
from pytracer_sdk.seam import acall_through, call_through
from pytracer_sdk.tracer import SpanContext

# Resolves the pyTracer span a seam call should nest under (set by install).
ParentResolver = Callable[[], "SpanContext | None"]

_SEAM_ATTR = "__pytracer_seam__"
_CALL = inspect.signature(BaseLLM.call)


def _jsonable(value: Any) -> Any:
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return dump(mode="json", exclude_none=True)
        except Exception:  # noqa: BLE001 - fall back to a readable value
            return str(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, type):
        return value.__name__
    return value


def _tool_names(tools: Any) -> list[str]:
    names = []
    for tool in tools or ():
        if isinstance(tool, dict):
            fn = tool.get("function")
            names.append(str(fn.get("name") if isinstance(fn, dict) else tool.get("name", tool)))
        else:
            names.append(str(getattr(tool, "name", tool)))
    return sorted(names)


def _model_key(bound: inspect.BoundArguments) -> dict[str, Any]:
    args = bound.arguments
    return {
        "messages": _jsonable(args.get("messages")),
        "tools": _tool_names(args.get("tools")),
        "response_model": _jsonable(args.get("response_model")),
    }


def _model_name(llm: Any) -> str:
    value = getattr(llm, "model", None)
    return str(value) if isinstance(value, str) and value else type(llm).__name__


def _tool_input(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """A tool call's arguments, the same shape on both tool paths."""
    if kwargs:
        return {"args": _jsonable(kwargs)}
    if len(args) == 1 and isinstance(args[0], dict):
        return {"args": _jsonable(args[0])}
    if len(args) == 1 and isinstance(args[0], str):
        try:
            parsed = json.loads(args[0])
            return {"args": _jsonable(parsed)}
        except ValueError:
            return {"args": args[0]}
    return {"args": _jsonable(list(args))}


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
    setattr(wrapper, _SEAM_ATTR, True)
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
    setattr(wrapper, _SEAM_ATTR, True)
    return wrapper


_WRAPPERS: dict[type, tuple[tuple[str, Callable[..., Callable[..., Any]]], ...]] = {
    BaseLLM: (("call", _wrap_call), ("acall", _wrap_call)),
    BaseTool: (("run", _wrap_tool), ("arun", _wrap_tool)),
    CrewStructuredTool: (("invoke", _wrap_tool), ("ainvoke", _wrap_tool)),
}


class Seams:
    """The installed CrewAI seams; :meth:`uninstall` removes them."""

    def __init__(self, parent: ParentResolver) -> None:
        self._parent = parent
        self._patched: list[tuple[type, str, Any]] = []
        self._hooked: list[tuple[type, Any]] = []

    def install(self) -> None:
        for base in _WRAPPERS:
            for cls in [base, *_subclasses(base)]:
                self._patch(base, cls)
        # CrewAI imports its native provider LLM classes lazily (when an LLM for
        # that provider is first built), so classes defined later are patched too.
        self._hook_new_subclasses(BaseLLM)
        self._hook_new_subclasses(BaseTool)

    def _patch(self, base: type, cls: type) -> None:
        for attr, wrap in _WRAPPERS[base]:
            original = cls.__dict__.get(attr)
            if (
                original is None
                or getattr(original, _SEAM_ATTR, False)
                or getattr(original, "__isabstractmethod__", False)
            ):
                continue
            setattr(cls, attr, wrap(attr, original, self._parent))
            self._patched.append((cls, attr, original))

    def _hook_new_subclasses(self, base: type) -> None:
        # Pydantic models run __pydantic_init_subclass__ once a subclass is fully
        # built; hooking it (not __init_subclass__) leaves pydantic's setup intact.
        previous = base.__dict__.get("__pydantic_init_subclass__")
        seams = self

        def hook(cls: type, **kwargs: Any) -> None:
            if previous is not None:
                previous.__func__(cls, **kwargs)
            else:
                super(base, cls).__pydantic_init_subclass__(**kwargs)  # type: ignore[arg-type,misc]
            seams._patch(base, cls)

        base.__pydantic_init_subclass__ = classmethod(hook)  # type: ignore[attr-defined]
        self._hooked.append((base, previous))

    def uninstall(self) -> None:
        while self._patched:
            cls, attr, original = self._patched.pop()
            setattr(cls, attr, original)
        while self._hooked:
            base, previous = self._hooked.pop()
            if previous is None:
                if "__pydantic_init_subclass__" in base.__dict__:
                    delattr(base, "__pydantic_init_subclass__")
            else:
                base.__pydantic_init_subclass__ = previous  # type: ignore[attr-defined]


def _subclasses(base: type) -> list[type]:
    found: list[type] = []
    pending = list(base.__subclasses__())
    while pending:
        cls = pending.pop()
        found.append(cls)
        pending.extend(cls.__subclasses__())
    return found

"""Seams for LangChain / LangGraph, so a captured run replays without ``@op``.

The callback handler only observes. To make a run replayable, LangChain's own
call boundaries are routed through pyTracer's op path (``pytracer_sdk.seam``):

- **Chat models**: ``invoke`` / ``ainvoke`` / ``stream`` / ``astream`` on
  ``BaseChatModel`` (and any subclass that overrides them), recorded as LLM spans.
  A model bound with ``bind_tools`` goes through the same ``invoke``.
- **Tools**: ``BaseTool.run`` / ``arun``, which ``invoke``, ``ToolNode`` and agents
  all go through, recorded as tool spans.

Each call nests under the LangChain run that made it (the graph node), found
through the callback handler in the call's config. A model call's recorded input
is its messages without their ``id`` (LangGraph's ``add_messages`` gives a message
a fresh id each run), plus the bound tools and stop words.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables.config import ensure_config
from langchain_core.tools import BaseTool
from pytracer_sdk import SpanKind
from pytracer_sdk.seam import (
    ClassSeams,
    acall_through,
    astream_through,
    call_through,
    key_data,
    stream_through,
    tool_names,
)
from pytracer_sdk.tracer import SpanContext

from .handler import PyTracerCallbackHandler

_MODEL_CALL = inspect.signature(BaseChatModel.invoke)
_TOOL_RUN = inspect.signature(BaseTool.run)


def _message_key(message: BaseMessage) -> dict[str, Any]:
    data = message.model_dump(mode="json", exclude_none=True)
    data.pop("id", None)  # assigned fresh each run by add_messages
    return {k: v for k, v in data.items() if v not in ({}, [], "")}


def _model_key(model: BaseChatModel, bound: inspect.BoundArguments) -> dict[str, Any]:
    args = bound.arguments
    raw_input: Any = args.get("input")
    try:
        messages = model._convert_input(raw_input).to_messages()
        rendered: Any = [_message_key(m) for m in messages]
    except Exception:  # noqa: BLE001 - an input LangChain cannot convert is keyed as is
        rendered = key_data(args.get("input"))
    extra = dict(args.get("kwargs") or {})
    key: dict[str, Any] = {"messages": rendered, "stop": args.get("stop")}
    if "tools" in extra:
        key["tools"] = tool_names(extra.pop("tools"))
    if "tool_choice" in extra:
        key["tool_choice"] = key_data(extra.pop("tool_choice"))
    if extra:
        key["kwargs"] = key_data(extra)
    return key


def _model_name(model: Any) -> str:
    for attr in ("model_name", "model", "model_id", "deployment_name"):
        value = getattr(model, attr, None)
        if isinstance(value, str) and value:
            return value
    return type(model).__name__


def _parent_from(callbacks: Any, config: Any) -> SpanContext | None:
    """The pyTracer span of the LangChain run making this call: the parent run
    of the call's callback manager, looked up in the pyTracer handler on it."""
    if callbacks is None:
        callbacks = ensure_config(config).get("callbacks")
    parent_run_id = getattr(callbacks, "parent_run_id", None)
    handlers = getattr(callbacks, "handlers", None)
    if handlers is None and isinstance(callbacks, list):
        handlers = callbacks
    if parent_run_id is None or not handlers:
        return None
    for handler in handlers:
        if isinstance(handler, PyTracerCallbackHandler):
            found = handler.context_of(str(parent_run_id))
            if found is not None:
                return found
    return None


def _wrap_model(name: str, original: Callable[..., Any]) -> Callable[..., Any]:
    through = {
        "invoke": call_through, "ainvoke": acall_through,
        "stream": stream_through, "astream": astream_through,
    }[name]

    def route(self: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        bound = _MODEL_CALL.bind(self, *args, **kwargs)
        return through(
            original, (self, *args), kwargs, kind=SpanKind.LLM, name=_model_name(self),
            input=_model_key(self, bound), parent=_parent_from(None, bound.arguments.get("config")),
        )

    if name == "ainvoke":

        @functools.wraps(original)
        async def awrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            return await route(self, args, kwargs)

        wrapper: Callable[..., Any] = awrapper
    else:

        @functools.wraps(original)
        def swrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            return route(self, args, kwargs)

        wrapper = swrapper
    return wrapper


def _wrap_tool(name: str, original: Callable[..., Any]) -> Callable[..., Any]:
    def route_args(self: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
        bound = _TOOL_RUN.bind(self, *args, **kwargs)
        return {
            "kind": SpanKind.TOOL, "name": self.name, "tool_name": self.name,
            "input": {"input": key_data(bound.arguments.get("tool_input"))},
            "parent": _parent_from(bound.arguments.get("callbacks"), bound.arguments.get("config")),
        }

    if name == "arun":

        @functools.wraps(original)
        async def arun(self: Any, *args: Any, **kwargs: Any) -> Any:
            return await acall_through(original, (self, *args), kwargs, **route_args(self, args, kwargs))

        wrapper: Callable[..., Any] = arun
    else:

        @functools.wraps(original)
        def run(self: Any, *args: Any, **kwargs: Any) -> Any:
            return call_through(original, (self, *args), kwargs, **route_args(self, args, kwargs))

        wrapper = run
    return wrapper


class Seams:
    """The installed LangChain seams: every BaseChatModel and BaseTool class,
    including ones defined later; :meth:`uninstall` removes them."""

    def __init__(self) -> None:
        self._classes = ClassSeams({
            BaseChatModel: [(name, _wrap_model) for name in ("invoke", "ainvoke", "stream", "astream")],
            BaseTool: [("run", _wrap_tool), ("arun", _wrap_tool)],
        })

    def install(self) -> None:
        self._classes.install()

    def uninstall(self) -> None:
        self._classes.uninstall()


_installed: Seams | None = None


def install_seams() -> None:
    """Route LangChain chat models and tools through pyTracer so a captured run
    replays without ``@op``. Use together with :class:`PyTracerCallbackHandler`."""
    global _installed
    uninstall_seams()
    _installed = Seams()
    _installed.install()


def uninstall_seams() -> None:
    """Remove the seams installed by :func:`install_seams`."""
    global _installed
    if _installed is not None:
        _installed.uninstall()
        _installed = None

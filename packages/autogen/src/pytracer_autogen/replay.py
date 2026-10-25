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
from pytracer_sdk.seam import acall_through, astream_through
from pytracer_sdk.tracer import SpanContext

# Resolves the pyTracer span a seam call should nest under (set by install).
ParentResolver = Callable[[], "SpanContext | None"]

_SEAM_ATTR = "__pytracer_seam__"

# Bind against the interface, not the subclass, so the key names the same
# arguments however a client declares its signature.
_CREATE = inspect.signature(ChatCompletionClient.create)
_CREATE_STREAM = inspect.signature(ChatCompletionClient.create_stream)


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


def _tool_name(tool: Any) -> str:
    name = getattr(tool, "name", None)
    if name is None and isinstance(tool, dict):
        name = tool.get("name")
    return str(name if name is not None else tool)


def _model_key(bound: inspect.BoundArguments) -> dict[str, Any]:
    args = bound.arguments
    tool_choice = args.get("tool_choice", "auto")
    return {
        "messages": _jsonable(list(args.get("messages") or ())),
        "tools": sorted(_tool_name(t) for t in args.get("tools") or ()),
        "tool_choice": tool_choice if isinstance(tool_choice, str) else _tool_name(tool_choice),
        "json_output": _jsonable(args.get("json_output")),
        "extra_create_args": _jsonable(dict(args.get("extra_create_args") or {})),
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

    setattr(create, _SEAM_ATTR, True)
    return create


def _wrap_create_stream(original: Callable[..., Any], parent: ParentResolver) -> Callable[..., Any]:
    @functools.wraps(original)
    def create_stream(self: Any, *args: Any, **kwargs: Any) -> Any:
        bound = _CREATE_STREAM.bind(self, *args, **kwargs)
        return astream_through(
            original, (self, *args), kwargs, kind=SpanKind.LLM, name=_model_name(self),
            input=_model_key(bound), parent=parent(),
        )

    setattr(create_stream, _SEAM_ATTR, True)
    return create_stream


def _wrap_run_json(original: Callable[..., Any], parent: ParentResolver) -> Callable[..., Any]:
    @functools.wraps(original)
    async def run_json(self: Any, args: Any, *rest: Any, **kwargs: Any) -> Any:
        return await acall_through(
            original, (self, args, *rest), kwargs, kind=SpanKind.TOOL, name=self.name,
            tool_name=self.name, input={"args": _jsonable(dict(args))}, parent=parent(),
        )

    setattr(run_json, _SEAM_ATTR, True)
    return run_json


_WRAPPERS: dict[type, tuple[tuple[str, Callable[..., Any]], ...]] = {
    ChatCompletionClient: (("create", _wrap_create), ("create_stream", _wrap_create_stream)),
    BaseTool: (("run_json", _wrap_run_json),),
}


class Seams:
    """The installed AutoGen seams; :meth:`uninstall` removes them."""

    def __init__(self, parent: ParentResolver) -> None:
        self._parent = parent
        self._patched: list[tuple[type, str, Any]] = []
        self._hooks: list[tuple[type, Any]] = []

    def install(self) -> None:
        for base in _WRAPPERS:
            for cls in [base, *_subclasses(base)]:
                self._patch(base, cls)
            self._hook_new_subclasses(base)

    def _patch(self, base: type, cls: type) -> None:
        for attr, wrap in _WRAPPERS[base]:
            original = cls.__dict__.get(attr)
            if (
                original is None
                or getattr(original, _SEAM_ATTR, False)
                or getattr(original, "__isabstractmethod__", False)
            ):
                continue
            setattr(cls, attr, wrap(original, self._parent))
            self._patched.append((cls, attr, original))

    def _hook_new_subclasses(self, base: type) -> None:
        """Patch classes defined after install (a custom client or tool)."""
        previous = base.__dict__.get("__init_subclass__")
        seams = self

        def __init_subclass__(cls: type, **kwargs: Any) -> None:  # noqa: N807
            if previous is not None:
                previous.__func__(cls, **kwargs)
            else:
                super(base, cls).__init_subclass__(**kwargs)  # type: ignore[arg-type,misc]
            seams._patch(base, cls)

        base.__init_subclass__ = classmethod(__init_subclass__)  # type: ignore[assignment]
        self._hooks.append((base, previous))

    def uninstall(self) -> None:
        while self._patched:
            cls, attr, original = self._patched.pop()
            setattr(cls, attr, original)
        while self._hooks:
            base, previous = self._hooks.pop()
            if previous is None:
                if "__init_subclass__" in base.__dict__:
                    delattr(base, "__init_subclass__")
            else:
                base.__init_subclass__ = previous  # type: ignore[method-assign]


def _subclasses(base: type) -> list[type]:
    found: list[type] = []
    pending = list(base.__subclasses__())
    while pending:
        cls = pending.pop()
        found.append(cls)
        pending.extend(cls.__subclasses__())
    return found

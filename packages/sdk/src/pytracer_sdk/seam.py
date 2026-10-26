"""Seams - route a framework's own model and tool calls through the op path.

``@op`` makes a function replayable because the call goes through the tracer,
where the replay engine can return a recorded result instead of running it. A
framework adapter gets the same for calls the user never decorated (a LangChain
chat model, a CrewAI tool, an OpenAI Agents model) by wrapping the framework's
call boundary with :func:`call_through` / :func:`acall_through` (or the stream
variants). Capture and replay then share one path, so everything replay does for
``@op`` (matching, recorded errors, streams, the network sandbox, side-effect
logging) applies to the framework's calls unchanged.

Each wrapper takes:

- ``input``: the call's input, **normalized** by the adapter, i.e. with values the
  framework generates fresh each run (message ids, run ids, timestamps) removed,
  so a recorded call and its replay hash to the same key.
- ``parent``: the span this call belongs under, which the adapter knows from the
  framework (its open agent or node span). ``None`` nests under the active span.
- ``tokens`` (optional): reads the call's token count from its output (for a
  stream, the list of items), for a framework that reports usage on the result.

While a seam call runs, :func:`active_seam` reports it, so the adapter's
observe-only hook (callback, event listener, trace processor) can step aside
instead of recording the same call twice; :meth:`SpanRecorder.adopt` does that
for recorder-based adapters. A seam call made inside another seam call of the
same kind (a framework's ``invoke`` that calls its own ``stream``) runs directly,
so only the outermost call is recorded.

:class:`ClassSeams` installs wrappers on framework classes reversibly, and
:func:`key_data` / :func:`tool_names` help an adapter build a call's match key.
"""

from __future__ import annotations

import contextvars
import dataclasses
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .schema import SpanKind
from .tracer import _FROM_ARGS, SpanContext, Tracer, _current, _OpMeta


@dataclass(frozen=True)
class SeamCall:
    """A framework call currently running through a seam."""

    kind: SpanKind
    span: SpanContext | None  # the span recording this call


_active: contextvars.ContextVar[SeamCall | None] = contextvars.ContextVar(
    "pytracer_active_seam", default=None
)


def active_seam() -> SeamCall | None:
    """The seam call running in this context, if any."""
    return _active.get()


def _meta(
    kind: SpanKind,
    name: str,
    tool_name: str | None,
    agent_name: str | None,
    tokens: Callable[[Any], int | None] | None,
) -> _OpMeta:
    return _OpMeta(
        name=name, kind=kind, agent_name=agent_name, tool_name=tool_name, capture_io=True,
        tokens=tokens,
    )


def _nested(kind: SpanKind) -> bool:
    seam = _active.get()
    return seam is not None and seam.kind is kind


def _resolve(tracer: Tracer | None) -> Tracer:
    if tracer is not None:
        return tracer
    from . import get_tracer  # late: the package imports this module

    return get_tracer()


def call_through(
    fn: Callable[..., Any],
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    *,
    kind: SpanKind,
    name: str,
    tool_name: str | None = None,
    agent_name: str | None = None,
    tokens: Callable[[Any], int | None] | None = None,
    input: Any = _FROM_ARGS,
    parent: SpanContext | None = None,
    tracer: Tracer | None = None,
) -> Any:
    """Run a framework's sync call through the op path (recordable, replayable)."""
    kwargs = kwargs or {}
    if _nested(kind):
        return fn(*args, **kwargs)

    def body(*a: Any, **k: Any) -> Any:
        token = _active.set(SeamCall(kind, _current.get()))
        try:
            return fn(*a, **k)
        finally:
            _active.reset(token)

    return _resolve(tracer).call_op(
        body, args, kwargs, _meta(kind, name, tool_name, agent_name, tokens), op_input=input, parent=parent
    )


async def acall_through(
    fn: Callable[..., Any],
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    *,
    kind: SpanKind,
    name: str,
    tool_name: str | None = None,
    agent_name: str | None = None,
    tokens: Callable[[Any], int | None] | None = None,
    input: Any = _FROM_ARGS,
    parent: SpanContext | None = None,
    tracer: Tracer | None = None,
) -> Any:
    """Async counterpart of :func:`call_through`."""
    kwargs = kwargs or {}
    if _nested(kind):
        return await fn(*args, **kwargs)

    async def body(*a: Any, **k: Any) -> Any:
        token = _active.set(SeamCall(kind, _current.get()))
        try:
            return await fn(*a, **k)
        finally:
            _active.reset(token)

    return await _resolve(tracer).acall_op(
        body, args, kwargs, _meta(kind, name, tool_name, agent_name, tokens), op_input=input, parent=parent
    )


def stream_through(
    fn: Callable[..., Any],
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    *,
    kind: SpanKind,
    name: str,
    tool_name: str | None = None,
    agent_name: str | None = None,
    tokens: Callable[[Any], int | None] | None = None,
    input: Any = _FROM_ARGS,
    parent: SpanContext | None = None,
    tracer: Tracer | None = None,
) -> Any:
    """Run a framework's sync streaming call through the op path.

    The stream's items are recorded and replay chunk by chunk. While the live
    stream is consumed, :func:`active_seam` reports it, so a call the stream makes
    internally (a client whose ``stream`` calls its own ``create``) runs directly
    instead of being recorded as a second call under the same key. The marker
    holds between the stream's items too, which is harmless for the usual
    consumer that only collects chunks.
    """
    kwargs = kwargs or {}
    if _nested(kind):
        return fn(*args, **kwargs)

    def body(*a: Any, **k: Any) -> Any:
        token = _active.set(SeamCall(kind, _current.get()))
        try:
            yield from fn(*a, **k)
        finally:
            _reset(token)

    return _resolve(tracer).call_gen_op(
        body, args, kwargs, _meta(kind, name, tool_name, agent_name, tokens), op_input=input, parent=parent
    )


def astream_through(
    fn: Callable[..., Any],
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    *,
    kind: SpanKind,
    name: str,
    tool_name: str | None = None,
    agent_name: str | None = None,
    tokens: Callable[[Any], int | None] | None = None,
    input: Any = _FROM_ARGS,
    parent: SpanContext | None = None,
    tracer: Tracer | None = None,
) -> Any:
    """Async-generator counterpart of :func:`stream_through`."""
    kwargs = kwargs or {}
    if _nested(kind):
        return fn(*args, **kwargs)

    async def body(*a: Any, **k: Any) -> Any:
        token = _active.set(SeamCall(kind, _current.get()))
        try:
            async for item in fn(*a, **k):
                yield item
        finally:
            _reset(token)

    return _resolve(tracer).call_async_gen_op(
        body, args, kwargs, _meta(kind, name, tool_name, agent_name, tokens), op_input=input, parent=parent
    )


def _reset(token: contextvars.Token[SeamCall | None]) -> None:
    """Clear a stream's marker. A stream finished from another context (closed by
    the garbage collector, or iterated across tasks) cannot reset the token, and
    there the marker never leaked into the caller's context anyway."""
    try:
        _active.reset(token)
    except ValueError:
        pass


_SEAM_ATTR = "__pytracer_seam__"

# Builds the seam wrapper for one method: (method name, original) -> wrapper.
WrapperFactory = Callable[[str, Callable[..., Any]], Callable[..., Any]]


class ClassSeams:
    """Wraps framework methods on base classes and all their subclasses.

    ``ClassSeams({BaseTool: [("run", make)]})`` replaces ``run`` wherever a class
    in ``BaseTool``'s family defines it with ``make("run", original)``. Abstract
    methods and methods already wrapped are skipped, so installing twice never
    double-records. Classes defined after :meth:`install` (a provider class a
    framework imports lazily, a user's custom client) are wrapped too, through the
    class-creation hook (pydantic's ``__pydantic_init_subclass__`` for pydantic
    models, ``__init_subclass__`` otherwise). :meth:`uninstall` restores it all.
    """

    def __init__(self, wrappers: Mapping[type, Sequence[tuple[str, WrapperFactory]]]) -> None:
        self._wrappers = wrappers
        self._patched: list[tuple[type, str, Any]] = []
        self._hooks: list[tuple[type, str, Any]] = []

    def install(self) -> None:
        for base in self._wrappers:
            for cls in [base, *_subclasses(base)]:
                self._wrap(base, cls)
            self._hook_new_subclasses(base)

    def uninstall(self) -> None:
        while self._patched:
            cls, attr, original = self._patched.pop()
            setattr(cls, attr, original)
        while self._hooks:
            base, hook, previous = self._hooks.pop()
            if previous is not None:
                setattr(base, hook, previous)
            elif hook in base.__dict__:
                delattr(base, hook)

    def _wrap(self, base: type, cls: type) -> None:
        for attr, make in self._wrappers[base]:
            original = cls.__dict__.get(attr)
            if (
                original is None
                or getattr(original, _SEAM_ATTR, False)
                or getattr(original, "__isabstractmethod__", False)
            ):
                continue
            wrapper = make(attr, original)
            setattr(wrapper, _SEAM_ATTR, True)
            setattr(cls, attr, wrapper)
            self._patched.append((cls, attr, original))

    def _hook_new_subclasses(self, base: type) -> None:
        hook = "__pydantic_init_subclass__" if _is_pydantic(base) else "__init_subclass__"
        previous = base.__dict__.get(hook)
        seams = self

        def on_subclass(cls: type, **kwargs: Any) -> None:
            if previous is not None:
                previous.__func__(cls, **kwargs)
            else:
                getattr(super(base, cls), hook)(**kwargs)
            seams._wrap(base, cls)

        setattr(base, hook, classmethod(on_subclass))
        self._hooks.append((base, hook, previous))


def _subclasses(base: type) -> list[type]:
    found: list[type] = []
    pending = list(base.__subclasses__())
    while pending:
        cls = pending.pop()
        found.append(cls)
        pending.extend(cls.__subclasses__())
    return found


def _is_pydantic(cls: type) -> bool:
    from pydantic import BaseModel

    return issubclass(cls, BaseModel)


def key_data(value: Any) -> Any:
    """JSON-friendly data for a seam call's match key: pydantic models and
    dataclasses as their fields, classes by name, containers element by element."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: key_data(getattr(value, f.name)) for f in dataclasses.fields(value)}
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return dump(mode="json", exclude_none=True)
        except Exception:  # noqa: BLE001 - fall back to a readable value
            return str(value)
    if isinstance(value, (list, tuple)):
        return [key_data(v) for v in value]
    if isinstance(value, dict):
        return {str(k): key_data(v) for k, v in value.items()}
    if isinstance(value, type):
        return value.__name__
    return value


def tool_names(tools: Any) -> list[str]:
    """Sorted names of the tools offered to a model call (objects or schema dicts)."""
    names = []
    for tool in tools or ():
        if isinstance(tool, dict):
            fn = tool.get("function")
            names.append(str(fn.get("name") if isinstance(fn, dict) else tool.get("name", tool)))
        else:
            names.append(str(getattr(tool, "name", tool)))
    return sorted(names)

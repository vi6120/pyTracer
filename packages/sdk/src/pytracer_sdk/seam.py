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

While a seam call runs, :func:`active_seam` reports it, so the adapter's
observe-only hook (callback, event listener, trace processor) can step aside
instead of recording the same call twice; :meth:`SpanRecorder.adopt` does that
for recorder-based adapters. A seam call made inside another seam call of the
same kind (a framework's ``invoke`` that calls its own ``stream``) runs directly,
so only the outermost call is recorded.

:class:`Patcher` installs wrappers on framework classes reversibly.
"""

from __future__ import annotations

import contextvars
import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .schema import SpanKind
from .tracer import _FROM_ARGS, SpanContext, Tracer, _current, _OpMeta


@dataclass(frozen=True)
class SeamCall:
    """A framework call currently running through a seam."""

    kind: SpanKind
    span: SpanContext  # the span recording this call


_active: contextvars.ContextVar[SeamCall | None] = contextvars.ContextVar(
    "pytracer_active_seam", default=None
)


def active_seam() -> SeamCall | None:
    """The seam call running in this context, if any."""
    return _active.get()


def _meta(kind: SpanKind, name: str, tool_name: str | None) -> _OpMeta:
    return _OpMeta(name=name, kind=kind, agent_name=None, tool_name=tool_name, capture_io=True)


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
    input: Any = _FROM_ARGS,
    parent: SpanContext | None = None,
    tracer: Tracer | None = None,
) -> Any:
    """Run a framework's sync call through the op path (recordable, replayable)."""
    kwargs = kwargs or {}
    if _nested(kind):
        return fn(*args, **kwargs)

    def body(*a: Any, **k: Any) -> Any:
        token = _active.set(SeamCall(kind, _current.get() or SpanContext("", "")))
        try:
            return fn(*a, **k)
        finally:
            _active.reset(token)

    return _resolve(tracer).call_op(
        body, args, kwargs, _meta(kind, name, tool_name), op_input=input, parent=parent
    )


async def acall_through(
    fn: Callable[..., Any],
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    *,
    kind: SpanKind,
    name: str,
    tool_name: str | None = None,
    input: Any = _FROM_ARGS,
    parent: SpanContext | None = None,
    tracer: Tracer | None = None,
) -> Any:
    """Async counterpart of :func:`call_through`."""
    kwargs = kwargs or {}
    if _nested(kind):
        return await fn(*args, **kwargs)

    async def body(*a: Any, **k: Any) -> Any:
        token = _active.set(SeamCall(kind, _current.get() or SpanContext("", "")))
        try:
            return await fn(*a, **k)
        finally:
            _active.reset(token)

    return await _resolve(tracer).acall_op(
        body, args, kwargs, _meta(kind, name, tool_name), op_input=input, parent=parent
    )


def stream_through(
    fn: Callable[..., Any],
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    *,
    kind: SpanKind,
    name: str,
    tool_name: str | None = None,
    input: Any = _FROM_ARGS,
    parent: SpanContext | None = None,
    tracer: Tracer | None = None,
) -> Any:
    """Run a framework's sync streaming call through the op path.

    The stream's items are recorded and replay chunk by chunk. The call is not
    reported by :func:`active_seam` while it is being iterated (a context variable
    cannot follow a generator across the caller's ``next`` calls safely), so an
    adapter that streams marks its own dedupe around the stream it wraps.
    """
    kwargs = kwargs or {}
    if _nested(kind):
        return fn(*args, **kwargs)
    return _resolve(tracer).call_gen_op(
        fn, args, kwargs, _meta(kind, name, tool_name), op_input=input, parent=parent
    )


def astream_through(
    fn: Callable[..., Any],
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    *,
    kind: SpanKind,
    name: str,
    tool_name: str | None = None,
    input: Any = _FROM_ARGS,
    parent: SpanContext | None = None,
    tracer: Tracer | None = None,
) -> Any:
    """Async-generator counterpart of :func:`stream_through`."""
    kwargs = kwargs or {}
    if _nested(kind):
        return fn(*args, **kwargs)
    return _resolve(tracer).call_async_gen_op(
        fn, args, kwargs, _meta(kind, name, tool_name), op_input=input, parent=parent
    )


_PATCHED = "__pytracer_seam__"


class Patcher:
    """Installs wrappers on framework attributes and can remove them again.

    ``patch(owner, "invoke", make_wrapper)`` replaces ``owner.invoke`` with
    ``make_wrapper(original)``. Patching an attribute that already carries a
    pyTracer wrapper is a no-op, so installing twice never double-records.
    """

    def __init__(self) -> None:
        self._originals: list[tuple[Any, str, Any]] = []

    def patch(
        self, owner: Any, attr: str, make_wrapper: Callable[[Callable[..., Any]], Callable[..., Any]]
    ) -> bool:
        """Wrap ``owner.attr``; returns False when it was already wrapped or absent."""
        original = owner.__dict__.get(attr) if isinstance(owner, type) else getattr(owner, attr, None)
        if original is None or getattr(original, _PATCHED, False):
            return False
        wrapper = make_wrapper(original)
        if callable(original):
            functools.update_wrapper(wrapper, original)
        setattr(wrapper, _PATCHED, True)
        self._originals.append((owner, attr, original))
        setattr(owner, attr, wrapper)
        return True

    def unpatch_all(self) -> None:
        """Restore every attribute this patcher replaced, newest first."""
        while self._originals:
            owner, attr, original = self._originals.pop()
            setattr(owner, attr, original)

    def __len__(self) -> int:
        return len(self._originals)

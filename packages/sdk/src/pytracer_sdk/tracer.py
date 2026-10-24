"""Tracer, active-span handle, and the ``@op`` decorator.

Trace/span context propagates through :mod:`contextvars`, so a span opened
inside another span automatically becomes its child - across sync calls and
within a single async task. Capture is fire-and-forget: closing a span redacts
its payload and hands it to the buffer without blocking.
"""

from __future__ import annotations

import contextvars
import functools
import inspect
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from types import TracebackType
from typing import Any, TypeVar

from .buffer import DEFAULT_FLUSH_TIMEOUT, DropCallback, SpanBuffer
from .ids import new_span_id, new_trace_id
from .redaction import Redactor
from .schema import Resource, Span, SpanKind, SpanStatus
from .transport import Transport

F = TypeVar("F", bound=Callable[..., Any])


@dataclass(frozen=True)
class SpanContext:
    """The identity of the currently-active span, propagated via contextvars."""

    trace_id: str
    span_id: str


@dataclass(frozen=True)
class Intercept:
    """An interceptor's decision to short-circuit an op with a recorded output.

    Returning this from a :data:`Interceptor` makes the ``@op`` skip the real
    function body and use ``output`` instead. This is how the replay engine
    injects recorded tool/model responses without the agent code knowing.

    When ``stream`` is set, ``output`` is a recorded sequence delivered as an
    iterator (async ops get an async iterator), so a streaming response replays
    chunk by chunk the way the agent consumed it live.
    """

    output: Any
    stream: bool = False
    # How the recording was matched: "exact", or a substitution ("fuzzy" by input
    # similarity, "positional" by call order) that the span is marked with.
    match: str = "exact"


# Called with (kind, name, tool_name, redacted_input); returns Intercept to
# inject a recorded output, or None to run the op live.
Interceptor = Callable[["SpanKind", str, "str | None", Any], "Intercept | None"]


async def _as_async_iter(items: Any) -> AsyncIterator[Any]:
    """Re-emit a recorded sequence as an async stream for an intercepted async op."""
    for item in items:
        yield item


@dataclass(frozen=True)
class _OpMeta:
    """Static metadata for a decorated op, computed once at decoration time."""

    name: str
    kind: SpanKind
    agent_name: str | None
    tool_name: str | None
    capture_io: bool


def _make_op_input(meta: _OpMeta, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any | None:
    if not meta.capture_io:
        return None
    return {"args": list(args), "kwargs": dict(kwargs)}


_current: contextvars.ContextVar[SpanContext | None] = contextvars.ContextVar(
    "pytracer_current_span", default=None
)


class ActiveSpan:
    """A span in progress. Use as a context manager; closing it records the span."""

    def __init__(self, tracer: Tracer, ctx: SpanContext, span: Span) -> None:
        self._tracer = tracer
        self._ctx = ctx
        self._span = span
        self._token: contextvars.Token[SpanContext | None] | None = None
        self._closed = False

    # --- context propagation ------------------------------------------------
    def __enter__(self) -> ActiveSpan:  # noqa: PYI034 (Self needs py311; SDK targets py310)
        self._token = _current.set(self._ctx)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc is not None:
            self.set_error(exc)
        self.close()

    # --- mutation while open ------------------------------------------------
    def set_output(self, output: Any) -> None:
        self._span.output = self._tracer.redactor.redact(output)

    def set_tokens(self, count: int) -> None:
        self._span.token_count = count

    def set_cost(self, cost_usd: float) -> None:
        self._span.cost_usd = cost_usd

    def set_attribute(self, key: str, value: Any) -> None:
        self._span.attributes[key] = value

    def set_error(self, exc: BaseException) -> None:
        self._span.status = SpanStatus.ERROR
        self._span.error_type = type(exc).__name__
        self._span.error_module = type(exc).__module__
        self._span.error_message = self._tracer.redactor.redact_text(str(exc))

    # --- finalize -----------------------------------------------------------
    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._span.end_time_ns = time.time_ns()
        if self._span.status is SpanStatus.UNSET:
            self._span.status = SpanStatus.OK
        # Re-validate to derive latency_ms from the timestamps.
        finalized = Span.model_validate(self._span.model_dump())
        self._tracer.buffer.record(finalized)
        if self._token is not None:
            _current.reset(self._token)
            self._token = None

    @property
    def span(self) -> Span:
        return self._span


class Tracer:
    """Creates spans, wires up parent/child context, and buffers them."""

    def __init__(
        self,
        transport: Transport,
        *,
        resource: Resource | None = None,
        redactor: Redactor | None = None,
        buffer: SpanBuffer | None = None,
        interceptor: Interceptor | None = None,
        on_drop: DropCallback | None = None,
    ) -> None:
        self.resource = resource or Resource()
        self.redactor = redactor or Redactor()
        # on_drop applies to the buffer built here; a caller-supplied buffer keeps its own.
        self.buffer = buffer or SpanBuffer(transport, on_drop=on_drop)
        # When set, consulted before every op to optionally inject a recorded
        # output (used by the replay engine). Settable at runtime.
        self.interceptor = interceptor

    def start_span(
        self,
        name: str,
        *,
        kind: SpanKind = SpanKind.UNKNOWN,
        agent_name: str | None = None,
        tool_name: str | None = None,
        input: Any | None = None,
    ) -> ActiveSpan:
        parent = _current.get()
        trace_id = parent.trace_id if parent else new_trace_id()
        span_id = new_span_id()
        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent.span_id if parent else None,
            name=name,
            kind=kind,
            agent_name=agent_name,
            tool_name=tool_name,
            input=self.redactor.redact(input) if input is not None else None,
            start_time_ns=time.time_ns(),
            resource=self.resource,
        )
        return ActiveSpan(self, SpanContext(trace_id, span_id), span)

    def op(
        self,
        func: F | None = None,
        *,
        name: str | None = None,
        kind: SpanKind = SpanKind.UNKNOWN,
        agent_name: str | None = None,
        tool_name: str | None = None,
        capture_io: bool = True,
    ) -> Any:
        """Decorator that captures a function call as a span, bound to this tracer.

        Works on both sync and async functions. Input is the call's args/kwargs
        and output is the return value, each redacted before storage.
        """

        def decorator(fn: F) -> F:
            meta = _OpMeta(name or fn.__name__, kind, agent_name, tool_name, capture_io)
            # An async generator function is not a coroutine function, so it must
            # be checked first or it would fall through to the sync path and never
            # capture its streamed items.
            if inspect.isasyncgenfunction(fn):

                @functools.wraps(fn)
                def async_gen_wrapper(*args: Any, **kwargs: Any) -> Any:
                    return self.call_async_gen_op(fn, args, kwargs, meta)

                return async_gen_wrapper  # type: ignore[return-value]

            if inspect.iscoroutinefunction(fn):

                @functools.wraps(fn)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    return await self.acall_op(fn, args, kwargs, meta)

                return async_wrapper  # type: ignore[return-value]

            if inspect.isgeneratorfunction(fn):

                @functools.wraps(fn)
                def gen_wrapper(*args: Any, **kwargs: Any) -> Any:
                    return self.call_gen_op(fn, args, kwargs, meta)

                return gen_wrapper  # type: ignore[return-value]

            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                return self.call_op(fn, args, kwargs, meta)

            return sync_wrapper  # type: ignore[return-value]

        # Support both @op and @op(...) usage.
        if func is not None:
            return decorator(func)
        return decorator

    def call_op(
        self, fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any], meta: _OpMeta
    ) -> Any:
        """Execute one sync op call through this tracer (used by decorators)."""
        with self.start_span(
            meta.name,
            kind=meta.kind,
            agent_name=meta.agent_name,
            tool_name=meta.tool_name,
            input=_make_op_input(meta, args, kwargs),
        ) as active:
            injected = self._maybe_intercept(active, meta.kind, meta.name, meta.tool_name)
            if injected is not None:
                return iter(injected.output) if injected.stream else injected.output
            result = fn(*args, **kwargs)
            if meta.capture_io:
                active.set_output(result)
            return result

    async def acall_op(
        self, fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any], meta: _OpMeta
    ) -> Any:
        """Execute one async op call through this tracer (used by decorators)."""
        with self.start_span(
            meta.name,
            kind=meta.kind,
            agent_name=meta.agent_name,
            tool_name=meta.tool_name,
            input=_make_op_input(meta, args, kwargs),
        ) as active:
            injected = self._maybe_intercept(active, meta.kind, meta.name, meta.tool_name)
            if injected is not None:
                return _as_async_iter(injected.output) if injected.stream else injected.output
            result = await fn(*args, **kwargs)
            if meta.capture_io:
                active.set_output(result)
            return result

    def call_gen_op(
        self, fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any], meta: _OpMeta
    ) -> Any:
        """Execute a sync generator op, capturing the streamed items as the output.

        The span stays open for the whole iteration and records the assembled
        sequence of yielded items, so a streaming response is captured instead of
        the bare generator object. Items are still yielded lazily, so the caller's
        streaming behavior is unchanged. The span is created (fixing its parent)
        when the op is called, but its context is not entered during the deferred
        iteration, so a span cannot leak into an unrelated context.
        """
        active = self.start_span(
            meta.name,
            kind=meta.kind,
            agent_name=meta.agent_name,
            tool_name=meta.tool_name,
            input=_make_op_input(meta, args, kwargs),
        )
        injected = self._maybe_intercept(active, meta.kind, meta.name, meta.tool_name)
        source = injected.output if injected is not None else fn(*args, **kwargs)
        return self._stream_sync(active, source, meta, live=injected is None)

    def _stream_sync(self, active: ActiveSpan, source: Any, meta: _OpMeta, *, live: bool) -> Any:
        collected: list[Any] = []
        try:
            for item in source:
                if live:
                    collected.append(item)
                yield item
        except Exception as exc:
            active.set_error(exc)
            raise
        finally:
            if live and meta.capture_io:
                active.set_output(collected)
            active.close()

    def call_async_gen_op(
        self, fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any], meta: _OpMeta
    ) -> Any:
        """Async-generator counterpart of :meth:`call_gen_op`."""
        active = self.start_span(
            meta.name,
            kind=meta.kind,
            agent_name=meta.agent_name,
            tool_name=meta.tool_name,
            input=_make_op_input(meta, args, kwargs),
        )
        injected = self._maybe_intercept(active, meta.kind, meta.name, meta.tool_name)
        source = injected.output if injected is not None else fn(*args, **kwargs)
        return self._stream_async(active, source, meta, live=injected is None)

    async def _stream_async(
        self, active: ActiveSpan, source: Any, meta: _OpMeta, *, live: bool
    ) -> Any:
        collected: list[Any] = []
        try:
            if live:
                async for item in source:
                    collected.append(item)
                    yield item
            else:
                for item in source:  # a recorded stream replays as a plain sequence
                    yield item
        except Exception as exc:
            active.set_error(exc)
            raise
        finally:
            if live and meta.capture_io:
                active.set_output(collected)
            active.close()

    def _maybe_intercept(
        self,
        active: ActiveSpan,
        kind: SpanKind,
        name: str,
        tool_name: str | None,
    ) -> Intercept | None:
        """Consult the interceptor; on a hit, record the injected output on the span."""
        if self.interceptor is None:
            return None
        decision = self.interceptor(kind, name, tool_name, active.span.input)
        if decision is not None:
            active.set_output(decision.output)
            active.set_attribute("pytracer.replayed", True)
            if decision.match != "exact":
                # Mark a substituted (non-exact) match so it is visible, not silent.
                active.set_attribute("pytracer.mock_match", decision.match)
        return decision

    def flush(self, timeout: float | None = DEFAULT_FLUSH_TIMEOUT) -> None:
        self.buffer.flush(timeout)

    def shutdown(self, timeout: float = 5.0) -> None:
        self.buffer.shutdown(timeout)


# Typing helper so callers can annotate a decorated function nicely.
_AnyFunc = Callable[..., Any] | Callable[..., Awaitable[Any]]

"""Joining AutoGen tool spans to the result AutoGen sends in a later message."""

from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict
from typing import Any

from pytracer_sdk import Span, SpanStatus, Tracer

from .genai import _MAX_TRACKED, _record

# AutoGen's tool span carries the call id but not the result; the result travels
# in a later runtime message (a ToolCallExecutionEvent) with the same call id.
_TOOL_CALL_ID = "gen_ai.tool.call.id"
_RESULT_EVENT = "ToolCallExecutionEvent"
_MESSAGE_ATTR = "message"
# How long a tool span waits for its result message before it is recorded without
# one (e.g. a tool run outside AgentChat, where no such message is published).
_RESULT_WAIT_SECONDS = 2.0


def _tool_results(message: Any) -> list[tuple[str, Any, bool]]:
    """``(call_id, content, is_error)`` for every tool result in a runtime message."""
    if not isinstance(message, str) or _RESULT_EVENT not in message:
        return []
    try:
        data = json.loads(message)
    except ValueError:
        return []
    found: list[tuple[str, Any, bool]] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == _RESULT_EVENT and isinstance(node.get("content"), list):
                for result in node["content"]:
                    if isinstance(result, dict) and result.get("call_id") is not None:
                        found.append(
                            (str(result["call_id"]), result.get("content"), bool(result.get("is_error")))
                        )
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(data)
    return found


class _ToolResultJoiner:
    """Records each AutoGen tool span once its result message has been seen.

    A tool span ends before the message carrying its result is delivered, so the
    span is held (per run and call id) until that message is exported, then
    recorded with the result as its output. A span whose result has not arrived
    within ``_RESULT_WAIT_SECONDS`` is recorded without it by :meth:`sweep`, which
    the exporter runs on every export (AutoGen exports continuously during a run),
    by :meth:`flush` at flush and shutdown, and by one background thread that runs
    only while spans are held, so a span is never held indefinitely in a process
    that goes idle.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Wakes the sweeper; it shares the lock that guards ``_pending``.
        self._wake = threading.Condition(self._lock)
        self._sweeper: threading.Thread | None = None
        # Held tool spans, oldest first, with the time each stops waiting.
        self._pending: OrderedDict[tuple[str, str], tuple[Span, Tracer, float]] = OrderedDict()
        # Results seen before their tool span ended (bounded; duplicates are common,
        # since AutoGen forwards the same result to several recipients).
        self._early: OrderedDict[tuple[str, str], tuple[Any, bool]] = OrderedDict()

    def hold(self, key: tuple[str, str], span: Span, tracer: Tracer) -> None:
        with self._lock:
            early = self._early.pop(key, None)
            if early is None:
                self._pending[key] = (span, tracer, time.monotonic() + _RESULT_WAIT_SECONDS)
                if self._sweeper is None:
                    # One thread per joiner, started on the first held span.
                    self._sweeper = threading.Thread(
                        target=self._run_sweeper, name="pytracer-autogen-tool-results", daemon=True
                    )
                    self._sweeper.start()
                return
        _record(tracer, _with_result(span, tracer, *early))

    def resolve(self, key: tuple[str, str], content: Any, is_error: bool) -> None:
        with self._lock:
            entry = self._pending.pop(key, None)
            if entry is None:
                self._early[key] = (content, is_error)
                self._early.move_to_end(key)
                while len(self._early) > _MAX_TRACKED:
                    self._early.popitem(last=False)
                return
        span, tracer, _ = entry
        _record(tracer, _with_result(span, tracer, content, is_error))

    def sweep(self) -> None:
        """Record, without a result, every held span whose wait is over."""
        now = time.monotonic()
        expired: list[tuple[Span, Tracer, float]] = []
        with self._lock:
            while self._pending:
                key, entry = next(iter(self._pending.items()))
                if entry[2] > now:
                    break  # held in arrival order, so the rest are still waiting
                del self._pending[key]
                expired.append(entry)
        for span, tracer, _ in expired:
            _record(tracer, span)

    def flush(self) -> None:
        """Record every held span now, with or without its result."""
        with self._lock:
            entries = list(self._pending.values())
            self._pending.clear()
            self._wake.notify()  # nothing is held now, so the sweeper stops
        for span, tracer, _ in entries:
            _record(tracer, span)

    def _run_sweeper(self) -> None:
        """Sweep at each held span's deadline; stop once nothing is held."""
        while True:
            with self._wake:
                if not self._pending:
                    self._sweeper = None
                    return
                delay = next(iter(self._pending.values()))[2] - time.monotonic()
                if delay > 0:
                    self._wake.wait(delay)
                    continue
            self.sweep()


def _with_result(span: Span, tracer: Tracer, content: Any, is_error: bool) -> Span:
    update: dict[str, Any] = {"output": tracer.redactor.redact(content)}
    if is_error:
        update["status"] = SpanStatus.ERROR
        update["error_message"] = tracer.redactor.redact_text(str(content))
    return span.model_copy(update=update)

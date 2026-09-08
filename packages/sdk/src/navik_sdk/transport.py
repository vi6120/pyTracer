"""Transports carry batches of spans off the agent's machine.

The buffer is decoupled from *where* spans go via this small protocol, so the
same buffering logic works against an in-memory sink (tests), a local file, or
the ingestion gateway (added once Layer 2 exists).
"""

from __future__ import annotations

import threading
from typing import Protocol, runtime_checkable

from .schema import Span


@runtime_checkable
class Transport(Protocol):
    """Anything that can accept a batch of spans."""

    def send(self, spans: list[Span]) -> None: ...


class InMemoryTransport:
    """Collects spans in a list. Used by tests and local debugging.

    Thread-safe: the buffer flushes from a background thread while assertions
    read from the main thread.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._spans: list[Span] = []

    def send(self, spans: list[Span]) -> None:
        with self._lock:
            self._spans.extend(spans)

    @property
    def spans(self) -> list[Span]:
        with self._lock:
            return list(self._spans)

    def clear(self) -> None:
        with self._lock:
            self._spans.clear()

"""Shared test helpers: span factory and fake writers."""

from __future__ import annotations

import threading

from navik_sdk import new_span_id, new_trace_id
from navik_sdk.schema import Resource, Span, SpanKind

RES = Resource(service_name="test", branch="main", commit="abc123")


def make_span(name: str = "op") -> Span:
    return Span(
        trace_id=new_trace_id(),
        span_id=new_span_id(),
        name=name,
        kind=SpanKind.TOOL,
        start_time_ns=1_000_000_000,
        end_time_ns=1_001_000_000,
        resource=RES,
    )


class FakeWriter:
    """Thread-safe in-memory writer; records (project, span) pairs."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.records: list[tuple[str, Span]] = []

    def write(self, project: str, spans: list[Span]) -> None:
        with self._lock:
            self.records.extend((project, s) for s in spans)

    @property
    def count(self) -> int:
        with self._lock:
            return len(self.records)


class FailingWriter:
    """Fails its first ``fail_times`` write calls, then succeeds (chaos test)."""

    def __init__(self, fail_times: int) -> None:
        self._remaining = fail_times
        self._lock = threading.Lock()
        self.inner = FakeWriter()

    def write(self, project: str, spans: list[Span]) -> None:
        with self._lock:
            if self._remaining > 0:
                self._remaining -= 1
                raise ConnectionError("simulated storage outage")
        self.inner.write(project, spans)


class BlockingWriter:
    """Blocks in ``write`` until released, to force queue backpressure."""

    def __init__(self) -> None:
        self._gate = threading.Event()
        self.inner = FakeWriter()

    def release(self) -> None:
        self._gate.set()

    def write(self, project: str, spans: list[Span]) -> None:
        self._gate.wait(timeout=10.0)
        self.inner.write(project, spans)

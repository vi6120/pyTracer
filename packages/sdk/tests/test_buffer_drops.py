"""Span loss is never silent (L1).

A full queue and a failed delivery both count the lost spans, call ``on_drop``,
and log a rate-limited warning; a broken callback never reaches the agent.
"""

from __future__ import annotations

import logging
import threading

import pytest

import pytracer_sdk as pytracer
from pytracer_sdk import new_span_id, new_trace_id
from pytracer_sdk.buffer import SpanBuffer
from pytracer_sdk.schema import Span, SpanKind
from pytracer_sdk.transport import InMemoryTransport

TRACE_ID = new_trace_id()


def _span() -> Span:
    return Span(
        trace_id=TRACE_ID,
        span_id=new_span_id(),
        name="x",
        kind=SpanKind.TOOL,
        start_time_ns=1,
        end_time_ns=2,
    )


class _BlockedSink(InMemoryTransport):
    """Holds the worker in send() until released, so the queue fills up."""

    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def send(self, spans: list[Span]) -> None:
        self.entered.set()
        self.release.wait(5.0)
        super().send(spans)


class _FailingSink(InMemoryTransport):
    def send(self, spans: list[Span]) -> None:
        raise ConnectionError("gateway unreachable")


def _drop_warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == "pytracer_sdk" and "dropped" in r.getMessage()]


def test_queue_full_is_counted_reported_and_warned_once(caplog: pytest.LogCaptureFixture) -> None:
    calls: list[tuple[str, int, BaseException | None]] = []
    sink = _BlockedSink()
    buf = SpanBuffer(
        sink,
        max_queue_size=2,
        batch_size=1,
        flush_interval=0.01,
        on_drop=lambda reason, count, error: calls.append((reason, count, error)),
    )
    with caplog.at_level(logging.WARNING, logger="pytracer_sdk"):
        buf.record(_span())
        assert sink.entered.wait(2.0)  # worker holds one span, queue is empty again
        for _ in range(12):
            buf.record(_span())
    sink.release.set()
    buf.shutdown(timeout=2.0)

    assert buf.dropped == 10
    assert calls == [("queue_full", 1, None)] * 10
    warnings = _drop_warnings(caplog)
    assert len(warnings) == 1  # rate limited: one warning, not one per span
    assert "queue_full" in warnings[0].getMessage()


def test_send_failure_is_counted_reported_and_warned(caplog: pytest.LogCaptureFixture) -> None:
    calls: list[tuple[str, int, BaseException | None]] = []
    buf = SpanBuffer(
        _FailingSink(),
        batch_size=3,
        flush_interval=0.01,
        on_drop=lambda reason, count, error: calls.append((reason, count, error)),
    )
    with caplog.at_level(logging.WARNING, logger="pytracer_sdk"):
        for _ in range(3):
            buf.record(_span())
        buf.flush(timeout=2.0)
    buf.shutdown(timeout=1.0)

    assert buf.send_errors == 3
    assert sum(count for _, count, _ in calls) == 3
    assert all(reason == "send_failed" for reason, _, _ in calls)
    assert all(isinstance(error, ConnectionError) for _, _, error in calls)
    warnings = _drop_warnings(caplog)
    assert len(warnings) == 1
    assert "send_failed" in warnings[0].getMessage()
    assert "gateway unreachable" in warnings[0].getMessage()


def test_loss_is_warned_without_a_callback(caplog: pytest.LogCaptureFixture) -> None:
    buf = SpanBuffer(_FailingSink(), batch_size=1, flush_interval=0.01)
    with caplog.at_level(logging.WARNING, logger="pytracer_sdk"):
        buf.record(_span())
        buf.flush(timeout=2.0)
    buf.shutdown(timeout=1.0)
    assert len(_drop_warnings(caplog)) == 1


def test_a_raising_callback_does_not_reach_the_agent(caplog: pytest.LogCaptureFixture) -> None:
    def broken(reason: str, count: int, error: BaseException | None) -> None:
        raise RuntimeError("callback bug")

    sink = _BlockedSink()
    buf = SpanBuffer(sink, max_queue_size=1, batch_size=1, flush_interval=0.01, on_drop=broken)
    with caplog.at_level(logging.WARNING, logger="pytracer_sdk"):
        buf.record(_span())
        assert sink.entered.wait(2.0)
        for _ in range(3):
            buf.record(_span())  # must not raise
    sink.release.set()
    buf.shutdown(timeout=2.0)
    assert buf.dropped == 2
    assert any("on_drop callback raised" in r.getMessage() for r in caplog.records)


def test_configure_wires_on_drop_through_to_the_buffer() -> None:
    calls: list[str] = []
    tracer = pytracer.configure(
        _FailingSink(), service_name="t", on_drop=lambda reason, count, error: calls.append(reason)
    )
    try:
        with tracer.start_span("work"):
            pass
        tracer.flush(timeout=2.0)
        assert calls == ["send_failed"]
    finally:
        tracer.shutdown(timeout=1.0)
        pytracer.set_tracer(None)

"""Span loss is never silent (L1).

A full queue and a failed delivery both count the lost spans, call ``on_drop``,
and log a rate-limited warning; a broken callback never reaches the agent.
"""

from __future__ import annotations

import logging
import threading
import time

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


def test_queue_full_is_warned_at_once_and_reported_by_the_worker(
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls: list[tuple[str, int, BaseException | None]] = []
    threads: list[str] = []

    def on_drop(reason: str, count: int, error: BaseException | None) -> None:
        calls.append((reason, count, error))
        threads.append(threading.current_thread().name)

    sink = _BlockedSink()
    buf = SpanBuffer(sink, max_queue_size=2, batch_size=1, flush_interval=0.01, on_drop=on_drop)
    with caplog.at_level(logging.WARNING, logger="pytracer_sdk"):
        buf.record(_span())
        assert sink.entered.wait(2.0)  # worker holds one span, queue is empty again
        started = time.monotonic()
        for _ in range(12):
            buf.record(_span())
        # Warned while send is still blocked, without waiting for the worker.
        warnings = _drop_warnings(caplog)
        assert len(warnings) == 1 and "queue_full" in warnings[0].getMessage()
        assert time.monotonic() - started < 0.2
        assert warnings[0].threadName == threading.current_thread().name
        assert calls == []  # dropping never calls back on the agent's thread
        sink.release.set()
        buf.shutdown(timeout=2.0)

    assert buf.dropped == 10
    assert sum(count for _, count, _ in calls) == 10  # every drop, once send returned
    assert all(reason == "queue_full" and error is None for reason, _, error in calls)
    assert set(threads) == {"pytracer-span-flush"}  # reported by the worker
    # Rate limited: the first drop is warned at once, the rest in one line later.
    messages = [r.getMessage() for r in _drop_warnings(caplog)]
    assert len(messages) == 2
    assert "dropped 1 span(s) (queue_full)" in messages[0]
    assert "9 more span(s) (queue_full) before shutdown" in messages[1]


def test_losses_suppressed_by_the_rate_limit_are_logged_at_shutdown(
    caplog: pytest.LogCaptureFixture,
) -> None:
    buf = SpanBuffer(_FailingSink(), batch_size=1, flush_interval=0.01)
    with caplog.at_level(logging.WARNING, logger="pytracer_sdk"):
        buf.record(_span())
        buf.flush(timeout=2.0)  # first loss: warned
        buf.record(_span())
        buf.record(_span())
        buf.flush(timeout=2.0)  # within the interval: counted, not yet warned
        buf.shutdown(timeout=2.0)
    messages = [r.getMessage() for r in _drop_warnings(caplog)]
    assert len(messages) == 2
    assert "2 more span(s) (send_failed) before shutdown" in messages[-1]


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
    # The worker may send the 3 spans as one batch or several; however they are
    # batched, the warnings (the first one, plus any rate-limited remainder logged
    # at shutdown) account for exactly the 3 lost spans.
    import re

    messages = [r.getMessage() for r in _drop_warnings(caplog)]
    assert "send_failed" in messages[0] and "gateway unreachable" in messages[0]
    assert sum(int(re.search(r"dropped (\d+)", m).group(1)) for m in messages) == 3  # type: ignore[union-attr]


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

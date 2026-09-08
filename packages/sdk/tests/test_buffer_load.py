"""Buffer load & non-blocking behavior (guide section 2 / section 10).

Confirms: no span dropped under high volume with an adequate queue, recording
does not block the calling thread, and a bounded queue applies backpressure by
dropping+counting rather than stalling the agent.
"""

from __future__ import annotations

import time

from navik_sdk import new_span_id, new_trace_id
from navik_sdk.buffer import SpanBuffer
from navik_sdk.schema import Span, SpanKind
from navik_sdk.transport import InMemoryTransport

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


def test_no_span_dropped_under_high_volume() -> None:
    sink = InMemoryTransport()
    buf = SpanBuffer(sink, batch_size=500, flush_interval=0.05)
    n = 20_000
    for _ in range(n):
        buf.record(_span())
    buf.flush(timeout=10.0)
    buf.shutdown()
    assert buf.dropped == 0
    assert len(sink.spans) == n


def test_record_does_not_block_caller() -> None:
    sink = InMemoryTransport()
    buf = SpanBuffer(sink, batch_size=256, flush_interval=0.05)
    start = time.perf_counter()
    for _ in range(10_000):
        buf.record(_span())
    elapsed = time.perf_counter() - start
    # 10k non-blocking enqueues should be well under a quarter second.
    assert elapsed < 0.25, f"recording blocked: {elapsed:.3f}s"
    buf.flush(timeout=10.0)
    buf.shutdown()


def test_bounded_queue_drops_and_counts_instead_of_blocking() -> None:
    # A never-draining transport + tiny queue: overflow must be dropped, not block.
    class SlowSink(InMemoryTransport):
        def send(self, spans: list[Span]) -> None:
            time.sleep(1.0)
            super().send(spans)

    buf = SpanBuffer(SlowSink(), max_queue_size=10, batch_size=1, flush_interval=0.01)
    start = time.perf_counter()
    for _ in range(1000):
        buf.record(_span())
    elapsed = time.perf_counter() - start
    assert elapsed < 0.25, f"recording blocked under backpressure: {elapsed:.3f}s"
    assert buf.dropped > 0
    buf.shutdown(timeout=0.1)

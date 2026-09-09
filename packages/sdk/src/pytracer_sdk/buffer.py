"""Non-blocking span buffer with a background flush worker.

Design goal (guide section 10): instrumentation is fire-and-forget. Recording a span
does a single ``queue.put_nowait`` on the calling (agent) thread and returns
immediately; a daemon worker thread batches spans and hands them to the
transport. The agent is never blocked waiting on the observability layer.
"""

from __future__ import annotations

import queue
import threading
from typing import Final

from .schema import Span
from .transport import Transport

_SENTINEL: Final = object()  # signals the worker to drain and stop


class SpanBuffer:
    """Buffers spans and flushes them in batches on a background thread.

    Parameters
    ----------
    transport:
        Where flushed batches are sent.
    max_queue_size:
        Bound on in-flight spans. ``0`` means unbounded. When bounded and full,
        new spans are dropped (and counted in :attr:`dropped`) rather than
        blocking the agent - backpressure without stalling execution.
    batch_size:
        Maximum spans handed to the transport in one ``send`` call.
    flush_interval:
        Seconds the worker waits for a batch to fill before flushing what it has.
    """

    def __init__(
        self,
        transport: Transport,
        *,
        max_queue_size: int = 100_000,
        batch_size: int = 256,
        flush_interval: float = 0.5,
    ) -> None:
        self._transport = transport
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._queue: queue.Queue[object] = queue.Queue(maxsize=max_queue_size)
        self._dropped = 0
        self._send_errors = 0
        self._dropped_lock = threading.Lock()
        self._stopping = threading.Event()
        self._worker = threading.Thread(
            target=self._run, name="pytracer-span-flush", daemon=True
        )
        self._worker.start()

    @property
    def dropped(self) -> int:
        """Number of spans dropped because the queue was full."""
        with self._dropped_lock:
            return self._dropped

    @property
    def send_errors(self) -> int:
        """Number of spans whose delivery raised (e.g. gateway unreachable)."""
        with self._dropped_lock:
            return self._send_errors

    def record(self, span: Span) -> None:
        """Enqueue a span without blocking the caller."""
        try:
            self._queue.put_nowait(span)
        except queue.Full:
            with self._dropped_lock:
                self._dropped += 1

    def flush(self, timeout: float | None = None) -> None:
        """Block until all currently-queued spans have been handed to the transport."""
        self._queue.join() if timeout is None else _join_with_timeout(self._queue, timeout)

    def shutdown(self, timeout: float = 5.0) -> None:
        """Drain remaining spans and stop the worker thread."""
        self._stopping.set()
        self._queue.put(_SENTINEL)
        self._worker.join(timeout=timeout)

    # --- worker -------------------------------------------------------------
    def _run(self) -> None:
        while True:
            batch, saw_sentinel = self._collect_batch()
            if batch:
                try:
                    self._transport.send(batch)
                except Exception:  # noqa: BLE001 - fire-and-forget: never kill the flush worker
                    with self._dropped_lock:
                        self._send_errors += len(batch)
                finally:
                    for _ in batch:
                        self._queue.task_done()
            if saw_sentinel:
                self._queue.task_done()  # account for the sentinel itself
                return

    def _collect_batch(self) -> tuple[list[Span], bool]:
        """Gather up to ``batch_size`` spans, waiting up to ``flush_interval``."""
        batch: list[Span] = []
        try:
            first = self._queue.get(timeout=self._flush_interval)
        except queue.Empty:
            return batch, False
        if first is _SENTINEL:
            return batch, True
        assert isinstance(first, Span)
        batch.append(first)
        while len(batch) < self._batch_size:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is _SENTINEL:
                return batch, True
            assert isinstance(item, Span)
            batch.append(item)
        return batch, False


def _join_with_timeout(q: queue.Queue[object], timeout: float) -> None:
    """Best-effort bounded join (queue.Queue.join has no native timeout)."""
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if q.unfinished_tasks == 0:  # type: ignore[attr-defined]
            return
        time.sleep(0.005)

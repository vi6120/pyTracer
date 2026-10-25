"""Non-blocking span buffer with a background flush worker.

Design goal (guide section 10): instrumentation is fire-and-forget. Recording a span
does a single ``queue.put_nowait`` on the calling (agent) thread and returns
immediately; a daemon worker thread batches spans and hands them to the
transport. The agent is never blocked waiting on the observability layer.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from typing import Final

from .schema import Span
from .transport import Transport

_log = logging.getLogger("pytracer_sdk")

_SENTINEL: Final = object()  # signals the worker to drain and stop

# A span that cannot be delivered is never lost silently: it is counted, reported
# to ``on_drop``, and logged. The warning is rate limited (at most one per reason
# per interval, carrying the count since the last one) so a sustained outage does
# not flood the agent's log.
DROP_WARNING_INTERVAL: Final = 30.0

DropCallback = Callable[[str, int, "BaseException | None"], None]
"""``on_drop(reason, count, error)``; ``reason`` is ``"queue_full"`` or ``"send_failed"``."""

# Default bound on flush() so an unreachable or no-answer gateway cannot stall the
# caller (for example a flush at the end of a request). The background worker keeps
# delivering regardless; pass timeout=None to wait for a full drain instead.
DEFAULT_FLUSH_TIMEOUT: Final = 2.0


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
    on_drop:
        Called as ``on_drop(reason, count, error)`` whenever spans are lost:
        ``"queue_full"`` (``error`` is ``None``) or ``"send_failed"`` (``error``
        is what the transport raised). It runs on the buffer's background thread,
        never on the agent's: spans dropped from a full queue are counted when they
        are dropped and reported together by the worker, so while the worker is
        inside one ``send`` (up to about 35 s against a gateway answering 429 or
        503) the callback lags behind. Every loss is also logged as a rate-limited
        warning on the ``pytracer_sdk`` logger, a full queue's at the moment of the
        first drop, and losses still waiting for the next warning are logged at
        shutdown. An exception raised by the callback is logged and never reaches
        the agent.
    """

    def __init__(
        self,
        transport: Transport,
        *,
        max_queue_size: int = 100_000,
        batch_size: int = 256,
        flush_interval: float = 0.5,
        on_drop: DropCallback | None = None,
    ) -> None:
        self._transport = transport
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._on_drop = on_drop
        self._queue: queue.Queue[object] = queue.Queue(maxsize=max_queue_size)
        self._dropped = 0
        self._send_errors = 0
        self._dropped_lock = threading.Lock()
        # Spans dropped from a full queue that the worker has not reported yet.
        self._queue_full_pending = 0
        # Per reason: spans lost since the last warning, and when it was logged.
        self._unreported: dict[str, int] = {}
        self._last_warning: dict[str, float] = {}
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
            # This runs on the agent's thread: count, and warn now (rate limited, so
            # at most one log call per interval); the callback is user code and is
            # left to the worker.
            with self._dropped_lock:
                self._dropped += 1
                self._queue_full_pending += 1
            self._warn("queue_full", 1, None)

    def flush(self, timeout: float | None = DEFAULT_FLUSH_TIMEOUT) -> None:
        """Wait until queued spans are handed to the transport, up to ``timeout`` s.

        ``timeout`` defaults to a short bound so a stalled gateway cannot hang the
        caller; the background worker keeps delivering after flush returns. Pass
        ``None`` to wait until the queue is fully drained (used at shutdown).
        """
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
            self._report_queue_full()
            if batch:
                try:
                    self._transport.send(batch)
                except Exception as exc:  # noqa: BLE001 - fire-and-forget: never kill the flush worker
                    with self._dropped_lock:
                        self._send_errors += len(batch)
                    self._report("send_failed", len(batch), exc)
                finally:
                    for _ in batch:
                        self._queue.task_done()
            if saw_sentinel:
                self._queue.task_done()  # account for the sentinel itself
                self._report_queue_full()
                self._log_unreported()
                return

    def _report_queue_full(self) -> None:
        with self._dropped_lock:
            count, self._queue_full_pending = self._queue_full_pending, 0
        if count:
            # Already warned in record(); only the callback is left.
            self._call_on_drop("queue_full", count, None)

    def _log_unreported(self) -> None:
        """At shutdown, log losses still waiting for the next rate-limited warning."""
        with self._dropped_lock:
            pending = {reason: n for reason, n in self._unreported.items() if n}
            self._unreported.clear()
        for reason, count in pending.items():
            _log.warning("pytracer dropped %d more span(s) (%s) before shutdown", count, reason)

    def _report(self, reason: str, count: int, error: BaseException | None) -> None:
        """Surface a span loss: callback every time, warning at most once per interval."""
        self._warn(reason, count, error)
        self._call_on_drop(reason, count, error)

    def _warn(self, reason: str, count: int, error: BaseException | None) -> None:
        """Log a loss, at most once per reason per interval, with the count since."""
        now = time.monotonic()
        with self._dropped_lock:
            pending = self._unreported.get(reason, 0) + count
            last = self._last_warning.get(reason)
            due = last is None or now - last >= DROP_WARNING_INTERVAL
            if due:
                self._unreported[reason] = 0
                self._last_warning[reason] = now
            else:
                self._unreported[reason] = pending
        if due:
            detail = f": {type(error).__name__}: {error}" if error is not None else ""
            _log.warning(
                "pytracer dropped %d span(s) (%s)%s; totals dropped=%d send_errors=%d",
                pending,
                reason,
                detail,
                self.dropped,
                self.send_errors,
            )

    def _call_on_drop(self, reason: str, count: int, error: BaseException | None) -> None:
        if self._on_drop is not None:
            try:
                self._on_drop(reason, count, error)
            except Exception:  # noqa: BLE001 - a broken callback must not reach the agent
                _log.exception("pytracer on_drop callback raised")

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
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if q.unfinished_tasks == 0:  # type: ignore[attr-defined]
            return
        time.sleep(0.005)

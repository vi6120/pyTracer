"""Ingest pipeline: validate -> enqueue -> async write-through to storage.

The request path only validates and enqueues, then returns immediately, so a
write spike to storage never blocks incoming requests (guide section 3). A
background worker drains the queue in batches and writes through to the trace
store, retrying on failure so a transient storage outage does not lose spans.
Spans that still fail after the retry budget, or that fail schema validation, go
to a dead-letter store rather than silently disappearing.

The queue and dead-letter store live behind the ``SpanQueue`` seam (see
``queue.py``): in-process by default, or Redis Streams for durability across a
gateway restart.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Protocol

from navik_sdk.schema import Span
from pydantic import ValidationError

from .config import GatewayConfig
from .queue import DeadLetter, SpanQueue, build_queue

__all__ = [
    "DeadLetter",
    "IngestPipeline",
    "Stats",
    "TraceStoreWriter",
    "Writer",
    "validate_spans",
]


class Writer(Protocol):
    """Sync sink that persists a batch of spans for one project."""

    def write(self, project: str, spans: list[Span]) -> None: ...


@dataclass
class Stats:
    """Running counters for one gateway process, exposed at ``/v1/stats``."""

    accepted: int = 0  # enqueued for writing
    written: int = 0  # persisted to storage
    validation_failed: int = 0
    backpressured: int = 0  # could not enqueue (queue full)
    storage_dlq: int = 0  # gave up writing after retries

    def snapshot(self) -> dict[str, int]:
        return {
            "accepted": self.accepted,
            "written": self.written,
            "validation_failed": self.validation_failed,
            "backpressured": self.backpressured,
            "storage_dlq": self.storage_dlq,
        }


def validate_spans(raw: list[Any]) -> tuple[list[Span], list[dict[str, Any]]]:
    """Split a raw payload into valid spans and per-item validation errors."""
    spans: list[Span] = []
    errors: list[dict[str, Any]] = []
    for i, item in enumerate(raw):
        try:
            spans.append(Span.model_validate(item))
        except ValidationError as exc:
            # Drop context/input so the error payload is always JSON-serializable
            # (ctx can hold the original exception object).
            detail = exc.errors(include_url=False, include_context=False, include_input=False)
            errors.append({"index": i, "error": detail})
    return spans, errors


class IngestPipeline:
    """Owns the background writer task and counters; delegates queueing to a backend."""

    def __init__(
        self,
        writer: Writer,
        config: GatewayConfig | None = None,
        queue: SpanQueue | None = None,
    ) -> None:
        self._writer = writer
        self._config = config or GatewayConfig()
        # Build the backend from config unless one is injected (tests do this).
        self._queue = queue if queue is not None else build_queue(self._config)
        self._worker: asyncio.Task[None] | None = None
        self._stopping = asyncio.Event()
        self.stats = Stats()

    @property
    def running(self) -> bool:
        return self._worker is not None and not self._worker.done()

    async def start(self) -> None:
        await self._queue.start()
        self._stopping.clear()
        self._worker = asyncio.create_task(self._run(), name="navik-ingest-writer")

    async def stop(self) -> None:
        self._stopping.set()
        if self._worker is not None:
            await self._worker
            self._worker = None
        await self._queue.close()

    async def enqueue(self, project: str, spans: list[Span]) -> tuple[int, int]:
        """Enqueue valid spans. Returns ``(accepted, backpressured)``."""
        accepted, backpressured = await self._queue.enqueue(project, spans)
        self.stats.accepted += accepted
        self.stats.backpressured += backpressured
        return accepted, backpressured

    async def record_validation_failures(
        self, project: str, errors: list[dict[str, Any]]
    ) -> None:
        """Count schema failures and dead-letter them so nothing is lost silently."""
        self.stats.validation_failed += len(errors)
        if errors:
            # No project trust from an unvalidated span; record the auth'd project.
            await self._queue.push_dead_letters(
                [DeadLetter("validation", project, str(err.get("error")), None) for err in errors]
            )

    async def dead_letter_count(self) -> int:
        return await self._queue.dead_letter_count()

    async def dead_letters(self, limit: int) -> list[DeadLetter]:
        return await self._queue.peek_dead_letters(limit)

    # --- worker -------------------------------------------------------------
    async def _run(self) -> None:
        while True:
            batch = await self._queue.reserve(self._config.batch_size, self._config.flush_interval)
            if batch:
                await self._write_batch(batch)
                # Ack only after the batch is terminal (written or dead-lettered),
                # so a crash mid-write replays it instead of dropping it.
                await self._queue.ack(batch)
            elif self._stopping.is_set() and await self._queue.is_drained():
                return

    async def _write_batch(self, batch: list[Any]) -> None:
        by_project: dict[str, list[Span]] = defaultdict(list)
        for item in batch:
            by_project[item.project].append(item.span)
        for project, spans in by_project.items():
            await self._write_project(project, spans)

    async def _write_project(self, project: str, spans: list[Span]) -> None:
        attempt = 0
        while True:
            try:
                # Writer is sync (e.g. ClickHouse client); keep the loop free.
                await asyncio.to_thread(self._writer.write, project, spans)
                self.stats.written += len(spans)
                return
            except Exception as exc:  # noqa: BLE001 - retry any storage error, then DLQ
                attempt += 1
                if attempt > self._config.max_write_retries:
                    self.stats.storage_dlq += len(spans)
                    await self._queue.push_dead_letters(
                        [
                            DeadLetter("storage", project, str(exc), s.model_dump(mode="json"))
                            for s in spans
                        ]
                    )
                    return
                await asyncio.sleep(self._config.retry_backoff * attempt)


class TraceStoreWriter:
    """Default writer: persists spans into the ClickHouse trace store."""

    def __init__(self, trace_store: Any) -> None:
        self._store = trace_store

    def write(self, project: str, spans: list[Span]) -> None:
        self._store.insert(spans, project=project)

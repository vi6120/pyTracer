"""Ingest pipeline: validate -> enqueue -> async write-through to storage.

The request path only validates and enqueues, then returns immediately, so a
write spike to storage never blocks incoming requests (guide section 3). A background
worker drains the queue in batches and writes through to the trace store,
retrying on failure so a transient storage outage does not lose spans. Spans
that still fail after the retry budget, or that fail schema validation, go to a
dead-letter list rather than silently disappearing.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Protocol

from navik_sdk.schema import Span
from pydantic import ValidationError

from .config import GatewayConfig


class Writer(Protocol):
    """Sync sink that persists a batch of spans for one project."""

    def write(self, project: str, spans: list[Span]) -> None: ...


@dataclass
class _Queued:
    project: str
    span: Span


@dataclass
class DeadLetter:
    """A span that could not be persisted, kept so nothing disappears silently."""

    reason: str  # "validation" | "storage"
    project: str | None
    detail: str
    span: dict[str, Any] | None = None


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
    """Owns the queue, the background writer task, counters, and the DLQ."""

    def __init__(self, writer: Writer, config: GatewayConfig | None = None) -> None:
        self._writer = writer
        self._config = config or GatewayConfig()
        self._queue: asyncio.Queue[_Queued] | None = None
        self._worker: asyncio.Task[None] | None = None
        self._stopping = asyncio.Event()
        self.stats = Stats()
        self.dead_letters: list[DeadLetter] = []

    @property
    def running(self) -> bool:
        return self._worker is not None and not self._worker.done()

    async def start(self) -> None:
        # Create the queue inside the running loop so it binds correctly.
        self._queue = asyncio.Queue(maxsize=self._config.queue_maxsize)
        self._stopping.clear()
        self._worker = asyncio.create_task(self._run(), name="navik-ingest-writer")

    async def stop(self) -> None:
        self._stopping.set()
        if self._queue is not None:
            await self._queue.join()
        if self._worker is not None:
            await self._worker
            self._worker = None

    def enqueue(self, project: str, spans: list[Span]) -> tuple[int, int]:
        """Enqueue valid spans without blocking. Returns (accepted, backpressured)."""
        assert self._queue is not None, "pipeline not started"
        accepted = 0
        backpressured = 0
        for span in spans:
            try:
                self._queue.put_nowait(_Queued(project, span))
                accepted += 1
            except asyncio.QueueFull:
                backpressured += 1
        self.stats.accepted += accepted
        self.stats.backpressured += backpressured
        return accepted, backpressured

    # --- worker -------------------------------------------------------------
    async def _run(self) -> None:
        assert self._queue is not None
        while True:
            batch = await self._collect_batch()
            if batch:
                await self._write_batch(batch)
                for _ in batch:
                    self._queue.task_done()
            elif self._stopping.is_set() and self._queue.empty():
                return

    async def _collect_batch(self) -> list[_Queued]:
        assert self._queue is not None
        batch: list[_Queued] = []
        try:
            first = await asyncio.wait_for(self._queue.get(), timeout=self._config.flush_interval)
        except asyncio.TimeoutError:
            return batch
        batch.append(first)
        while len(batch) < self._config.batch_size:
            try:
                batch.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return batch

    async def _write_batch(self, batch: list[_Queued]) -> None:
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
                    self.dead_letters.extend(
                        DeadLetter("storage", project, str(exc), s.model_dump(mode="json"))
                        for s in spans
                    )
                    return
                await asyncio.sleep(self._config.retry_backoff * attempt)


class TraceStoreWriter:
    """Default writer: persists spans into the ClickHouse trace store."""

    def __init__(self, trace_store: Any) -> None:
        self._store = trace_store

    def write(self, project: str, spans: list[Span]) -> None:
        self._store.insert(spans, project=project)

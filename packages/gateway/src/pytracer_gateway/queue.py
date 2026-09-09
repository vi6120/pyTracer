"""Queue backends for the ingest pipeline: in-process or durable (Redis Streams).

The gateway accepts spans on the request path and drains them to storage on a
background worker. This module owns the queue that sits between the two, plus
the dead-letter store for spans that cannot be persisted.

Two backends implement the same ``SpanQueue`` protocol:

- ``InProcessQueue`` keeps the queue and dead letters in memory. It is the
  default and needs no external service, which suits local development and the
  test suite. A gateway restart loses whatever it is holding.
- ``RedisQueue`` keeps both the queue and the dead letters in Redis Streams, so
  a gateway restart replays in-flight spans (unacknowledged stream entries are
  reclaimed) and the dead-letter list survives. Select it by setting
  ``PYTRACER_GATEWAY_QUEUE_BACKEND=redis``.

Delivery is at-least-once: a span is acknowledged only after it has been written
(or moved to the dead-letter store), so a crash between reserve and acknowledge
replays the span rather than dropping it.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import uuid4

from pytracer_sdk.schema import Span

from .config import GatewayConfig


@dataclass
class DeadLetter:
    """A span that could not be persisted, kept so nothing disappears silently."""

    reason: str  # "validation" | "storage"
    project: str | None
    detail: str
    span: dict[str, Any] | None = None


@dataclass
class Reserved:
    """A span pulled from the queue for writing, plus the token used to ack it."""

    project: str
    span: Span
    receipt: Any  # backend token: an in-process marker, or a Redis stream id


class SpanQueue(Protocol):
    """The queue seam between the request path and the storage worker.

    Implementations must make ``reserve`` -> ``ack`` at-least-once: an item that
    is reserved but not acked (a crash mid-write) is redelivered later.
    """

    async def start(self) -> None: ...

    async def enqueue(self, project: str, spans: list[Span]) -> tuple[int, int]:
        """Add spans. Returns ``(accepted, backpressured)`` without blocking long."""
        ...

    async def reserve(self, max_items: int, timeout: float) -> list[Reserved]:
        """Take up to ``max_items`` for writing, waiting up to ``timeout`` seconds."""
        ...

    async def ack(self, items: list[Reserved]) -> None:
        """Mark reserved items as done (written or dead-lettered)."""
        ...

    async def push_dead_letters(self, letters: list[DeadLetter]) -> None: ...

    async def dead_letter_count(self) -> int: ...

    async def peek_dead_letters(self, limit: int) -> list[DeadLetter]: ...

    async def is_drained(self) -> bool:
        """True when nothing is queued or in flight (used for graceful shutdown)."""
        ...

    async def close(self) -> None: ...


class InProcessQueue:
    """Default backend: an ``asyncio.Queue`` plus an in-memory dead-letter list.

    Fast and dependency-free, but everything it holds is lost on restart.
    """

    def __init__(self, config: GatewayConfig) -> None:
        self._config = config
        self._queue: asyncio.Queue[Reserved] | None = None
        self._inflight = 0
        self._dead_letters: list[DeadLetter] = []

    async def start(self) -> None:
        # Build the queue inside the running loop so it binds to it correctly.
        self._queue = asyncio.Queue(maxsize=self._config.queue_maxsize)

    async def enqueue(self, project: str, spans: list[Span]) -> tuple[int, int]:
        assert self._queue is not None, "queue not started"
        accepted = 0
        backpressured = 0
        for span in spans:
            try:
                self._queue.put_nowait(Reserved(project, span, receipt=None))
                accepted += 1
            except asyncio.QueueFull:
                backpressured += 1
        return accepted, backpressured

    async def reserve(self, max_items: int, timeout: float) -> list[Reserved]:
        assert self._queue is not None
        batch: list[Reserved] = []
        try:
            first = await asyncio.wait_for(self._queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return batch
        batch.append(first)
        while len(batch) < max_items:
            try:
                batch.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        self._inflight += len(batch)
        return batch

    async def ack(self, items: list[Reserved]) -> None:
        assert self._queue is not None
        for _ in items:
            self._queue.task_done()
        self._inflight -= len(items)

    async def push_dead_letters(self, letters: list[DeadLetter]) -> None:
        self._dead_letters.extend(letters)

    async def dead_letter_count(self) -> int:
        return len(self._dead_letters)

    async def peek_dead_letters(self, limit: int) -> list[DeadLetter]:
        return list(self._dead_letters[:limit])

    async def is_drained(self) -> bool:
        assert self._queue is not None
        return self._queue.empty() and self._inflight == 0

    async def close(self) -> None:
        return None


class RedisQueue:
    """Durable backend backed by two Redis Streams: the work queue and the DLQ.

    The work queue uses a consumer group so unacknowledged entries left by a
    crashed process are reclaimed on the next start and replayed. Acknowledged
    entries are deleted from the stream to reclaim memory, so the stream length
    reflects the live backlog and doubles as the backpressure signal.
    """

    def __init__(self, config: GatewayConfig) -> None:
        self._config = config
        self._redis: Any = None
        self._stream = config.redis_stream
        self._dlq = f"{config.redis_stream}:dlq"
        self._group = config.redis_group
        # A unique consumer name per process so reclaim can tell a dead prior
        # consumer's pending entries from a live one's.
        self._consumer = f"gw-{os.getpid()}-{uuid4().hex[:8]}"
        self._recovered: list[Reserved] = []

    async def start(self) -> None:
        redis = _import_redis()
        self._redis = redis.from_url(self._config.redis_url, decode_responses=True)
        # Create the stream and consumer group if they do not exist yet.
        try:
            await self._redis.xgroup_create(self._stream, self._group, id="0", mkstream=True)
        except Exception as exc:  # BUSYGROUP just means the group already exists.
            if "BUSYGROUP" not in str(exc):
                raise
        await self._recover()

    async def _recover(self) -> None:
        # Claim entries left pending by a previous consumer so in-flight spans
        # are replayed instead of lost. min_idle_time=0 claims all pending, which
        # is correct for the single-instance deployment this backend targets.
        cursor = "0-0"
        while True:
            result = await self._redis.xautoclaim(
                self._stream,
                self._group,
                self._consumer,
                min_idle_time=self._config.reclaim_min_idle_ms,
                start_id=cursor,
                count=self._config.batch_size,
            )
            # redis-py returns (next_cursor, messages) or (next_cursor, messages,
            # deleted_ids) depending on version; the first two are all we need.
            cursor, messages = result[0], result[1]
            self._recovered.extend(await self._parse(messages))
            if cursor == "0-0":
                break

    async def enqueue(self, project: str, spans: list[Span]) -> tuple[int, int]:
        # Reject the overflow once the live backlog reaches the configured cap.
        backlog = int(await self._redis.xlen(self._stream))
        capacity = max(0, self._config.queue_maxsize - backlog)
        accepted = spans[:capacity]
        backpressured = len(spans) - len(accepted)
        if accepted:
            pipe = self._redis.pipeline(transaction=False)
            for span in accepted:
                pipe.xadd(self._stream, {"project": project, "span": span.model_dump_json()})
            await pipe.execute()
        return len(accepted), backpressured

    async def reserve(self, max_items: int, timeout: float) -> list[Reserved]:
        # Serve reclaimed entries before reading new ones.
        if self._recovered:
            batch = self._recovered[:max_items]
            del self._recovered[:max_items]
            return batch
        block_ms = max(1, int(timeout * 1000))
        resp = await self._redis.xreadgroup(
            self._group,
            self._consumer,
            {self._stream: ">"},
            count=max_items,
            block=block_ms,
        )
        if not resp:
            return []
        # resp is [[stream_name, [(id, {fields}), ...]]]; one stream requested.
        return await self._parse(resp[0][1])

    async def _parse(self, messages: Any) -> list[Reserved]:
        reserved: list[Reserved] = []
        malformed: list[str] = []
        for msg_id, fields in messages:
            raw = fields.get("span") if fields else None
            if not raw:
                malformed.append(msg_id)
                continue
            try:
                span = Span.model_validate_json(raw)
            except Exception:  # noqa: BLE001 - a corrupt entry must not wedge the queue.
                malformed.append(msg_id)
                continue
            reserved.append(Reserved(fields.get("project", ""), span, receipt=msg_id))
        if malformed:
            # Drop unparseable entries so reclaim does not return them forever.
            pipe = self._redis.pipeline(transaction=False)
            pipe.xack(self._stream, self._group, *malformed)
            pipe.xdel(self._stream, *malformed)
            await pipe.execute()
        return reserved

    async def ack(self, items: list[Reserved]) -> None:
        if not items:
            return
        ids = [item.receipt for item in items]
        pipe = self._redis.pipeline(transaction=False)
        pipe.xack(self._stream, self._group, *ids)
        pipe.xdel(self._stream, *ids)
        await pipe.execute()

    async def push_dead_letters(self, letters: list[DeadLetter]) -> None:
        if not letters:
            return
        pipe = self._redis.pipeline(transaction=False)
        for letter in letters:
            pipe.xadd(
                self._dlq,
                {
                    "reason": letter.reason,
                    "project": letter.project or "",
                    "detail": letter.detail,
                    "span": json.dumps(letter.span) if letter.span is not None else "",
                },
            )
        await pipe.execute()

    async def dead_letter_count(self) -> int:
        return int(await self._redis.xlen(self._dlq))

    async def peek_dead_letters(self, limit: int) -> list[DeadLetter]:
        entries = await self._redis.xrange(self._dlq, count=limit)
        out: list[DeadLetter] = []
        for _id, fields in entries:
            raw = fields.get("span")
            span = json.loads(raw) if raw else None
            out.append(
                DeadLetter(
                    fields.get("reason", ""),
                    fields.get("project") or None,
                    fields.get("detail", ""),
                    span,
                )
            )
        return out

    async def is_drained(self) -> bool:
        # Acked entries are deleted, so an empty stream means nothing is ready or
        # in flight. Recovered-but-unserved entries also count as not drained.
        if self._recovered:
            return False
        return int(await self._redis.xlen(self._stream)) == 0

    async def close(self) -> None:
        if self._redis is None:
            return
        # redis-py 5 renamed close() to aclose(); support both.
        closer = getattr(self._redis, "aclose", None) or self._redis.close
        await closer()
        self._redis = None


def _import_redis() -> Any:
    try:
        import redis.asyncio as redis
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The Redis queue backend needs the 'redis' package. "
            "Install it with: pip install pytracer-gateway[redis]"
        ) from exc
    return redis


def build_queue(config: GatewayConfig) -> SpanQueue:
    """Construct the queue backend named by ``config.queue_backend``."""
    backend = config.queue_backend.lower()
    if backend == "redis":
        return RedisQueue(config)
    if backend == "memory":
        return InProcessQueue(config)
    raise ValueError(
        f"unknown queue backend {config.queue_backend!r} (want 'memory' or 'redis')"
    )

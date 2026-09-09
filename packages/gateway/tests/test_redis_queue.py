"""Durability tests for the Redis Streams queue backend.

These prove the property the in-process backend cannot offer: spans survive a
gateway restart. They are skipped when Redis is not reachable, so the suite
still runs anywhere. Each test uses a unique stream name and deletes it in a
finally block, so runs never interfere with each other or leave state behind.

Each test drives its async scenario with ``asyncio.run`` rather than a bare
``async def`` body, so it runs the same way whether the suite is invoked at the
repo root or inside this package (the two use different pytest-asyncio modes).
"""

from __future__ import annotations

import asyncio
import time
import uuid

import pytest
from helpers import FakeWriter, make_span

from pytracer_gateway import GatewayConfig, IngestPipeline, RedisQueue
from pytracer_gateway.queue import DeadLetter


def _redis_up() -> bool:
    try:
        import redis

        client = redis.from_url("redis://localhost:6379/0")
        client.ping()
        client.close()
        return True
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(not _redis_up(), reason="Redis not reachable")


def _config() -> GatewayConfig:
    # A unique stream per test isolates it from every other run.
    stream = f"pytracer:test:{uuid.uuid4().hex[:12]}"
    return GatewayConfig(
        queue_backend="redis",
        redis_stream=stream,
        batch_size=64,
        flush_interval=0.1,
    )


async def _drop(cfg: GatewayConfig) -> None:
    import redis.asyncio as redis

    client = redis.from_url(cfg.redis_url, decode_responses=True)
    await client.delete(cfg.redis_stream, f"{cfg.redis_stream}:dlq")
    await client.aclose()


def test_reserve_ack_roundtrip() -> None:
    async def scenario() -> None:
        cfg = _config()
        queue = RedisQueue(cfg)
        try:
            await queue.start()
            accepted, backpressured = await queue.enqueue("proj", [make_span() for _ in range(3)])
            assert (accepted, backpressured) == (3, 0)

            batch = await queue.reserve(max_items=10, timeout=1.0)
            assert len(batch) == 3
            assert all(item.project == "proj" for item in batch)

            await queue.ack(batch)
            assert await queue.is_drained()
        finally:
            await queue.close()
            await _drop(cfg)

    asyncio.run(scenario())


def test_in_flight_spans_survive_restart() -> None:
    # Reserve spans but do NOT ack them, then close the queue: this is exactly
    # what a crash mid-write looks like. A fresh queue on the same stream must
    # reclaim and redeliver them.
    async def scenario() -> None:
        cfg = _config()
        first = RedisQueue(cfg)
        try:
            await first.start()
            await first.enqueue("proj", [make_span(f"s{i}") for i in range(5)])
            reserved = await first.reserve(max_items=5, timeout=1.0)
            assert len(reserved) == 5
            # Simulate a crash: the process goes away without acking.
            await first.close()

            second = RedisQueue(cfg)
            try:
                await second.start()
                recovered = await second.reserve(max_items=5, timeout=1.0)
                assert len(recovered) == 5, "in-flight spans were lost across the restart"
                await second.ack(recovered)
                assert await second.is_drained()
            finally:
                await second.close()
        finally:
            await _drop(cfg)

    asyncio.run(scenario())


def test_dead_letters_survive_restart() -> None:
    async def scenario() -> None:
        cfg = _config()
        first = RedisQueue(cfg)
        try:
            await first.start()
            await first.push_dead_letters([DeadLetter("storage", "proj", "boom", {"name": "op"})])
            assert await first.dead_letter_count() == 1
            await first.close()

            second = RedisQueue(cfg)
            try:
                await second.start()
                assert await second.dead_letter_count() == 1
                sample = await second.peek_dead_letters(10)
                assert sample[0].reason == "storage"
                assert sample[0].span == {"name": "op"}
            finally:
                await second.close()
        finally:
            await _drop(cfg)

    asyncio.run(scenario())


def test_backpressure_when_backlog_is_full() -> None:
    async def scenario() -> None:
        cfg = GatewayConfig(
            queue_backend="redis",
            redis_stream=f"pytracer:test:{uuid.uuid4().hex[:12]}",
            queue_maxsize=4,
        )
        queue = RedisQueue(cfg)
        try:
            await queue.start()
            accepted, backpressured = await queue.enqueue("proj", [make_span() for _ in range(10)])
            assert accepted == 4
            assert backpressured == 6
        finally:
            await queue.close()
            await _drop(cfg)

    asyncio.run(scenario())


def test_pipeline_drains_stream_enqueued_before_it_started() -> None:
    # Spans land in Redis while no gateway is consuming (a restart window). A
    # pipeline that starts afterwards must pick them up and persist them.
    async def scenario() -> None:
        cfg = _config()
        producer = RedisQueue(cfg)
        await producer.start()
        await producer.enqueue("proj", [make_span(f"s{i}") for i in range(6)])
        await producer.close()

        writer = FakeWriter()
        pipeline = IngestPipeline(writer, cfg, queue=RedisQueue(cfg))
        try:
            await pipeline.start()
            deadline = asyncio.get_event_loop().time() + 10.0
            while writer.count < 6 and asyncio.get_event_loop().time() < deadline:
                await asyncio.sleep(0.05)
            assert writer.count == 6
        finally:
            await pipeline.stop()
            await _drop(cfg)

    asyncio.run(scenario())


def test_http_ingest_over_redis_backend() -> None:
    # The full app (create_app -> pipeline -> RedisQueue) accepts a POST and
    # drains it to the writer, proving the wiring end to end over Redis.
    from fastapi.testclient import TestClient

    from pytracer_gateway import create_app

    cfg = _config()
    writer = FakeWriter()
    spans = [make_span(f"s{i}").model_dump(mode="json") for i in range(8)]
    try:
        with TestClient(create_app(writer=writer, api_keys={"k": "proj"}, config=cfg)) as client:
            resp = client.post("/v1/spans", headers={"X-API-Key": "k"}, json={"spans": spans})
            assert resp.status_code == 202
            assert resp.json()["accepted"] == 8

            deadline = time.monotonic() + 10.0
            while writer.count < 8 and time.monotonic() < deadline:
                time.sleep(0.05)
            assert writer.count == 8
            assert client.get("/v1/stats").json()["stats"]["written"] == 8
    finally:
        asyncio.run(_drop(cfg))


def test_unknown_backend_is_rejected() -> None:
    from pytracer_gateway import build_queue

    with pytest.raises(ValueError, match="unknown queue backend"):
        build_queue(GatewayConfig(queue_backend="rabbitmq"))

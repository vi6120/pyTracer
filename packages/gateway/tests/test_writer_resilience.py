"""The background writer survives queue errors instead of dying silently.

Before this, one exception from the queue (e.g. Redis restarting) ended the writer
task for good: ingest kept answering 202 while nothing reached storage. These
drive the pipeline with a queue that fails on purpose and check the writer logs,
backs off, keeps running and loses or duplicates nothing.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient
from helpers import FailingWriter, FakeWriter, make_span

from pytracer_gateway import GatewayConfig, IngestPipeline, create_app
from pytracer_gateway.queue import DeadLetter, InProcessQueue, Reserved

API_KEYS = {"k": "proj"}
HEADERS = {"X-API-Key": "k"}


class FlakyQueue(InProcessQueue):
    """An in-process queue whose operations fail a set number of times."""

    def __init__(
        self,
        config: GatewayConfig,
        *,
        reserve_failures: int = 0,
        ack_failures: int = 0,
        dead_letter_failures: int = 0,
    ) -> None:
        super().__init__(config)
        self.reserve_failures = reserve_failures
        self.ack_failures = ack_failures
        self.dead_letter_failures = dead_letter_failures

    async def reserve(self, max_items: int, timeout: float) -> list[Reserved]:
        if self.reserve_failures:
            self.reserve_failures -= 1
            raise ConnectionError("simulated queue outage on reserve")
        return await super().reserve(max_items, timeout)

    async def ack(self, items: list[Reserved]) -> None:
        if self.ack_failures:
            self.ack_failures -= 1
            raise ConnectionError("simulated queue outage on ack")
        await super().ack(items)

    async def push_dead_letters(self, letters: list[DeadLetter]) -> None:
        if self.dead_letter_failures:
            self.dead_letter_failures -= 1
            raise ConnectionError("simulated queue outage on dead-letter push")
        await super().push_dead_letters(letters)


def _config(**overrides: float) -> GatewayConfig:
    return GatewayConfig(**{"flush_interval": 0.05, "retry_backoff": 0.01, **overrides})


def test_reserve_errors_are_logged_and_the_writer_keeps_running(
    wait_until: Callable[..., bool], caplog: pytest.LogCaptureFixture
) -> None:
    cfg = _config()
    writer = FakeWriter()
    app = create_app(writer=writer, api_keys=API_KEYS, config=cfg)
    app.state.pipeline._queue = FlakyQueue(cfg, reserve_failures=3)
    spans = [make_span(f"s{i}").model_dump(mode="json") for i in range(5)]

    with caplog.at_level(logging.WARNING, logger="pytracer_gateway"), TestClient(app) as client:
        assert client.post("/v1/spans", headers=HEADERS, json={"spans": spans}).status_code == 202
        assert wait_until(lambda: writer.count == 5)
        assert client.get("/readyz").status_code == 200
        stats = client.get("/v1/stats").json()["stats"]
        assert stats["written"] == 5
        assert stats["writer_errors"] == 3

    errors = [r for r in caplog.records if "ingest writer error" in r.getMessage()]
    assert len(errors) == 3
    # The traceback once per outage, not on every retry.
    assert errors[0].exc_info is not None
    assert not any(r.exc_info for r in errors[1:])


def test_ack_error_retries_the_ack_without_writing_the_batch_twice(
    wait_until: Callable[..., bool],
) -> None:
    cfg = _config()
    writer = FakeWriter()
    app = create_app(writer=writer, api_keys=API_KEYS, config=cfg)
    queue = FlakyQueue(cfg, ack_failures=2)
    app.state.pipeline._queue = queue
    spans = [make_span(f"s{i}").model_dump(mode="json") for i in range(4)]

    with TestClient(app) as client:
        client.post("/v1/spans", headers=HEADERS, json={"spans": spans})
        assert wait_until(lambda: queue.ack_failures == 0 and queue._inflight == 0)
        assert writer.count == 4
        assert client.get("/v1/stats").json()["stats"]["written"] == 4


def test_dead_letter_error_keeps_the_batch_until_it_is_dead_lettered(
    wait_until: Callable[..., bool],
) -> None:
    # Storage never recovers and the first dead-letter push fails too: the spans
    # must still end up dead-lettered, not dropped with the failed batch.
    cfg = _config(max_write_retries=1)
    writer = FailingWriter(fail_times=10_000)
    app = create_app(writer=writer, api_keys=API_KEYS, config=cfg)
    app.state.pipeline._queue = FlakyQueue(cfg, dead_letter_failures=1)
    spans = [make_span(f"s{i}").model_dump(mode="json") for i in range(3)]

    with TestClient(app) as client:
        client.post("/v1/spans", headers=HEADERS, json={"spans": spans})
        assert wait_until(lambda: client.get("/v1/stats").json()["dead_letters"] == 3)
        assert client.get("/v1/stats").json()["stats"]["writer_errors"] == 1


def test_stop_does_not_hang_on_a_queue_that_stays_down() -> None:
    async def scenario() -> float:
        cfg = _config(shutdown_timeout=0.5, retry_backoff_max=0.05)
        queue = FlakyQueue(cfg)
        pipeline = IngestPipeline(FakeWriter(), cfg, queue=queue)
        await pipeline.start()
        await pipeline.enqueue("proj", [make_span()])
        queue.reserve_failures = 10**9  # the queue goes away with work still in it
        queue._queue.put_nowait(Reserved("proj", make_span(), receipt=None))
        started = time.monotonic()
        await pipeline.stop()
        return time.monotonic() - started

    elapsed = asyncio.run(scenario())
    assert elapsed < 3.0

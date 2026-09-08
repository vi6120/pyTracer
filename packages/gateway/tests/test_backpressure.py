"""Backpressure: a full queue is reported (429), the service does not crash,
and once storage drains, the accepted spans are written.
"""

from __future__ import annotations

from collections.abc import Callable

from fastapi.testclient import TestClient
from helpers import BlockingWriter, make_span

from navik_gateway import GatewayConfig, create_app

API_KEYS = {"k": "proj"}
HEADERS = {"X-API-Key": "k"}


def test_full_queue_returns_429_then_drains(wait_until: Callable[..., bool]) -> None:
    writer = BlockingWriter()
    # Tiny queue + batch_size 1 so the worker takes one span, blocks writing it,
    # and the rest of a flood cannot be enqueued.
    cfg = GatewayConfig(queue_maxsize=3, batch_size=1, flush_interval=0.05)
    spans = [make_span().model_dump(mode="json") for _ in range(30)]

    with TestClient(create_app(writer=writer, api_keys=API_KEYS, config=cfg)) as client:
        # Let the worker pick up and block on the first span.
        wait_until(lambda: True, timeout=0.1)
        resp = client.post("/v1/spans", headers=HEADERS, json={"spans": spans})
        assert resp.status_code == 429  # some spans did not fit
        body = resp.json()
        assert body["backpressured"] > 0
        assert body["accepted"] + body["backpressured"] == 30

        # Service is still alive and serving other routes.
        assert client.get("/healthz").json() == {"status": "ok"}

        # Release storage; everything that was accepted gets written.
        accepted = body["accepted"]
        writer.release()
        assert wait_until(lambda: writer.inner.count == accepted, timeout=10.0)

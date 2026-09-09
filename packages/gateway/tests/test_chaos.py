"""Chaos: storage fails mid-write, the queue retries, and no span is lost."""

from __future__ import annotations

from collections.abc import Callable

from fastapi.testclient import TestClient
from helpers import FailingWriter, make_span

from pytracer_gateway import GatewayConfig, create_app

API_KEYS = {"k": "proj"}
HEADERS = {"X-API-Key": "k"}


def test_transient_storage_outage_recovers_without_loss(wait_until: Callable[..., bool]) -> None:
    # Fail the first 3 write attempts, then succeed; fast backoff to keep the test quick.
    writer = FailingWriter(fail_times=3)
    cfg = GatewayConfig(max_write_retries=10, retry_backoff=0.01, batch_size=64)
    spans = [make_span(f"s{i}").model_dump(mode="json") for i in range(20)]

    with TestClient(create_app(writer=writer, api_keys=API_KEYS, config=cfg)) as client:
        resp = client.post("/v1/spans", headers=HEADERS, json={"spans": spans})
        assert resp.status_code == 202
        # Despite the simulated outage, all 20 spans eventually persist.
        assert wait_until(lambda: writer.inner.count == 20, timeout=10.0)
        stats = client.get("/v1/stats").json()
        assert stats["stats"]["written"] == 20
        assert stats["stats"]["storage_dlq"] == 0
        assert stats["dead_letters"] == 0


def test_permanent_failure_goes_to_dead_letter(wait_until: Callable[..., bool]) -> None:
    # Never recovers within the retry budget -> spans land in the DLQ, not lost silently.
    writer = FailingWriter(fail_times=10_000)
    cfg = GatewayConfig(max_write_retries=2, retry_backoff=0.01, batch_size=8)
    spans = [make_span().model_dump(mode="json") for _ in range(8)]

    with TestClient(create_app(writer=writer, api_keys=API_KEYS, config=cfg)) as client:
        client.post("/v1/spans", headers=HEADERS, json={"spans": spans})
        assert wait_until(lambda: client.get("/v1/stats").json()["dead_letters"] == 8, timeout=10.0)
        stats = client.get("/v1/stats").json()
        assert stats["stats"]["storage_dlq"] == 8
        assert stats["stats"]["written"] == 0

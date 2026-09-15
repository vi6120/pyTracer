"""The optional Prometheus /metrics endpoint: real output, and the no-dep stub."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient
from helpers import FakeWriter, make_span

from pytracer_gateway import create_app
from pytracer_gateway.metrics import metrics_available

API_KEYS = {"k": "proj"}
HEADERS = {"X-API-Key": "k"}


def _metric_value(body: str, name: str) -> float:
    """Read a single scalar sample (``name value``) out of the exposition text."""
    for line in body.splitlines():
        if line.startswith(name + " "):
            return float(line.split()[1])
    raise AssertionError(f"metric {name!r} not present in /metrics output")


def _written(client: TestClient) -> int:
    return int(client.get("/v1/stats").json()["stats"]["written"])


@pytest.mark.skipif(not metrics_available(), reason="prometheus-client not installed")
def test_metrics_exposes_span_counters_and_latency(
    wait_until: Callable[..., bool],
) -> None:
    writer = FakeWriter()
    spans = [make_span(f"s{i}").model_dump(mode="json") for i in range(5)]
    with TestClient(create_app(writer=writer, api_keys=API_KEYS)) as client:
        assert client.post("/v1/spans", headers=HEADERS, json=spans).status_code == 202
        # Wait for the writer to drain so the written counter is settled.
        assert wait_until(lambda: _written(client) == 5)
        resp = client.get("/metrics")

    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]
    body = resp.text
    # Counters read live from the pipeline stats at scrape time.
    assert _metric_value(body, "pytracer_spans_accepted_total") == 5.0
    assert _metric_value(body, "pytracer_spans_written_total") == 5.0
    # The request-latency histogram recorded the ingest call.
    assert "pytracer_ingest_request_seconds_bucket" in body
    assert "/v1/spans" in body


def test_metrics_stub_returns_501_without_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Force the no-dependency path even when prometheus-client is installed, so
    # the stub behavior (a clear 501, not a crash) is covered everywhere.
    monkeypatch.setattr("pytracer_gateway.metrics.metrics_available", lambda: False)
    with TestClient(create_app(writer=FakeWriter(), api_keys=API_KEYS)) as client:
        resp = client.get("/metrics")
    assert resp.status_code == 501
    assert "metrics" in resp.text.lower()

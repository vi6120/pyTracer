"""Per-key rate limiting: the token bucket, and the 429 it produces on ingest."""

from __future__ import annotations

from fastapi.testclient import TestClient
from helpers import FakeWriter, make_span

from navik_gateway import GatewayConfig, TokenBucketLimiter, create_app

API_KEYS = {"k": "proj"}
HEADERS = {"X-API-Key": "k"}


def test_disabled_limiter_always_allows() -> None:
    limiter = TokenBucketLimiter(rate=0, burst=0)
    assert limiter.enabled is False
    assert all(limiter.allow("k") for _ in range(1000))


def test_bucket_admits_burst_then_denies() -> None:
    # rate is tiny, so almost nothing refills between calls: exactly `burst`
    # requests get through, then the bucket is dry.
    limiter = TokenBucketLimiter(rate=0.001, burst=3)
    assert limiter.enabled is True
    assert [limiter.allow("k") for _ in range(5)] == [True, True, True, False, False]
    # A different key has its own bucket.
    assert limiter.allow("other") is True


def test_bucket_refills_over_time() -> None:
    # A high rate refills a spent bucket quickly.
    limiter = TokenBucketLimiter(rate=1000.0, burst=1)
    assert limiter.allow("k") is True
    # Draining is possible only if we outrun the refill; here we just prove a
    # spent token comes back after a short wait.
    import time

    limiter.allow("k")
    time.sleep(0.01)
    assert limiter.allow("k") is True


def test_ingest_returns_429_when_rate_limited() -> None:
    cfg = GatewayConfig(rate_limit_per_sec=0.001, rate_limit_burst=2)
    writer = FakeWriter()
    span = make_span().model_dump(mode="json")
    with TestClient(create_app(writer=writer, api_keys=API_KEYS, config=cfg)) as client:
        r1 = client.post("/v1/spans", headers=HEADERS, json={"spans": [span]})
        r2 = client.post("/v1/spans", headers=HEADERS, json={"spans": [span]})
        r3 = client.post("/v1/spans", headers=HEADERS, json={"spans": [span]})
    assert r1.status_code == 202
    assert r2.status_code == 202
    assert r3.status_code == 429
    assert r3.json()["detail"] == "rate limit exceeded"
    assert r3.headers.get("Retry-After") == "1"

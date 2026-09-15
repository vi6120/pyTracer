"""Prometheus metrics for the ingestion gateway (optional).

Enabled when ``prometheus-client`` is installed (the ``[metrics]`` extra). When
it is, the gateway mounts ``GET /metrics`` exposing:

- ``pytracer_ingest_request_seconds`` - a latency histogram per endpoint and
  status code, recorded by an HTTP middleware, and
- ``pytracer_spans_*`` counters (accepted, written, validation_failed,
  backpressured, storage_dlq) read live from the pipeline's ``Stats`` at scrape
  time, plus ``process_*`` metrics for the gateway's CPU and memory.

When the dependency is absent the mount is a no-op stub that returns 501, so the
base install stays lean and nothing else about the gateway changes. Each app
gets its own registry, so building several apps in one process (tests) never
collides on the global default registry.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Iterator
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse, Response
from starlette.requests import Request

if TYPE_CHECKING:
    from .ingest import IngestPipeline

__all__ = ["install_metrics", "metrics_available"]

# Buckets tuned for a fast in-memory ingest path: sub-millisecond up to a few
# seconds, so p50/p95/p99 stay meaningful under load.
_LATENCY_BUCKETS = (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)

_STATS_DOCS = {
    "accepted": "Spans enqueued for writing.",
    "written": "Spans persisted to the trace store.",
    "validation_failed": "Spans rejected by schema validation.",
    "backpressured": "Spans dropped because the queue was full.",
    "storage_dlq": "Spans dead-lettered after write retries were exhausted.",
}


def metrics_available() -> bool:
    """True when ``prometheus-client`` is importable (the ``[metrics]`` extra)."""
    try:
        import prometheus_client  # noqa: F401
    except ImportError:
        return False
    return True


def install_metrics(app: FastAPI, pipeline: IngestPipeline) -> None:
    """Mount ``GET /metrics`` on ``app``, backed by ``pipeline``'s counters.

    A no-op stub (returning 501) is mounted when ``prometheus-client`` is not
    installed, so callers never have to branch on availability.
    """
    if not metrics_available():
        @app.get("/metrics")
        async def _metrics_unavailable() -> PlainTextResponse:
            return PlainTextResponse(
                "metrics require the [metrics] extra: "
                "pip install 'pytracer-gateway[metrics]'\n",
                status_code=501,
            )

        return

    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        Histogram,
        ProcessCollector,
        generate_latest,
    )
    from prometheus_client.core import CounterMetricFamily

    registry = CollectorRegistry()
    try:
        # CPU and RSS for the gateway process (reads /proc; harmless if absent).
        ProcessCollector(registry=registry)
    except Exception:  # noqa: BLE001 - process metrics are best-effort only.
        pass

    request_seconds = Histogram(
        "pytracer_ingest_request_seconds",
        "Gateway HTTP request latency in seconds.",
        ("endpoint", "status"),
        buckets=_LATENCY_BUCKETS,
        registry=registry,
    )

    class _StatsCollector:
        """Read the live ingest counters from ``Stats`` on each scrape."""

        def collect(self) -> Iterator[CounterMetricFamily]:
            for key, value in pipeline.stats.snapshot().items():
                yield CounterMetricFamily(
                    f"pytracer_spans_{key}", _STATS_DOCS.get(key, ""), value=value
                )

    registry.register(_StatsCollector())

    @app.middleware("http")
    async def _record_request(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # Do not measure the scrape endpoint itself.
        if request.url.path == "/metrics":
            return await call_next(request)
        start = time.perf_counter()
        response = await call_next(request)
        # Only the gateway's own fixed routes reach here, so path cardinality is
        # bounded and safe to use as a label.
        request_seconds.labels(
            endpoint=request.url.path, status=str(response.status_code)
        ).observe(time.perf_counter() - start)
        return response

    @app.get("/metrics")
    async def _metrics() -> Response:
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

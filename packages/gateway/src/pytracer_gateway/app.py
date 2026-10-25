"""FastAPI application for the ingestion gateway.

Endpoints:
    POST /v1/spans   ingest a batch of spans (X-API-Key required)
    GET  /healthz    liveness, with the gateway's version
    GET  /readyz     readiness (ingest worker running)
    GET  /v1/stats   ingest counters (JSON)
    GET  /metrics    Prometheus metrics (with the [metrics] extra)
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse

from .auth import API_KEY_HEADER, APIKeyAuth, AuthBackend
from .config import GatewayConfig, load_api_keys
from .ingest import IngestPipeline, TraceStoreWriter, Writer, validate_spans
from .metrics import install_metrics
from .ratelimit import TokenBucketLimiter

_log = logging.getLogger("pytracer_gateway")

try:
    # The installed distribution's version, so /healthz, /openapi.json and /docs
    # report the release actually deployed instead of a stale hardcoded value.
    GATEWAY_VERSION = _pkg_version("pytracer-gateway")
except PackageNotFoundError:  # running from a source tree without an installed dist
    GATEWAY_VERSION = "0.0.0+unknown"


def create_app(
    *,
    writer: Writer,
    api_keys: dict[str, str] | None = None,
    auth: AuthBackend | None = None,
    config: GatewayConfig | None = None,
) -> FastAPI:
    """Build a gateway app around a writer and an auth backend.

    Pass ``auth`` to use a specific backend (e.g. the store-backed one); when it
    is omitted, a static backend is built from ``api_keys``.
    """
    cfg = config or GatewayConfig()
    resolver: AuthBackend = auth if auth is not None else APIKeyAuth(api_keys or {})
    limiter = TokenBucketLimiter(cfg.rate_limit_per_sec, cfg.rate_limit_burst)
    pipeline = IngestPipeline(writer, cfg)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        # Log the resolved config so an operator can confirm the deploy settings
        # actually reached the container (e.g. the managed key store is active).
        _log.info(
            "pytracer gateway %s starting: auth=%s rate_limit_per_sec=%s queue_maxsize=%s",
            GATEWAY_VERSION,
            type(resolver).__name__,
            cfg.rate_limit_per_sec,
            cfg.queue_maxsize,
        )
        await pipeline.start()
        try:
            yield
        finally:
            await pipeline.stop()

    app = FastAPI(title="pyTracer Ingestion Gateway", version=GATEWAY_VERSION, lifespan=lifespan)
    app.state.pipeline = pipeline

    def require_project(x_api_key: str | None = Header(default=None, alias=API_KEY_HEADER)) -> str:
        project = resolver.project_for(x_api_key)
        if project is None:
            # Reject before any span processing happens.
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="missing or invalid API key",
            )
        # Authenticated: apply the per-key rate limit before accepting work.
        if limiter.enabled and x_api_key is not None and not limiter.allow(x_api_key):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="rate limit exceeded",
                headers={"Retry-After": "1"},
            )
        return project

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        # The version is public so an operator (or a client) can confirm which
        # release a gateway runs, e.g. before upgrading the SDKs that send to it.
        return {"status": "ok", "version": GATEWAY_VERSION}

    @app.get("/readyz")
    async def readyz() -> JSONResponse:
        ready = pipeline.running
        return JSONResponse(
            {"ready": ready},
            status_code=status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    @app.get("/v1/stats")
    async def stats() -> dict[str, Any]:
        return {"stats": pipeline.stats.snapshot(), "dead_letters": await pipeline.dead_letter_count()}

    @app.get("/v1/dead-letters")
    async def dead_letters(limit: int = 20) -> dict[str, Any]:
        # Inspect the dead-letter store (durable when the Redis backend is used).
        capped = min(max(limit, 1), 200)
        sample = await pipeline.dead_letters(capped)
        return {
            "count": await pipeline.dead_letter_count(),
            "sample": [
                {"reason": d.reason, "project": d.project, "detail": d.detail} for d in sample
            ],
        }

    @app.post("/v1/dead-letters/requeue")
    async def requeue_dead_letters(limit: int = 200) -> dict[str, Any]:
        # Re-enqueue storage dead letters for another write (after an outage recovers).
        capped = min(max(limit, 1), 10_000)
        requeued = await pipeline.requeue_dead_letters(capped)
        return {"requeued": requeued, "remaining": await pipeline.dead_letter_count()}

    @app.post("/v1/spans")
    async def ingest_spans(request: Request, project: str = Depends(require_project)) -> JSONResponse:
        body = await _parse_body(request)
        spans, errors = validate_spans(body)
        await pipeline.record_validation_failures(project, errors)

        accepted, backpressured = await pipeline.enqueue(project, spans)

        payload: dict[str, Any] = {
            "accepted": accepted,
            "rejected": len(errors),
            "backpressured": backpressured,
        }
        if errors:
            payload["errors"] = errors[: cfg.max_errors_in_response]

        if backpressured > 0:
            # Some spans did not fit; ask the client to retry the remainder. The
            # Retry-After tells the SDK how long to wait, and the accepted count
            # lets it resend only the backpressured tail, not the stored prefix.
            return JSONResponse(
                payload,
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                headers={"Retry-After": "1"},
            )
        return JSONResponse(payload, status_code=status.HTTP_202_ACCEPTED)

    # Prometheus scrape endpoint (no-op stub unless the [metrics] extra is present).
    install_metrics(app, pipeline)

    return app


async def _parse_body(request: Request) -> list[Any]:
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid JSON body") from exc
    if isinstance(body, list):
        return body
    if isinstance(body, dict):
        spans = body.get("spans")
        if isinstance(spans, list):
            return spans
    raise HTTPException(
        status.HTTP_400_BAD_REQUEST,
        "body must be a list of spans or an object with a 'spans' list",
    )


def build_default_app() -> FastAPI:
    """Factory used by uvicorn: wires the ClickHouse trace store from env config.

    Migrates the spans table on startup (idempotent), so a freshly deployed
    gateway can accept writes without a separate migration step. The auth backend
    is chosen by ``PYTRACER_GATEWAY_AUTH_BACKEND`` (static map or Postgres key store).
    """
    from pytracer_stores import TraceStore

    cfg = GatewayConfig()
    store = TraceStore()
    store.migrate()
    return create_app(
        writer=TraceStoreWriter(store),
        api_keys=load_api_keys(),
        auth=_build_auth(cfg),
        config=cfg,
    )


def _build_auth(cfg: GatewayConfig) -> AuthBackend | None:
    """Build the store-backed auth for the Postgres backend; None keeps static."""
    if cfg.auth_backend.lower() == "postgres":
        from pytracer_stores import ApiKeyStore

        from .auth import StoreApiKeyAuth

        key_store = ApiKeyStore()
        key_store.migrate()
        return StoreApiKeyAuth(key_store, ttl=cfg.auth_cache_ttl)
    return None

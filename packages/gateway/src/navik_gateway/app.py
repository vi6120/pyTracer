"""FastAPI application for the ingestion gateway.

Endpoints:
    POST /v1/spans   ingest a batch of spans (X-API-Key required)
    GET  /healthz    liveness
    GET  /readyz     readiness (ingest worker running)
    GET  /v1/stats   ingest counters
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse

from .auth import API_KEY_HEADER, APIKeyAuth
from .config import GatewayConfig, load_api_keys
from .ingest import IngestPipeline, TraceStoreWriter, Writer, validate_spans


def create_app(
    *,
    writer: Writer,
    api_keys: dict[str, str],
    config: GatewayConfig | None = None,
) -> FastAPI:
    """Build a gateway app around a given writer and API-key map."""
    cfg = config or GatewayConfig()
    auth = APIKeyAuth(api_keys)
    pipeline = IngestPipeline(writer, cfg)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await pipeline.start()
        try:
            yield
        finally:
            await pipeline.stop()

    app = FastAPI(title="Navik Ingestion Gateway", version="0.0.1", lifespan=lifespan)
    app.state.pipeline = pipeline

    def require_project(x_api_key: str | None = Header(default=None, alias=API_KEY_HEADER)) -> str:
        project = auth.project_for(x_api_key)
        if project is None:
            # Reject before any span processing happens.
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="missing or invalid API key",
            )
        return project

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

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
            # Some spans did not fit; ask the client to retry the remainder.
            return JSONResponse(payload, status_code=status.HTTP_429_TOO_MANY_REQUESTS)
        return JSONResponse(payload, status_code=status.HTTP_202_ACCEPTED)

    return app


async def _parse_body(request: Request) -> list[Any]:
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid JSON body") from exc
    if isinstance(body, list):
        return body
    if isinstance(body, dict) and isinstance(body.get("spans"), list):
        return body["spans"]
    raise HTTPException(
        status.HTTP_400_BAD_REQUEST,
        "body must be a list of spans or an object with a 'spans' list",
    )


def build_default_app() -> FastAPI:
    """Factory used by uvicorn: wires the ClickHouse trace store from env config.

    Migrates the spans table on startup (idempotent), so a freshly deployed
    gateway can accept writes without a separate migration step.
    """
    from navik_stores import TraceStore

    store = TraceStore()
    store.migrate()
    return create_app(
        writer=TraceStoreWriter(store),
        api_keys=load_api_keys(),
    )

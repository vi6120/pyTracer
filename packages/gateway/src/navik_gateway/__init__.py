"""Navik ingestion gateway - receive spans and write them through to storage."""

from __future__ import annotations

from .app import build_default_app, create_app
from .auth import APIKeyAuth, AuthBackend, StoreApiKeyAuth
from .config import GatewayConfig, load_api_keys
from .ingest import (
    IngestPipeline,
    Stats,
    TraceStoreWriter,
    Writer,
    validate_spans,
)
from .queue import (
    DeadLetter,
    InProcessQueue,
    RedisQueue,
    Reserved,
    SpanQueue,
    build_queue,
)
from .ratelimit import TokenBucketLimiter

__all__ = [
    "APIKeyAuth",
    "AuthBackend",
    "DeadLetter",
    "GatewayConfig",
    "InProcessQueue",
    "IngestPipeline",
    "RedisQueue",
    "Reserved",
    "SpanQueue",
    "Stats",
    "StoreApiKeyAuth",
    "TokenBucketLimiter",
    "TraceStoreWriter",
    "Writer",
    "build_default_app",
    "build_queue",
    "create_app",
    "load_api_keys",
    "validate_spans",
]

__version__ = "0.0.1"

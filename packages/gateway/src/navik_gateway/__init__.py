"""Navik ingestion gateway — receive spans and write them through to storage."""

from __future__ import annotations

from .app import build_default_app, create_app
from .auth import APIKeyAuth
from .config import GatewayConfig, load_api_keys
from .ingest import (
    DeadLetter,
    IngestPipeline,
    Stats,
    TraceStoreWriter,
    Writer,
    validate_spans,
)

__all__ = [
    "APIKeyAuth",
    "DeadLetter",
    "GatewayConfig",
    "IngestPipeline",
    "Stats",
    "TraceStoreWriter",
    "Writer",
    "build_default_app",
    "create_app",
    "load_api_keys",
    "validate_spans",
]

__version__ = "0.0.1"

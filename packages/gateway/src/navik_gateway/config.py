"""Gateway configuration and API-key loading.

API keys map to projects. For local development the mapping is read from
``NAVIK_GATEWAY_API_KEYS`` in the form ``key1:project1,key2:project2``. In
production this would come from a database; the gateway only needs the mapping.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class GatewayConfig:
    """Tunables for the gateway's ingest queue, batching, and write retries."""

    queue_maxsize: int = int(os.getenv("NAVIK_GATEWAY_QUEUE_MAXSIZE", "10000"))
    batch_size: int = int(os.getenv("NAVIK_GATEWAY_BATCH_SIZE", "256"))
    flush_interval: float = float(os.getenv("NAVIK_GATEWAY_FLUSH_INTERVAL", "0.5"))
    max_write_retries: int = int(os.getenv("NAVIK_GATEWAY_MAX_WRITE_RETRIES", "5"))
    retry_backoff: float = float(os.getenv("NAVIK_GATEWAY_RETRY_BACKOFF", "0.1"))
    max_errors_in_response: int = int(os.getenv("NAVIK_GATEWAY_MAX_ERRORS_IN_RESPONSE", "20"))

    # Queue backend: "memory" (default, in-process) or "redis" (durable, so a
    # restart replays in-flight spans and keeps the dead-letter list).
    queue_backend: str = os.getenv("NAVIK_GATEWAY_QUEUE_BACKEND", "memory")
    redis_url: str = os.getenv("NAVIK_REDIS_URL", "redis://localhost:6379/0")
    redis_stream: str = os.getenv("NAVIK_REDIS_STREAM", "navik:spans")
    redis_group: str = os.getenv("NAVIK_REDIS_GROUP", "navik-writers")
    # Idle time (ms) before reclaiming another consumer's pending entries. 0 suits
    # a single instance; raise it when running several gateways on one stream.
    reclaim_min_idle_ms: int = int(os.getenv("NAVIK_REDIS_RECLAIM_MIN_IDLE_MS", "0"))

    # Auth backend: "static" (default, the NAVIK_GATEWAY_API_KEYS map) or
    # "postgres" (keys created/revoked at runtime via the ApiKeyStore).
    auth_backend: str = os.getenv("NAVIK_GATEWAY_AUTH_BACKEND", "static")
    # Seconds a resolved key is cached; also the window a revoked key keeps working.
    auth_cache_ttl: float = float(os.getenv("NAVIK_GATEWAY_AUTH_CACHE_TTL", "30"))

    # Per-key rate limit (token bucket). rate <= 0 disables limiting; burst
    # defaults to the rate when left at 0.
    rate_limit_per_sec: float = float(os.getenv("NAVIK_GATEWAY_RATE_LIMIT_PER_SEC", "0"))
    rate_limit_burst: float = float(os.getenv("NAVIK_GATEWAY_RATE_LIMIT_BURST", "0"))


def load_api_keys() -> dict[str, str]:
    """Parse ``NAVIK_GATEWAY_API_KEYS`` into a ``{api_key: project}`` mapping."""
    raw = os.getenv("NAVIK_GATEWAY_API_KEYS", "").strip()
    mapping: dict[str, str] = {}
    if not raw:
        return mapping
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair or ":" not in pair:
            continue
        key, project = pair.split(":", 1)
        key, project = key.strip(), project.strip()
        if key and project:
            mapping[key] = project
    return mapping

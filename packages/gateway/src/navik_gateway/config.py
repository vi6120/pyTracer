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
    queue_maxsize: int = int(os.getenv("NAVIK_GATEWAY_QUEUE_MAXSIZE", "10000"))
    batch_size: int = int(os.getenv("NAVIK_GATEWAY_BATCH_SIZE", "256"))
    flush_interval: float = float(os.getenv("NAVIK_GATEWAY_FLUSH_INTERVAL", "0.5"))
    max_write_retries: int = int(os.getenv("NAVIK_GATEWAY_MAX_WRITE_RETRIES", "5"))
    retry_backoff: float = float(os.getenv("NAVIK_GATEWAY_RETRY_BACKOFF", "0.1"))
    max_errors_in_response: int = int(os.getenv("NAVIK_GATEWAY_MAX_ERRORS_IN_RESPONSE", "20"))


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

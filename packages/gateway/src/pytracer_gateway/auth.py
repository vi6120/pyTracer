"""Per-project API-key authentication.

Two backends resolve an API key to its project:

- ``APIKeyAuth`` reads a fixed key -> project map (the ``PYTRACER_GATEWAY_API_KEYS``
  environment variable). Simple, but a key cannot be rotated or revoked without
  a redeploy. This is the default and suits local development.
- ``StoreApiKeyAuth`` resolves against the PostgreSQL ``ApiKeyStore``, so keys
  are created and revoked at runtime. Resolved keys are cached briefly to keep
  the request path fast, so a revoked key stops working within the cache TTL.

Either way a key is resolved before any span is processed, so an invalid or
missing key is rejected up front (guide section 3: reject before any processing).
"""

from __future__ import annotations

import hashlib
import time
from threading import Lock
from typing import Any, Protocol

API_KEY_HEADER = "X-API-Key"


class AuthBackend(Protocol):
    """Resolves an API key to its project, or ``None`` if the key is unknown."""

    def project_for(self, api_key: str | None) -> str | None: ...


class APIKeyAuth:
    """Static backend: resolves an API key against a fixed in-memory map."""

    def __init__(self, keys: dict[str, str]) -> None:
        self._keys = dict(keys)

    def project_for(self, api_key: str | None) -> str | None:
        """Return the project an API key maps to, or None if the key is unknown."""
        if not api_key:
            return None
        return self._keys.get(api_key)

    def __len__(self) -> int:
        return len(self._keys)


def _cache_key(api_key: str) -> str:
    # Cache by hash so plaintext secrets are not held in the cache dict.
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


class StoreApiKeyAuth:
    """Store-backed backend: resolves against the ApiKeyStore, with a TTL cache.

    Postgres is only queried on a cache miss, so the hot path stays fast. A key
    that is revoked keeps working until its cache entry expires (at most ``ttl``
    seconds). Negative results are cached too, so a flood of bad keys does not
    hammer the store.
    """

    def __init__(self, store: Any, *, ttl: float = 30.0, max_entries: int = 10_000) -> None:
        self._store = store
        self._ttl = ttl
        self._max_entries = max_entries
        self._cache: dict[str, tuple[str | None, float]] = {}
        self._cache_lock = Lock()
        # The ApiKeyStore holds one psycopg connection, which is not safe for
        # concurrent use; serialize the (rare, cache-missing) DB hits.
        self._db_lock = Lock()

    def project_for(self, api_key: str | None) -> str | None:
        if not api_key:
            return None
        ck = _cache_key(api_key)
        now = time.monotonic()
        with self._cache_lock:
            hit = self._cache.get(ck)
            if hit is not None and hit[1] > now:
                return hit[0]
        with self._db_lock:
            project: str | None = self._store.resolve(api_key)
        with self._cache_lock:
            if len(self._cache) >= self._max_entries:
                self._cache.clear()
            self._cache[ck] = (project, now + self._ttl)
        return project

    def invalidate(self) -> None:
        """Drop the cache so revocations take effect immediately (used in tests)."""
        with self._cache_lock:
            self._cache.clear()

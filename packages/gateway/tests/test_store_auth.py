"""Store-backed auth: TTL cache behavior, and a live path against Postgres.

The cache-behavior tests use a fake store and need no services. The integration
tests build a real ApiKeyStore and are skipped when Postgres is not reachable.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient
from helpers import FakeWriter, make_span

from navik_gateway import GatewayConfig, StoreApiKeyAuth, create_app


class FakeKeyStore:
    """Minimal ApiKeyStore stand-in that counts how often resolve() is called."""

    def __init__(self, keys: dict[str, str]) -> None:
        self._keys = keys
        self.resolve_calls = 0

    def resolve(self, key: str | None) -> str | None:
        self.resolve_calls += 1
        return self._keys.get(key) if key else None


def test_cache_avoids_repeated_store_hits() -> None:
    store = FakeKeyStore({"good": "proj"})
    auth = StoreApiKeyAuth(store, ttl=60.0)
    assert auth.project_for("good") == "proj"
    assert auth.project_for("good") == "proj"
    assert auth.project_for("good") == "proj"
    assert store.resolve_calls == 1  # only the first missed the cache


def test_negative_results_are_cached() -> None:
    store = FakeKeyStore({})
    auth = StoreApiKeyAuth(store, ttl=60.0)
    assert auth.project_for("bad") is None
    assert auth.project_for("bad") is None
    assert store.resolve_calls == 1  # a bad key does not hammer the store


def test_empty_key_short_circuits() -> None:
    store = FakeKeyStore({"good": "proj"})
    auth = StoreApiKeyAuth(store)
    assert auth.project_for(None) is None
    assert auth.project_for("") is None
    assert store.resolve_calls == 0


def test_invalidate_forces_a_refresh() -> None:
    store = FakeKeyStore({"good": "proj"})
    auth = StoreApiKeyAuth(store, ttl=60.0)
    assert auth.project_for("good") == "proj"
    # Revoke in the backing store, then clear the cache: the next lookup misses.
    store._keys.clear()
    auth.invalidate()
    assert auth.project_for("good") is None
    assert store.resolve_calls == 2


# --- integration against a real Postgres ApiKeyStore ------------------------


def _postgres_up() -> bool:
    try:
        import psycopg
        from navik_stores.config import PostgresConfig

        conn = psycopg.connect(PostgresConfig().conninfo(), connect_timeout=3)
        conn.close()
        return True
    except Exception:  # noqa: BLE001
        return False


pg = pytest.mark.skipif(not _postgres_up(), reason="PostgreSQL not reachable")


@pg
def test_live_gateway_accepts_then_rejects_a_revoked_key() -> None:
    from navik_stores import ApiKeyStore

    table = f"api_keys_test_{uuid.uuid4().hex[:8]}"
    store = ApiKeyStore(table=table)
    store.migrate()
    try:
        record, full_key = store.create("live-proj", name="test")
        # ttl=0 so the cache never masks a revocation in this test.
        auth = StoreApiKeyAuth(store, ttl=0.0)
        writer = FakeWriter()
        span = make_span().model_dump(mode="json")
        with TestClient(create_app(writer=writer, auth=auth, config=GatewayConfig())) as client:
            ok = client.post("/v1/spans", headers={"X-API-Key": full_key}, json={"spans": [span]})
            assert ok.status_code == 202

            store.revoke(record.key_id)
            auth.invalidate()
            denied = client.post(
                "/v1/spans", headers={"X-API-Key": full_key}, json={"spans": [span]}
            )
            assert denied.status_code == 401
    finally:
        store.drop()
        store.close()

"""Shared fixtures. Integration tests run against the docker-compose services;
they skip cleanly (not fail) when those services are not reachable.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import pytest

from navik_stores import MockStore, TraceStore
from navik_stores.config import ClickHouseConfig, PostgresConfig


def _clickhouse_up() -> bool:
    try:
        import clickhouse_connect

        cfg = ClickHouseConfig()
        client = clickhouse_connect.get_client(
            host=cfg.host, port=cfg.port, username=cfg.username,
            password=cfg.password, database=cfg.database,
        )
        client.command("SELECT 1")
        client.close()
        return True
    except Exception:  # noqa: BLE001 — any failure means "not reachable, skip"
        return False


def _postgres_up() -> bool:
    try:
        import psycopg

        conn = psycopg.connect(PostgresConfig().conninfo(), connect_timeout=3)
        conn.close()
        return True
    except Exception:  # noqa: BLE001 — any failure means "not reachable, skip"
        return False


CLICKHOUSE_UP = _clickhouse_up()
POSTGRES_UP = _postgres_up()


@pytest.fixture()
def trace_store() -> Iterator[TraceStore]:
    if not CLICKHOUSE_UP:
        pytest.skip("ClickHouse not reachable (run `docker compose up -d`)")
    table = f"spans_test_{uuid.uuid4().hex[:8]}"
    store = TraceStore(table=table)
    store.migrate()
    try:
        yield store
    finally:
        store.drop()
        store.close()


@pytest.fixture()
def mock_store() -> Iterator[MockStore]:
    if not POSTGRES_UP:
        pytest.skip("PostgreSQL not reachable (run `docker compose up -d`)")
    table = f"mocks_test_{uuid.uuid4().hex[:8]}"
    store = MockStore(table=table)
    store.migrate()
    try:
        yield store
    finally:
        store.drop()
        store.close()

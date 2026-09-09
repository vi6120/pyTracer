"""Connection settings for the trace and mock stores.

Defaults match the local ``docker-compose.yml`` (ClickHouse + PostgreSQL, both
``pytracer``/``pytracer``). Every value can be overridden by an environment variable
so the same code runs against local containers, CI services, or managed tiers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ClickHouseConfig:
    """ClickHouse connection settings for the trace store (env-overridable)."""

    host: str = os.getenv("PYTRACER_CLICKHOUSE_HOST", "localhost")
    port: int = int(os.getenv("PYTRACER_CLICKHOUSE_PORT", "8123"))
    username: str = os.getenv("PYTRACER_CLICKHOUSE_USER", "pytracer")
    password: str = os.getenv("PYTRACER_CLICKHOUSE_PASSWORD", "pytracer")
    database: str = os.getenv("PYTRACER_CLICKHOUSE_DB", "pytracer")


@dataclass(frozen=True)
class PostgresConfig:
    """PostgreSQL connection settings for the mock store (env-overridable)."""

    host: str = os.getenv("PYTRACER_POSTGRES_HOST", "localhost")
    port: int = int(os.getenv("PYTRACER_POSTGRES_PORT", "5432"))
    dbname: str = os.getenv("PYTRACER_POSTGRES_DB", "pytracer")
    user: str = os.getenv("PYTRACER_POSTGRES_USER", "pytracer")
    password: str = os.getenv("PYTRACER_POSTGRES_PASSWORD", "pytracer")

    def conninfo(self) -> str:
        """Return a libpq connection string for psycopg."""
        return (
            f"host={self.host} port={self.port} dbname={self.dbname} "
            f"user={self.user} password={self.password}"
        )

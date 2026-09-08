"""Connection settings for the trace and mock stores.

Defaults match the local ``docker-compose.yml`` (ClickHouse + PostgreSQL, both
``navik``/``navik``). Every value can be overridden by an environment variable
so the same code runs against local containers, CI services, or managed tiers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ClickHouseConfig:
    host: str = os.getenv("NAVIK_CLICKHOUSE_HOST", "localhost")
    port: int = int(os.getenv("NAVIK_CLICKHOUSE_PORT", "8123"))
    username: str = os.getenv("NAVIK_CLICKHOUSE_USER", "navik")
    password: str = os.getenv("NAVIK_CLICKHOUSE_PASSWORD", "navik")
    database: str = os.getenv("NAVIK_CLICKHOUSE_DB", "navik")


@dataclass(frozen=True)
class PostgresConfig:
    host: str = os.getenv("NAVIK_POSTGRES_HOST", "localhost")
    port: int = int(os.getenv("NAVIK_POSTGRES_PORT", "5432"))
    dbname: str = os.getenv("NAVIK_POSTGRES_DB", "navik")
    user: str = os.getenv("NAVIK_POSTGRES_USER", "navik")
    password: str = os.getenv("NAVIK_POSTGRES_PASSWORD", "navik")

    def conninfo(self) -> str:
        return (
            f"host={self.host} port={self.port} dbname={self.dbname} "
            f"user={self.user} password={self.password}"
        )

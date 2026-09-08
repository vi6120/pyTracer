"""Mock store — PostgreSQL.

Holds recorded tool responses used to reconstruct deterministic replays. Each
mock is keyed by ``(project, tool_name, input_hash)`` and *versioned*, so the
same tool call can carry multiple recorded outcomes for different scenarios,
each flagged as a success or a failure case (guide §4). Mock volume is far
lower than trace volume, so a row store is the right fit here.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from .config import PostgresConfig
from .hashing import input_hash


def _advisory_lock_key(project: str, tool_name: str, input_hash_hex: str) -> int:
    """Stable signed 64-bit key for a per-mock-key Postgres advisory lock."""
    digest = hashlib.sha256(f"{project}\x00{tool_name}\x00{input_hash_hex}".encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=True)


@dataclass(frozen=True)
class MockRecord:
    """One recorded tool response."""

    id: int
    project: str
    tool_name: str
    input_hash: str
    input: Any
    output: Any
    is_failure: bool
    version: int
    scenario: str | None
    created_at: datetime


def _mocks_ddl(table: str) -> str:
    return f"""
    CREATE TABLE IF NOT EXISTS {table} (
        id          BIGSERIAL PRIMARY KEY,
        project     TEXT        NOT NULL DEFAULT 'default',
        tool_name   TEXT        NOT NULL,
        input_hash  TEXT        NOT NULL,
        input       JSONB,
        output      JSONB,
        is_failure  BOOLEAN     NOT NULL DEFAULT FALSE,
        version     INTEGER     NOT NULL,
        scenario    TEXT,
        created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
        UNIQUE (project, tool_name, input_hash, version)
    );
    CREATE INDEX IF NOT EXISTS {table}_key_idx
        ON {table} (project, tool_name, input_hash);
    """


_SELECT_COLS = (
    "id, project, tool_name, input_hash, input, output, is_failure, version, scenario, created_at"
)


class MockStore:
    """Read/write access to the PostgreSQL mock store."""

    def __init__(
        self,
        config: PostgresConfig | None = None,
        *,
        table: str = "mocks",
        conn: psycopg.Connection[Any] | None = None,
    ) -> None:
        self._config = config or PostgresConfig()
        self._table = table
        self._conn = conn or psycopg.connect(self._config.conninfo(), autocommit=True)

    # --- schema -------------------------------------------------------------
    def migrate(self) -> None:
        """Create the mocks table + index if absent (idempotent)."""
        self._conn.execute(_mocks_ddl(self._table))  # type: ignore[arg-type]

    # --- writes -------------------------------------------------------------
    def record(
        self,
        tool_name: str,
        input: Any,
        output: Any,
        *,
        is_failure: bool = False,
        scenario: str | None = None,
        project: str = "default",
    ) -> MockRecord:
        """Record a new mock, auto-assigning the next version for its key.

        A transaction-scoped advisory lock keyed by ``(project, tool_name,
        input_hash)`` serializes concurrent writers on the *same* key, so the
        ``MAX(version) + 1`` read-then-insert cannot race — versions stay
        contiguous with no unique-constraint collisions. Writers on different
        keys never contend.

        The lock is taken in its own statement *before* the insert, inside an
        explicit transaction: under READ COMMITTED each statement takes a fresh
        snapshot at its start, so once a waiter acquires the lock its INSERT
        sees the previous writer's committed row. (Acquiring the lock inside
        the same statement as the ``MAX`` read would not — that snapshot is
        frozen before the wait.) The lock releases at commit.
        """
        key_hash = input_hash(input)
        lock_key = _advisory_lock_key(project, tool_name, key_hash)
        insert_sql = f"""
            INSERT INTO {self._table}
                (project, tool_name, input_hash, input, output, is_failure, version, scenario)
            VALUES (
                %(project)s, %(tool_name)s, %(input_hash)s, %(input)s, %(output)s, %(is_failure)s,
                (SELECT COALESCE(MAX(version), 0) + 1 FROM {self._table}
                 WHERE project = %(project)s AND tool_name = %(tool_name)s
                   AND input_hash = %(input_hash)s),
                %(scenario)s
            )
            RETURNING {_SELECT_COLS}
        """
        params = {
            "project": project,
            "tool_name": tool_name,
            "input_hash": key_hash,
            "input": Jsonb(input),
            "output": Jsonb(output),
            "is_failure": is_failure,
            "scenario": scenario,
        }
        with self._conn.transaction():
            self._conn.execute("SELECT pg_advisory_xact_lock(%s)", (lock_key,))
            row = self._conn.execute(insert_sql, params).fetchone()  # type: ignore[arg-type]
        assert row is not None
        return self._row_to_record(row)

    # --- reads --------------------------------------------------------------
    def lookup(
        self,
        tool_name: str,
        input: Any,
        *,
        project: str = "default",
        version: int | None = None,
        scenario: str | None = None,
        is_failure: bool | None = None,
    ) -> MockRecord | None:
        """Return the newest matching mock, or a specific version/scenario."""
        key_hash = input_hash(input)
        where = [
            "project = %(project)s",
            "tool_name = %(tool_name)s",
            "input_hash = %(input_hash)s",
        ]
        params: dict[str, Any] = {
            "project": project,
            "tool_name": tool_name,
            "input_hash": key_hash,
        }
        if version is not None:
            where.append("version = %(version)s")
            params["version"] = version
        if scenario is not None:
            where.append("scenario = %(scenario)s")
            params["scenario"] = scenario
        if is_failure is not None:
            where.append("is_failure = %(is_failure)s")
            params["is_failure"] = is_failure
        sql = (
            f"SELECT {_SELECT_COLS} FROM {self._table} "
            f"WHERE {' AND '.join(where)} ORDER BY version DESC LIMIT 1"
        )
        row = self._conn.execute(sql, params).fetchone()  # type: ignore[arg-type]
        return self._row_to_record(row) if row else None

    def list_versions(
        self, tool_name: str, input: Any, *, project: str = "default"
    ) -> list[MockRecord]:
        """All recorded versions for a key, oldest first."""
        key_hash = input_hash(input)
        sql = (
            f"SELECT {_SELECT_COLS} FROM {self._table} "
            "WHERE project = %(project)s AND tool_name = %(tool_name)s "
            "AND input_hash = %(input_hash)s ORDER BY version ASC"
        )
        rows = self._conn.execute(  # type: ignore[arg-type]
            sql, {"project": project, "tool_name": tool_name, "input_hash": key_hash}
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def count(self, *, project: str | None = None) -> int:
        if project is None:
            row = self._conn.execute(f"SELECT count(*) FROM {self._table}").fetchone()  # type: ignore[arg-type]
        else:
            row = self._conn.execute(  # type: ignore[arg-type]
                f"SELECT count(*) FROM {self._table} WHERE project = %s", (project,)
            ).fetchone()
        assert row is not None
        return int(row[0])

    @staticmethod
    def _row_to_record(row: tuple[Any, ...]) -> MockRecord:
        return MockRecord(
            id=row[0],
            project=row[1],
            tool_name=row[2],
            input_hash=row[3],
            input=row[4],
            output=row[5],
            is_failure=row[6],
            version=row[7],
            scenario=row[8],
            created_at=row[9],
        )

    def _iter_all(self) -> Iterator[MockRecord]:
        rows = self._conn.execute(f"SELECT {_SELECT_COLS} FROM {self._table}").fetchall()  # type: ignore[arg-type]
        yield from (self._row_to_record(r) for r in rows)

    # --- lifecycle ----------------------------------------------------------
    def truncate(self) -> None:
        self._conn.execute(f"TRUNCATE TABLE {self._table} RESTART IDENTITY")  # type: ignore[arg-type]

    def drop(self) -> None:
        self._conn.execute(f"DROP TABLE IF EXISTS {self._table}")  # type: ignore[arg-type]

    def close(self) -> None:
        self._conn.close()

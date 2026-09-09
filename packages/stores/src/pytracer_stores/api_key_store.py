"""API-key store - PostgreSQL.

Holds the per-project API keys the gateway authenticates against. Keys are stored
hashed (SHA-256), never in plaintext: the full key is shown once at creation and
cannot be recovered afterwards, the same way a password would be handled. A key
carries a short public identifier (``key_id``) used to list and revoke it without
knowing the secret.

This replaces the static ``PYTRACER_GATEWAY_API_KEYS`` environment map for real
deployments, which cannot rotate or revoke a key without a redeploy.
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import psycopg

from .config import PostgresConfig

# Full keys look like ``nvk_<key_id>_<secret>``; the prefix makes them easy to
# spot in logs and config, and the key_id lets an operator revoke one by name.
KEY_PREFIX = "nvk"


def generate_key() -> tuple[str, str]:
    """Return ``(key_id, full_key)`` for a brand-new key. ``full_key`` is secret."""
    key_id = secrets.token_hex(4)  # 8 hex chars, the public identifier
    secret = secrets.token_urlsafe(24)  # the unguessable part
    return key_id, f"{KEY_PREFIX}_{key_id}_{secret}"


def hash_key(full_key: str) -> str:
    """SHA-256 hex digest of a full key, as stored and looked up."""
    return hashlib.sha256(full_key.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ApiKeyRecord:
    """One API key's metadata. Never carries the secret or its hash."""

    id: int
    key_id: str
    project: str
    name: str
    active: bool
    created_at: datetime
    revoked_at: datetime | None
    last_used_at: datetime | None


def _api_keys_ddl(table: str) -> str:
    return f"""
    CREATE TABLE IF NOT EXISTS {table} (
        id           BIGSERIAL PRIMARY KEY,
        key_id       TEXT        NOT NULL UNIQUE,
        key_hash     TEXT        NOT NULL UNIQUE,
        project      TEXT        NOT NULL,
        name         TEXT        NOT NULL DEFAULT '',
        active       BOOLEAN     NOT NULL DEFAULT TRUE,
        created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
        revoked_at   TIMESTAMPTZ,
        last_used_at TIMESTAMPTZ
    );
    CREATE INDEX IF NOT EXISTS {table}_project_idx ON {table} (project);
    """


# Columns that populate an ApiKeyRecord, in order (key_hash is never selected).
_SELECT_COLS = (
    "id, key_id, project, name, active, created_at, revoked_at, last_used_at"
)


class ApiKeyStore:
    """Read/write access to the PostgreSQL API-key store."""

    def __init__(
        self,
        config: PostgresConfig | None = None,
        *,
        table: str = "api_keys",
        conn: psycopg.Connection[Any] | None = None,
    ) -> None:
        self._config = config or PostgresConfig()
        self._table = table
        self._conn = conn or psycopg.connect(self._config.conninfo(), autocommit=True)

    # --- schema -------------------------------------------------------------
    def migrate(self) -> None:
        """Create the api_keys table + index if absent (idempotent)."""
        self._conn.execute(_api_keys_ddl(self._table))  # type: ignore[arg-type]

    # --- writes -------------------------------------------------------------
    def create(self, project: str, *, name: str = "") -> tuple[ApiKeyRecord, str]:
        """Mint a key for a project. Returns ``(record, full_key)``.

        ``full_key`` is the only time the secret is available; only its hash is
        stored, so it cannot be shown again.
        """
        key_id, full_key = generate_key()
        sql = f"""
            INSERT INTO {self._table} (key_id, key_hash, project, name)
            VALUES (%(key_id)s, %(key_hash)s, %(project)s, %(name)s)
            RETURNING {_SELECT_COLS}
        """
        params = {
            "key_id": key_id,
            "key_hash": hash_key(full_key),
            "project": project,
            "name": name,
        }
        row = self._conn.execute(sql, params).fetchone()  # type: ignore[arg-type]
        assert row is not None
        return self._row_to_record(row), full_key

    def revoke(self, key_id: str) -> bool:
        """Deactivate a key by its public id. Returns True if a key changed state."""
        sql = f"""
            UPDATE {self._table}
            SET active = FALSE, revoked_at = now()
            WHERE key_id = %s AND active
        """
        cur = self._conn.execute(sql, (key_id,))  # type: ignore[arg-type]
        return cur.rowcount > 0

    # --- reads --------------------------------------------------------------
    def resolve(self, presented_key: str | None) -> str | None:
        """Return the project an active key maps to, or None.

        Stamps ``last_used_at`` in the same statement, so it doubles as a
        liveness signal without an extra round-trip. The gateway caches the
        result, so this hits Postgres only on a cache miss.
        """
        if not presented_key:
            return None
        sql = f"""
            UPDATE {self._table}
            SET last_used_at = now()
            WHERE key_hash = %s AND active
            RETURNING project
        """
        row = self._conn.execute(sql, (hash_key(presented_key),)).fetchone()  # type: ignore[arg-type]
        return str(row[0]) if row else None

    def get(self, key_id: str) -> ApiKeyRecord | None:
        sql = f"SELECT {_SELECT_COLS} FROM {self._table} WHERE key_id = %s"
        row = self._conn.execute(sql, (key_id,)).fetchone()  # type: ignore[arg-type]
        return self._row_to_record(row) if row else None

    def list(
        self, *, project: str | None = None, include_revoked: bool = False
    ) -> list[ApiKeyRecord]:
        where: list[str] = []
        params: list[Any] = []
        if project is not None:
            where.append("project = %s")
            params.append(project)
        if not include_revoked:
            where.append("active")
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        sql = f"SELECT {_SELECT_COLS} FROM {self._table} {clause} ORDER BY created_at DESC"
        rows = self._conn.execute(sql, params).fetchall()  # type: ignore[arg-type]
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
    def _row_to_record(row: tuple[Any, ...]) -> ApiKeyRecord:
        return ApiKeyRecord(
            id=row[0],
            key_id=row[1],
            project=row[2],
            name=row[3],
            active=row[4],
            created_at=row[5],
            revoked_at=row[6],
            last_used_at=row[7],
        )

    # --- lifecycle ----------------------------------------------------------
    def truncate(self) -> None:
        self._conn.execute(f"TRUNCATE TABLE {self._table} RESTART IDENTITY")  # type: ignore[arg-type]

    def drop(self) -> None:
        self._conn.execute(f"DROP TABLE IF EXISTS {self._table}")  # type: ignore[arg-type]

    def close(self) -> None:
        self._conn.close()

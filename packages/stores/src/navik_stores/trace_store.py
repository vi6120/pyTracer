"""Trace store — ClickHouse.

Holds every captured span for querying, analysis, and replay. ClickHouse is a
columnar engine tuned for the append-heavy, high-cardinality write pattern that
traces produce (guide §4, §10). ``branch`` and ``commit`` are enforced as
required on every span so a captured failure is always tied to the exact code
state that produced it — the foundation cross-branch replay is built on.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import Any

import clickhouse_connect
from clickhouse_connect.driver.client import Client
from navik_sdk.schema import Resource, Span, SpanKind, SpanStatus

from .config import ClickHouseConfig
from .hashing import canonical_json

NANOS_PER_SECOND = 1_000_000_000
SECONDS_PER_DAY = 86_400

# Column order used for both the DDL and every INSERT.
_COLUMNS: tuple[str, ...] = (
    "trace_id",
    "span_id",
    "parent_span_id",
    "name",
    "kind",
    "agent_name",
    "tool_name",
    "input",
    "output",
    "start_time_ns",
    "end_time_ns",
    "latency_ms",
    "token_count",
    "cost_usd",
    "status",
    "error_type",
    "error_message",
    "attributes",
    "service_name",
    "branch",
    "commit",
    "environment",
    "sdk_name",
    "sdk_version",
    "project",
)


def _spans_ddl(database: str, table: str) -> str:
    return f"""
    CREATE TABLE IF NOT EXISTS {database}.{table}
    (
        trace_id        String,
        span_id         String,
        parent_span_id  Nullable(String),
        name            String,
        kind            LowCardinality(String),
        agent_name      Nullable(String),
        tool_name       Nullable(String),
        input           Nullable(String),
        output          Nullable(String),
        start_time_ns   UInt64,
        end_time_ns     Nullable(UInt64),
        latency_ms      Nullable(Float64),
        token_count     Nullable(UInt32),
        cost_usd        Nullable(Float64),
        status          LowCardinality(String),
        error_type      Nullable(String),
        error_message   Nullable(String),
        attributes      String DEFAULT '{{}}',
        service_name    LowCardinality(String),
        branch          LowCardinality(String),
        commit          String,
        environment     Nullable(String),
        sdk_name        LowCardinality(String),
        sdk_version     LowCardinality(String),
        project         LowCardinality(String) DEFAULT 'default',
        start_time      DateTime64(9) MATERIALIZED fromUnixTimestamp64Nano(toInt64(start_time_ns)),
        ingested_at     DateTime DEFAULT now(),
        INDEX idx_agent  agent_name TYPE bloom_filter GRANULARITY 4,
        INDEX idx_status status     TYPE set(0)        GRANULARITY 4
    )
    ENGINE = MergeTree
    PARTITION BY toYYYYMM(start_time)
    ORDER BY (project, trace_id, start_time_ns, span_id)
    SETTINGS index_granularity = 8192
    """


class TraceStore:
    """Read/write access to the ClickHouse span store."""

    def __init__(
        self,
        config: ClickHouseConfig | None = None,
        *,
        table: str = "spans",
        client: Client | None = None,
    ) -> None:
        self._config = config or ClickHouseConfig()
        self._table = table
        self._client = client or clickhouse_connect.get_client(
            host=self._config.host,
            port=self._config.port,
            username=self._config.username,
            password=self._config.password,
            database=self._config.database,
        )

    @property
    def qualified_table(self) -> str:
        return f"{self._config.database}.{self._table}"

    # --- schema -------------------------------------------------------------
    def migrate(self) -> None:
        """Create the spans table if it does not already exist (idempotent)."""
        self._client.command(_spans_ddl(self._config.database, self._table))

    # --- writes -------------------------------------------------------------
    def insert(self, spans: Span | Iterable[Span], *, project: str = "default") -> int:
        """Insert one span or a batch. Returns the number of rows written.

        Raises ``ValueError`` if any span is missing branch/commit, since the
        trace store treats those as required (guide §4).
        """
        batch = [spans] if isinstance(spans, Span) else list(spans)
        if not batch:
            return 0
        rows = [self._span_to_row(s, project) for s in batch]
        self._client.insert(self.qualified_table, rows, column_names=list(_COLUMNS))
        return len(rows)

    @staticmethod
    def _span_to_row(span: Span, project: str) -> list[Any]:
        r = span.resource
        if not r.branch or not r.commit:
            raise ValueError(
                f"trace store requires branch and commit on every span "
                f"(span {span.span_id!r} has branch={r.branch!r} commit={r.commit!r})"
            )
        return [
            span.trace_id,
            span.span_id,
            span.parent_span_id,
            span.name,
            span.kind.value,
            span.agent_name,
            span.tool_name,
            canonical_json(span.input) if span.input is not None else None,
            canonical_json(span.output) if span.output is not None else None,
            span.start_time_ns,
            span.end_time_ns,
            span.latency_ms,
            span.token_count,
            span.cost_usd,
            span.status.value,
            span.error_type,
            span.error_message,
            canonical_json(span.attributes),
            r.service_name,
            r.branch,
            r.commit,
            r.environment,
            r.sdk_name,
            r.sdk_version,
            project,
        ]

    # --- reads --------------------------------------------------------------
    def query(
        self,
        *,
        trace_id: str | None = None,
        agent_name: str | None = None,
        status: str | None = None,
        failed_only: bool = False,
        start: datetime | None = None,
        end: datetime | None = None,
        project: str | None = None,
        limit: int = 1000,
    ) -> list[Span]:
        """Filter spans by trace id, agent, failure status, and date range."""
        where: list[str] = []
        params: dict[str, Any] = {}
        if project is not None:
            where.append("project = {project:String}")
            params["project"] = project
        if trace_id is not None:
            where.append("trace_id = {trace_id:String}")
            params["trace_id"] = trace_id
        if agent_name is not None:
            where.append("agent_name = {agent_name:String}")
            params["agent_name"] = agent_name
        if failed_only:
            where.append("status = 'error'")
        elif status is not None:
            where.append("status = {status:String}")
            params["status"] = status
        if start is not None:
            where.append("start_time_ns >= {start_ns:UInt64}")
            params["start_ns"] = int(start.timestamp() * NANOS_PER_SECOND)
        if end is not None:
            where.append("start_time_ns < {end_ns:UInt64}")
            params["end_ns"] = int(end.timestamp() * NANOS_PER_SECOND)

        clause = f" WHERE {' AND '.join(where)}" if where else ""
        sql = (
            f"SELECT {', '.join(_COLUMNS)} FROM {self.qualified_table}{clause} "
            f"ORDER BY start_time_ns, span_id LIMIT {int(limit)}"
        )
        result = self._client.query(sql, parameters=params)
        return [
            self._row_to_span(dict(zip(result.column_names, row, strict=True)))
            for row in result.result_rows
        ]

    def get_trace(self, trace_id: str, *, project: str | None = None) -> list[Span]:
        """All spans of one trace, ordered for replay reconstruction."""
        return self.query(trace_id=trace_id, project=project, limit=100_000)

    @staticmethod
    def _row_to_span(d: dict[str, Any]) -> Span:
        return Span(
            trace_id=d["trace_id"],
            span_id=d["span_id"],
            parent_span_id=d["parent_span_id"] or None,
            name=d["name"],
            kind=SpanKind(d["kind"]),
            agent_name=d["agent_name"],
            tool_name=d["tool_name"],
            input=json.loads(d["input"]) if d["input"] else None,
            output=json.loads(d["output"]) if d["output"] else None,
            start_time_ns=int(d["start_time_ns"]),
            end_time_ns=int(d["end_time_ns"]) if d["end_time_ns"] is not None else None,
            token_count=int(d["token_count"]) if d["token_count"] is not None else None,
            cost_usd=float(d["cost_usd"]) if d["cost_usd"] is not None else None,
            status=SpanStatus(d["status"]),
            error_type=d["error_type"],
            error_message=d["error_message"],
            attributes=json.loads(d["attributes"]) if d["attributes"] else {},
            resource=Resource(
                service_name=d["service_name"],
                branch=d["branch"],
                commit=d["commit"],
                environment=d["environment"],
                sdk_name=d["sdk_name"],
                sdk_version=d["sdk_version"],
            ),
        )

    def count(self, *, project: str | None = None) -> int:
        clause = " WHERE project = {project:String}" if project else ""
        params = {"project": project} if project else {}
        result = self._client.query(
            f"SELECT count() FROM {self.qualified_table}{clause}", parameters=params
        )
        return int(result.result_rows[0][0])

    # --- retention & export -------------------------------------------------
    def purge_older_than(self, days: float, *, project: str | None = None) -> None:
        """Delete spans older than ``days`` (optionally for one project).

        Runs synchronously so callers/tests observe the effect immediately.
        """
        cutoff_ns = int(time.time() * NANOS_PER_SECOND) - int(days * SECONDS_PER_DAY * NANOS_PER_SECOND)
        clause = "start_time_ns < {cutoff:UInt64}"
        params: dict[str, Any] = {"cutoff": cutoff_ns}
        if project is not None:
            clause += " AND project = {project:String}"
            params["project"] = project
        self._client.command(
            f"ALTER TABLE {self.qualified_table} DELETE WHERE {clause}",
            parameters=params,
            settings={"mutations_sync": 1},
        )

    def export_jsonl(self, spans: Sequence[Span] | None = None, **filters: Any) -> str:
        """Export spans as newline-delimited JSON (no lock-in, guide §4).

        Pass an explicit ``spans`` sequence, or filter keywords forwarded to
        :meth:`query`. Returns the JSONL text.
        """
        rows = list(spans) if spans is not None else self.query(**filters)
        return "\n".join(s.to_json() for s in rows)

    # --- lifecycle ----------------------------------------------------------
    def truncate(self) -> None:
        """Remove all rows (used by tests)."""
        self._client.command(f"TRUNCATE TABLE IF EXISTS {self.qualified_table}")

    def drop(self) -> None:
        self._client.command(f"DROP TABLE IF EXISTS {self.qualified_table}")

    def close(self) -> None:
        self._client.close()

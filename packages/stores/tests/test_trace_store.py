"""Trace store integration tests (guide section 4): write throughput, query
correctness, concurrency integrity, retention purge, and required git context.
"""

from __future__ import annotations

import threading
import time

from pytracer_sdk import new_span_id, new_trace_id
from pytracer_sdk.schema import Resource, Span, SpanKind, SpanStatus

from pytracer_stores import TraceStore
from pytracer_stores.config import ClickHouseConfig

NOW_NS = time.time_ns()
DAY_NS = 86_400 * 1_000_000_000
RES = Resource(service_name="test", branch="main", commit="abc123")


def _span(
    *,
    trace_id: str | None = None,
    agent_name: str | None = None,
    status: SpanStatus = SpanStatus.OK,
    start_time_ns: int = NOW_NS,
    resource: Resource = RES,
    name: str = "op",
    kind: SpanKind = SpanKind.TOOL,
) -> Span:
    return Span(
        trace_id=trace_id or new_trace_id(),
        span_id=new_span_id(),
        name=name,
        kind=kind,
        agent_name=agent_name,
        status=status,
        start_time_ns=start_time_ns,
        end_time_ns=start_time_ns + 1_000_000,
        resource=resource,
    )


def test_insert_and_get_trace_roundtrip(trace_store: TraceStore) -> None:
    tid = new_trace_id()
    spans = [_span(trace_id=tid, name=f"s{i}", start_time_ns=NOW_NS + i) for i in range(5)]
    assert trace_store.insert(spans, project="p1") == 5
    got = trace_store.get_trace(tid, project="p1")
    assert [s.name for s in got] == ["s0", "s1", "s2", "s3", "s4"]  # ordered by time
    assert all(s.resource.branch == "main" and s.resource.commit == "abc123" for s in got)


def test_branch_and_commit_are_required(trace_store: TraceStore) -> None:
    bad = _span(resource=Resource(service_name="x", branch=None, commit=None))
    try:
        trace_store.insert(bad)
        assert False, "expected ValueError for missing branch/commit"
    except ValueError as exc:
        assert "branch and commit" in str(exc)


def test_query_filters_return_exactly_expected(trace_store: TraceStore) -> None:
    trace_store.insert(
        [
            _span(agent_name="planner", status=SpanStatus.OK),
            _span(agent_name="planner", status=SpanStatus.ERROR),
            _span(agent_name="researcher", status=SpanStatus.OK),
        ],
        project="p2",
    )
    assert len(trace_store.query(project="p2", agent_name="planner")) == 2
    assert len(trace_store.query(project="p2", failed_only=True)) == 1
    assert len(trace_store.query(project="p2", agent_name="researcher", failed_only=True)) == 0
    assert len(trace_store.query(project="p2")) == 3


def test_date_range_query(trace_store: TraceStore) -> None:
    from datetime import datetime, timezone

    old = _span(start_time_ns=NOW_NS - 10 * DAY_NS)
    recent = _span(start_time_ns=NOW_NS)
    trace_store.insert([old, recent], project="p3")
    cutoff = datetime.fromtimestamp((NOW_NS - DAY_NS) / 1e9, tz=timezone.utc)
    within = trace_store.query(project="p3", start=cutoff)
    assert len(within) == 1
    assert within[0].span_id == recent.span_id


def test_write_throughput_batch(trace_store: TraceStore) -> None:
    n = 5_000
    spans = [_span(name="bulk") for _ in range(n)]
    start = time.perf_counter()
    trace_store.insert(spans, project="bulk")
    elapsed = time.perf_counter() - start
    assert trace_store.count(project="bulk") == n
    # Loose bound: a 5k batch should insert well under 5s locally.
    assert elapsed < 5.0, f"bulk insert too slow: {elapsed:.2f}s"


def test_concurrent_writes_no_loss_or_duplication(trace_store: TraceStore) -> None:
    threads_n, per_thread = 8, 500
    cfg = ClickHouseConfig()
    errors: list[Exception] = []

    def worker() -> None:
        # Each thread uses its own client/store instance.
        store = TraceStore(cfg, table=trace_store._table)  # type: ignore[attr-defined]
        try:
            store.insert([_span(name="c") for _ in range(per_thread)], project="conc")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            store.close()

    threads = [threading.Thread(target=worker) for _ in range(threads_n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    rows = trace_store.query(project="conc", limit=1_000_000)
    assert len(rows) == threads_n * per_thread
    assert len({s.span_id for s in rows}) == threads_n * per_thread  # no duplicates


def test_retention_purge(trace_store: TraceStore) -> None:
    old = _span(start_time_ns=NOW_NS - 40 * DAY_NS)
    recent = _span(start_time_ns=NOW_NS)
    trace_store.insert([old, recent], project="ret")
    assert trace_store.count(project="ret") == 2
    trace_store.purge_older_than(30, project="ret")
    remaining = trace_store.query(project="ret", limit=10)
    assert len(remaining) == 1
    assert remaining[0].span_id == recent.span_id


def test_list_traces_summarizes_and_filters(trace_store: TraceStore) -> None:
    ok_trace, bad_trace = new_trace_id(), new_trace_id()
    trace_store.insert(
        [
            _span(trace_id=ok_trace, agent_name="writer", name="run", kind=SpanKind.AGENT),
            _span(trace_id=ok_trace, name="chat"),
            _span(trace_id=bad_trace, agent_name="planner", name="run", kind=SpanKind.AGENT),
            _span(trace_id=bad_trace, name="search", status=SpanStatus.ERROR),
        ],
        project="ls",
    )
    all_rows = {r.trace_id: r for r in trace_store.list_traces(project="ls")}
    assert set(all_rows) == {ok_trace, bad_trace}
    assert all_rows[ok_trace].span_count == 2
    assert all_rows[ok_trace].failed is False
    assert all_rows[bad_trace].failed is True
    assert all_rows[bad_trace].root_agent == "planner"

    failed = trace_store.list_traces(project="ls", failed_only=True)
    assert [r.trace_id for r in failed] == [bad_trace]


def test_export_jsonl_roundtrips(trace_store: TraceStore) -> None:
    tid = new_trace_id()
    trace_store.insert([_span(trace_id=tid, name="e")], project="exp")
    jsonl = trace_store.export_jsonl(project="exp")
    assert jsonl
    restored = Span.from_json(jsonl.splitlines()[0])
    assert restored.trace_id == tid

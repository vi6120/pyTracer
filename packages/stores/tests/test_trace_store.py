"""Trace store integration tests (guide section 4): write throughput, query
correctness, concurrency integrity, retention purge, and required git context.
"""

from __future__ import annotations

import datetime as dt
import threading
import time
from decimal import Decimal
from pathlib import Path

from pytracer_sdk import new_span_id, new_trace_id
from pytracer_sdk.schema import Resource, Span, SpanKind, SpanStatus

from pytracer_stores import TraceStore
from pytracer_stores.config import ClickHouseConfig

NOW_NS = time.time_ns()
DAY_NS = 86_400 * 1_000_000_000
RES = Resource(service_name="test", branch="main", commit="abc123")


class _PilotDbError(Exception):
    """A non-builtin exception, module-level so replay can import it back."""


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


def test_error_module_round_trips_for_replay(trace_store: TraceStore) -> None:
    # A gateway-captured failure must keep the exception's defining module, so
    # replay rebuilds the real type (e.g. neo4j.exceptions.ServiceUnavailable)
    # instead of a generic ReplayedError (V2/R4).
    tid = new_trace_id()
    span = Span(
        trace_id=tid,
        span_id=new_span_id(),
        name="db.query",
        kind=SpanKind.TOOL,
        status=SpanStatus.ERROR,
        start_time_ns=NOW_NS,
        end_time_ns=NOW_NS + 1_000_000,
        error_type="ServiceUnavailable",
        error_module="neo4j.exceptions",
        error_message="unable to retrieve routing information",
        resource=RES,
    )
    assert trace_store.insert(span, project="p1") == 1
    (got,) = trace_store.get_trace(tid, project="p1")
    assert got.error_module == "neo4j.exceptions"
    assert got.error_type == "ServiceUnavailable"
    assert got.error_message == "unable to retrieve routing information"


def test_captured_failure_survives_gateway_store_record_and_replays_as_real_type(
    trace_store: TraceStore, tmp_path: Path
) -> None:
    # The whole path V2/R4 asks for: capture -> gateway -> store -> record -> replay.
    # A tool fails with a non-builtin exception; after a round trip through the
    # gateway and ClickHouse and a record to JSONL, replay must reconstruct the
    # real type so the agent's own `except` fires, not a generic ReplayedError.
    import pytracer_sdk as pytracer
    from pytracer_cli.traces import record_trace
    from pytracer_gateway.ingest import validate_spans
    from pytracer_replay import MockSet, ReplayEngine, ReplayMode

    @pytracer.op(kind=SpanKind.TOOL, tool_name="query")
    def query(x: int) -> str:
        raise _PilotDbError("db down")

    @pytracer.op(kind=SpanKind.AGENT, agent_name="a")
    def agent() -> str:
        try:
            return str(query(1))
        except _PilotDbError:
            return "handled"

    rec = ReplayEngine(ReplayMode.LIVE).replay(agent, MockSet())
    assert rec.return_value == "handled"

    # Capture -> gateway: stamp branch/commit (the store requires them) and ingest.
    stamped = [s.model_copy(update={"resource": RES}) for s in rec.spans]
    valid, errors = validate_spans([s.model_dump(mode="json", exclude_none=True) for s in stamped])
    assert errors == []
    assert len(valid) == len(stamped)

    # Store -> read back from ClickHouse.
    tid = valid[0].trace_id
    trace_store.insert(valid, project="r4")
    read_back = trace_store.get_trace(tid, project="r4")
    assert any(s.error_module == f"{__name__}" and s.error_type == "_PilotDbError" for s in read_back)

    # Record -> JSONL, then reload exactly as replay reads a recording.
    record_trace(read_back, agent="m:agent", entry={}, out_dir=tmp_path)
    jsonl = next((tmp_path / "traces").glob("*.jsonl"))
    reloaded = [Span.from_json(line) for line in jsonl.read_text().splitlines() if line.strip()]

    # Replay: the real exception is re-raised, so the agent's except path fires.
    replayed = ReplayEngine(ReplayMode.FULL).replay(agent, MockSet.from_trace(reloaded))
    assert replayed.return_value == "handled"


def test_typed_inputs_and_outputs_round_trip_through_the_store(trace_store: TraceStore) -> None:
    # A span captured through the gateway and read back for `record` keeps its
    # Decimal / date / tuple values, so the replay gets the types it had live.
    tid = new_trace_id()
    span = Span(
        trace_id=tid,
        span_id=new_span_id(),
        name="get_item",
        kind=SpanKind.TOOL,
        status=SpanStatus.OK,
        start_time_ns=NOW_NS,
        end_time_ns=NOW_NS + 1_000_000,
        input={"since": dt.date(2026, 1, 1)},
        output={"price": Decimal("19.99"), "ids": (1, 2)},
        resource=RES,
    )
    trace_store.insert(span, project="types")
    (got,) = trace_store.get_trace(tid, project="types")
    assert got.input == {"since": dt.date(2026, 1, 1)}
    assert got.output == {"price": Decimal("19.99"), "ids": (1, 2)}
    assert type(got.output["price"]) is Decimal
    assert type(got.output["ids"]) is tuple


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
        store = TraceStore(cfg, table=trace_store.table)
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


def test_a_tag_shaped_value_round_trips_through_the_store_as_the_same_dict(
    trace_store: TraceStore,
) -> None:
    # An end user's data shaped like an encoded enum is data, not a tag: the store
    # returns the same dict, not the enum it names.
    tag_shaped = {"__pytracer__": "enum", "v": 2, "cls": "re:RegexFlag"}
    tid = new_trace_id()
    span = _span(trace_id=tid)
    span.input = {"payload": dict(tag_shaped)}
    span.output = dict(tag_shaped)
    trace_store.insert(span, project="tag-shaped")
    (got,) = trace_store.get_trace(tid, project="tag-shaped")
    assert got.input == {"payload": tag_shaped}
    assert type(got.output) is dict and got.output == tag_shaped


def test_a_span_stored_twice_is_read_once(trace_store: TraceStore) -> None:
    # Ingestion is at least once: after a gateway crash a peer can write a batch
    # the crashed gateway had already written. Reads must still show each span once.
    tid = new_trace_id()
    spans = [_span(trace_id=tid, name=f"s{i}", start_time_ns=NOW_NS + i) for i in range(3)]
    trace_store.insert(spans, project="dup")
    trace_store.insert(spans[:2], project="dup")  # the retried part of the batch
    assert trace_store.count(project="dup") == 5  # physically stored twice
    assert [s.name for s in trace_store.get_trace(tid, project="dup")] == ["s0", "s1", "s2"]
    (summary,) = trace_store.list_traces(project="dup")
    assert summary.span_count == 3

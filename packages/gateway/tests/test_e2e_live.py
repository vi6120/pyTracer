"""Live end-to-end tests over a real HTTP server.

Proves the full path the SDK actually uses: SDK buffer -> HTTPTransport -> real
uvicorn server -> gateway -> writer. The last test carries it all the way into
ClickHouse and is skipped when the trace store is not reachable.
"""

from __future__ import annotations

import socket
import threading
import time
import uuid
from collections.abc import Callable

import navik_sdk as navik
import pytest
import uvicorn
from fastapi import FastAPI
from helpers import FakeWriter

from navik_gateway import TraceStoreWriter, create_app

API_KEYS = {"livekey": "live-proj"}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class _LiveServer:
    def __init__(self, app: FastAPI) -> None:
        self.port = _free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        self._server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="warning")
        )
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    def __enter__(self) -> _LiveServer:  # noqa: PYI034
        self._thread.start()
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if self._server.started:
                return self
            time.sleep(0.02)
        raise RuntimeError("live server failed to start")

    def __exit__(self, *_: object) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=5.0)


def _clickhouse_up() -> bool:
    try:
        import clickhouse_connect
        from navik_stores.config import ClickHouseConfig

        cfg = ClickHouseConfig()
        client = clickhouse_connect.get_client(
            host=cfg.host, port=cfg.port, username=cfg.username,
            password=cfg.password, database=cfg.database,
        )
        client.command("SELECT 1")
        client.close()
        return True
    except Exception:  # noqa: BLE001
        return False


def test_sdk_http_transport_to_gateway(wait_until: Callable[..., bool]) -> None:
    writer = FakeWriter()
    with _LiveServer(create_app(writer=writer, api_keys=API_KEYS)) as server:
        transport = navik.HTTPTransport(server.base_url, api_key="livekey")
        tracer = navik.Tracer(
            transport,
            resource=navik.Resource(service_name="live", branch="main", commit="c0ffee"),
        )

        @tracer.op(kind=navik.SpanKind.TOOL)
        def work(x: int) -> int:
            return x + 1

        for i in range(5):
            work(i)
        tracer.flush()
        tracer.shutdown()

        assert wait_until(lambda: writer.count == 5, timeout=10.0)
    assert all(project == "live-proj" for project, _ in writer.records)


def test_sdk_http_transport_survives_unreachable_gateway() -> None:
    # Point at a dead port: the SDK must not crash; spans count as send errors.
    transport = navik.HTTPTransport(f"http://127.0.0.1:{_free_port()}", api_key="x")
    tracer = navik.Tracer(
        transport,
        resource=navik.Resource(service_name="live", branch="main", commit="c0ffee"),
    )

    @tracer.op()
    def work() -> str:
        return "ok"

    assert work() == "ok"  # agent keeps working despite the gateway being down
    tracer.flush()
    tracer.shutdown()
    assert tracer.buffer.send_errors >= 1


@pytest.mark.skipif(not _clickhouse_up(), reason="ClickHouse not reachable")
def test_end_to_end_into_clickhouse(wait_until: Callable[..., bool]) -> None:
    from navik_stores import TraceStore

    table = f"spans_e2e_{uuid.uuid4().hex[:8]}"
    # The ClickHouse client is not safe for concurrent queries on one session, so
    # the writer (used by the gateway's worker thread) and the test's own reads
    # use separate TraceStore instances against the same table.
    admin = TraceStore(table=table)
    admin.migrate()
    writer_store = TraceStore(table=table)
    try:
        app = create_app(writer=TraceStoreWriter(writer_store), api_keys=API_KEYS)
        with _LiveServer(app) as server:
            transport = navik.HTTPTransport(server.base_url, api_key="livekey")
            tracer = navik.Tracer(
                transport,
                resource=navik.Resource(service_name="e2e", branch="feature/x", commit="deadbeef"),
            )

            @tracer.op(kind=navik.SpanKind.AGENT, agent_name="planner")
            def run() -> str:
                return "done"

            run()
            tracer.flush()
            tracer.shutdown()

            assert wait_until(lambda: admin.count(project="live-proj") == 1, timeout=15.0)
        persisted = admin.query(project="live-proj")
        assert persisted[0].resource.commit == "deadbeef"
        assert persisted[0].agent_name == "planner"
    finally:
        admin.drop()
        admin.close()
        writer_store.close()

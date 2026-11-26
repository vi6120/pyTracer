"""HTTP requests inside a tool or model call are recorded on its span, retries marked."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from pytracer_sdk import Resource, SpanKind, Tracer, http
from pytracer_sdk.transport import InMemoryTransport

httpx = pytest.importorskip("httpx")


@pytest.fixture
def tracer() -> Iterator[tuple[Tracer, InMemoryTransport]]:
    sink = InMemoryTransport()
    yield Tracer(sink, resource=Resource(service_name="http", branch="main")), sink


def _spans(tracer: Tracer, sink: InMemoryTransport, kind: SpanKind = SpanKind.TOOL) -> list[Any]:
    tracer.flush()
    return [s for s in sink.spans if s.kind is kind]


def _flaky_transport(first: Exception | int) -> Any:
    """Fails the first request (an error, or that status), then answers 200."""
    calls = {"n": 0}

    def handle(request: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            if isinstance(first, Exception):
                raise first
            return httpx.Response(first)
        return httpx.Response(200, json={"id": "re_1"})

    return httpx.MockTransport(handle)


def test_a_retried_post_inside_a_tool_is_recorded_as_a_retry(tracer: Any) -> None:
    t, sink = tracer
    client = httpx.Client(transport=_flaky_transport(httpx.ReadTimeout("timed out")))

    @t.op(kind=SpanKind.TOOL, tool_name="refund_payment")
    def refund_payment(order_id: str) -> str:
        for _ in range(3):  # the tool's own retry, invisible from outside the call
            try:
                client.post("https://pay.example/v1/refunds?api_key=secret", json={"order": order_id})
                return "refunded"
            except httpx.ReadTimeout:
                continue
        return "failed"

    assert refund_payment("A1") == "refunded"
    [span] = _spans(t, sink)
    first, second = span.attributes["pytracer.http"]
    assert first["method"] == "POST" and first["url"] == "https://pay.example/v1/refunds"
    assert first["error"] == "ReadTimeout" and "attempt" not in first
    assert second["status"] == 200 and second["attempt"] == 2 and second["retry"] is True
    assert "secret" not in str(span.attributes)  # no query string, headers or body


def test_a_request_from_a_helper_op_belongs_to_the_enclosing_tool(tracer: Any) -> None:
    t, sink = tracer
    client = httpx.Client(transport=_flaky_transport(503))

    @t.op  # a plain helper, not a tool
    def post(path: str) -> int:
        return int(client.post(f"https://pay.example{path}", content=b"x").status_code)

    @t.op(kind=SpanKind.TOOL, tool_name="charge")
    def charge() -> int:
        post("/v1/charges")
        return post("/v1/charges")

    assert charge() == 200
    [span] = _spans(t, sink)
    statuses = [(r["status"], r.get("retry")) for r in span.attributes["pytracer.http"]]
    assert statuses == [(503, None), (200, True)]


def test_different_bodies_are_different_requests(tracer: Any) -> None:
    t, sink = tracer
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200)))

    @t.op(kind=SpanKind.TOOL, tool_name="notify")
    def notify() -> None:
        client.post("https://hooks.example/send", json={"to": "a"})
        client.post("https://hooks.example/send", json={"to": "b"})

    notify()
    [span] = _spans(t, sink)
    assert all("attempt" not in r for r in span.attributes["pytracer.http"])


def test_requests_outside_a_tool_or_model_call_are_not_recorded(tracer: Any) -> None:
    t, sink = tracer
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200)))

    @t.op(kind=SpanKind.AGENT)
    def agent() -> None:
        client.get("https://example.test/")

    agent()
    t.flush()
    assert all("pytracer.http" not in s.attributes for s in sink.spans)


def test_async_requests_inside_a_model_call_are_recorded(tracer: Any) -> None:
    t, sink = tracer

    @t.op(kind=SpanKind.LLM)
    async def complete() -> int:
        client = httpx.AsyncClient(transport=_flaky_transport(429))
        first = await client.post("https://llm.example/v1/chat", json={"q": 1})
        second = await client.post("https://llm.example/v1/chat", json={"q": 1})
        return int(first.status_code + second.status_code)

    assert asyncio.run(complete()) == 629
    [span] = _spans(t, sink, SpanKind.LLM)
    assert [r["status"] for r in span.attributes["pytracer.http"]] == [429, 200]
    assert span.attributes["pytracer.http"][1]["retry"] is True


class _Flaky(BaseHTTPRequestHandler):
    hits = 0

    def do_POST(self) -> None:  # noqa: N802 - the http.server hook name
        type(self).hits += 1
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.send_response(503 if type(self).hits == 1 else 200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *args: Any) -> None:
        pass


def test_requests_retry_adapter_attempts_are_each_recorded(tracer: Any) -> None:
    requests = pytest.importorskip("requests")
    from urllib3.util.retry import Retry

    server = HTTPServer(("127.0.0.1", 0), _Flaky)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    session = requests.Session()
    retry = Retry(total=3, status_forcelist=[503], allowed_methods=None, backoff_factor=0)
    session.mount("http://", requests.adapters.HTTPAdapter(max_retries=retry))
    t, sink = tracer

    @t.op(kind=SpanKind.TOOL, tool_name="refund_payment")
    def refund_payment() -> int:
        return int(session.post(f"http://127.0.0.1:{server.server_port}/refunds", data=b"A1").status_code)

    try:
        assert refund_payment() == 200
    finally:
        server.shutdown()
    [span] = _spans(t, sink)
    recorded = [(r["status"], r.get("retry")) for r in span.attributes["pytracer.http"]]
    assert recorded == [(503, None), (200, True)]  # the adapter's retry, one entry per attempt


def test_capture_can_be_turned_off(monkeypatch: pytest.MonkeyPatch) -> None:
    http.uninstall()
    monkeypatch.setenv("PYTRACER_HTTP_CAPTURE", "0")
    try:
        http.install()
        assert not getattr(httpx.Client.send, "__pytracer_http__", False)
    finally:
        monkeypatch.delenv("PYTRACER_HTTP_CAPTURE")
        http.install()
    assert getattr(httpx.Client.send, "__pytracer_http__", False)


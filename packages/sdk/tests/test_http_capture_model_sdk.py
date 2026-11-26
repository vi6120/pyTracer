"""A model SDK's own retry loop is recorded on the model call that hides it.

Kept apart from test_http_capture.py, which needs httpx: OpenAI's 3.x SDK sends
through httpx2, so this runs with whichever HTTP package the installed SDK uses.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from pytracer_sdk import Resource, SpanKind, Tracer
from pytracer_sdk.transport import InMemoryTransport


@pytest.fixture
def tracer() -> Iterator[tuple[Tracer, InMemoryTransport]]:
    sink = InMemoryTransport()
    yield Tracer(sink, resource=Resource(service_name="http", branch="main")), sink


def _spans(tracer: Tracer, sink: InMemoryTransport, kind: SpanKind) -> list[Any]:
    tracer.flush()
    return [s for s in sink.spans if s.kind is kind]


def test_a_model_sdks_own_retry_is_recorded_on_the_model_call(tracer: Any) -> None:
    openai = pytest.importorskip("openai")
    import inspect

    # The HTTP package the installed OpenAI SDK is built on (httpx, or httpx2 in 3.x).
    annotation = str(inspect.signature(openai.OpenAI.__init__).parameters["http_client"].annotation)
    client_http = pytest.importorskip("httpx2" if "httpx2" in annotation else "httpx")
    calls = {"n": 0}

    def handle(request: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            return client_http.Response(500, json={"error": {"message": "overloaded"}})
        return client_http.Response(200, json={
            "id": "c", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "hi"}}],
        })

    t, sink = tracer
    client = openai.OpenAI(
        api_key="test", max_retries=2,
        http_client=client_http.Client(transport=client_http.MockTransport(handle)),
    )

    @t.op(kind=SpanKind.LLM)
    def complete() -> str:
        reply = client.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])
        return str(reply.choices[0].message.content)

    assert complete() == "hi"  # one clean call, from the outside
    [span] = _spans(t, sink, SpanKind.LLM)
    recorded = [(r["status"], r.get("retry")) for r in span.attributes["pytracer.http"]]
    assert recorded == [(500, None), (200, True)]  # the SDK's hidden retry

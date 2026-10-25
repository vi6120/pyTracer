"""A GenAI-instrumented model client keeps its prompt and reply (P7), for real.

``opentelemetry-instrumentation-openai-v2`` leaves the model span empty and sends
the prompt and reply as OpenTelemetry log records. This drives the real library
against a local fake OpenAI server (no network) and checks the recorded model
span has both. Skipped when the instrumentation or ``openai`` is not installed.
"""

from __future__ import annotations

import importlib.util
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest
import pytracer_sdk as pytracer
from pytracer_sdk import SpanKind
from pytracer_sdk.transport import InMemoryTransport

from pytracer_autogen import pytracer_tracer_provider


def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:  # a parent package is missing
        return False


pytestmark = pytest.mark.skipif(
    not (_installed("opentelemetry.instrumentation.openai_v2") and _installed("openai")),
    reason="opentelemetry-instrumentation-openai-v2 or openai not installed",
)

_REPLY = {
    "id": "c1", "object": "chat.completion", "created": 1, "model": "gpt-4o-mini",
    "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "sunny"}}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
}


class _FakeOpenAI(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 - the http.server hook name
        self.rfile.read(int(self.headers["Content-Length"]))
        body = json.dumps(_REPLY).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: Any) -> None:
        pass


@pytest.fixture
def server() -> Iterator[str]:
    httpd = HTTPServer(("127.0.0.1", 0), _FakeOpenAI)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}/v1"
    finally:
        httpd.shutdown()


def test_instrumented_model_call_keeps_its_prompt_and_reply(
    server: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import openai
    from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor
    from opentelemetry.sdk._logs import LoggerProvider

    from pytracer_autogen import genai_log_processor

    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true")
    sink = InMemoryTransport()
    tracer = pytracer.Tracer(sink, resource=pytracer.Resource(service_name="p7", branch="main", commit="c0"))
    logs = LoggerProvider()
    logs.add_log_record_processor(genai_log_processor())
    instrumentor = OpenAIInstrumentor()
    instrumentor.instrument(tracer_provider=pytracer_tracer_provider(tracer), logger_provider=logs)
    try:
        openai.OpenAI(base_url=server, api_key="test").chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "be brief"}, {"role": "user", "content": "weather?"}],
        )
    finally:
        instrumentor.uninstrument()
    tracer.flush()
    (llm,) = [s for s in sink.spans if s.kind is SpanKind.LLM]
    assert llm.input == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "weather?"},
    ]
    assert llm.output == {"role": "assistant", "content": "sunny"}
    assert llm.token_count == 6

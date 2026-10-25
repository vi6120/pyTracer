"""Model content that GenAI instrumentation sends as OpenTelemetry log records."""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from typing import Any

from .genai import _MAX_TRACKED

_log = logging.getLogger("pytracer_autogen")

class _GenAIContentStore:
    """Model input and output that GenAI instrumentation sends as log records.

    Libraries such as ``opentelemetry-instrumentation-openai-v2`` leave the model
    span empty and emit the prompt (``gen_ai.<role>.message``) and the reply
    (``gen_ai.choice``) as OpenTelemetry log records carrying the span's id. They
    are kept here by span id until that span is exported.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._content: OrderedDict[int, tuple[list[Any], list[Any]]] = OrderedDict()

    def add(self, span_id: int, event_name: str, body: Any) -> None:
        if event_name == "gen_ai.choice":
            item = body.get("message", body) if isinstance(body, dict) else body
            slot = 1
        elif event_name.startswith("gen_ai.") and event_name.endswith(".message"):
            role = event_name[len("gen_ai.") : -len(".message")]
            item = {"role": role, **body} if isinstance(body, dict) else {"role": role, "content": body}
            slot = 0
        else:
            return
        with self._lock:
            entry = self._content.setdefault(span_id, ([], []))
            entry[slot].append(item)
            self._content.move_to_end(span_id)
            while len(self._content) > _MAX_TRACKED:
                self._content.popitem(last=False)

    def take(self, span_id: int) -> tuple[list[Any], list[Any]] | None:
        with self._lock:
            return self._content.pop(span_id, None)


_GENAI_CONTENT = _GenAIContentStore()


def _genai_log_processor() -> Any:
    """A LogRecordProcessor feeding ``_GENAI_CONTENT`` (built lazily: the logs SDK
    is optional and its module is still underscored in OpenTelemetry)."""
    from opentelemetry.sdk._logs import LogRecordProcessor

    class _GenAIContentProcessor(LogRecordProcessor):
        def on_emit(self, log_data: Any) -> None:
            record = getattr(log_data, "log_record", log_data)
            span_id = getattr(record, "span_id", None)
            attributes = dict(getattr(record, "attributes", None) or {})
            name = getattr(record, "event_name", None) or attributes.get("event.name")
            if span_id and isinstance(name, str):
                _GENAI_CONTENT.add(span_id, name, getattr(record, "body", None))

        emit = on_emit  # the SDK's method name before on_emit

        def shutdown(self) -> None:
            pass

        def force_flush(self, timeout_millis: int = 30_000) -> bool:
            return True

    return _GenAIContentProcessor()


_genai_logs_attached = False


def _attach_genai_log_records() -> None:
    """Receive GenAI content log records: add a processor to the global
    LoggerProvider, or register one if none is set. Done once per process."""
    global _genai_logs_attached
    if _genai_logs_attached:
        return
    try:
        from opentelemetry import _logs
        from opentelemetry.sdk._logs import LoggerProvider
    except ImportError:  # the logs SDK is not available
        return
    provider = _logs.get_logger_provider()
    if not isinstance(provider, LoggerProvider):
        provider = LoggerProvider()
        _logs.set_logger_provider(provider)
        _log.info(
            "pytracer: registered an OpenTelemetry LoggerProvider to read the model "
            "content GenAI instrumentation sends as log records; to use your own, "
            "register it before install() and pyTracer adds its processor to it"
        )
    provider.add_log_record_processor(_genai_log_processor())
    _genai_logs_attached = True

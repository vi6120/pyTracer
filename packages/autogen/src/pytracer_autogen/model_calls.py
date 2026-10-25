"""Model calls from the events AutoGen's model clients log, or derived from replies."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .exporter import PyTracerSpanExporter

# AutoGen's model clients (autogen_ext: OpenAI, Azure, Anthropic, Ollama, ...) log
# each call as an event on this logger; AutoGen itself emits no model-call span.
_EVENT_LOGGER = "autogen_core.events"


def _model_output(response: Any) -> Any:
    """The reply in a logged model response: the first choice's message for the
    OpenAI shape, else the response's content, else the whole response."""
    if isinstance(response, dict):
        choices = response.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            return choices[0].get("message", choices[0])
        if "content" in response:
            return response["content"]
    return response


def _usage_messages(message: Any) -> list[dict[str, Any]]:
    """Messages in a runtime message that carry model usage (a model reply)."""
    if not isinstance(message, str) or '"models_usage"' not in message:
        return []
    try:
        data = json.loads(message)
    except ValueError:
        return []
    found: list[dict[str, Any]] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if isinstance(node.get("models_usage"), dict) and node.get("id") is not None:
                found.append(node)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(data)
    return found


class _ModelCallLogHandler(logging.Handler):
    """Turns the model-call events AutoGen's clients log into pyTracer LLM spans.

    Events go to the exporter of the global tracer provider when that is a
    pyTracer one (the provider AutoGen traces through), else to the exporter the
    handler was attached with.
    """

    def __init__(self, exporter: PyTracerSpanExporter) -> None:
        super().__init__(logging.INFO)
        self._exporter = exporter

    def emit(self, record: logging.LogRecord) -> None:
        from opentelemetry import trace as otel_trace

        from .exporter import exporter_for

        try:
            target = exporter_for(otel_trace.get_tracer_provider()) or self._exporter
            target.record_model_event(record.msg)
        except Exception:  # noqa: BLE001 - capture must never break the agent
            _log.debug("pytracer could not record an AutoGen model event", exc_info=True)


_log = logging.getLogger("pytracer_autogen")


def _attach_model_call_handler(exporter: PyTracerSpanExporter) -> None:
    """Route AutoGen's model-call events to ``exporter`` (the latest provider wins).

    The events are logged at INFO. When the event logger would drop them (its
    effective level is above INFO, the default), its level is lowered and its
    propagation turned off, so other handlers see exactly what they saw before.
    """
    logger = logging.getLogger(_EVENT_LOGGER)
    for handler in list(logger.handlers):
        if isinstance(handler, _ModelCallLogHandler):
            logger.removeHandler(handler)
    logger.addHandler(_ModelCallLogHandler(exporter))
    if logger.getEffectiveLevel() > logging.INFO:
        logger.setLevel(logging.INFO)
        logger.propagate = False
        _log.info(
            "pytracer: set the %r logger to INFO so pyTracer receives AutoGen's model-call "
            "events, and turned off its propagation, since those events were being dropped "
            "before; set that logger's level yourself to manage it",
            _EVENT_LOGGER,
        )

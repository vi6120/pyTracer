"""pyTracer adapter for AutoGen (via OpenTelemetry)."""

from __future__ import annotations

from .exporter import (
    PyTracerSpanExporter,
    exporter_for,
    install,
    install_seams,
    pytracer_tracer_provider,
    uninstall_seams,
)
from .genai_logs import genai_log_processor

__all__ = [
    "PyTracerSpanExporter",
    "exporter_for",
    "genai_log_processor",
    "install",
    "install_seams",
    "pytracer_tracer_provider",
    "uninstall_seams",
]

__version__ = "0.1.3"

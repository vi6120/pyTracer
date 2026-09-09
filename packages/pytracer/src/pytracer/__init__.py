"""pyTracer - meta-package for the full agent test and replay platform.

Installing ``pytracer`` pulls in every component (SDK, gateway, stores, replay
engine, cross-branch runner, registry, and CLI). The SDK's core entrypoints are
re-exported here so ``import pytracer`` is enough to instrument an agent; the other
components are imported from their own packages (``pytracer_gateway``,
``pytracer_stores``, ``pytracer_replay``, ``pytracer_runner``, ``pytracer_registry``,
``pytracer_cli``).
"""

from __future__ import annotations

from pytracer_sdk import (
    HTTPTransport,
    Resource,
    Span,
    SpanKind,
    SpanStatus,
    configure,
    flush,
    op,
    shutdown,
)

__all__ = [
    "HTTPTransport",
    "Resource",
    "Span",
    "SpanKind",
    "SpanStatus",
    "configure",
    "flush",
    "op",
    "shutdown",
]

__version__ = "0.0.1"

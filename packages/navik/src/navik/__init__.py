"""Navik - meta-package for the full agent test and replay platform.

Installing ``navik`` pulls in every component (SDK, gateway, stores, replay
engine, cross-branch runner, registry, and CLI). The SDK's core entrypoints are
re-exported here so ``import navik`` is enough to instrument an agent; the other
components are imported from their own packages (``navik_gateway``,
``navik_stores``, ``navik_replay``, ``navik_runner``, ``navik_registry``,
``navik_cli``).
"""

from __future__ import annotations

from navik_sdk import (
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

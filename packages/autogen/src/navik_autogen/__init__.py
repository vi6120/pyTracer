"""Navik adapter for AutoGen (via OpenTelemetry)."""

from __future__ import annotations

from .exporter import NavikSpanExporter, navik_tracer_provider

__all__ = ["NavikSpanExporter", "navik_tracer_provider"]

__version__ = "0.0.1"

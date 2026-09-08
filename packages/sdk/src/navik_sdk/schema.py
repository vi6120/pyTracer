"""Span schema — the core data contract for the whole platform.

Every layer (ingestion, trace store, replay engine, CLI) speaks this schema,
so it is defined once here and reused everywhere. Fields follow the guide's
span definition and map cleanly onto the OpenTelemetry / OTLP model.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .ids import is_valid_span_id, is_valid_trace_id

NANOS_PER_MILLI = 1_000_000


class SpanKind(str, Enum):
    """What an agent did during this span."""

    AGENT = "agent"  # a top-level agent invocation
    LLM = "llm"  # a model call
    TOOL = "tool"  # a tool / function call
    REASONING = "reasoning"  # an internal reasoning / planning step
    HANDOFF = "handoff"  # control passed between agents
    UNKNOWN = "unknown"


class SpanStatus(str, Enum):
    """Outcome of a span, aligned with OTel status codes."""

    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


class Resource(BaseModel):
    """Origin metadata shared by all spans in a process.

    ``branch`` and ``commit`` are captured here so a trace can later be tied to
    the exact code state that produced it (required by the trace store for
    cross-branch replay).
    """

    model_config = ConfigDict(extra="allow")

    service_name: str = "unknown-agent"
    sdk_name: str = "navik-sdk"
    sdk_version: str = "0.0.1"
    branch: str | None = None
    commit: str | None = None
    environment: str | None = None


class Span(BaseModel):
    """A single captured agent action.

    Timestamps are Unix-epoch nanoseconds (OTel convention). ``latency_ms`` is
    derived from the start/end timestamps and should not be set by hand.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # --- identity -----------------------------------------------------------
    trace_id: str
    span_id: str
    parent_span_id: str | None = None

    # --- semantics ----------------------------------------------------------
    name: str
    kind: SpanKind = SpanKind.UNKNOWN
    agent_name: str | None = None
    tool_name: str | None = None

    # --- payload ------------------------------------------------------------
    input: Any | None = None
    output: Any | None = None

    # --- timing & cost ------------------------------------------------------
    start_time_ns: int
    end_time_ns: int | None = None
    latency_ms: float | None = None
    token_count: int | None = None
    cost_usd: float | None = None

    # --- status -------------------------------------------------------------
    status: SpanStatus = SpanStatus.UNSET
    error_type: str | None = None
    error_message: str | None = None

    # --- context ------------------------------------------------------------
    attributes: dict[str, Any] = Field(default_factory=dict)
    resource: Resource = Field(default_factory=Resource)

    # --- validation ---------------------------------------------------------
    @field_validator("trace_id")
    @classmethod
    def _check_trace_id(cls, v: str) -> str:
        if not is_valid_trace_id(v):
            raise ValueError(f"invalid trace_id: {v!r} (expected 32 lowercase hex chars)")
        return v

    @field_validator("span_id")
    @classmethod
    def _check_span_id(cls, v: str) -> str:
        if not is_valid_span_id(v):
            raise ValueError(f"invalid span_id: {v!r} (expected 16 lowercase hex chars)")
        return v

    @field_validator("parent_span_id")
    @classmethod
    def _check_parent_span_id(cls, v: str | None) -> str | None:
        if v is not None and not is_valid_span_id(v):
            raise ValueError(f"invalid parent_span_id: {v!r}")
        return v

    @model_validator(mode="after")
    def _derive_latency(self) -> Span:
        """Fill in latency_ms from the timestamps when the span is closed."""
        if self.end_time_ns is not None:
            if self.end_time_ns < self.start_time_ns:
                raise ValueError("end_time_ns must be >= start_time_ns")
            # Avoid recursion: validate_assignment would re-trigger this hook.
            object.__setattr__(
                self,
                "latency_ms",
                (self.end_time_ns - self.start_time_ns) / NANOS_PER_MILLI,
            )
        return self

    # --- serialization ------------------------------------------------------
    def to_json(self) -> str:
        """Serialize to a compact JSON string (used by the flush transport)."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, raw: str) -> Span:
        """Deserialize from a JSON string (used by the ingestion gateway)."""
        return cls.model_validate_json(raw)

    def to_otlp(self) -> dict[str, Any]:
        """Map to an OTLP-style span dict for OpenTelemetry interoperability.

        Keys follow the OTLP/JSON encoding so the span can be forwarded to any
        OTel-compatible collector without a translation layer.
        """
        status_code = {
            SpanStatus.UNSET: "STATUS_CODE_UNSET",
            SpanStatus.OK: "STATUS_CODE_OK",
            SpanStatus.ERROR: "STATUS_CODE_ERROR",
        }[self.status]
        return {
            "traceId": self.trace_id,
            "spanId": self.span_id,
            "parentSpanId": self.parent_span_id or "",
            "name": self.name,
            "kind": f"SPAN_KIND_{self.kind.value.upper()}",
            "startTimeUnixNano": str(self.start_time_ns),
            "endTimeUnixNano": str(self.end_time_ns) if self.end_time_ns is not None else "",
            "attributes": _to_otlp_attributes(self._otlp_attribute_source()),
            "status": {"code": status_code, "message": self.error_message or ""},
        }

    def _otlp_attribute_source(self) -> dict[str, Any]:
        """Merge semantic fields into the attribute bag for OTLP export."""
        merged: dict[str, Any] = dict(self.attributes)
        for key, value in {
            "navik.agent_name": self.agent_name,
            "navik.tool_name": self.tool_name,
            "navik.token_count": self.token_count,
            "navik.cost_usd": self.cost_usd,
            "navik.kind": self.kind.value,
        }.items():
            if value is not None:
                merged[key] = value
        return merged


def _to_otlp_attributes(attrs: dict[str, Any]) -> list[dict[str, Any]]:
    """Encode a flat dict as an OTLP KeyValue list."""
    out: list[dict[str, Any]] = []
    for key, value in attrs.items():
        out.append({"key": key, "value": _otlp_any_value(value)})
    return out


def _otlp_any_value(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, str):
        return {"stringValue": value}
    # Fall back to a string encoding for anything richer.
    return {"stringValue": str(value)}

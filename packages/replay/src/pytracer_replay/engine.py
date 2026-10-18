"""Replay engine - re-run an agent with recorded responses injected.

Reconstructs a run by swapping the SDK's default tracer for one carrying a
replay interceptor. The agent's ``@op``-decorated tools and model calls consult
the interceptor and receive recorded outputs instead of executing live, so the
run is deterministic. Three modes control what is mocked.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from enum import Enum

import pytracer_sdk as pytracer
from pytracer_sdk import InMemoryTransport, Intercept, Resource, Span, SpanKind, SpanStatus, Tracer

from .fidelity import NondeterminismControls, SideEffectKind, SideEffectLog, controlled
from .mocks import MockSet, _identifier
from .sandbox import NetworkSandbox


class ReplayMode(str, Enum):
    """What gets replaced with recorded responses during a replay."""

    FULL = "full"  # recorded model output AND recorded tools - fully deterministic
    PARTIAL = "partial"  # recorded tools, live model
    LIVE = "live"  # nothing mocked - re-run everything (drift detection)


_MOCKED_KINDS: dict[ReplayMode, frozenset[SpanKind]] = {
    ReplayMode.FULL: frozenset({SpanKind.TOOL, SpanKind.LLM}),
    ReplayMode.PARTIAL: frozenset({SpanKind.TOOL}),
    ReplayMode.LIVE: frozenset(),
}


class MissingMockError(RuntimeError):
    """Raised in strict mode when a mocked op has no recorded response."""


@dataclass
class ReplayResult:
    """Outcome of a single replay run."""

    return_value: object
    status: SpanStatus
    mode: ReplayMode
    spans: list[Span] = field(default_factory=list)
    error_type: str | None = None
    error_message: str | None = None
    # External (unmocked) tool calls that ran live during this replay.
    side_effects: SideEffectLog = field(default_factory=SideEffectLog)

    @property
    def failed(self) -> bool:
        return self.status is SpanStatus.ERROR


class ReplayEngine:
    """Runs agents under a chosen replay mode against a mock set."""

    def __init__(
        self,
        mode: ReplayMode = ReplayMode.FULL,
        *,
        strict: bool = False,
        sandbox: bool = False,
        network_allow: Iterable[str] = (),
        fuzzy_match: bool = False,
        controls: NondeterminismControls | None = None,
    ) -> None:
        self.mode = mode
        self.strict = strict
        # When sandbox is on, agent execution runs with outbound network blocked
        # (except network_allow), so a replay cannot fire a real side effect.
        self.sandbox = sandbox
        self.network_allow = tuple(network_allow)
        # When on, an exact mock miss falls back to the identifier's recordings
        # (nearest by input similarity), so small input drift still replays.
        self.fuzzy_match = fuzzy_match
        # When set, the clock is frozen and the RNG seeded around agent execution,
        # so a live (or partially live) replay is reproducible beyond the mocks.
        self.controls = controls

    def replay(
        self,
        agent: Callable[[], object],
        mocks: MockSet,
        *,
        resource: Resource | None = None,
    ) -> ReplayResult:
        """Run ``agent`` with recorded responses injected per the mode."""
        mocked = _MOCKED_KINDS[self.mode]
        sink = InMemoryTransport()
        side_effects = SideEffectLog()
        se_lock = threading.Lock()

        def interceptor(
            kind: SpanKind, name: str, tool_name: str | None, inp: object
        ) -> Intercept | None:
            identifier = _identifier(kind, name, tool_name)
            if kind in mocked:
                hit, output = mocks.lookup(identifier, inp, fuzzy=self.fuzzy_match)
                if hit:
                    return Intercept(output)
                if self.strict:
                    raise MissingMockError(f"no recorded mock for {kind.value} {tool_name or name!r}")
            # Not mocked (or a non-strict mock miss): the op runs live. A live tool
            # call is a real external side effect; record it for the safety checks.
            if kind is SpanKind.TOOL:
                with se_lock:
                    side_effects.record(identifier, SideEffectKind.EXTERNAL_CALL, inp)
            return None  # fall back to a live call

        tracer = Tracer(
            sink,
            resource=resource or Resource(service_name="replay", branch="replay", commit="replay"),
            interceptor=interceptor,
        )
        previous = pytracer.get_tracer()
        pytracer.set_tracer(tracer)
        status = SpanStatus.OK
        error_type: str | None = None
        error_message: str | None = None
        return_value: object = None
        run_ctx: AbstractContextManager[None] = (
            controlled(self.controls) if self.controls is not None else nullcontext()
        )
        try:
            if self.sandbox:
                with NetworkSandbox(self.network_allow), run_ctx:
                    return_value = agent()
            else:
                with run_ctx:
                    return_value = agent()
        except Exception as exc:  # noqa: BLE001 - capture agent failure as a result, not a raise
            status = SpanStatus.ERROR
            error_type = type(exc).__name__
            error_message = str(exc)
        finally:
            tracer.flush()
            tracer.shutdown()
            pytracer.set_tracer(previous)

        spans = sorted(sink.spans, key=lambda s: s.start_time_ns)
        if any(s.status is SpanStatus.ERROR for s in spans):
            status = SpanStatus.ERROR
            if error_type is None:
                first_error = next(s for s in spans if s.status is SpanStatus.ERROR)
                error_type = first_error.error_type
                error_message = first_error.error_message
        return ReplayResult(
            return_value=return_value,
            status=status,
            mode=self.mode,
            spans=spans,
            error_type=error_type,
            error_message=error_message,
            side_effects=side_effects,
        )

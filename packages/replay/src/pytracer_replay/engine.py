"""Replay engine - re-run an agent with recorded responses injected.

Reconstructs a run by swapping the SDK's default tracer for one carrying a
replay interceptor. The agent's ``@op``-decorated tools and model calls consult
the interceptor and receive recorded outputs instead of executing live, so the
run is deterministic. Three modes control what is mocked.
"""

from __future__ import annotations

import builtins
import importlib
import threading
from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from enum import Enum

import pytracer_sdk as pytracer
from pytracer_sdk import InMemoryTransport, Intercept, Resource, Span, SpanKind, SpanStatus, Tracer
from pytracer_sdk.tracer import intercepted_span

from .fidelity import (
    ConcurrencyLedger,
    NondeterminismControls,
    RecordedStream,
    SideEffectKind,
    SideEffectLog,
    controlled,
)
from .hashing import input_hash
from .mocks import MockSet, RecordedError, _identifier
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


class ReplayedError(Exception):
    """Stand-in raised on replay for a recorded failure whose original exception
    type is not a builtin and so cannot be reconstructed exactly."""


def _resolve_exception_type(module: str | None, name: str) -> type[BaseException] | None:
    """Find the real exception class named ``name`` from ``module`` (or builtins)."""
    builtin = getattr(builtins, name, None)
    if isinstance(builtin, type) and issubclass(builtin, BaseException):
        return builtin
    if module and module != "builtins":
        try:
            mod = importlib.import_module(module)
        except Exception:  # noqa: BLE001 - the module may not be importable at replay
            return None
        cls = getattr(mod, name, None)
        if isinstance(cls, type) and issubclass(cls, BaseException):
            return cls
    return None


def _rebuild_exception(rec: RecordedError) -> BaseException:
    """Reconstruct a recorded failure as a live exception to re-raise on replay.

    The real exception type is re-created when it can be found - a builtin, or a
    class imported from its recorded module (e.g. ``neo4j.exceptions.
    ServiceUnavailable``) - so the agent's own ``except`` for that type still fires.
    When the type cannot be resolved or constructed from the message alone, it
    falls back to a :class:`ReplayedError` carrying the original name and message.
    """
    exc_cls = _resolve_exception_type(rec.error_module, rec.error_type)
    if exc_cls is not None:
        try:
            return exc_cls(rec.error_message or "")
        except Exception:  # noqa: BLE001 - some exceptions need other constructor args
            pass
    label = f"{rec.error_type}: {rec.error_message}" if rec.error_message else rec.error_type
    return ReplayedError(label)


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
    # Identifiers of frozen model (LLM) ops whose input no longer matched any
    # recording: the model input changed (e.g. a prompt edit), so the recorded
    # response does not apply and the step cannot be judged by a frozen replay.
    model_input_changes: list[str] = field(default_factory=list)
    # Identifiers of ops called concurrently from different OS threads with the same
    # input, under the same caller (or no caller op), but differing recorded
    # responses: nothing distinguishes those calls, so the match may be swapped
    # (surfaced, not silent). Concurrent calls from different callers resolve by
    # caller; sequential and single-threaded-async concurrency is not affected.
    concurrency_ambiguities: list[str] = field(default_factory=list)
    # True when there were recordings to inject but the agent hit no @op seam, so
    # nothing could be replayed (e.g. a run captured via an adapter, which records
    # only; replay needs @op instrumentation). The run therefore executed live.
    uninstrumented: bool = False
    # True when this replay ran with --evaluate-changed-prompts, so a changed model
    # op was run live rather than failing frozen. Lets a report explain the verdict
    # without advising an option that is already on.
    evaluated_changed_prompts: bool = False

    @property
    def failed(self) -> bool:
        return self.status is SpanStatus.ERROR

    @property
    def changed_models(self) -> list[str]:
        """Distinct model ops whose input changed, in first-seen order."""
        return list(dict.fromkeys(self.model_input_changes))


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
        live_ops: Iterable[str] = (),
        frozen_ops: Iterable[str] = (),
        writes: Iterable[str] = (),
        evaluate_changed_prompts: bool = False,
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
        # Per-op selective freeze: ops named in live_ops run live (freeze the rest),
        # ops named in frozen_ops are served from the recording, each overriding the
        # mode's kind-based default. An op is matched by its identifier (tool name,
        # else span name). This is how a user freezes a whole run except the one step
        # they changed.
        self.live_ops = frozenset(live_ops)
        self.frozen_ops = frozenset(frozen_ops)
        # Identifiers of tools known to mutate external state. A live call to one
        # is logged as a WRITE (so a repeat is flagged non-idempotent); every other
        # live tool call is logged as an unclassified EXTERNAL_CALL, never as a
        # write, so read-only tools are not falsely flagged.
        self.writes = frozenset(writes)
        # When on, a frozen model op whose input changed (a prompt edit) runs live
        # instead of failing the scenario, so the change is actually evaluated. It
        # makes a real model call, so the caller also drops the network sandbox.
        self.evaluate_changed_prompts = evaluate_changed_prompts
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
        # Frozen model ops whose input no longer matched a recording (a prompt edit).
        model_input_changes: list[str] = []
        # Numbers each (identifier, input) call so concurrent/repeat calls resolve
        # to their recording in order; thread-safe for concurrent tool execution.
        ledger = ConcurrencyLedger()
        # Each live op's span id -> its (identifier, input hash), so a call can name
        # its caller the same way the recording does. Same-input calls racing on
        # different threads then resolve by caller instead of by arrival order.
        callers: dict[str, tuple[str, str]] = {}
        # Flags whether any @op was seen, to detect replaying a run that has
        # recordings but never hits the @op seam (adapter-captured, record-only).
        seen_op = {"any": False}

        def caller_of(identifier: str, inp: object) -> tuple[str, str] | None:
            """Register this op and return its calling op's key, if it has one."""
            span = intercepted_span()
            if span is None:
                return None
            with se_lock:
                callers[span.span_id] = (identifier, input_hash(inp))
                return callers.get(span.parent_span_id) if span.parent_span_id else None

        def interceptor(
            kind: SpanKind, name: str, tool_name: str | None, inp: object
        ) -> Intercept | None:
            seen_op["any"] = True
            identifier = _identifier(kind, name, tool_name)
            caller = caller_of(identifier, inp)
            # Per-op selective freeze overrides the mode's kind-based default.
            if identifier in self.frozen_ops:
                mock_this = True
            elif identifier in self.live_ops:
                mock_this = False
            else:
                mock_this = kind in mocked
            if mock_this:
                # Resolve by caller when the recording has calls under this caller;
                # the call is still numbered on its plain key so a later call that
                # has no caller scope keeps its place in the sequence.
                if caller is not None and not mocks.has_caller_scope(identifier, inp, caller):
                    caller = None
                occurrence = ledger.key_for(
                    identifier, inp, track_threads=caller is None
                ).occurrence
                caller_occurrence = (
                    ledger.scoped_key_for(identifier, inp, caller).occurrence
                    if caller is not None
                    else 0
                )
                # In partial mode the model runs live and words tool inputs
                # differently each run, so a frozen tool matches by call order
                # (its per-identifier index), not by the re-worded input.
                positional = self.mode is ReplayMode.PARTIAL and kind is SpanKind.TOOL
                pos_occurrence = ledger.positional_for(identifier) if positional else None
                hit, output, match = mocks.lookup(
                    identifier,
                    inp,
                    occurrence=occurrence,
                    positional_occurrence=pos_occurrence,
                    fuzzy=self.fuzzy_match,
                    positional=positional,
                    caller=caller,
                    caller_occurrence=caller_occurrence,
                )
                if hit:
                    if isinstance(output, RecordedError):
                        # Re-raise the recorded failure so the agent's error path
                        # replays exactly, instead of serving it as a success.
                        raise _rebuild_exception(output)
                    if isinstance(output, RecordedStream):
                        return Intercept(list(output.replay()), stream=True, match=match)
                    return Intercept(output, match=match)
                # Miss on a frozen op. A frozen model op that misses means the model
                # input changed (a prompt edit): the recorded response does not
                # apply, so note it - and, if asked, run it live to evaluate it.
                if kind is SpanKind.LLM:
                    with se_lock:
                        model_input_changes.append(identifier)
                    if self.evaluate_changed_prompts:
                        return None  # fall through to a live model call
                if self.strict:
                    raise MissingMockError(
                        f"no recording for {kind.value} {tool_name or name!r} "
                        f"(occurrence {occurrence}): its input did not match any recorded "
                        f"call, so there is nothing to replay. Re-record the trace, or run "
                        f"this scenario in partial/live mode."
                    )
            # Not mocked (or a non-strict mock miss): the op runs live. A live tool
            # call is a real external side effect; record it for the safety checks,
            # as a WRITE when the tool was declared a writer, else as an
            # unclassified external call.
            if kind is SpanKind.TOOL:
                effect = (
                    SideEffectKind.WRITE
                    if identifier in self.writes
                    else SideEffectKind.EXTERNAL_CALL
                )
                with se_lock:
                    side_effects.record(identifier, effect, inp)
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
                # Block loopback too, not just remote hosts: a framework's model
                # call (no @op seam) would otherwise reach a local model server
                # (Ollama, vLLM, LM Studio) during replay. A loopback host that is
                # genuinely needed can still be added to network_allow.
                with NetworkSandbox(self.network_allow, allow_loopback=False), run_ctx:
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
        # The run's verdict is the agent's own outcome (the value it returned or the
        # exception it raised, captured above), not the status of child spans. A tool
        # error the agent caught and recovered from leaves the run OK, so a scenario
        # like "the app survives a database outage" can pass. An unhandled child error
        # still fails the run, because it propagates out of agent() into the except
        # above. A handled child error that should still fail is left to the
        # assertions (no_errors with scope="any").
        # A key (same input, and same caller when resolved by caller) reached from
        # more than one OS thread, whose recordings differ, could have had its
        # matches swapped by arrival order; surface it.
        order_sensitive = mocks.order_sensitive_keys()
        concurrency_ambiguities = sorted(
            {key[0] for key in ledger.multi_thread_keys() if key in order_sensitive}
        )
        return ReplayResult(
            return_value=return_value,
            status=status,
            mode=self.mode,
            spans=spans,
            error_type=error_type,
            error_message=error_message,
            side_effects=side_effects,
            model_input_changes=model_input_changes,
            concurrency_ambiguities=concurrency_ambiguities,
            uninstrumented=(len(mocks) > 0 and not seen_op["any"]),
            evaluated_changed_prompts=self.evaluate_changed_prompts,
        )

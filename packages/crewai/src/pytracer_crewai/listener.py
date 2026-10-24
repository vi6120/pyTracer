"""CrewAI event-bus listener that records pyTracer spans.

CrewAI emits typed events on a global event bus for crew kickoffs, task and
agent execution, tool usage, and LLM calls. The bus stamps each event with an
``event_id``, a ``parent_event_id`` (the enclosing event), and, on completion
events, a ``started_event_id`` (the matching start event). This listener uses
those ids to drive :class:`~pytracer_sdk.SpanRecorder`: start events open a span
keyed by ``event_id`` under ``parent_event_id``, and completion/failure events
close the span keyed by ``started_event_id``.

Instrumentation is automatic: constructing the listener registers it on the
global bus, so any crew run is captured with no ``@op`` in the agent code.
"""

from __future__ import annotations

from typing import Any

from crewai.events import BaseEventListener
from crewai.events.types.agent_events import (
    AgentExecutionCompletedEvent,
    AgentExecutionErrorEvent,
    AgentExecutionStartedEvent,
)
from crewai.events.types.crew_events import (
    CrewKickoffCompletedEvent,
    CrewKickoffStartedEvent,
)
from crewai.events.types.llm_events import (
    LLMCallCompletedEvent,
    LLMCallFailedEvent,
    LLMCallStartedEvent,
)
from crewai.events.types.task_events import (
    TaskCompletedEvent,
    TaskFailedEvent,
    TaskStartedEvent,
)
from crewai.events.types.tool_usage_events import (
    ToolUsageErrorEvent,
    ToolUsageFinishedEvent,
    ToolUsageStartedEvent,
)
from pytracer_sdk import SpanKind, SpanRecorder, Tracer, get_tracer

_MAX_DEPTH = 6


def _agent_role(event: Any) -> str | None:
    """The agent's role, so its span is named for the role rather than "agent".

    CrewAI carries the role on the event (``agent_role``) or on a nested ``agent``
    object (``event.agent.role``); fall back to a plain ``role`` attribute.
    """
    role = getattr(event, "agent_role", None)
    if role:
        return str(role)
    agent = getattr(event, "agent", None)
    if agent is not None:
        nested = getattr(agent, "role", None)
        if nested:
            return str(nested)
    role = getattr(event, "role", None)
    return str(role) if role else None


def _jsonify(value: Any, depth: int = 0) -> Any:
    if depth > _MAX_DEPTH:
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonify(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v, depth + 1) for v in value]
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return _jsonify(dump(mode="json"), depth + 1)
        except Exception:  # noqa: BLE001 - fall back to a string form
            return str(value)
    return str(value)


def _tokens(usage: Any) -> int | None:
    if not isinstance(usage, dict):
        return None
    if usage.get("total_tokens") is not None:
        return int(usage["total_tokens"])
    inp = usage.get("prompt_tokens") or usage.get("input_tokens")
    out = usage.get("completion_tokens") or usage.get("output_tokens")
    if inp is not None or out is not None:
        return int(inp or 0) + int(out or 0)
    return None


class PyTracerEventListener(BaseEventListener):
    """Records pyTracer spans from CrewAI's event bus."""

    def __init__(self, tracer: Tracer | None = None, *, register: bool = True) -> None:
        self._recorder = SpanRecorder(tracer or get_tracer())
        # Open agent runs, innermost last. CrewAI emits LLM and tool events without a
        # parent_event_id, so they are attached to the current agent instead of each
        # starting its own trace (which hid which agent made the call).
        self._agent_ids: list[str] = []
        # ``register`` is False in unit tests that call the handlers directly and
        # do not want to attach to the global bus.
        if register:
            super().__init__()

    @property
    def recorder(self) -> SpanRecorder:
        return self._recorder

    # --- registration -------------------------------------------------------
    def setup_listeners(self, bus: Any) -> None:
        bus.on(CrewKickoffStartedEvent)(self.on_crew_start)
        bus.on(CrewKickoffCompletedEvent)(self.on_crew_end)
        bus.on(TaskStartedEvent)(self.on_task_start)
        bus.on(TaskCompletedEvent)(self.on_task_end)
        bus.on(TaskFailedEvent)(self.on_task_error)
        bus.on(AgentExecutionStartedEvent)(self.on_agent_start)
        bus.on(AgentExecutionCompletedEvent)(self.on_agent_end)
        bus.on(AgentExecutionErrorEvent)(self.on_agent_error)
        bus.on(ToolUsageStartedEvent)(self.on_tool_start)
        bus.on(ToolUsageFinishedEvent)(self.on_tool_end)
        bus.on(ToolUsageErrorEvent)(self.on_tool_error)
        bus.on(LLMCallStartedEvent)(self.on_llm_start)
        bus.on(LLMCallCompletedEvent)(self.on_llm_end)
        bus.on(LLMCallFailedEvent)(self.on_llm_error)

    # --- shared start/end ---------------------------------------------------
    def _start(
        self,
        event: Any,
        *,
        name: str,
        kind: SpanKind,
        agent_name: str | None = None,
        tool_name: str | None = None,
        input: Any | None = None,
        fallback_parent: str | None = None,
    ) -> None:
        parent = getattr(event, "parent_event_id", None)
        parent_id = str(parent) if parent else None
        # CrewAI can point parent_event_id at an intermediate event we never opened
        # as a span (seen in 1.15), which would start a new root trace. Nest under
        # the innermost open agent instead whenever the reported parent is not open.
        if parent_id is None or not self._recorder.is_open(parent_id):
            parent_id = fallback_parent
        self._recorder.start(
            str(event.event_id),
            name=name,
            kind=kind,
            agent_name=agent_name,
            tool_name=tool_name,
            parent_run_id=parent_id,
            input=_jsonify(input) if input is not None else None,
            attributes={"framework": "crewai"},
        )

    def _current_agent(self) -> str | None:
        """The innermost open agent run, used as the parent for LLM/tool events."""
        return self._agent_ids[-1] if self._agent_ids else None

    def _end(
        self, event: Any, *, output: Any = None, tokens: int | None = None, error: Any = None
    ) -> None:
        started = getattr(event, "started_event_id", None)
        if started is None:
            return  # cannot correlate to a start event
        self._recorder.end(
            str(started),
            output=_jsonify(output) if output is not None else None,
            token_count=tokens,
            error=error if error is not None else None,
        )

    # --- crew ---------------------------------------------------------------
    def on_crew_start(self, source: Any, event: Any) -> None:
        self._start(
            event, name=getattr(event, "crew_name", None) or "crew", kind=SpanKind.AGENT,
            input=getattr(event, "inputs", None),
        )

    def on_crew_end(self, source: Any, event: Any) -> None:
        self._end(event, output=getattr(event, "output", None))

    # --- task ---------------------------------------------------------------
    def on_task_start(self, source: Any, event: Any) -> None:
        self._start(event, name=getattr(event, "task_name", None) or "task", kind=SpanKind.AGENT)

    def on_task_end(self, source: Any, event: Any) -> None:
        self._end(event, output=getattr(event, "output", None))

    def on_task_error(self, source: Any, event: Any) -> None:
        self._end(event, error=str(getattr(event, "error", "task failed")))

    # --- agent --------------------------------------------------------------
    def on_agent_start(self, source: Any, event: Any) -> None:
        role = _agent_role(event)
        self._start(
            event, name=role or "agent", kind=SpanKind.AGENT, agent_name=role,
            input=getattr(event, "task_prompt", None),
        )
        self._agent_ids.append(str(event.event_id))

    def on_agent_end(self, source: Any, event: Any) -> None:
        self._pop_agent(getattr(event, "started_event_id", None))
        self._end(event, output=getattr(event, "output", None))

    def on_agent_error(self, source: Any, event: Any) -> None:
        self._pop_agent(getattr(event, "started_event_id", None))
        self._end(event, error=str(getattr(event, "error", "agent failed")))

    def _pop_agent(self, started: Any) -> None:
        if started is not None and str(started) in self._agent_ids:
            self._agent_ids.remove(str(started))

    # --- tool ---------------------------------------------------------------
    def on_tool_start(self, source: Any, event: Any) -> None:
        name = getattr(event, "tool_name", None) or "tool"
        self._start(
            event, name=name, kind=SpanKind.TOOL, tool_name=name,
            input=getattr(event, "tool_args", None),
            fallback_parent=self._current_agent(),
        )

    def on_tool_end(self, source: Any, event: Any) -> None:
        failure = getattr(event, "failure", None)
        if failure:
            self._end(event, error=str(failure))
        else:
            self._end(event, output=getattr(event, "output", None))

    def on_tool_error(self, source: Any, event: Any) -> None:
        self._end(event, error=str(getattr(event, "error", "tool failed")))

    # --- llm ----------------------------------------------------------------
    def on_llm_start(self, source: Any, event: Any) -> None:
        self._start(
            event, name=getattr(event, "model", None) or "llm", kind=SpanKind.LLM,
            input=getattr(event, "messages", None),
            fallback_parent=self._current_agent(),
        )

    def on_llm_end(self, source: Any, event: Any) -> None:
        self._end(
            event, output=getattr(event, "response", None), tokens=_tokens(getattr(event, "usage", None))
        )

    def on_llm_error(self, source: Any, event: Any) -> None:
        self._end(event, error=str(getattr(event, "error", "llm call failed")))

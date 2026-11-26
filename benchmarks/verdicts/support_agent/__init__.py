"""A customer-support agent for tau2-bench's retail domain.

The agent loop is fixed; ``glue.py`` is the code between the model and the tools
(argument preparation, tool calls and their errors, the history the model sees),
and is where each seeded bug lives. The run returns the writes the agent made, in
order, which is what the benchmark's assertions judge.
"""

from __future__ import annotations

from typing import Any

import pytracer_sdk as pytracer
from pytracer_sdk import SpanKind
from tau2.data_model.message import ToolMessage

from . import glue, harness

MAX_TURNS = 15  # customer turns; a retail task needs far fewer, so a stuck loop ends early
MAX_STEPS = 12  # model calls per customer turn
GREETING = "Hi! How can I help you today?"


@pytracer.op(kind=SpanKind.AGENT, agent_name="support")
def run(task_id: str) -> dict[str, Any]:
    session = harness.start(task_id)
    run.last_session = session  # type: ignore[attr-defined]  # for the live check, even if the run raises
    history: list[Any] = [harness.system_message(session.policy)]
    agent_text = GREETING
    for _ in range(MAX_TURNS):
        customer = session.user_says(agent_text)
        if customer is None:
            break
        history.append(customer)
        for _ in range(MAX_STEPS):
            reply = harness.model(glue.history_for_model(history), session.tools)
            history.append(reply)
            if not reply.tool_calls:
                agent_text = reply.content or ""
                break
            for call in reply.tool_calls:
                result = glue.call_tool(session, call.name, glue.prepare_args(call.name, call.arguments))
                history.append(ToolMessage(id=call.id, role="tool", content=result, requestor="assistant"))
    return {"task": task_id, "last_reply": agent_text}

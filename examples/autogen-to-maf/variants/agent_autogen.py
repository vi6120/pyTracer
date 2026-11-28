"""The support agent on AutoGen (archived upstream). A scripted model client stands
in for the real one, so the recipe runs with no API key: it asks for one refund,
then confirms it."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import ToolCallExecutionEvent
from autogen_core import FunctionCall
from autogen_core.models import CreateResult, ModelFamily, RequestUsage
from autogen_ext.models.replay import ReplayChatCompletionClient

from . import refunds

_installed = False


def _model() -> ReplayChatCompletionClient:
    call = FunctionCall(id="call_1", name="refund_order",
                        arguments=json.dumps({"order_id": "#W1001", "amount": 49.99}))
    return ReplayChatCompletionClient(
        [CreateResult(finish_reason="function_calls", content=[call],
                      usage=RequestUsage(prompt_tokens=40, completion_tokens=12), cached=False)],
        model_info={"vision": False, "function_calling": True, "json_output": False,
                    "family": ModelFamily.UNKNOWN, "structured_output": False},
    )


def run(order_id: str) -> dict[str, Any]:
    global _installed
    if not _installed:  # route AutoGen's model and tool calls through pyTracer, once
        from pytracer_autogen import install

        install(replayable=True)
        _installed = True
    agent = AssistantAgent("support", model_client=_model(), tools=[refunds.refund_order],
                           system_message="You handle refunds for the shop.")

    async def go() -> list[str]:
        result = await agent.run(task=f"Please refund my order {order_id}.")
        return [r.content for m in result.messages if isinstance(m, ToolCallExecutionEvent)
                for r in m.content]

    return refunds.receipt(asyncio.run(go()))

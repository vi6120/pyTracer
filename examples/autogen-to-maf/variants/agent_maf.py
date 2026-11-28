"""The same support agent, ported to Microsoft Agent Framework. A scripted chat
client stands in for the real one, so the recipe runs with no API key: it asks for
one refund, then confirms it."""

from __future__ import annotations

import asyncio
from typing import Any

from agent_framework import (
    Agent,
    BaseChatClient,
    ChatMiddlewareLayer,
    ChatResponse,
    Content,
    FunctionInvocationLayer,
    Message,
    tool,
)

from . import refunds

_installed = False
USAGE = {"input_token_count": 40, "output_token_count": 12, "total_token_count": 52}


class ScriptedClient(FunctionInvocationLayer, ChatMiddlewareLayer, BaseChatClient):  # type: ignore[misc]
    """Replies from a script, one reply per model call."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.replies = [
            [Content.from_function_call(call_id="call_1", name="refund_order",
                                        arguments='{"order_id": "#W1001", "amount": 49.99}')],
            [Content.from_text("Your order #W1001 is refunded: $49.99.")],
        ]

    def _inner_get_response(self, *, messages: Any, stream: bool, options: Any, **kwargs: Any) -> Any:
        contents = self.replies.pop(0)

        async def respond() -> Any:
            return ChatResponse(messages=[Message(role="assistant", contents=contents)],
                                model="scripted", usage_details=USAGE)

        return respond()


def run(order_id: str) -> dict[str, Any]:
    global _installed
    if not _installed:  # capture and replay Agent Framework's model and tool calls, once
        from pytracer_agent_framework import install

        install()
        _installed = True
    agent = Agent(client=ScriptedClient(), name="support", tools=[tool(refunds.refund_order)],
                  instructions="You handle refunds for the shop.")

    async def go() -> list[str]:
        response = await agent.run(f"Please refund my order {order_id}.")
        return [str(c.result) for m in response.messages for c in m.contents
                if getattr(c, "type", None) == "function_result"]

    return refunds.receipt(asyncio.run(go()))

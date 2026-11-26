"""A refund agent that retries a timed-out refund, for the side-effect assertions.

The first refund call times out after the refund went through, so the retry
refunds the customer a second time: the run passes, and only the write calls
show the duplicate.
"""

from __future__ import annotations

from typing import Any

import pytracer_sdk as pytracer
from pytracer_sdk import SpanKind

TIMEOUTS: list[Exception] = []


@pytracer.op(kind=SpanKind.TOOL, tool_name="lookup_order")
def lookup_order(order_id: str) -> dict[str, Any]:
    return {"order_id": order_id, "amount": 30}


@pytracer.op(kind=SpanKind.TOOL, tool_name="refund_payment")
def refund_payment(order_id: str, amount: int) -> dict[str, str]:
    if TIMEOUTS:
        raise TIMEOUTS.pop()
    return {"status": "refunded"}


@pytracer.op(kind=SpanKind.AGENT, agent_name="support")
def agent(order_id: str) -> dict[str, Any]:
    order = lookup_order(order_id=order_id)
    for _ in range(3):
        try:
            refund_payment(order_id=order_id, amount=order["amount"])
            return {"order_id": order_id, "status": "refunded"}
        except TimeoutError:
            continue
    return {"order_id": order_id, "status": "failed"}


# The same refund, retried below the tool: its HTTP client retries the timed-out
# POST, so the agent sees one successful call while the refund went out twice.
HTTP_TIMEOUTS: list[bool] = []


def _gateway(request: Any) -> Any:
    import httpx

    if HTTP_TIMEOUTS:
        HTTP_TIMEOUTS.pop()
        raise httpx.ReadTimeout("payment gateway timed out")
    return httpx.Response(200, json={"status": "refunded"})


@pytracer.op(kind=SpanKind.TOOL, tool_name="refund_via_api")
def refund_via_api(order_id: str) -> dict[str, str]:
    import httpx

    client = httpx.Client(transport=httpx.MockTransport(_gateway))
    for _ in range(3):
        try:
            return dict(client.post("https://pay.example/v1/refunds", json={"order": order_id}).json())
        except httpx.ReadTimeout:
            continue
    return {"status": "failed"}


@pytracer.op(kind=SpanKind.AGENT, agent_name="support")
def http_agent(order_id: str) -> dict[str, Any]:
    return {"order_id": order_id, **refund_via_api(order_id=order_id)}

"""The shop's refund tool and the receipt the agent returns. Framework-neutral: the
AutoGen agent and its Agent Framework port both import this file unchanged.

This version is fixed: the receipt keeps the cents.
"""

from __future__ import annotations

import json

ORDERS = {"#W1001": 49.99}


def refund_order(order_id: str, amount: float) -> str:
    """Refund an order. Returns what the payment service confirmed, as JSON."""
    return json.dumps({"order_id": order_id, "refunded": f"{amount:.2f}"})


def receipt(tool_results: list[str]) -> dict[str, object]:
    """What the agent reports once the conversation is over."""
    for text in tool_results:
        done = json.loads(text)
        return {"status": "refunded", "order_id": done["order_id"],
                "amount_cents": round(float(done["refunded"]) * 100)}
    return {"status": "no_action"}

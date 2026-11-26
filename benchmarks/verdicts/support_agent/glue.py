"""Glue between the support agent's model and the retail tools.

This is the correct version. Each seeded bug in ``bugs.py`` is a small edit to
this file, made on the baseline branch; its branches then fix it, leave it, or
fix it while breaking something else.
"""

from __future__ import annotations

from typing import Any

from .harness import Session

# Tools whose call changes an order, so a timeout may hide a committed write.
ORDER_WRITES = (
    "cancel_pending_order",
    "exchange_delivered_order_items",
    "modify_pending_order_address",
    "modify_pending_order_items",
    "modify_pending_order_payment",
    "return_delivered_order_items",
)


def prepare_args(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """The tool arguments, as the model gave them."""
    return dict(args)


def call_tool(session: Session, name: str, args: dict[str, Any]) -> str:
    """Run a tool and give the model its result, or the error it raised."""
    try:
        return session.tool(name, args)
    except TimeoutError:
        if name in ORDER_WRITES:
            # The write may have committed before the timeout: check the order
            # before trying again, so a retry never repeats a write.
            order = session.tool("get_order_details", {"order_id": args["order_id"]})
            if '"status": "pending"' not in order and '"status": "delivered"' not in order:
                return order
        return session.tool(name, args)
    except Exception as exc:  # noqa: BLE001 - the model reads the error and recovers
        return f"Error: {exc}"


def history_for_model(history: list[Any]) -> list[Any]:
    """The conversation the model sees: all of it, policy first."""
    return list(history)

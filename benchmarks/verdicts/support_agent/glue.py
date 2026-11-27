"""Glue between the support agent's model and the retail tools.

This is the correct version. Each seeded bug in ``bugs.py`` is a small edit to
this file, made on the baseline branch; its branches then fix it, leave it, or
fix it while breaking something else.
"""

from __future__ import annotations

import re
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

# An order id as the retail tools write it.
ORDER_ID = re.compile(r"#W\d{7}")


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


def run_record(calls: list[tuple[str, dict[str, Any], str]]) -> dict[str, Any]:
    """What the run reports to its caller once the conversation is over: each order
    write that worked (its tool and the order id as sent), and a status.

    ``calls`` is every tool call of the run, in order: its name, the arguments as
    sent, and the result the model was given. Nothing here reaches the model or a
    tool, so a change to it never changes the run's calls.
    """
    made = [(name, args, result) for name, args, result in calls if name in ORDER_WRITES]
    failed = [name for name, _, result in made if result.startswith("Error")]
    writes = []
    for name, args, result in made:
        if result.startswith("Error"):
            continue
        order_id = str(args.get("order_id"))
        if not ORDER_ID.fullmatch(order_id):
            raise ValueError(f"not an order id: {order_id!r}")
        writes.append({"tool": name, "order_id": order_id})
    status = "failed" if failed else "resolved" if writes else "no_action"
    return {"status": status, "writes": writes}

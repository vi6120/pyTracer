"""The fixed part of the benchmark: tau2-bench's retail world, instrumented.

Every branch of every seeded bug imports this unchanged; only ``glue.py`` differs
between branches. It provides:

- the model calls (the support agent's and tau2's user simulator's), routed
  through pyTracer as LLM ops, so a captured conversation replays frozen;
- the retail tools, routed through pyTracer as tool ops;
- a task session: tau2's environment, its LLM user simulator for the task, and the
  task's gold outcome (the database after tau2's gold actions), for the live check.

A model call's match key is the conversation without the per-run ids (tool call
ids, timestamps, costs), so a replay of the same conversation finds its recording.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import tau2.user.user_simulator as user_module
import tau2.utils.llm_utils as llm_utils
from pytracer_sdk import SpanKind
from pytracer_sdk.seam import call_through, tool_names
from tau2.data_model.message import AssistantMessage, SystemMessage, UserMessage
from tau2.domains.retail.environment import get_environment, get_tasks
from tau2.user.user_simulator import UserSimulator

MODEL = os.environ.get("BENCH_MODEL", "anthropic/claude-haiku-5-5")
# The simulated customer's model; the agent's by default.
USER_MODEL = os.environ.get("BENCH_USER_MODEL") or MODEL
# Extra model-call arguments for the agent and the user simulator, as JSON; e.g.
# '{"reasoning_effort": "none"}' turns off a local Ollama model's thinking.
LLM_ARGS: dict[str, Any] = json.loads(os.environ.get("BENCH_LLM_ARGS") or "{}")

# Spending cap. Every live model call is charged to a ledger file shared by all
# the benchmark's processes, at the configured prices (USD per million tokens,
# all input at the full price, so the real bill is at or below it); once the
# ledger reaches BENCH_MAX_USD, the next call refuses to start.
LEDGER = os.environ.get("BENCH_USAGE_LEDGER")
MAX_USD = float(os.environ.get("BENCH_MAX_USD") or 0)
PRICES = {
    "support-agent": (float(os.environ.get("BENCH_PRICE_IN") or 0), float(os.environ.get("BENCH_PRICE_OUT") or 0)),
    "user-simulator": (
        float(os.environ.get("BENCH_USER_PRICE_IN") or os.environ.get("BENCH_PRICE_IN") or 0),
        float(os.environ.get("BENCH_USER_PRICE_OUT") or os.environ.get("BENCH_PRICE_OUT") or 0),
    ),
}


class BudgetExceeded(RuntimeError):
    """The spending cap is reached; the benchmark stops instead of spending more."""


def spent_usd() -> float:
    """What the ledger has charged so far."""
    if not LEDGER or not os.path.exists(LEDGER):
        return 0.0
    with open(LEDGER, encoding="utf-8") as ledger:
        return sum(float(json.loads(line)["usd"]) for line in ledger if line.strip())


def _check_budget() -> None:
    if MAX_USD and spent_usd() >= MAX_USD:
        raise BudgetExceeded(f"spending cap of ${MAX_USD:.2f} reached (ledger {LEDGER})")


def _charge(role: str, usage: Any) -> None:
    if not LEDGER:
        return
    usage = usage or {}
    prompt, completion = int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
    price_in, price_out = PRICES[role]
    usd = (prompt * price_in + completion * price_out) / 1_000_000
    with open(LEDGER, "a", encoding="utf-8") as ledger:
        ledger.write(json.dumps({"role": role, "prompt": prompt, "completion": completion, "usd": usd}) + "\n")

# Retail tools that change the database.
WRITE_TOOLS = (
    "cancel_pending_order",
    "exchange_delivered_order_items",
    "modify_pending_order_address",
    "modify_pending_order_items",
    "modify_pending_order_payment",
    "modify_user_address",
    "return_delivered_order_items",
)

_ROLES = {"user_simulator_response": "user-simulator"}
_generate = llm_utils.generate


def _scripted(path: str) -> Any:
    """A model that replays fixed replies (per role, in order) from a JSON file:
    ``BENCH_MODEL=scripted:<file>`` checks the benchmark's plumbing at no cost."""
    from tau2.data_model.message import ToolCall

    script = json.loads(open(path, encoding="utf-8").read())
    turns = {role: iter(replies) for role, replies in script.items()}

    def generate(model: str, messages: list[Any], tools: Any = None, **kwargs: Any) -> Any:
        role = _ROLES.get(str(kwargs.get("call_name")), "support-agent")
        reply = next(turns[role])
        calls = [
            ToolCall(id=f"call_{i}", name=c["name"], arguments=c["arguments"])
            for i, c in enumerate(reply.get("tool_calls") or ())
        ]
        usage = reply.get("usage") or {"prompt_tokens": 1000, "completion_tokens": 100}
        return AssistantMessage(
            role="assistant", content=reply.get("content"), tool_calls=calls or None, usage=usage
        )

    return generate


if MODEL.startswith("scripted:"):
    _generate = _scripted(MODEL.split(":", 1)[1])


def _message_key(message: Any) -> dict[str, Any]:
    data: dict[str, Any] = {"role": message.role, "content": message.content}
    calls = getattr(message, "tool_calls", None)
    if calls:
        data["tool_calls"] = [{"name": c.name, "arguments": c.arguments} for c in calls]
    return data


def generate(model: str, messages: list[Any], tools: Any = None, **kwargs: Any) -> Any:
    """tau2's model call, as a pyTracer LLM op (agent or user simulator)."""
    role = _ROLES.get(str(kwargs.get("call_name")), "support-agent")

    def call() -> Any:
        _check_budget()
        reply = _generate(model, messages, tools=tools, **kwargs)
        _charge(role, reply.usage)
        # The provider's raw response is per-run noise; keep the message itself.
        return reply.model_copy(update={"raw_data": None})

    def tokens(reply: Any) -> int | None:
        usage = getattr(reply, "usage", None) or {}
        total = int(usage.get("prompt_tokens") or 0) + int(usage.get("completion_tokens") or 0)
        return total or None

    key = {
        "model": model,
        "messages": [_message_key(m) for m in messages],
        "tools": tool_names(tools),
    }
    return call_through(call, kind=SpanKind.LLM, name=role, tokens=tokens, input=key)


# Every model call in the run goes through the op above, tau2's user simulator too.
llm_utils.generate = generate
user_module.generate = generate


@dataclass
class Session:
    """One tau2 retail task: its database, its user simulator, and its gold outcome."""

    task_id: str
    env: Any
    task: Any
    user: Any
    user_state: Any
    flaky_writes: int = 0  # writes still to time out after they commit (bug B1)
    writes: list[dict[str, Any]] = field(default_factory=list)

    @property
    def policy(self) -> str:
        return str(self.env.get_policy())

    @property
    def tools(self) -> list[Any]:
        return list(self.env.get_tools())

    def user_says(self, agent_text: str) -> UserMessage | None:
        """The simulated customer's next message, or None when they end the call."""
        message, self.user_state = self.user.generate_next_message(
            AssistantMessage(role="assistant", content=agent_text), self.user_state
        )
        return None if UserSimulator.is_stop(message) else message

    def tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Run a retail tool as a pyTracer tool op; returns its JSON text."""

        def call() -> str:
            result = self.env.use_tool(name, **arguments)
            if name in WRITE_TOOLS:
                self.writes.append({"tool": name, "args": dict(arguments)})
                if self.flaky_writes:
                    # The gateway timed out after the write committed (bug B1).
                    self.flaky_writes -= 1
                    raise TimeoutError("payment gateway timed out")
            return str(self.env.to_json_str(result))

        return str(call_through(
            call, kind=SpanKind.TOOL, name=name, tool_name=name, input={"arguments": arguments},
        ))

    def db_matches_gold(self) -> bool:
        """Whether the database ends as tau2's gold actions leave it (the live check)."""
        gold = get_environment()
        for action in self.task.evaluation_criteria.actions or ():
            try:
                gold.use_tool(action.name, **action.arguments)
            except Exception:  # noqa: BLE001 - a gold action that fails changes nothing
                pass
        return bool(gold.get_db_hash() == self.env.get_db_hash())


def start(task_id: str) -> Session:
    task = next(t for t in get_tasks("base") if t.id == task_id)
    user = UserSimulator(llm=USER_MODEL, instructions=str(task.user_scenario), llm_args=dict(LLM_ARGS))
    session = Session(
        task_id=task_id, env=get_environment(), task=task, user=user,
        user_state=user.get_init_state(),
        flaky_writes=int(os.environ.get("BENCH_FLAKY_WRITES", "0")),
    )
    return session


def system_message(text: str) -> SystemMessage:
    return SystemMessage(role="system", content=text)


def model(messages: list[Any], tools: list[Any]) -> Any:
    """The support agent's model call."""
    return llm_utils.generate(MODEL, messages, tools=tools, call_name="support_agent", **LLM_ARGS)


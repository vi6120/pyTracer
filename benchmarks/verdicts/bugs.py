"""The seeded bugs: each is a small edit to ``support_agent/glue.py``.

For every bug the benchmark builds a repository whose ``main`` branch carries the
bug, and three candidate branches:

- ``fix``: the correct glue (the bug is fixed);
- ``nofix``: the bug, with a cosmetic edit (nothing fixed);
- ``other``: the bug fixed, and a different bug added.

The expected verdicts are ``fixed``, ``still_failing`` and ``diverged``. They are not
assumed: the benchmark runs every branch live and records what tau2's own check
(the database against the gold actions) and the scenario's assertions say, and
scores pyTracer against that.

Each edit is ``(old, new)``, applied to the correct glue; an edit whose ``old``
text is missing fails loudly, so the catalog cannot drift from the glue.
"""

from __future__ import annotations

from dataclasses import dataclass, field

Edit = tuple[str, str]

_RETRY_CHECK = '''    except TimeoutError:
        if name in ORDER_WRITES:
            # The write may have committed before the timeout: check the order
            # before trying again, so a retry never repeats a write.
            order = session.tool("get_order_details", {"order_id": args["order_id"]})
            if '"status": "pending"' not in order and '"status": "delivered"' not in order:
                return order
        return session.tool(name, args)
'''
_BLIND_RETRY = '''    except TimeoutError:
        return session.tool(name, args)  # a timeout is transient: try again
'''
_PREPARE = '''    """The tool arguments, as the model gave them."""
    return dict(args)
'''
_CALL = '''        return session.tool(name, args)
    except TimeoutError:'''
_HISTORY = '''    """The conversation the model sees: all of it, policy first."""
    return list(history)
'''
_SWALLOW = '''    except Exception as exc:  # noqa: BLE001 - the model reads the error and recovers
        return f"Error: {exc}"
'''


def _prepare(body: str) -> Edit:
    return (_PREPARE, f'    """The tool arguments, as the model gave them."""\n    args = dict(args)\n{body}    return args\n')


@dataclass(frozen=True)
class Bug:
    key: str
    title: str
    task_id: str  # the tau2 retail task the bug is exercised on
    bug: list[Edit]
    cosmetic: list[Edit]  # applied on top of ``bug`` for the nofix branch
    other: list[Edit]  # a different bug, applied to the correct glue
    env: dict[str, str] = field(default_factory=dict)


BUGS: list[Bug] = [
    Bug(
        key="B1",
        title="a write retried after a timeout that had already committed it",
        task_id="38",
        env={"BENCH_FLAKY_WRITES": "1"},  # the first write times out after it commits
        bug=[(_RETRY_CHECK, _BLIND_RETRY)],
        cosmetic=[("# a timeout is transient: try again", "# timeouts are transient, so retry")],
        # Fixed retry, but every cancellation is filed with the wrong reason.
        other=[_prepare('    if name == "cancel_pending_order":\n        args["reason"] = "ordered by mistake"\n')],
    ),
    Bug(
        key="B2",
        title="order ids lose their '#', so every order lookup fails",
        task_id="0",
        bug=[_prepare('    if "order_id" in args:\n        args["order_id"] = str(args["order_id"]).lstrip("#")\n')],
        cosmetic=[('str(args["order_id"]).lstrip("#")', 'str(args["order_id"]).replace("#", "", 1)')],
        # Ids kept, but replacement items are paired with the wrong originals.
        other=[_prepare('    if "new_item_ids" in args:\n        args["new_item_ids"] = list(reversed(args["new_item_ids"]))\n')],
    ),
    Bug(
        key="B3",
        title="tool results cut to 400 characters to save context",
        task_id="1",
        bug=[(_CALL, "        return session.tool(name, args)[:400]\n    except TimeoutError:")],
        cosmetic=[("session.tool(name, args)[:400]", "session.tool(name, args)[: 400]")],
        # Full results, but a tool error reaches the model as success.
        other=[(_SWALLOW, '    except Exception:  # noqa: BLE001\n        return "ok"\n')],
    ),
    Bug(
        key="B4",
        title="only the last 6 messages are sent to the model",
        task_id="5",
        bug=[(_HISTORY, '    """The conversation the model sees: the recent part."""\n    return list(history[-6:])\n')],
        cosmetic=[("list(history[-6:])", "history[-6:][:]")],
        # Whole conversation, but without the policy.
        other=[(_HISTORY, '    """The conversation the model sees: all of it."""\n    return list(history[1:])\n')],
    ),
    Bug(
        key="B5",
        title="a 'batch limit' that returns at most 2 items per call",
        task_id="13",
        bug=[_prepare('    if "item_ids" in args:\n        args["item_ids"] = list(args["item_ids"])[:2]\n')],
        cosmetic=[('list(args["item_ids"])[:2]', 'list(args["item_ids"][:2])')],
        # All items, but the refund's payment method is dropped.
        other=[_prepare('    if name == "return_delivered_order_items":\n        args.pop("payment_method_id", None)\n')],
    ),
]

BRANCHES = ("fix", "nofix", "other")
EXPECTED = {"fix": "fixed", "nofix": "still_failing", "other": "diverged"}


def apply(source: str, edits: list[Edit]) -> str:
    for old, new in edits:
        if source.count(old) != 1:
            raise ValueError(f"edit target not found exactly once: {old[:60]!r}")
        source = source.replace(old, new)
    return source


def variants(bug: Bug, correct: str) -> dict[str, str]:
    """The glue source for the baseline (``main``) and each candidate branch."""
    buggy = apply(correct, bug.bug)
    return {
        "main": buggy,
        "fix": correct,
        "nofix": apply(buggy, bug.cosmetic),
        "other": apply(correct, bug.other),
    }

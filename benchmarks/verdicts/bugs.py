"""The seeded bugs: each is a small edit to ``support_agent/glue.py``.

For every bug the benchmark builds a repository whose ``main`` branch carries the
bug, and three candidate branches:

- ``fix``: the correct glue (the bug is fixed);
- ``nofix``: the bug, with a cosmetic edit (nothing fixed);
- ``other``: the bug fixed, and a different bug added.

The expected verdicts are ``fixed``, ``still_failing`` and ``diverged``. Whether a
branch is fixed is measured, not assumed: the benchmark runs every branch live and
records what tau2's own check (the database against the gold actions) and the
scenario's assertions say. How a failing branch fails follows from how it was
built: ``nofix`` behaves exactly as ``main``, so it is still failing; ``other``
carries a different bug, so if it fails, it diverged. A bug the live model hides
(``main`` passes anyway, or ``fix`` fails anyway) cannot be scored and is reported
as masked.

Each edit is ``(old, new)``, applied to the correct glue; an edit whose ``old``
text is missing fails loudly, so the catalog cannot drift from the glue.

Each bug is tagged with the shape of its fix (see PREREGISTRATION.md): whether the
fix adds a call, changes a call's arguments, changes the model's input, or leaves
every call unchanged. B6 to B10 are of the last kind: they live in
``run_record``, which runs after the conversation, so their scenarios also check
the record the run returns.
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


# run_record's pieces, each the target of a v2 bug (B6 to B10).
_RECORD_WRITES = '        writes.append({"tool": name, "order_id": order_id})\n'
_RECORD_RETURN = '    return {"status": status, "writes": writes}\n'
_RECORD_FAILED = '    failed = [name for name, _, result in made if result.startswith("Error")]\n'
_RECORD_ID = 'ORDER_ID = re.compile(r"#W\\d{7}")\n'
_RECORD_MADE = '    made = [(name, args, result) for name, args, result in calls if name in ORDER_WRITES]\n'

_DROP_LAST = (_RECORD_RETURN, '    return {"status": status, "writes": writes[:-1]}\n')
_ID_LOSES_HASH = (_RECORD_WRITES, '        writes.append({"tool": name, "order_id": order_id.lstrip("#")})\n')
_INVERTED_STATUS = (
    _RECORD_FAILED,
    '    failed = [name for name, _, result in made if not result.startswith("Error")]\n',
)

# B11 to B15: ported from real issues (PREREGISTRATION.md section 3, the seeded draw).
_LOOKUPS = "get_order_details"

# The shape of a bug's fix.
ADDS_CALL = "adds a call"
CHANGES_ARGS = "changes arguments"
CHANGES_MODEL_INPUT = "changes the model's input"
KEEPS_CALLS = "leaves every call unchanged"


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
    shape: str  # the shape of the fix (ADDS_CALL, CHANGES_ARGS, ...)
    env: dict[str, str] = field(default_factory=dict)
    checks_record: bool = False  # the scenario also checks the record the run returns


BUGS: list[Bug] = [
    Bug(
        key="B1",
        shape=ADDS_CALL,
        title="a write retried after a timeout that had already committed it",
        # Task 88: the customer cancels outright once the agent finds no 4-foot
        # shelf. Task 38 left the cancellation to the model's judgement, so most
        # live runs never cancelled and the bug never showed.
        task_id="88",
        env={"BENCH_FLAKY_WRITES": "1"},  # the first write times out after it commits
        bug=[(_RETRY_CHECK, _BLIND_RETRY)],
        cosmetic=[("# a timeout is transient: try again", "# timeouts are transient, so retry")],
        # Fixed retry, but every cancellation is filed with the wrong reason (the
        # task's gold reason is "ordered by mistake").
        other=[_prepare('    if name == "cancel_pending_order":\n        args["reason"] = "no longer needed"\n')],
    ),
    Bug(
        key="B2",
        shape=CHANGES_ARGS,
        title="order ids lose their '#', so every order lookup fails",
        task_id="0",
        bug=[_prepare('    if "order_id" in args:\n        args["order_id"] = str(args["order_id"]).lstrip("#")\n')],
        cosmetic=[('str(args["order_id"]).lstrip("#")', 'str(args["order_id"]).replace("#", "", 1)')],
        # Ids kept, but replacement items are paired with the wrong originals.
        other=[_prepare('    if "new_item_ids" in args:\n        args["new_item_ids"] = list(reversed(args["new_item_ids"]))\n')],
    ),
    Bug(
        key="B3",
        shape=CHANGES_MODEL_INPUT,
        title="tool results cut to 400 characters to save context",
        task_id="1",
        bug=[(_CALL, "        return session.tool(name, args)[:400]\n    except TimeoutError:")],
        cosmetic=[("session.tool(name, args)[:400]", "session.tool(name, args)[: 400]")],
        # Full results, but a tool error reaches the model as success.
        other=[(_SWALLOW, '    except Exception:  # noqa: BLE001\n        return "ok"\n')],
    ),
    Bug(
        key="B4",
        shape=CHANGES_MODEL_INPUT,
        title="only the last 6 messages are sent to the model",
        task_id="5",
        bug=[(_HISTORY, '    """The conversation the model sees: the recent part."""\n    return list(history[-6:])\n')],
        cosmetic=[("list(history[-6:])", "history[-6:][:]")],
        # Whole conversation, but without the policy.
        other=[(_HISTORY, '    """The conversation the model sees: all of it."""\n    return list(history[1:])\n')],
    ),
    Bug(
        key="B5",
        shape=CHANGES_ARGS,
        title="a 'batch limit' that returns at most 2 items per call",
        task_id="13",
        bug=[_prepare('    if "item_ids" in args:\n        args["item_ids"] = list(args["item_ids"])[:2]\n')],
        cosmetic=[('list(args["item_ids"])[:2]', 'list(args["item_ids"][:2])')],
        # All items, but the refund's payment method is dropped.
        other=[_prepare('    if name == "return_delivered_order_items":\n        args.pop("payment_method_id", None)\n')],
    ),
    # v2: bugs whose fix leaves every call unchanged (see PREREGISTRATION.md).
    Bug(
        key="B6",
        title="the run's record drops its last write",
        task_id="88",
        shape=KEEPS_CALLS,
        checks_record=True,
        bug=[_DROP_LAST],
        cosmetic=[("writes[:-1]", "writes[: len(writes) - 1]")],
        other=[_ID_LOSES_HASH],
    ),
    Bug(
        key="B7",
        title="order ids lose their '#' in the run's record",
        task_id="13",
        shape=KEEPS_CALLS,
        checks_record=True,
        bug=[_ID_LOSES_HASH],
        cosmetic=[('order_id.lstrip("#")', 'order_id.replace("#", "", 1)')],
        other=[_DROP_LAST],
    ),
    Bug(
        key="B8",
        title="a write counts as failed when it worked",
        task_id="0",
        shape=KEEPS_CALLS,
        checks_record=True,
        bug=[_INVERTED_STATUS],
        cosmetic=[('not result.startswith("Error")', 'result[:5] != "Error"')],
        other=[_DROP_LAST],
    ),
    Bug(
        key="B9",
        title="a guard rejects valid order ids",
        task_id="1",
        shape=KEEPS_CALLS,
        checks_record=True,
        bug=[(_RECORD_ID, 'ORDER_ID = re.compile(r"#W\\d{6}")\n')],
        cosmetic=[('r"#W\\d{6}"', 'r"#W[0-9]{6}"')],
        other=[_ID_LOSES_HASH],
    ),
    Bug(
        key="B10",
        title="cancellations are left out of the run's record",
        task_id="88",
        shape=KEEPS_CALLS,
        checks_record=True,
        bug=[(_RECORD_MADE, _RECORD_MADE.replace("in ORDER_WRITES]", "in ORDER_WRITES[1:]]"))],
        cosmetic=[("ORDER_WRITES[1:]]", "ORDER_WRITES[1:len(ORDER_WRITES)]]")],
        other=[_INVERTED_STATUS],
    ),
    # v2: ported from real, fixed issues in agent frameworks (the seeded draw).
    Bug(
        key="B11",
        # openai-agents-python#879: an upstream HTTP error from a tool is raised and
        # ends the run instead of reaching the model as an error it can handle.
        title="an upstream error from a tool ends the run instead of reaching the model",
        task_id="13",
        shape=ADDS_CALL,  # the fixed branch goes on to calls the bug never reaches
        env={"BENCH_FLAKY_LOOKUPS": "1"},  # the first order lookup fails upstream (502)
        bug=[(_SWALLOW, '    except Exception as exc:  # noqa: BLE001 - the model reads the error and recovers\n'
                        '        if "upstream" in str(exc):\n'
                        '            raise  # a failing upstream service is not the tool\'s error: give up\n'
                        '        return f"Error: {exc}"\n')],
        cosmetic=[('if "upstream" in str(exc):', 'if str(exc).find("upstream") >= 0:')],
        other=[_prepare('    if "order_id" in args:\n        args["order_id"] = str(args["order_id"]).lstrip("#")\n')],
    ),
    Bug(
        key="B12",
        # llama_index#21124: assistant text that comes with tool calls is dropped
        # when the conversation is sent back to the model.
        title="text the model wrote alongside a tool call is dropped from the history",
        task_id="0",
        shape=CHANGES_MODEL_INPUT,
        bug=[(_HISTORY, '    """The conversation the model sees: all of it, policy first."""\n'
                        '    return [m.model_copy(update={"content": None}) if getattr(m, "tool_calls", None) else m\n'
                        '            for m in history]\n')],
        cosmetic=[('if getattr(m, "tool_calls", None) else m', 'if getattr(m, "tool_calls", None) is not None and m.tool_calls else m')],
        # Text kept, but the customer's first message is dropped.
        other=[(_HISTORY, '    """The conversation the model sees: all of it, policy first."""\n'
                          '    return list(history[:1]) + list(history[2:])\n')],
    ),
    Bug(
        key="B13",
        # openai-agents-python#2110: the caller gets an empty string instead of the
        # tool's result.
        title="the model gets an empty string instead of a tool's result",
        task_id="88",
        shape=CHANGES_MODEL_INPUT,
        bug=[(_CALL, f'        result = session.tool(name, args)\n'
                     f'        return "" if name == "{_LOOKUPS}" else result\n    except TimeoutError:')],
        cosmetic=[(f'"" if name == "{_LOOKUPS}" else result', f'"" if name in ("{_LOOKUPS}",) else result')],
        # Results kept, but every cancellation is filed with the wrong reason.
        other=[_prepare('    if name == "cancel_pending_order":\n        args["reason"] = "no longer needed"\n')],
    ),
    Bug(
        key="B14",
        # crewAI#5275 (redrawn for #5460, which cannot show in tau2): the arguments
        # are looked up under a key with a truthy default ("{}"), so the fallback to
        # the real arguments never runs and the tool is called with none.
        title="tool arguments silently dropped: the tool runs with empty input",
        task_id="0",
        shape=CHANGES_ARGS,
        bug=[_prepare('    import json\n'
                      '    found = args.get("arguments", "{}") or args  # "{}" is truthy: never falls back\n'
                      '    args = json.loads(found) if isinstance(found, str) else dict(found)\n')],
        cosmetic=[('args.get("arguments", "{}")', 'args.get("arguments", "{" + "}")')],
        # Arguments kept, but replacement items are paired with the wrong originals.
        other=[_prepare('    if "new_item_ids" in args:\n        args["new_item_ids"] = list(reversed(args["new_item_ids"]))\n')],
    ),
    Bug(
        key="B15",
        # llama_index#20872: the system prompt (here, the retail policy) never
        # reaches the model.
        title="the policy (system prompt) never reaches the model",
        task_id="13",
        shape=CHANGES_MODEL_INPUT,
        bug=[(_HISTORY, '    """The conversation the model sees: all of it."""\n    return list(history[1:])\n')],
        cosmetic=[("list(history[1:])", "history[1:][:]")],
        # Policy kept, but tool results are dropped, so the tool calls have no answer.
        other=[(_HISTORY, '    """The conversation the model sees: all of it, policy first."""\n'
                          '    return [m for m in history if getattr(m, "role", None) != "tool"]\n')],
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

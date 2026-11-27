"""Runs inside one branch's checkout: capture the baseline, or one live check.

``python child.py capture <checkout> <task_id> <trace.jsonl> <collection.yaml>``
    Runs the agent live, writes its spans as the recording to replay, and prints
    the same JSON line as ``live`` for that run, so the harness can check that the
    recording shows the bug.

``python child.py live <checkout> <task_id> <collection.yaml>``
    Runs the agent live and prints one JSON line: each assertion of the
    scenario (judged on the live run), tau2's own check (does the database end
    as the gold actions leave it), the error if the run raised, and the tokens.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pytracer_replay import ReplayResult


def _agent(checkout: str) -> object:
    sys.path.insert(0, checkout)
    import support_agent

    return support_agent


def capture(checkout: str, task_id: str, trace: str, collection: str) -> None:
    import pytracer_sdk as pytracer
    from pytracer_replay import ReplayMode, ReplayResult
    from pytracer_sdk import SpanStatus
    from pytracer_sdk.transport import InMemoryTransport

    agent = _agent(checkout)
    sink = InMemoryTransport()
    pytracer.configure(sink, service_name="seeded-bugs", branch="main", commit="baseline")
    error: Exception | None = None
    returned: object = None
    try:
        returned = agent.run(task_id)  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001 - a failing baseline is still recorded
        if type(exc).__name__ == "BudgetExceeded":
            sys.exit(f"stopped: {exc}")
        print(f"baseline raised {type(exc).__name__}: {exc}", file=sys.stderr)
        error = exc
    pytracer.get_tracer().flush()
    Path(trace).parent.mkdir(parents=True, exist_ok=True)
    Path(trace).write_text("".join(s.to_json() + "\n" for s in sink.spans), encoding="utf-8")
    result = ReplayResult(
        return_value=returned,  # the run's record, which B6 to B10 check
        status=SpanStatus.ERROR if error else SpanStatus.OK,
        mode=ReplayMode.LIVE,
        spans=sorted(sink.spans, key=lambda s: s.start_time_ns),
        error_type=type(error).__name__ if error else None,
        error_message=str(error) if error else None,
    )
    _report(agent, result, collection)


def live(checkout: str, task_id: str, collection: str) -> None:
    from pytracer_replay import MockSet, ReplayEngine, ReplayMode

    agent = _agent(checkout)
    result = ReplayEngine(ReplayMode.LIVE).replay(lambda: agent.run(task_id), MockSet())  # type: ignore[attr-defined]
    if result.error_type == "BudgetExceeded":
        sys.exit(f"stopped: {result.error_message}")  # not a run outcome; the benchmark stops
    _report(agent, result, collection)


def _report(agent: object, result: ReplayResult, collection: str) -> None:
    """Print one JSON line: the scenario's assertions, tau2's check, the error, the tokens."""
    from pytracer_cli.collection import load_collection
    from pytracer_cli.runner import build_assertion
    from pytracer_replay import run_assertions

    loaded, _ = load_collection(Path(collection))
    scenario = loaded.scenarios[0]
    assertions = [build_assertion(a, scenario.writes) for a in scenario.assertions]
    checks = run_assertions(result, assertions)
    session = getattr(agent.run, "last_session", None)  # type: ignore[attr-defined]
    print(json.dumps({
        "assertions": {c.name: {"passed": c.passed, "detail": c.detail} for c in checks},
        "db_matches_gold": bool(session.db_matches_gold()) if session is not None else False,
        "error_type": result.error_type,
        "tokens": sum(s.token_count or 0 for s in result.spans),
    }))


if __name__ == "__main__":
    mode, *rest = sys.argv[1:]
    if mode == "capture":
        capture(*rest)
    else:
        live(*rest)

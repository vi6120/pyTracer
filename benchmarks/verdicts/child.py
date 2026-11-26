"""Runs inside one branch's checkout: capture the baseline, or one live check.

``python child.py capture <checkout> <task_id> <trace.jsonl>``
    Runs the agent live and writes its spans as the recording to replay.

``python child.py live <checkout> <task_id> <collection.yaml>``
    Runs the agent live and prints one JSON line: each assertion of the
    scenario (judged on the live run), tau2's own check (does the database end
    as the gold actions leave it), the error if the run raised, and the tokens.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _agent(checkout: str) -> object:
    sys.path.insert(0, checkout)
    import support_agent

    return support_agent


def capture(checkout: str, task_id: str, trace: str) -> None:
    import pytracer_sdk as pytracer
    from pytracer_sdk.transport import InMemoryTransport

    agent = _agent(checkout)
    sink = InMemoryTransport()
    pytracer.configure(sink, service_name="seeded-bugs", branch="main", commit="baseline")
    try:
        agent.run(task_id)  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001 - a failing baseline is still recorded
        print(f"baseline raised {type(exc).__name__}: {exc}", file=sys.stderr)
    pytracer.get_tracer().flush()
    Path(trace).parent.mkdir(parents=True, exist_ok=True)
    Path(trace).write_text("".join(s.to_json() + "\n" for s in sink.spans), encoding="utf-8")


def live(checkout: str, task_id: str, collection: str) -> None:
    from pytracer_cli.collection import load_collection
    from pytracer_cli.runner import build_assertion
    from pytracer_replay import MockSet, ReplayEngine, ReplayMode, run_assertions

    agent = _agent(checkout)
    loaded, _ = load_collection(Path(collection))
    scenario = loaded.scenarios[0]
    result = ReplayEngine(ReplayMode.LIVE).replay(lambda: agent.run(task_id), MockSet())  # type: ignore[attr-defined]
    checks = run_assertions(result, [build_assertion(a, scenario.writes) for a in scenario.assertions])
    session = getattr(agent.run, "last_session", None)  # type: ignore[attr-defined]
    print(json.dumps({
        "assertions": {c.name: {"passed": c.passed, "detail": c.detail} for c in checks},
        "db_matches_gold": bool(session.db_matches_gold()) if session is not None else False,
        "error_type": result.error_type,
        "tokens": sum(s.token_count or 0 for s in result.spans),
    }))


if __name__ == "__main__":
    mode, *rest = sys.argv[1:]
    {"capture": capture, "live": live}[mode](*rest)

"""Seeded-bug verdict benchmark: does pyTracer's verdict match what really happened?

For each seeded bug (``bugs.py``) on its tau2-bench retail task:

1. Build a throwaway git repository: ``main`` carries the bug; ``fix``, ``nofix``
   and ``other`` branch from it (see ``bugs.py``).
2. Capture ``main`` live once: the recording every replay freezes against.
3. Ground truth: run every branch live ``--repeats`` times and judge each run by
   the scenario's assertions and tau2's own database check. A branch whose runs
   pass is truly fixed; one that fails the way ``main`` fails is still failing;
   one that fails another way diverged (the majority over its repeats).
4. Run ``pytracer reproduce`` for every branch under each setting:
   ``strict`` (everything frozen, strict), ``fallback`` (frozen but not strict: a
   call with no recording runs live, a changed model call too with
   ``--evaluate-changed-prompts``) and ``partial`` (tools frozen, the model live).
5. Score each reported verdict against the ground truth: correct, an abstention
   (``cannot_judge`` / ``cannot_reproduce``, with its reason), or wrong; and count
   the unsafe errors, a branch that is not fixed reported ``fixed``.

With ``--max-usd`` every live model call is charged to ``<out>/usage.jsonl`` at the
prices in ``BENCH_PRICE_IN`` / ``BENCH_PRICE_OUT`` (USD per million tokens; all input
at the full price, so the real bill is at or below the ledger), and the benchmark
stops before a call that would start past the cap. The ledger persists across
runs in the same ``--out`` folder, so one budget covers several invocations.

Needs tau2-bench installed (``TAU2_DATA_DIR`` pointing at its ``data`` folder), the
pyTracer CLI, and the model's API key in the environment (``BENCH_MODEL``, default
``anthropic/claude-haiku-5-5``). The live steps call the model; ``--plan`` prints
what would run without calling it.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

from bugs import BRANCHES, BUGS, Bug, variants

HERE = Path(__file__).resolve().parent
AGENT = HERE / "support_agent"
SETTINGS = {
    "strict": {"mode": "full", "strict": True, "flags": []},
    "fallback": {"mode": "full", "strict": False, "flags": ["--evaluate-changed-prompts"]},
    "partial": {"mode": "partial", "strict": False, "flags": []},
}
ABSTAIN = {"cannot_judge", "cannot_reproduce"}


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _write_tools() -> list[str]:
    source = (AGENT / "harness.py").read_text()
    start = source.index("WRITE_TOOLS = (")
    block = source[start:source.index(")", start)]
    return [line.strip().strip('",') for line in block.splitlines()[1:] if line.strip()]


def _collection(bug: Bug, setting: str) -> dict[str, Any]:
    tasks = json.loads((Path(os.environ["TAU2_DATA_DIR"]) / "tau2/domains/retail/tasks.json").read_text())
    task = next(t for t in tasks if t["id"] == bug.task_id)
    writes = _write_tools()
    gold = [a for a in task["evaluation_criteria"]["actions"] if a["name"] in writes]
    expected = [{"tool": a["name"], "times": 1, "args": a["arguments"]} for a in gold]
    config = SETTINGS[setting]
    scenario: dict[str, Any] = {
        "name": f"task-{bug.task_id}",
        "agent": "support_agent:run",
        "entry": {"task_id": bug.task_id},
        "trace": "traces/run.jsonl",
        "mode": config["mode"],
        "strict": config["strict"],
        "writes": writes,
        "assertions": [
            {"type": "no_errors"},
            {"type": "writes", "only": True, "expected": expected},
            {"type": "no_duplicate_writes"},
        ],
    }
    return {"name": f"seeded-{bug.key}-{setting}", "scenarios": [scenario]}


def build_repo(bug: Bug, root: Path) -> Path:
    repo = root / bug.key
    (repo / "support_agent").mkdir(parents=True)
    for name in ("__init__.py", "harness.py"):
        shutil.copy(AGENT / name, repo / "support_agent" / name)
    for setting in SETTINGS:
        (repo / f"collection-{setting}.yaml").write_text(yaml.safe_dump(_collection(bug, setting), sort_keys=False))
    glue = variants(bug, (AGENT / "glue.py").read_text())
    (repo / "support_agent" / "glue.py").write_text(glue["main"])
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "-c", "user.name=bench", "-c", "user.email=bench@pytracer.local", "add", "-A")
    _git(repo, "-c", "user.name=bench", "-c", "user.email=bench@pytracer.local", "commit", "-qm", f"{bug.key}: {bug.title}")
    for branch in BRANCHES:
        _git(repo, "checkout", "-q", "-b", branch, "main")
        (repo / "support_agent" / "glue.py").write_text(glue[branch])
        _git(repo, "-c", "user.name=bench", "-c", "user.email=bench@pytracer.local", "commit", "-qam", f"{branch}: {bug.key}")
    _git(repo, "checkout", "-q", "main")
    return repo


class Stop(Exception):
    """The spending cap was reached; nothing more runs."""


def _env(bug: Bug) -> dict[str, str]:
    return {**os.environ, **bug.env}


def _spent() -> float:
    ledger = os.environ.get("BENCH_USAGE_LEDGER")
    if not ledger or not Path(ledger).exists():
        return 0.0
    return sum(json.loads(line)["usd"] for line in Path(ledger).read_text().splitlines() if line.strip())


def _check_budget() -> None:
    cap = float(os.environ.get("BENCH_MAX_USD") or 0)
    if cap and _spent() >= cap:
        raise Stop(f"spending cap of ${cap:.2f} reached: ${_spent():.2f} charged")


def _checkout(repo: Path, branch: str, root: Path) -> Path:
    path = root / f"{repo.name}-{branch}"
    if not path.exists():
        _git(repo, "worktree", "add", "-q", str(path), branch)
    return path


def capture(bug: Bug, repo: Path, out: Path) -> Path:
    trace = out / bug.key / "trace.jsonl"
    done = subprocess.run(
        [sys.executable, str(HERE / "child.py"), "capture", str(repo), bug.task_id, str(trace)],
        env=_env(bug),
    )
    _check_budget()
    if done.returncode != 0:
        raise RuntimeError(f"capture of {bug.key} failed")
    return trace


def live_truth(bug: Bug, repo: Path, root: Path, repeats: int) -> dict[str, Any]:
    runs: dict[str, list[dict[str, Any]]] = {}
    for branch in ("main", *BRANCHES):
        checkout = repo if branch == "main" else _checkout(repo, branch, root)
        runs[branch] = []
        for _ in range(repeats):
            done = subprocess.run(
                [sys.executable, str(HERE / "child.py"), "live", str(checkout), bug.task_id,
                 str(checkout / "collection-strict.yaml")],
                env=_env(bug), capture_output=True, text=True,
            )
            # The agent's own failures are inside the JSON; a child that exits
            # non-zero is a broken harness, and must stop the benchmark rather
            # than count as a failed run.
            _check_budget()
            lines = [line for line in done.stdout.splitlines() if line.startswith("{")]
            if done.returncode != 0 or not lines:
                raise RuntimeError(f"live check of {bug.key}/{branch} failed:\n{done.stderr[-2000:]}")
            runs[branch].append(json.loads(lines[-1]))
    return {"runs": runs, "labels": _labels(runs)}


def _signature(run: dict[str, Any]) -> tuple[Any, ...]:
    """How a live run failed: which assertions failed and with what detail (as
    pyTracer compares failures), whether the database matched the gold one, and
    the error. Names alone are too coarse: an order never exchanged and one
    exchanged wrongly both fail ``writes``, with different details."""
    failed = sorted((n, a["detail"]) for n, a in (run.get("assertions") or {}).items() if not a["passed"])
    return (tuple(failed), run.get("db_matches_gold"), run.get("error_type"))


def _passed(run: dict[str, Any]) -> bool:
    checks = run.get("assertions") or {}
    return bool(checks) and all(a["passed"] for a in checks.values()) and bool(run.get("db_matches_gold"))


def _labels(runs: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Each candidate's true outcome, by majority over its live repeats."""
    baseline = collections.Counter(_signature(r) for r in runs["main"] if not _passed(r))
    main_signature = baseline.most_common(1)[0][0] if baseline else None
    labels: dict[str, Any] = {"baseline_fails": sum(not _passed(r) for r in runs["main"])}
    for branch in BRANCHES:
        votes = collections.Counter(
            "fixed" if _passed(r) else ("still_failing" if _signature(r) == main_signature else "diverged")
            for r in runs[branch]
        )
        label, count = votes.most_common(1)[0]
        labels[branch] = {"label": label, "votes": dict(votes), "unanimous": count == len(runs[branch])}
    return labels


def reproduce(bug: Bug, repo: Path, trace: Path, setting: str) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for branch in BRANCHES:
        command = [
            "pytracer", "reproduce", "--repo", str(repo), "--baseline", "main", "--branch", branch,
            "--collection", f"collection-{setting}.yaml", "--isolation", "local", "--no-install",
            "--freeze-trace", str(trace), "--trace-dest", "traces/run.jsonl", *SETTINGS[setting]["flags"],
        ]
        done = subprocess.run(command, env=_env(bug), capture_output=True, text=True)
        _check_budget()  # a reproduce stopped by the cap has no verdict to score
        verdict, reason = None, None
        name = f"task-{bug.task_id}"
        for line in done.stdout.splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0] == name:
                verdict = parts[1]
            elif line.startswith(f"{name}: "):
                reason = line[len(name) + 2:]
        results[branch] = {"verdict": verdict, "reason": reason, "exit": done.returncode}
        if verdict is None:
            results[branch]["output"] = (done.stdout + done.stderr)[-800:]
    return results


def score(truth: dict[str, Any], reported: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for branch in BRANCHES:
        label, verdict = truth[branch]["label"], reported[branch]["verdict"]
        if verdict in ABSTAIN:
            outcome = "abstained"
        elif verdict == label:
            outcome = "correct"
        else:
            outcome = "wrong"
        out[branch] = {
            "truth": label, "verdict": verdict, "outcome": outcome,
            "unsafe": verdict == "fixed" and label != "fixed",
        }
    return out


def summary(results: dict[str, Any]) -> str:
    lines = ["# Seeded-bug verdict benchmark", ""]
    for setting in SETTINGS:
        rows = [(k, b, r["scores"][setting][b]) for k, r in results.items() if setting in r["scores"] for b in BRANCHES]
        if not rows:
            continue
        counts = collections.Counter(s["outcome"] for _, _, s in rows)
        unsafe = sum(s["unsafe"] for _, _, s in rows)
        lines += [
            f"## {setting}", "",
            f"Broken branches reported fixed: **{unsafe}** of {sum(s['truth'] != 'fixed' for _, _, s in rows)}. "
            f"Correct {counts['correct']}, abstained {counts['abstained']}, wrong {counts['wrong']} (of {len(rows)}).",
            "", "| Bug | Branch | True outcome | pyTracer | Result | Reason |", "|---|---|---|---|---|---|",
        ]
        for key, branch, s in rows:
            reason = (results[key]["reproduce"][setting][branch].get("reason") or "").replace("|", "/")
            lines.append(f"| {key} | {branch} | {s['truth']} | {s['verdict']} | {s['outcome']} | {reason} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bugs", default=",".join(b.key for b in BUGS))
    parser.add_argument("--settings", default=",".join(SETTINGS))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    parser.add_argument("--plan", action="store_true", help="print what would run, call no model")
    parser.add_argument("--max-usd", type=float, default=0.0, help="stop before spending more than this")
    args = parser.parse_args()
    if args.max_usd:
        if not (os.environ.get("BENCH_PRICE_IN") and os.environ.get("BENCH_PRICE_OUT")):
            sys.exit("--max-usd needs BENCH_PRICE_IN and BENCH_PRICE_OUT (USD per million tokens)")
        args.out.mkdir(parents=True, exist_ok=True)
        os.environ["BENCH_MAX_USD"] = str(args.max_usd)
        os.environ["BENCH_USAGE_LEDGER"] = str((args.out / "usage.jsonl").resolve())
        print(f"spending cap ${args.max_usd:.2f}; already charged ${_spent():.2f}")
    chosen = [b for b in BUGS if b.key in args.bugs.split(",")]
    settings = [s for s in args.settings.split(",") if s in SETTINGS]
    if "TAU2_DATA_DIR" not in os.environ:
        sys.exit("set TAU2_DATA_DIR to tau2-bench's data folder")
    root = Path(tempfile.mkdtemp(prefix="pytracer-seeded-"))
    results: dict[str, Any] = {}
    try:
        _run_bugs(args, chosen, settings, root, results)
    except Stop as stop:
        print(f"stopped: {stop}")
    if results:
        (args.out / "summary.md").write_text(summary(results))
        print(summary(results))
    if args.max_usd:
        print(f"charged so far (conservative): ${_spent():.2f} of ${args.max_usd:.2f}")
    shutil.rmtree(root, ignore_errors=True)


def _run_bugs(
    args: argparse.Namespace, chosen: list[Bug], settings: list[str], root: Path, results: dict[str, Any]
) -> None:
    for bug in chosen:
        repo = build_repo(bug, root)
        if args.plan:
            live_runs = 1 + 4 * args.repeats
            print(f"{bug.key} task {bug.task_id}: repo {repo}; {live_runs} live runs, "
                  f"then reproduce x{len(BRANCHES)} per setting {settings}")
            continue
        args.out.mkdir(parents=True, exist_ok=True)
        trace = capture(bug, repo, args.out)
        truth = live_truth(bug, repo, root, args.repeats)
        reported = {s: reproduce(bug, repo, trace, s) for s in settings}
        results[bug.key] = {
            "title": bug.title, "task": bug.task_id, "truth": truth, "reproduce": reported,
            "scores": {s: score(truth["labels"], reported[s]) for s in settings},
        }
        (args.out / f"{bug.key}.json").write_text(json.dumps(results[bug.key], indent=1))


if __name__ == "__main__":
    main()

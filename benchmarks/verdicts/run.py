"""Seeded-bug verdict benchmark: does pyTracer's verdict match what really happened?

For each seeded bug (``bugs.py``) on its tau2-bench retail task:

1. Build a throwaway git repository: ``main`` carries the bug; ``fix``, ``nofix``
   and ``other`` branch from it (see ``bugs.py``).
2. Ground truth: run every branch live ``--repeats`` times and judge each run by
   the scenario's assertions and tau2's own database check. A branch whose runs
   pass (by majority) is truly fixed. How a failing branch fails follows from how
   it was built, not from comparing live runs (a live model fails a little
   differently every time): ``nofix`` behaves exactly as ``main``, so it is still
   failing; ``other`` carries a different bug, so it diverged. A bug the live
   model hides cannot be scored: if ``main`` passes anyway (the bug never shows)
   or ``fix`` fails anyway (the model fails the task without the bug), the bug is
   reported as masked, with its live counts, and left out of the score.
3. Record ``main`` live: the recording every replay freezes against. It must show
   the bug, failing the same checks as most of ``main``'s live runs; a try that
   fails another way is recorded again, up to ``CAPTURE_TRIES`` times, and a bug
   no try shows is reported as masked too.
4. Run ``pytracer reproduce`` for every branch under each setting:
   ``strict`` (everything frozen, strict), ``fallback`` (frozen but not strict: a
   call with no recording runs live, a changed model call too with
   ``--evaluate-changed-prompts``) and ``partial`` (tools frozen, the model live).
5. Score each reported verdict against the ground truth: correct, an abstention
   (``cannot_judge`` / ``cannot_reproduce``, with its reason), or wrong; and count
   the unsafe errors, a branch that is not fixed reported ``fixed``. In ``partial``
   the model runs live, so its verdicts are not frozen results; pyTracer says so.

``--rescore <folder>`` recomputes the labels and scores of a finished run from its
saved JSON, calling no model.

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
import math
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
# Seconds each reproduce run may take: with the model live (partial, or a call that
# falls back to live), a slow model needs far longer than pyTracer's default.
REPRODUCE_TIMEOUT = os.environ.get("BENCH_REPRODUCE_TIMEOUT", "3600")
# How many times the baseline is recorded before giving up on a recording that
# shows the bug (a live model does not fail the same way every run).
CAPTURE_TRIES = 3


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
    if bug.checks_record:
        # The record the run returns must list the gold writes, as sent, and say
        # they worked (B6 to B10 live in the code that builds it).
        record = [{"tool": a["name"], "order_id": a["arguments"]["order_id"]} for a in gold]
        scenario["assertions"].append(
            {"type": "returns", "name": "record", "expected": {"status": "resolved", "writes": record}}
        )
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


def capture(bug: Bug, repo: Path, out: Path, main_runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Record the baseline live until the recording fails the way ``main`` fails.

    A recording is only worth judging if it shows the bug. A live model sometimes
    fails another way (it never reaches the write the bug is in), and a recording of
    that failure cannot tell a fix from no fix. So each try is judged like a live run
    and kept only if its failure matches the most common failure of ``main``'s live
    runs; after ``CAPTURE_TRIES`` misses the bug is reported as not captured.
    """
    trace = out / bug.key / "trace.jsonl"
    target = _bug_failure(main_runs)
    tries: list[dict[str, Any]] = []
    for _ in range(CAPTURE_TRIES):
        done = subprocess.run(
            [sys.executable, str(HERE / "child.py"), "capture", str(repo), bug.task_id, str(trace),
             str(repo / "collection-strict.yaml")],
            env=_env(bug), capture_output=True, text=True,
        )
        _check_budget()
        lines = [line for line in done.stdout.splitlines() if line.startswith("{")]
        if done.returncode != 0 or not lines:
            raise RuntimeError(f"capture of {bug.key} failed:\n{done.stderr[-2000:]}")
        tries.append(json.loads(lines[-1]))
        if target is not None and _failure(tries[-1]) == target:
            return {"trace": str(trace), "tries": tries, "shows_bug": True}
    return {"trace": str(trace), "tries": tries, "shows_bug": False}


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


def _passed(run: dict[str, Any]) -> bool:
    checks = run.get("assertions") or {}
    return bool(checks) and all(a["passed"] for a in checks.values()) and bool(run.get("db_matches_gold"))


def _passes(runs: list[dict[str, Any]]) -> str:
    return f"{sum(_passed(r) for r in runs)}/{len(runs)}"


def _majority_pass(runs: list[dict[str, Any]]) -> bool:
    return 2 * sum(_passed(r) for r in runs) > len(runs)


def _failure(run: dict[str, Any]) -> tuple[str, ...] | None:
    """How a run failed: its failed checks, by name. None if it passed."""
    if _passed(run):
        return None
    failed = sorted(name for name, a in (run.get("assertions") or {}).items() if not a["passed"])
    return (*failed, *([] if run.get("db_matches_gold") else ["db_matches_gold"]))


def _bug_failure(runs: list[dict[str, Any]]) -> tuple[str, ...] | None:
    """The most common failure among the runs (the first seen on a tie)."""
    failures = [f for f in map(_failure, runs) if f is not None]
    return collections.Counter(failures).most_common(1)[0][0] if failures else None


def _labels(runs: dict[str, list[dict[str, Any]]], captured: dict[str, Any] | None = None) -> dict[str, Any]:
    """Each candidate's true outcome. Whether it is fixed is measured live (the
    majority of its repeats); how a failing branch fails follows from how it was
    built (see the module docstring). ``masked`` says why a bug cannot be scored."""
    labels: dict[str, Any] = {"passes": {b: _passes(r) for b, r in runs.items()}, "masked": None}
    if _majority_pass(runs["main"]):
        labels["masked"] = "main passes live: the bug does not show on this task"
    elif not _majority_pass(runs["fix"]):
        labels["masked"] = "fix fails live: the model fails the task without the bug"
    elif captured is not None and not captured["shows_bug"]:
        labels["masked"] = (
            f"no recording showed the bug: {len(captured['tries'])} tries, none failed as main fails live"
        )
    built = {"fix": "fixed", "nofix": "still_failing", "other": "diverged"}
    for branch in BRANCHES:
        label = "fixed" if _majority_pass(runs[branch]) else built[branch]
        if branch == "nofix":
            label = "still_failing"  # a cosmetic edit: it behaves exactly as main
        labels[branch] = {"label": label, "passes": labels["passes"][branch]}
    return labels


def reproduce(bug: Bug, repo: Path, trace: Path, setting: str) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for branch in BRANCHES:
        command = [
            "pytracer", "reproduce", "--repo", str(repo), "--baseline", "main", "--branch", branch,
            "--collection", f"collection-{setting}.yaml", "--isolation", "local", "--no-install",
            "--freeze-trace", str(trace), "--trace-dest", "traces/run.jsonl",
            "--timeout", REPRODUCE_TIMEOUT, *SETTINGS[setting]["flags"],
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
            # pyTracer's own output, without tau2's debug log, so the cause is kept.
            errors = [line for line in done.stderr.splitlines() if "| DEBUG" not in line]
            results[branch]["stdout"] = done.stdout[-3000:]
            results[branch]["stderr"] = "\n".join(errors)[-3000:]
    return results


def score(truth: dict[str, Any], reported: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for branch in BRANCHES:
        label, verdict = truth[branch]["label"], reported[branch]["verdict"]
        if truth.get("masked"):
            outcome = "masked"
        elif verdict in ABSTAIN:
            outcome = "abstained"
        elif verdict == label:
            outcome = "correct"
        else:
            outcome = "wrong"
        out[branch] = {
            "truth": label, "verdict": verdict, "outcome": outcome,
            "unsafe": outcome != "masked" and verdict == "fixed" and label != "fixed",
        }
    return out


def summary(results: dict[str, Any]) -> str:
    lines = ["# Seeded-bug verdict benchmark", ""]
    masked = {k: r["truth"]["labels"] for k, r in results.items() if r["truth"]["labels"].get("masked")}
    if masked:
        lines += ["## Masked by the model (not scored)", "", "| Bug | Task | Why | Live passes |", "|---|---|---|---|"]
        for key, labels in masked.items():
            passes = ", ".join(f"{b} {n}" for b, n in labels["passes"].items())
            lines.append(f"| {key} | {results[key]['task']} | {labels['masked']} | {passes} |")
        lines.append("")
    for setting in SETTINGS:
        rows = [
            (k, b, r["scores"][setting][b])
            for k, r in sorted(results.items(), key=lambda kv: _bug_order(kv[0]))
            if setting in r["scores"] and k not in masked
            for b in BRANCHES
        ]
        if not rows:
            continue
        unsafe = sum(s["unsafe"] for _, _, s in rows)
        broken = sum(s["truth"] != "fixed" for _, _, s in rows)
        lines += [f"## {setting}", ""]
        if SETTINGS[setting]["mode"] != "full":
            lines += ["The model runs live here, so these verdicts are not frozen results.", ""]
        lines += [
            f"Broken branches reported fixed: **{unsafe}** of {broken} "
            f"(95% upper bound on the unsafe rate: {_pct(_interval(unsafe, broken)[1])}).",
            "", *_scores(rows), "", *_by_shape(results, rows), "",
            "| Bug | Branch | True outcome | Live passes | pyTracer | Result | Reason |",
            "|---|---|---|---|---|---|---|",
        ]
        for key, branch, s in rows:
            reported = results[key]["reproduce"][setting][branch]
            reason = (reported.get("reason") or "").replace("|", "/")
            if reported.get("verdict") is None:
                reason = reason or _first_error(reported.get("stderr") or "")
            passes = results[key]["truth"]["labels"][branch].get("passes", "")
            lines.append(f"| {key} | {branch} | {s['truth']} | {passes} | {s['verdict']} | {s['outcome']} | {reason} |")
        lines.append("")
    return "\n".join(lines)


def _bug_order(key: str) -> tuple[int, str]:
    """B2 before B10."""
    digits = "".join(c for c in key if c.isdigit())
    return (int(digits) if digits else 0, key)


def _binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p)."""
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))


def _interval(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Two-sided exact (Clopper-Pearson) interval for k successes in n trials."""
    if n == 0:
        return 0.0, 1.0

    def solve(f: Any, target: float) -> float:  # f(p) rises with p: find f(p) = target
        lo, hi = 0.0, 1.0
        for _ in range(80):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) < target else (lo, mid)
        return (lo + hi) / 2

    # Lower: P(X >= k) = alpha/2. Upper: P(X <= k) = alpha/2, i.e. P(X > k) = 1 - alpha/2.
    lower = 0.0 if k == 0 else solve(lambda p: 1 - _binom_cdf(k - 1, n, p), alpha / 2)
    upper = 1.0 if k == n else solve(lambda p: 1 - _binom_cdf(k, n, p), 1 - alpha / 2)
    return lower, upper


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def _share(k: int, n: int) -> str:
    lo, hi = _interval(k, n)
    return f"{k} of {n} | {_pct(k / n) if n else '-'} | {_pct(lo)} to {_pct(hi)}"


def _scores(rows: list[tuple[str, str, dict[str, Any]]]) -> list[str]:
    """Correct, abstained and wrong with 95% intervals, and how far `fixed` can be trusted."""
    n = len(rows)
    counts = collections.Counter(s["outcome"] for _, _, s in rows)
    said_fixed = [s for _, _, s in rows if s["verdict"] == "fixed"]
    truly_fixed = [s for _, _, s in rows if s["truth"] == "fixed"]
    right = sum(s["truth"] == "fixed" for s in said_fixed)
    caught = sum(s["verdict"] == "fixed" for s in truly_fixed)
    return [
        "| Verdicts | Count | Share | 95% interval |",
        "|---|---|---|---|",
        f"| Correct | {_share(counts['correct'], n)} |",
        f"| Abstained | {_share(counts['abstained'], n)} |",
        f"| Wrong | {_share(counts['wrong'], n)} |",
        f"| `fixed` precision (reported fixed, and fixed) | {_share(right, len(said_fixed))} |",
        f"| `fixed` recall (fixed, and reported fixed) | {_share(caught, len(truly_fixed))} |",
    ]


def _by_shape(results: dict[str, Any], rows: list[tuple[str, str, dict[str, Any]]]) -> list[str]:
    """The same verdicts split by the shape of each bug's fix (bugs.py)."""
    shapes = {b.key: b.shape for b in BUGS}
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for key, _, s in rows:
        groups[results[key].get("shape") or shapes.get(key, "unknown")].append(s)
    lines = [
        "| Shape of the fix | Branches | Correct | Abstained | Wrong | Unsafe |",
        "|---|---|---|---|---|---|",
    ]
    for shape, scores in sorted(groups.items()):
        c = collections.Counter(s["outcome"] for s in scores)
        lines.append(
            f"| {shape} | {len(scores)} | {c['correct']} | {c['abstained']} | {c['wrong']} "
            f"| {sum(s['unsafe'] for s in scores)} |"
        )
    return lines


def _first_error(stderr: str) -> str:
    """pyTracer's own error line, for a reproduce that gave no verdict."""
    for line in stderr.splitlines():
        if line.startswith("error:"):
            return line.replace("|", "/")
    return ""


def rescore(folder: Path) -> None:
    """Recompute labels and scores of a finished run from its saved JSON."""
    results: dict[str, Any] = {}
    for path in sorted(folder.glob("B*.json")):
        result = json.loads(path.read_text())
        result["truth"]["labels"] = _labels(result["truth"]["runs"], result["truth"].get("capture"))
        result["scores"] = {s: score(result["truth"]["labels"], v) for s, v in result["reproduce"].items()}
        path.write_text(json.dumps(result, indent=1))
        results[path.stem] = result
    if not results:
        sys.exit(f"no B*.json in {folder}")
    (folder / "summary.md").write_text(summary(results))
    print(summary(results))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bugs", default=",".join(b.key for b in BUGS))
    parser.add_argument("--settings", default=",".join(SETTINGS))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    parser.add_argument("--plan", action="store_true", help="print what would run, call no model")
    parser.add_argument("--max-usd", type=float, default=0.0, help="stop before spending more than this")
    parser.add_argument("--rescore", type=Path, help="recompute a finished run's scores, call no model")
    args = parser.parse_args()
    if args.rescore:
        rescore(args.rescore)
        return
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
            live_runs = f"{4 * args.repeats} live runs, then 1 to {CAPTURE_TRIES} recordings"
            print(f"{bug.key} task {bug.task_id}: repo {repo}; {live_runs}, "
                  f"then reproduce x{len(BRANCHES)} per setting {settings}")
            continue
        args.out.mkdir(parents=True, exist_ok=True)
        # Live truth first: it says whether the bug shows at all, and how main fails,
        # which the recording must match. A masked bug is not recorded or judged.
        truth = live_truth(bug, repo, root, args.repeats)
        captured = None if truth["labels"]["masked"] else capture(bug, repo, args.out, truth["runs"]["main"])
        if captured is not None:
            truth["capture"] = captured
            truth["labels"] = _labels(truth["runs"], captured)
        reported = {}
        if captured is not None and captured["shows_bug"]:
            reported = {s: reproduce(bug, repo, Path(captured["trace"]), s) for s in settings}
        results[bug.key] = {
            "title": bug.title, "task": bug.task_id, "shape": bug.shape, "truth": truth, "reproduce": reported,
            "scores": {s: score(truth["labels"], v) for s, v in reported.items()},  # none for a masked bug
        }
        (args.out / f"{bug.key}.json").write_text(json.dumps(results[bug.key], indent=1))


if __name__ == "__main__":
    main()

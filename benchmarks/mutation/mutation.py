"""Scripted mutation benchmark: pyTracer's verdicts on mechanical bugs, at no model cost.

The seeded-bug benchmark (``benchmarks/verdicts``) judges 15 hand-written bugs with a
live model: realistic, but a small sample. This one judges many more bugs, made by
rule instead of by hand (``mutants.py``: every flipped comparison, emptied string,
removed statement and so on in the agent's glue), on scripted conversations: the
model and the simulated customer reply from a fixed script, so every run is free and
gives the same result each time. It proves pyTracer's verdict logic on many bug
shapes; it says nothing about how live models behave, and is reported apart from the
live benchmark.

For each scripted scenario (a retail task, a script and its injected faults) and
each mutant:

1. **Screen.** Run the mutant on the scenario. A mutant that still passes every
   check is not shown by that scenario and is left out there (a survivor). The
   correct glue must pass every scenario, or the benchmark stops.
2. **Build** a repository as the live benchmark does: ``main`` carries the mutant;
   ``fix`` is the correct glue; ``nofix`` is the mutant with a comment-only line
   added; ``other`` is a different mutant, drawn with a fixed seed from those that
   fail the same scenario in a different way (different failed checks or error),
   so it is truly a different failure. With none, there is no ``other`` branch.
3. **Truth.** ``fix`` passes (it is the correct glue), so it is fixed. ``nofix`` is
   run once and must fail exactly as ``main`` does, so it is still failing.
   ``other`` fails differently, so it diverged.
4. **Record** ``main`` and run ``pytracer reproduce`` for every branch in each
   setting (strict, fallback, partial, cut-point); score as the live benchmark does.

Each (mutant, scenario) pair where the scenario shows the mutant is one case. Cases
that share a mutant are not independent, so the summary also counts unsafe verdicts
per mutant.

    python benchmarks/mutation/mutation.py --jobs 4          # every mutant, every scenario
    python benchmarks/mutation/mutation.py --mutants M012,M040 --scenarios refund

Needs tau2-bench (``TAU2_DATA_DIR``) and the ``pytracer`` CLI on the PATH; no API key
(every model key is removed from the environment before anything runs).
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
VERDICTS = HERE.parent / "verdicts"
sys.path.insert(0, str(VERDICTS))
sys.path.insert(0, str(HERE))

import run as bench  # noqa: E402  (benchmarks/verdicts/run.py)
from bugs import BRANCHES, Bug  # noqa: E402
from mutants import Mutant, mutants  # noqa: E402

SETTINGS = ("strict", "fallback", "partial", "cut-point")
SEED = 20261128  # draws each case's ``other`` mutant
KEY_SUFFIXES = ("_API_KEY", "_API_BASE", "_API_TOKEN")


@dataclass(frozen=True)
class Scenario:
    name: str
    task_id: str
    script: str  # a file in benchmarks/verdicts
    env: dict[str, str] = field(default_factory=dict)


SCENARIOS = (
    Scenario("cancel", "88", "smoke_script_task88.json"),
    Scenario("cancel-timeout", "88", "smoke_script_task88.json", {"BENCH_FLAKY_WRITES": "1"}),
    Scenario("exchange-two", "0", "smoke_script_task0.json"),
    Scenario("exchange-one", "1", "smoke_script_task1.json"),
    Scenario("refund", "13", "smoke_script_task13.json"),
    Scenario("refund-upstream-502", "13", "smoke_script_task13_lookup.json", {"BENCH_FLAKY_LOOKUPS": "1"}),
)


def _bug(key: str, title: str, scenario: Scenario, operator: str) -> Bug:
    model = f"scripted:{VERDICTS / scenario.script}"
    env = {"BENCH_MODEL": model, "BENCH_USER_MODEL": model, **scenario.env}
    return Bug(key=key, title=title, task_id=scenario.task_id, bug=[], cosmetic=[], other=[],
               shape=operator, env=env, checks_record=True)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-c", "user.name=bench", "-c", "user.email=bench@pytracer.local", *args],
                   cwd=repo, check=True, capture_output=True)


def build_repo(bug: Bug, root: Path, glue: dict[str, str]) -> Path:
    """A throwaway repository: ``main`` carries the mutant, each branch its own glue."""
    repo = root / bug.key
    (repo / "support_agent").mkdir(parents=True)
    for name in ("__init__.py", "harness.py"):
        shutil.copy(bench.AGENT / name, repo / "support_agent" / name)
    for setting in SETTINGS:
        collection = bench._collection(bug, setting)
        (repo / f"collection-{setting}.yaml").write_text(yaml.safe_dump(collection, sort_keys=False))
    (repo / "support_agent" / "glue.py").write_text(glue["main"])
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", f"{bug.key}: {bug.title}")
    for branch in BRANCHES:
        if branch not in glue:
            continue
        _git(repo, "checkout", "-q", "-b", branch, "main")
        (repo / "support_agent" / "glue.py").write_text(glue[branch])
        _git(repo, "commit", "-qam", f"{branch}: {bug.key}")
    _git(repo, "checkout", "-q", "main")
    return repo


def _failure(run: dict[str, Any]) -> tuple[str, ...] | None:
    """How a run failed: its failed checks and its error type. None if it passed."""
    failed = bench._failure(run)
    return None if failed is None else (*failed, f"error={run.get('error_type')}")


def screen(scenario: Scenario, glue: str, key: str, root: Path) -> dict[str, Any]:
    """One live (scripted) run of ``glue`` on the scenario, judged."""
    bug = _bug(key, key, scenario, "screen")
    repo = build_repo(bug, root, {"main": glue})
    try:
        return bench._live_once(bug, repo, "main")
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def _screen_or_crash(scenario: Scenario, glue: str, key: str, root: Path) -> dict[str, Any]:
    """``screen``, or ``{"crashed": ...}`` when the run's process itself fails (a mutant
    that breaks the harness, not the agent): such a pair is not judged."""
    try:
        return screen(scenario, glue, key, root)
    except RuntimeError as exc:
        return {"crashed": str(exc)[-300:], "assertions": {}, "db_matches_gold": False}


def capture(bug: Bug, repo: Path, trace: Path) -> dict[str, Any]:
    done = subprocess.run(
        [sys.executable, str(VERDICTS / "child.py"), "capture", str(repo), bug.task_id, str(trace),
         str(repo / "collection-strict.yaml")],
        env=bench._env(bug), capture_output=True, text=True,
    )
    lines = [line for line in done.stdout.splitlines() if line.startswith("{")]
    if done.returncode != 0 or not lines:
        raise RuntimeError(f"capture of {bug.key} failed:\n{done.stderr[-2000:]}")
    return dict(json.loads(lines[-1]))


def judge_case(mutant: Mutant, other: Mutant | None, scenario: Scenario, main_run: dict[str, Any],
               correct: str, root: Path) -> dict[str, Any]:
    key = f"{mutant.id}-{scenario.name}"
    bug = _bug(key, mutant.description, scenario, mutant.operator)
    glue = {"main": mutant.source, "fix": correct, "nofix": mutant.cosmetic}
    if other is not None:
        glue["other"] = other.source
    repo = build_repo(bug, root, glue)
    try:
        nofix = bench._live_once(bug, bench._checkout(repo, "nofix", root), "nofix")
        if _failure(nofix) != _failure(main_run):
            raise RuntimeError(f"{key}: nofix (a comment-only edit) did not fail as main does")
        trace = root / "traces" / f"{key}.jsonl"  # deterministic: rerun to get it back
        recorded = capture(bug, repo, trace)
        if _failure(recorded) != _failure(main_run):
            raise RuntimeError(f"{key}: the recording did not fail as the screening run did")
        labels = {"masked": None, "fix": {"label": "fixed"}, "nofix": {"label": "still_failing"},
                  "other": {"label": "diverged"}}
        branches = [b for b in BRANCHES if b in glue]
        reported: dict[str, Any] = {}
        for setting in SETTINGS:
            got = bench.reproduce(bug, repo, trace, setting) if len(branches) == 3 else _reproduce(
                bug, repo, trace, setting, branches)
            reported[setting] = {b: got[b] for b in branches}
        scores = {s: {b: v for b, v in _score(labels, r).items()} for s, r in reported.items()}
        return {"case": key, "mutant": mutant.id, "operator": mutant.operator, "where": mutant.where,
                "description": mutant.description, "scenario": scenario.name,
                "other": other.id if other else None, "failure": list(_failure(main_run) or ()),
                "reproduce": reported, "scores": scores}
    finally:
        shutil.rmtree(repo, ignore_errors=True)
        for branch in BRANCHES:
            shutil.rmtree(root / f"{key}-{branch}", ignore_errors=True)


def _reproduce(bug: Bug, repo: Path, trace: Path, setting: str, branches: list[str]) -> dict[str, Any]:
    """``bench.reproduce`` for a repository without an ``other`` branch."""
    saved = bench.BRANCHES
    bench.BRANCHES = tuple(branches)  # type: ignore[assignment]
    try:
        return bench.reproduce(bug, repo, trace, setting)
    finally:
        bench.BRANCHES = saved  # type: ignore[assignment]


def _score(labels: dict[str, Any], reported: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for branch, got in reported.items():
        label, verdict = labels[branch]["label"], got["verdict"]
        outcome = ("abstained" if verdict in bench.ABSTAIN else "correct" if verdict == label else "wrong")
        out[branch] = {"truth": label, "verdict": verdict, "outcome": outcome,
                       "unsafe": verdict == "fixed" and label != "fixed"}
    return out


def _no_keys() -> None:
    for name in list(os.environ):
        if name.endswith(KEY_SUFFIXES) or name in ("BENCH_USAGE_LEDGER", "BENCH_MAX_USD"):
            del os.environ[name]


def summary(cases: list[dict[str, Any]], survivors: dict[str, list[str]], crashed: list[str], total: int,
            scenarios: list[Scenario], seconds: float) -> str:
    shown = {c["mutant"] for c in cases}
    never = sorted(m for m, names in survivors.items() if len(names) == len(scenarios))
    lines = [
        "# Scripted mutation benchmark",
        "",
        f"{total} mechanical mutants of the agent's glue, {len(scenarios)} scripted scenarios, "
        f"{len(cases)} cases (a mutant that a scenario shows), {len(shown)} mutants shown by at "
        f"least one scenario, {len(never)} shown by none (survivors). No model was called: the "
        f"model and the customer reply from fixed scripts. Took {seconds / 60:.0f} minutes.",
        "",
        "This tests pyTracer's verdict logic on many bug shapes. It is not a measure of live-model "
        "behaviour; see the seeded-bug benchmark for that.",
    ]
    for setting in SETTINGS:
        rows = [(c["case"], b, s) for c in cases for b, s in c["scores"][setting].items()]
        if not rows:
            continue
        unsafe = [r for r in rows if r[2]["unsafe"]]
        unsafe_mutants = sorted({c["mutant"] for c in cases
                                 if any(s["unsafe"] for s in c["scores"][setting].values())})
        lo, hi = bench._interval(len(unsafe), len(rows))
        lines += [
            "", f"## {setting}", "",
            *bench._scores(rows),
            "",
            f"Unsafe (reported `fixed`, not fixed): {len(unsafe)} of {len(rows)} verdicts "
            f"(95% interval {bench._pct(lo)} to {bench._pct(hi)}), in {len(unsafe_mutants)} of "
            f"{len(shown)} mutants{': ' + ', '.join(unsafe_mutants) if unsafe_mutants else ''}.",
            "",
            "| Branch | Correct | Abstained | Wrong |",
            "|---|---|---|---|",
        ]
        for branch in BRANCHES:
            counts = collections.Counter(s["outcome"] for _, b, s in rows if b == branch)
            lines.append(f"| {branch} | {counts['correct']} | {counts['abstained']} | {counts['wrong']} |")
        lines += ["", "| Operator | Cases | Correct | Abstained | Wrong | Unsafe |", "|---|---|---|---|---|---|"]
        by_op: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for c in cases:
            by_op[c["operator"]].extend(c["scores"][setting].values())
        for op, scores in sorted(by_op.items()):
            n = collections.Counter(s["outcome"] for s in scores)
            cases_n = sum(c["operator"] == op for c in cases)
            lines.append(f"| {op} | {cases_n} | {n['correct']} | {n['abstained']} | {n['wrong']} "
                         f"| {sum(s['unsafe'] for s in scores)} |")
    wrong = [(c["case"], s, b, v) for c in cases for s in SETTINGS for b, v in c["scores"][s].items()
             if v["outcome"] == "wrong"]
    lines += ["", "## Wrong verdicts", ""]
    lines += [f"- {case} {setting} {branch}: reported `{v['verdict']}`, truly `{v['truth']}`"
              for case, setting, branch, v in wrong] or ["None."]
    lines += ["", "## Survivors (no scenario shows them)", ""]
    lines += [f"- {m}" for m in never] or ["None."]
    lines += ["", "## Not judged: the agent's process crashed before it could be checked", ""]
    lines += [f"- {c}" for c in crashed] or ["None."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mutants", help="comma-separated ids, e.g. M012,M040 (default: all)")
    parser.add_argument("--scenarios", help="comma-separated names (default: all)")
    parser.add_argument("--jobs", type=int, default=4, help="cases run at once")
    parser.add_argument("--out", type=Path, default=HERE / "results")
    parser.add_argument("--list", action="store_true", help="print the mutants and stop")
    args = parser.parse_args()
    correct = (bench.AGENT / "glue.py").read_text()
    every = mutants(correct)
    if args.list:
        for m in every:
            print(f"{m.id} line {m.line} {m.where}: {m.operator}: {m.description}")
        return
    chosen = [m for m in every if not args.mutants or m.id in args.mutants.split(",")]
    scenarios = [s for s in SCENARIOS if not args.scenarios or s.name in args.scenarios.split(",")]
    if "TAU2_DATA_DIR" not in os.environ:
        sys.exit("set TAU2_DATA_DIR to tau2-bench's data folder")
    _no_keys()
    started = time.monotonic()
    args.out.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="pytracer-mutation-"))
    try:
        with ThreadPoolExecutor(args.jobs) as pool:
            baseline = list(pool.map(lambda s: screen(s, correct, f"correct-{s.name}", root), scenarios))
            broken = [s.name for s, r in zip(scenarios, baseline) if _failure(r) is not None]
            if broken:
                sys.exit(f"the correct glue fails scenarios {broken}: fix the scripts first")
            pairs = [(m, s) for m in every for s in scenarios]  # every mutant: ``other`` draws from all
            runs = {(m.id, s.name): r for (m, s), r in zip(
                pairs, pool.map(lambda p: _screen_or_crash(p[1], p[0].source, f"{p[0].id}-{p[1].name}", root), pairs))}
            print(f"screened {len(pairs)} pairs in {(time.monotonic() - started) / 60:.1f} min", file=sys.stderr)
            rng = random.Random(SEED)
            jobs = []
            for m in chosen:
                for s in scenarios:
                    failure = _failure(runs[(m.id, s.name)])
                    if failure is None or "crashed" in runs[(m.id, s.name)]:
                        continue
                    others = [o for o in every if o.id != m.id and "crashed" not in runs[(o.id, s.name)]
                              and _failure(runs[(o.id, s.name)]) not in (None, failure)]
                    jobs.append((m, rng.choice(others) if others else None, s))
            cases = list(pool.map(lambda j: judge_case(j[0], j[1], j[2], runs[(j[0].id, j[2].name)], correct,
                                                       root), jobs))
    finally:
        shutil.rmtree(root, ignore_errors=True)
    survivors = {m.id: [s.name for s in scenarios if _failure(runs[(m.id, s.name)]) is None] for m in chosen}
    crashed = sorted(f"{m}-{s}" for (m, s), r in runs.items() if "crashed" in r)
    screening = {f"{m}-{s}": r.get("crashed") or list(_failure(r) or ()) for (m, s), r in runs.items()}
    (args.out / "cases.json").write_text(json.dumps(
        {"mutants": [{k: v for k, v in m.__dict__.items() if k not in ("source", "cosmetic")} for m in every],
         "screening": screening, "cases": cases}, indent=1))
    text = summary(cases, survivors, crashed, len(chosen), scenarios, time.monotonic() - started)
    (args.out / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()

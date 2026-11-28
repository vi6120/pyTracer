"""Determinism: how often does a live test change its answer, and does a replay ever?

Live side (no model calls): reads a finished seeded-bug run
(``benchmarks/verdicts/run.py`` with ``--baseline-repeats``). Each branch of each
bug was run live several times (``main`` 5, ``fix``/``nofix``/``other`` 10: the
ground-truth runs plus the baseline runs). A live test is flaky on a branch when
its runs disagree, pass on some and fail on others. Counted per branch and per run:
how often one live run gives a different answer than the majority of that branch's
runs, which is how often a team re-running a test once would be misled.

Replay side (no model calls either): rebuilds each bug's repository and replays
its recording with ``pytracer reproduce`` in the strict setting ``--replays`` times
per branch, each a fresh process, with every model API key removed from the
environment so nothing can call a model. A replay is deterministic when every
repeat gives the same verdict and the same reason. The verdict the benchmark run
itself reported is compared too.

Masked bugs (the bug does not show live) have no recording; their live runs still
count on the live side.

    python benchmarks/determinism/determinism.py --results benchmarks/verdicts/results/v2-gpt-5.2
    python benchmarks/determinism/determinism.py --results ... --live-only   # seconds, no replays

Needs what ``benchmarks/verdicts/run.py`` needs for its replays: tau2-bench
(``TAU2_DATA_DIR``), the ``pytracer`` CLI on the PATH, and the model names the run
recorded (``BENCH_MODEL``, and ``BENCH_USER_MODEL`` if it differed); no API key.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
VERDICTS = HERE.parent / "verdicts"
sys.path.insert(0, str(VERDICTS))

import run as bench  # noqa: E402  (benchmarks/verdicts/run.py)
from bugs import BRANCHES, BUGS  # noqa: E402

KEY_SUFFIXES = ("_API_KEY", "_API_BASE", "_API_TOKEN")


def _live_runs(result: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Every live run of each branch: the ground-truth runs plus the baseline runs."""
    truth = result["truth"]
    extra = truth.get("baseline_runs") or {}
    return {b: [*truth["runs"][b], *extra.get(b, [])] for b in ("main", *BRANCHES)}


def live_side(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows, flaky, branches, misled, runs = [], 0, 0, 0, 0
    for key, result in results.items():
        for branch, branch_runs in _live_runs(result).items():
            passed = sum(bench._passed(r) for r in branch_runs)
            n = len(branch_runs)
            minority = min(passed, n - passed)  # runs that disagree with the majority
            rows.append({"bug": key, "branch": branch, "passed": passed, "runs": n, "minority": minority})
            branches += 1
            flaky += 0 < passed < n
            misled += minority
            runs += n
    return {"rows": rows, "branches": branches, "flaky": flaky, "misled": misled, "runs": runs}


def _no_keys() -> None:
    """Remove every model credential from this process, so its replays cannot call one."""
    for name in list(os.environ):
        if name.endswith(KEY_SUFFIXES):
            del os.environ[name]


def replay_side(results: dict[str, dict[str, Any]], replays: int, folder: Path) -> dict[str, Any]:
    _no_keys()
    by_key = {b.key: b for b in BUGS}
    root = Path(tempfile.mkdtemp(prefix="pytracer-determinism-"))
    rows = []
    try:
        for key, result in results.items():
            captured = result["truth"].get("capture")
            if not captured or not captured.get("shows_bug"):
                continue
            trace = (VERDICTS / captured["trace"]).resolve()
            if not trace.exists():  # the run was made from another folder: try beside the JSON
                trace = (folder / key / Path(captured["trace"]).name).resolve()
            repo = bench.build_repo(by_key[key], root)
            seen: dict[str, list[tuple[Any, Any, Any]]] = {b: [] for b in BRANCHES}
            seconds: dict[str, list[float]] = {b: [] for b in BRANCHES}
            for i in range(replays):
                for branch, got in bench.reproduce(by_key[key], repo, trace, "strict").items():
                    seen[branch].append((got["verdict"], got["reason"], got["exit"]))
                    seconds[branch].append(got["seconds"])
                print(f"{key}: replay {i + 1}/{replays} done", file=sys.stderr)
            for branch in BRANCHES:
                reported = result["reproduce"].get("strict", {}).get(branch, {})
                distinct = sorted({s for s in seen[branch]}, key=repr)
                rows.append({
                    "bug": key, "branch": branch, "replays": replays,
                    "verdicts": [s[0] for s in seen[branch]],
                    "distinct": [{"verdict": v, "reason": r, "exit": e} for v, r, e in distinct],
                    "matches_run": all(s[:2] == (reported.get("verdict"), reported.get("reason")) for s in seen[branch]),
                    "seconds_median": sorted(seconds[branch])[len(seconds[branch]) // 2],
                })
            shutil.rmtree(repo, ignore_errors=True)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    stable = sum(len(r["distinct"]) == 1 for r in rows)
    return {"rows": rows, "branches": len(rows), "stable": stable,
            "matches_run": sum(r["matches_run"] for r in rows)}


def _share(k: int, n: int) -> str:
    return f"{k}/{n} ({100 * k / n:.1f}%)" if n else "0/0"


def _shown(folder: Path) -> str:
    """The folder relative to the repository, so no local path is published."""
    try:
        return str(folder.relative_to(HERE.parent.parent))
    except ValueError:
        return folder.name


def summary(folder: Path, live: dict[str, Any], replay: dict[str, Any] | None) -> str:
    lines = [
        "# Determinism: live runs versus replays",
        "",
        f"Source: `{_shown(folder)}` (live runs as recorded by the seeded-bug benchmark). "
        f"Machine: {platform.machine()}, Python {platform.python_version()}.",
        "",
        "## Live",
        "",
        f"- Branches whose live runs disagreed (some passed, some failed): "
        f"{_share(live['flaky'], live['branches'])}",
        f"- Single live runs that gave a different answer than their branch's majority: "
        f"{_share(live['misled'], live['runs'])}",
        "",
        "| Bug | main | fix | nofix | other |",
        "|---|---|---|---|---|",
    ]
    by_bug: dict[str, dict[str, str]] = {}
    for r in live["rows"]:
        mark = " (flaky)" if 0 < r["passed"] < r["runs"] else ""
        by_bug.setdefault(r["bug"], {})[r["branch"]] = f"{r['passed']}/{r['runs']} pass{mark}"
    for bug in sorted(by_bug, key=bench._bug_order):
        cells = by_bug[bug]
        lines.append(f"| {bug} | " + " | ".join(cells.get(b, "") for b in ("main", *BRANCHES)) + " |")
    if replay is None:
        lines += ["", "Replays not run (`--live-only`)."]
        return "\n".join(lines) + "\n"
    n = replay["rows"][0]["replays"] if replay["rows"] else 0
    lines += [
        "",
        "## Replay (strict, no model key in the environment)",
        "",
        f"- Branches whose {n} replays all gave the same verdict and reason: "
        f"{_share(replay['stable'], replay['branches'])}",
        f"- Branches whose replays all matched the verdict the benchmark run reported: "
        f"{_share(replay['matches_run'], replay['branches'])}",
        "",
        "| Bug | Branch | Verdicts over the replays | Same every time | Matches the run | Median seconds |",
        "|---|---|---|---|---|---|",
    ]
    for r in replay["rows"]:
        counts: dict[str, int] = {}
        for v in r["verdicts"]:
            counts[str(v)] = counts.get(str(v), 0) + 1
        shown = ", ".join(f"{v} x{c}" for v, c in counts.items())
        lines.append(
            f"| {r['bug']} | {r['branch']} | {shown} | {'yes' if len(r['distinct']) == 1 else 'NO'} | "
            f"{'yes' if r['matches_run'] else 'NO'} | {r['seconds_median']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=VERDICTS / "results" / "v2-gpt-5.2",
                        help="a finished seeded-bug run folder (B*.json)")
    parser.add_argument("--replays", type=int, default=10, help="strict replays per branch")
    parser.add_argument("--bugs", help="comma-separated, e.g. B1,B2 (default: every bug in the folder)")
    parser.add_argument("--live-only", action="store_true", help="skip the replays")
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    folder = args.results.resolve()
    results = {p.stem: json.loads(p.read_text()) for p in folder.glob("B*.json")}
    if args.bugs:
        results = {k: v for k, v in results.items() if k in args.bugs.split(",")}
    results = dict(sorted(results.items(), key=lambda kv: bench._bug_order(kv[0])))
    if not results:
        sys.exit(f"no B*.json in {folder}")
    live = live_side(results)
    replay = None
    if not args.live_only:
        if "TAU2_DATA_DIR" not in os.environ:
            sys.exit("set TAU2_DATA_DIR to tau2-bench's data folder (or pass --live-only)")
        if "BENCH_MODEL" not in os.environ:
            # The model names are part of every recorded input: a replay with other
            # names leaves the recording at once and says nothing about determinism.
            sys.exit("set BENCH_MODEL (and BENCH_USER_MODEL if it differed) to the models the run recorded")
        replay = replay_side(results, args.replays, folder)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "raw.json").write_text(json.dumps({"live": live, "replay": replay}, indent=1))
    text = summary(folder, live, replay)
    (args.out / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()

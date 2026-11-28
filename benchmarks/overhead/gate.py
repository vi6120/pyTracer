"""CI performance gate: does a change make the SDK's hot path slower?

Compares two copies of the SDK source (the pull request's and its base branch's) on
the same workload as ``overhead.py``: the recorded 20-span agent run with the model
and tools stubbed, with and without ``@op`` and the default redaction. Absolute
times on a shared CI runner move by tens of percent from one job to the next, so
the gate never compares with a stored number. Both copies run in the same job, each
in its own process, in alternating rounds, and only their difference counts.

What is compared is the time the SDK adds to a run (instrumented minus plain, at the
median), so the plain agent's own cost cancels out. The gate fails when the pull
request adds both more than ``--max-slowdown`` times what the base adds and more than
``--min-delta-us`` per span, or when it adds more than ``--ceiling-us`` per span
however the base does. A failure is measured a second time before it counts, so a
noisy moment on the runner does not fail a pull request on its own.

    python benchmarks/overhead/gate.py --base ../base/packages/sdk/src --head packages/sdk/src

Exits 1 when the gate fails. Writes its table to ``$GITHUB_STEP_SUMMARY`` when set.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent


def worker() -> None:
    """Runs inside one SDK copy (``PYTHONPATH``): times plain and instrumented runs on request."""
    sys.path.insert(0, str(HERE))
    import overhead  # the same workload as the published benchmark

    import pytracer_sdk as pytracer
    from pytracer_sdk import Tracer
    from pytracer_sdk.transport import InMemoryTransport

    steps, _ = overhead.load_steps()
    plain = overhead.build_agent(steps, instrumented=False)
    traced = overhead.build_agent(steps, instrumented=True)
    print(json.dumps({"spans": len(steps) + 1, "sdk": pytracer.__file__}), flush=True)
    for line in sys.stdin:
        runs = int(line)
        times: dict[str, list[int]] = {"plain": [], "sdk": []}
        tracer = Tracer(InMemoryTransport())  # fresh each batch, so kept spans do not pile up
        for i in range(runs):
            for name in (("plain", "sdk") if i % 2 else ("sdk", "plain")):
                pytracer.set_tracer(tracer if name == "sdk" else None)
                start = time.perf_counter_ns()
                (traced if name == "sdk" else plain)()
                times[name].append(time.perf_counter_ns() - start)
        print(json.dumps(times), flush=True)


class Side:
    def __init__(self, name: str, src: Path) -> None:
        env = {**os.environ, "PYTHONPATH": str(src.resolve())}
        self.name = name
        self.proc = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "worker"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env,
        )
        hello = json.loads(self._read())
        self.spans: int = hello["spans"]
        self.sdk: str = hello["sdk"]
        if not Path(self.sdk).resolve().is_relative_to(src.resolve()):
            sys.exit(f"{name}: imported the SDK from {self.sdk}, not from {src}")
        self.times: dict[str, list[int]] = {"plain": [], "sdk": []}

    def _read(self) -> str:
        assert self.proc.stdout is not None
        line = self.proc.stdout.readline()
        if not line:
            sys.exit(f"{self.name}: the worker stopped (exit {self.proc.wait()})")
        return line

    def batch(self, runs: int, keep: bool) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(f"{runs}\n")
        self.proc.stdin.flush()
        got = json.loads(self._read())
        if keep:
            for name, values in got.items():
                self.times[name].extend(values)

    def added_us_per_span(self) -> float:
        added_ns = statistics.median(self.times["sdk"]) - statistics.median(self.times["plain"])
        return added_ns / 1000 / self.spans

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        self.proc.wait()


def measure(base_src: Path, head_src: Path, rounds: int, batch: int, warmup: int, seed: int) -> tuple[float, float]:
    """Added µs per span at the median, for the base and the head, measured in alternating rounds."""
    sides = [Side("base", base_src), Side("head", head_src)]
    try:
        for side in sides:
            side.batch(warmup, keep=False)
        rng = random.Random(seed)
        for _ in range(rounds):
            rng.shuffle(sides)
            for side in sides:
                side.batch(batch, keep=True)
        by_name = {s.name: s for s in sides}
        return by_name["base"].added_us_per_span(), by_name["head"].added_us_per_span()
    finally:
        for side in sides:
            side.close()


def verdict(base: float, head: float, args: argparse.Namespace) -> str | None:
    """Why the head fails the gate, or None if it passes."""
    if head > args.ceiling_us:
        return f"adds {head:.0f} µs per span, over the {args.ceiling_us:.0f} µs ceiling"
    if head > base * args.max_slowdown and head - base > args.min_delta_us:
        return (f"adds {head:.0f} µs per span against the base's {base:.0f} µs "
                f"({head / base:.2f}x; the gate allows up to {args.max_slowdown:.2f}x, or up to "
                f"{args.min_delta_us:.0f} µs more)")
    return None


def main() -> None:
    if sys.argv[1:2] == ["worker"]:
        worker()
        return
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", type=Path, required=True, help="the base branch's packages/sdk/src")
    parser.add_argument("--head", type=Path, required=True, help="the change's packages/sdk/src")
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--batch", type=int, default=40, help="runs of each kind per round")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--max-slowdown", type=float, default=1.25)
    parser.add_argument("--min-delta-us", type=float, default=15.0)
    parser.add_argument("--ceiling-us", type=float, default=1000.0)
    parser.add_argument("--seed", type=int, default=20261128)
    args = parser.parse_args()

    tries: list[tuple[float, float, str | None]] = []
    for attempt in range(2):  # a failure is measured again before it counts
        base, head = measure(args.base, args.head, args.rounds, args.batch, args.warmup, args.seed + attempt)
        tries.append((base, head, verdict(base, head, args)))
        if tries[-1][2] is None:
            break
    failed = all(reason is not None for _, _, reason in tries)
    lines = [
        "## SDK overhead gate",
        "",
        "Time the SDK adds per span (instrumented minus plain, at the median), base and change "
        "measured in the same job in alternating rounds.",
        "",
        "| Try | Base (µs per span) | Change (µs per span) | Ratio | Result |",
        "|---|---|---|---|---|",
        *[f"| {i + 1} | {b:.0f} | {h:.0f} | {h / b:.2f}x | {r or 'pass'} |" for i, (b, h, r) in enumerate(tries)],
        "",
        "Result: **" + ("fail" if failed else "pass") + "**",
    ]
    text = "\n".join(lines) + "\n"
    print(text)
    summary: Any = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(text)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

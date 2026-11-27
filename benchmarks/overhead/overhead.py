"""SDK overhead: how much time does pyTracer add to an agent run?

The agent replays a real recorded run (``fixture_run.jsonl``: a GPT-5.2 support
agent on tau2-bench retail task 88, 19 spans) with the model and the tools
stubbed: each call returns its recorded output at once. That leaves only the cost
of the agent's own code and of pyTracer's instrumentation, which a live model call
(about 2 seconds) would otherwise drown in noise.

Each configuration runs the same agent ``--runs`` times, interleaved run by run so
drift in the machine hits them all alike, and the time is measured on the agent's
own thread, which is the time a user waits:

- ``none``: no SDK, the plain functions;
- ``sdk``: ``@op`` on every call, the default redaction, spans kept in memory;
- ``sdk-no-redaction``: the same with redaction turned off;
- ``sdk-http``: the default redaction, spans shipped by the background buffer to a
  local HTTP endpoint that accepts them (the production path, minus the network).

It writes the raw times (``results/raw.csv``) and a summary (``results/summary.md``)
with the machine it ran on. Re-run it on your own machine; the numbers will differ.

    python benchmarks/overhead/overhead.py            # 1,000 runs per configuration
"""

from __future__ import annotations

import argparse
import csv
import http.server
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytracer_sdk as pytracer
from pytracer_sdk import SpanKind, Tracer
from pytracer_sdk.redaction import Redactor
from pytracer_sdk.transport import HTTPTransport, InMemoryTransport

HERE = Path(__file__).resolve().parent
KINDS = {"llm": SpanKind.LLM, "tool": SpanKind.TOOL}


def load_steps() -> tuple[list[dict[str, Any]], float]:
    """The recorded model and tool calls, in order, and the run's recorded model time (ms)."""
    spans = [json.loads(line) for line in (HERE / "fixture_run.jsonl").read_text().splitlines()]
    steps = [s for s in spans if s["kind"] in KINDS]
    model_ms = sum(s["latency_ms"] for s in steps if s["kind"] == "llm")
    return steps, model_ms


def build_agent(steps: list[dict[str, Any]], instrumented: bool) -> Callable[[], None]:
    """The agent: makes each recorded call in order; each stub returns the recorded output."""
    calls: list[tuple[Callable[[Any], Any], Any]] = []
    cache: dict[tuple[str, str], Callable[[Any], Any]] = {}
    for step in steps:
        key = (step["kind"], step.get("tool_name") or step["name"])
        if key not in cache:
            output = step["output"]

            def stub(payload: Any, _output: Any = output) -> Any:
                return _output

            if instrumented:
                stub = pytracer.op(
                    kind=KINDS[step["kind"]], name=step["name"], tool_name=step.get("tool_name")
                )(stub)
            cache[key] = stub
        calls.append((cache[key], step["input"]))

    def run() -> None:
        for fn, payload in calls:
            fn(payload)

    if instrumented:
        run = pytracer.op(kind=SpanKind.AGENT, agent_name="support")(run)
    return run


class _Accept(http.server.BaseHTTPRequestHandler):
    """A gateway stand-in: reads the batch and answers 202."""

    def do_POST(self) -> None:  # noqa: N802 - http.server's naming
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        body = b'{"accepted": 0}'
        self.send_response(202)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: Any) -> None:
        pass


def local_endpoint() -> str:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Accept)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}"


def machine() -> dict[str, str]:
    cpu = platform.processor() or platform.machine()
    if sys.platform == "darwin":
        try:
            cpu = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
            ).stdout.strip() or cpu
        except OSError:
            pass
    return {
        "cpu": cpu,
        "cores": str(os.cpu_count()),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "pytracer-sdk": pytracer.__version__,
    }


def pct(values: list[int], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(p * len(ordered)))] / 1e6  # ms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261128)
    args = parser.parse_args()

    steps, model_ms = load_steps()
    spans_per_run = len(steps) + 1  # every call plus the agent itself
    tracers: dict[str, Tracer | None] = {
        "none": None,
        "sdk": Tracer(InMemoryTransport()),
        "sdk-no-redaction": Tracer(InMemoryTransport(), redactor=Redactor(enabled=False)),
        "sdk-http": Tracer(HTTPTransport(local_endpoint(), api_key="benchmark")),
    }
    agents = {name: build_agent(steps, instrumented=t is not None) for name, t in tracers.items()}
    times: dict[str, list[int]] = {name: [] for name in tracers}
    order = list(tracers)
    rng = random.Random(args.seed)
    for i in range(args.warmup + args.runs):
        rng.shuffle(order)
        for name in order:
            pytracer.set_tracer(tracers[name])
            start = time.perf_counter_ns()
            agents[name]()
            elapsed = time.perf_counter_ns() - start
            if i >= args.warmup:
                times[name].append(elapsed)
    for tracer in tracers.values():
        if tracer is not None:
            tracer.flush()

    out = HERE / "results"
    out.mkdir(exist_ok=True)
    with (out / "raw.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["configuration", "run", "ns"])
        for name, values in times.items():
            writer.writerows((name, i, v) for i, v in enumerate(values))

    base = statistics.median(times["none"]) / 1e6
    lines = [
        "# pyTracer SDK overhead",
        "",
        f"A recorded {spans_per_run}-span agent run (model and tools stubbed), {args.runs} runs per "
        f"configuration after {args.warmup} warm-up runs, interleaved in a seeded random order "
        f"(seed {args.seed}). Times are on the agent's own thread, in milliseconds per run.",
        "",
        "| Machine | |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in machine().items()],
        "",
        "| Configuration | p50 | p95 | p99 | Added at p50 | Added per span at p50 | Share of the run's model time |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, values in times.items():
        p50 = pct(values, 0.50)
        added = p50 - base
        lines.append(
            f"| {name} | {p50:.3f} | {pct(values, 0.95):.3f} | {pct(values, 0.99):.3f} | "
            + ("- | - | - |" if name == "none" else
               f"{added:.3f} | {1000 * added / spans_per_run:.1f} µs | {100 * added / model_ms:.3f}% |")
        )
    lines += [
        "",
        f"The share compares the added time with the {model_ms / 1000:.1f} s the recorded run spent "
        "waiting on its model calls (GPT-5.2 agent and customer), which is what a live run adds "
        "on top of the code measured here.",
    ]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

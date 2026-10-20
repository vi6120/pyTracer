"""Capture a successful run of the analyst agent and save its frozen trace.

This records the tool/model responses that every branch replays against, so the
only thing that varies between branches is the agent's own glue code. Run it to
(re)generate ``traces/run.jsonl``. Capture uses the fixed variant because it does
not raise; the ``search`` and ``summarize`` outputs it records are identical
across all four variants.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytracer_sdk as pytracer

HERE = Path(__file__).parent


def main() -> None:
    sys.path.insert(0, str(HERE / "variants"))
    import agent_fixed as agent

    sink = pytracer.InMemoryTransport()
    pytracer.configure(sink, service_name="analyst", branch="main", commit="demo")
    result = agent.run("climate change")
    pytracer.flush()

    traces = HERE / "traces"
    traces.mkdir(exist_ok=True)
    jsonl = traces / "run.jsonl"
    jsonl.write_text("\n".join(s.to_json() for s in sink.spans) + "\n", encoding="utf-8")

    print(f"agent returned: {result!r}")
    print(f"captured {len(sink.spans)} spans -> {jsonl}")


if __name__ == "__main__":
    main()

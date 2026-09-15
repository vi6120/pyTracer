"""Capture a run of the analyst agent and save its trace (the frozen mocks).

This records the recorded tool/model responses that both branches replay
against, so the only thing that varies between branches is the agent's own code.
Run it to (re)generate ``traces/run.jsonl``; the committed copy is produced this
way. The op outputs are identical in both the buggy and fixed variants, so
capturing from either yields the same frozen mocks.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytracer_sdk as pytracer

HERE = Path(__file__).parent


def main() -> None:
    sys.path.insert(0, str(HERE / "variants"))
    import analyst_fixed as agent

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

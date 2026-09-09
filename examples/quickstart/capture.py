"""Run the agent once and save its trace, simulating a captured run.

In a real deployment the SDK ships spans to the gateway and they land in the
trace store. Here we capture them in-process and write them to a JSONL file, so
the quickstart needs no running services.
"""

from __future__ import annotations

from pathlib import Path

import pytracer_sdk as pytracer
from agent import run


def main() -> None:
    sink = pytracer.InMemoryTransport()
    pytracer.configure(sink, service_name="researcher", branch="main", commit="quickstart")

    result = run("climate change")
    pytracer.flush()

    traces_dir = Path(__file__).parent / "traces"
    traces_dir.mkdir(exist_ok=True)
    jsonl = traces_dir / "run.jsonl"
    jsonl.write_text("\n".join(s.to_json() for s in sink.spans) + "\n", encoding="utf-8")

    print(f"agent returned: {result!r}")
    print(f"captured {len(sink.spans)} spans -> {jsonl.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()

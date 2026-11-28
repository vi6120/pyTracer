"""Run the agent in the current directory once and save its trace: the recording
that ``pytracer reproduce`` freezes. ``python capture.py <trace.jsonl>``"""

from __future__ import annotations

import sys
from pathlib import Path

import pytracer_sdk as pytracer


def main() -> None:
    sys.path.insert(0, str(Path.cwd()))
    import support_agent

    sink = pytracer.InMemoryTransport()
    pytracer.configure(sink, service_name="support", branch="capture", commit="demo")
    result = support_agent.run("#W1001")
    pytracer.flush()
    out = Path(sys.argv[1])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(s.to_json() + "\n" for s in sink.spans), encoding="utf-8")
    kinds = ", ".join(f"{s.kind.value} {s.name}" for s in sink.spans)
    print(f"agent returned {result}; recorded {len(sink.spans)} spans ({kinds}) -> {out}")


if __name__ == "__main__":
    main()

"""End-to-end smoke check: capture a run, then replay the collection.

Runs the whole core loop in one process and exits non-zero if the replay fails,
so it doubles as a quick check that the platform is wired up correctly.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    here = Path(__file__).parent
    os.chdir(here)
    sys.path.insert(0, str(here))

    import capture

    capture.main()

    from pytracer_cli import run_path

    result = run_path(here / "collection.yaml")
    for scenario in result.scenarios:
        print(f"[{'PASS' if scenario.passed else 'FAIL'}] {scenario.name}")
    print("PASSED" if result.passed else "FAILED")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

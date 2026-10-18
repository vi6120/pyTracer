"""pytest plugin: run pyTracer collections as native pytest tests.

With ``pytracer-cli`` installed, ``pytest`` discovers pyTracer collection files -
by default ``*.pytracer.yaml`` (override with the ``pytracer_glob`` ini option) -
and runs each scenario in them as its own test item. Agent regression tests then
sit alongside your unit tests and gate CI the same way, with ``-k`` filtering,
reporting, and exit codes all working as usual::

    # agents.pytracer.yaml discovered automatically
    $ pytest
    agents.pytracer.yaml::researcher-happy-path PASSED
    agents.pytracer.yaml::refund-policy        FAILED

The plugin is auto-registered via the ``pytest11`` entry point; no conftest wiring
is needed. It reuses the same runner as ``pytracer run``, so a scenario behaves
identically here, on the CLI, and in CI.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from .collection import ScenarioSpec, load_collection
from .runner import ScenarioResult, run_scenario

if TYPE_CHECKING:
    from collections.abc import Iterator

DEFAULT_GLOB = "*.pytracer.yaml"


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the ``pytracer_glob`` ini option."""
    parser.addini(
        "pytracer_glob",
        help="glob matching pyTracer collection files to run as tests",
        default=DEFAULT_GLOB,
    )


def pytest_collect_file(parent: pytest.Collector, file_path: Path) -> pytest.Collector | None:
    """Collect a pyTracer collection file when its name matches the glob."""
    glob = str(parent.config.getini("pytracer_glob")) or DEFAULT_GLOB
    if fnmatch.fnmatch(file_path.name, glob):
        return CollectionFile.from_parent(parent, path=file_path)
    return None


def _format_failure(result: ScenarioResult) -> str:
    """A readable multi-line report of why a scenario failed."""
    lines = [f"pyTracer scenario {result.name!r} failed:"]
    if result.config_error:
        lines.append(f"  config error: {result.config_error}")
    elif result.replay.failed:
        lines.append(
            f"  replay errored: {result.replay.error_type}: {result.replay.error_message}"
        )
    for assertion in result.assertions:
        status = "ok" if assertion.passed else "FAIL"
        lines.append(f"  [{status}] {assertion.name}: {assertion.detail}".rstrip())
    return "\n".join(lines)


class _ScenarioFailed(AssertionError):
    """Raised with the full failure report so pytest shows it verbatim."""

    def __init__(self, result: ScenarioResult) -> None:
        super().__init__(_format_failure(result))


class CollectionFile(pytest.File):
    """A pyTracer collection file; each scenario in it becomes a test item."""

    def collect(self) -> Iterator[ScenarioItem]:
        collection, base_dir = load_collection(self.path)
        for scenario in collection.scenarios:
            yield ScenarioItem.from_parent(
                self, name=scenario.name, scenario=scenario, base_dir=base_dir
            )


class ScenarioItem(pytest.Item):
    """One scenario: replay it through the shared runner and assert it passed."""

    def __init__(self, *, scenario: ScenarioSpec, base_dir: Path, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.scenario = scenario
        self.base_dir = base_dir

    def runtest(self) -> None:
        result = run_scenario(self.scenario, self.base_dir)
        if not result.passed:
            raise _ScenarioFailed(result)

"""The meta-package re-exports the SDK entrypoints and every component imports."""

from __future__ import annotations

import importlib

import pytracer


def test_version() -> None:
    assert pytracer.__version__ == "0.0.1"


def test_reexports_sdk_entrypoints() -> None:
    # `import pytracer` alone is enough to instrument an agent.
    assert callable(pytracer.op)
    assert callable(pytracer.configure)
    assert pytracer.SpanKind.TOOL.value == "tool"


def test_all_components_are_installed() -> None:
    for module in (
        "pytracer_sdk",
        "pytracer_stores",
        "pytracer_gateway",
        "pytracer_replay",
        "pytracer_runner",
        "pytracer_registry",
        "pytracer_cli",
    ):
        assert importlib.import_module(module) is not None

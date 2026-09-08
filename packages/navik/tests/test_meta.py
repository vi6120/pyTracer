"""The meta-package re-exports the SDK entrypoints and every component imports."""

from __future__ import annotations

import importlib

import navik


def test_version() -> None:
    assert navik.__version__ == "0.0.1"


def test_reexports_sdk_entrypoints() -> None:
    # `import navik` alone is enough to instrument an agent.
    assert callable(navik.op)
    assert callable(navik.configure)
    assert navik.SpanKind.TOOL.value == "tool"


def test_all_components_are_installed() -> None:
    for module in (
        "navik_sdk",
        "navik_stores",
        "navik_gateway",
        "navik_replay",
        "navik_runner",
        "navik_registry",
        "navik_cli",
    ):
        assert importlib.import_module(module) is not None

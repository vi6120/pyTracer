"""Navik CLI — run agent test collections locally and in CI."""

from __future__ import annotations

from .collection import AssertionSpec, Collection, ScenarioSpec, load_collection
from .report import dumps_json, issue_body, to_json, to_junit, to_markdown
from .runner import (
    CollectionResult,
    ScenarioError,
    ScenarioResult,
    run_collection,
    run_path,
    run_scenario,
)

__all__ = [
    "AssertionSpec",
    "Collection",
    "CollectionResult",
    "ScenarioError",
    "ScenarioResult",
    "ScenarioSpec",
    "dumps_json",
    "issue_body",
    "load_collection",
    "run_collection",
    "run_path",
    "run_scenario",
    "to_json",
    "to_junit",
    "to_markdown",
]

__version__ = "0.0.1"

"""pyTracer CLI - run agent test collections locally and in CI."""

from __future__ import annotations

from .collection import AssertionSpec, Collection, ScenarioSpec, load_collection
from .github import GitHubClient, IssueRef, SyncResult, sync_issue
from .report import (
    dumps_json,
    issue_body,
    issue_body_from_json,
    issue_title,
    to_json,
    to_junit,
    to_markdown,
)
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
    "GitHubClient",
    "IssueRef",
    "ScenarioError",
    "ScenarioResult",
    "ScenarioSpec",
    "SyncResult",
    "dumps_json",
    "issue_body",
    "issue_body_from_json",
    "issue_title",
    "load_collection",
    "run_collection",
    "run_path",
    "run_scenario",
    "sync_issue",
    "to_json",
    "to_junit",
    "to_markdown",
]

__version__ = "0.0.1"

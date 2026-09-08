"""Navik replay & test engine — deterministic replay, assertions, diffing, and
cross-branch outcome classification.
"""

from __future__ import annotations

from .assertions import (
    Assertion,
    AssertionResult,
    Contains,
    ExactMatch,
    LLMJudge,
    NoErrors,
    Predicate,
    SemanticSimilarity,
    all_passed,
    run_assertions,
    token_jaccard,
)
from .classify import Outcome, classify
from .diff import diff_results, diff_traces
from .engine import MissingMockError, ReplayEngine, ReplayMode, ReplayResult
from .fingerprint import failure_fingerprint, fingerprint
from .hashing import canonical_json, input_hash
from .mocks import MockSet

__all__ = [
    "Assertion",
    "AssertionResult",
    "Contains",
    "ExactMatch",
    "LLMJudge",
    "MissingMockError",
    "MockSet",
    "NoErrors",
    "Outcome",
    "Predicate",
    "ReplayEngine",
    "ReplayMode",
    "ReplayResult",
    "SemanticSimilarity",
    "all_passed",
    "canonical_json",
    "classify",
    "diff_results",
    "diff_traces",
    "failure_fingerprint",
    "fingerprint",
    "input_hash",
    "run_assertions",
    "token_jaccard",
]

__version__ = "0.0.1"

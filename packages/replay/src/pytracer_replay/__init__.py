"""pyTracer replay & test engine - deterministic replay, assertions, diffing, and
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
    Route,
    SemanticSimilarity,
    StepCount,
    ToolsUsed,
    all_passed,
    run_assertions,
    token_jaccard,
)
from .classify import Outcome, classify
from .diff import diff_results, diff_traces
from .engine import MissingMockError, ReplayedError, ReplayEngine, ReplayMode, ReplayResult
from .fidelity import (
    CallKey,
    Chunk,
    ConcurrencyLedger,
    NondeterminismControls,
    RecordedStream,
    RetryDeduper,
    SideEffect,
    SideEffectKind,
    SideEffectLog,
    controlled,
    stream_from_text,
)
from .fingerprint import failure_fingerprint, fingerprint
from .hashing import canonical_json, input_hash
from .mocks import MockSet, RecordedError
from .sandbox import BlockedNetworkError, NetworkSandbox

__all__ = [
    "Assertion",
    "AssertionResult",
    "BlockedNetworkError",
    "CallKey",
    "Chunk",
    "ConcurrencyLedger",
    "Contains",
    "ExactMatch",
    "LLMJudge",
    "MissingMockError",
    "MockSet",
    "NetworkSandbox",
    "NoErrors",
    "NondeterminismControls",
    "Outcome",
    "Predicate",
    "RecordedError",
    "RecordedStream",
    "ReplayedError",
    "ReplayEngine",
    "ReplayMode",
    "ReplayResult",
    "RetryDeduper",
    "Route",
    "SemanticSimilarity",
    "StepCount",
    "SideEffect",
    "SideEffectKind",
    "SideEffectLog",
    "ToolsUsed",
    "all_passed",
    "canonical_json",
    "classify",
    "controlled",
    "diff_results",
    "diff_traces",
    "failure_fingerprint",
    "fingerprint",
    "input_hash",
    "run_assertions",
    "stream_from_text",
    "token_jaccard",
]

__version__ = "0.1.1"

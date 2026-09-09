"""pyTracer per-commit isolated runner - reproduce a trace against any branch."""

from __future__ import annotations

from .executor import DockerExecutor, ExecResult, Executor, LocalExecutor
from .runner import CrossBranchRunner, ReproduceResult, classify_reproduction
from .source import CheckoutError, GitSourceProvider, SourceProvider

__all__ = [
    "CheckoutError",
    "CrossBranchRunner",
    "DockerExecutor",
    "ExecResult",
    "Executor",
    "GitSourceProvider",
    "LocalExecutor",
    "ReproduceResult",
    "SourceProvider",
    "classify_reproduction",
]

__version__ = "0.0.1"

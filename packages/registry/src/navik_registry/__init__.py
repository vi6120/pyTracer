"""Navik collaboration registry — versioned, forkable, access-controlled
test collections.
"""

from __future__ import annotations

from .diffmerge import MergeConflict, diff_artifacts, three_way_merge
from .models import (
    Collection,
    CollectionArtifact,
    ForkRef,
    PullRequest,
    Version,
    Visibility,
)
from .rbac import can_edit, can_fork, can_view
from .registry import AccessDenied, NotFoundError, Registry
from .store import FilesystemStore, InMemoryStore, Store

__all__ = [
    "AccessDenied",
    "Collection",
    "CollectionArtifact",
    "FilesystemStore",
    "ForkRef",
    "InMemoryStore",
    "MergeConflict",
    "NotFoundError",
    "PullRequest",
    "Registry",
    "Store",
    "Version",
    "Visibility",
    "can_edit",
    "can_fork",
    "can_view",
    "diff_artifacts",
    "three_way_merge",
]

__version__ = "0.0.1"

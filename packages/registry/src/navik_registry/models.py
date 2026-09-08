"""Registry data models.

A **collection** is a versioned repository of test scenarios. Its content (the
:class:`CollectionArtifact`) is a diffable document; the collection tracks an
append-only history of :class:`Version` records, its owner and visibility, and
its fork lineage.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Visibility(str, Enum):
    """Who may see and interact with a collection."""

    PRIVATE = "private"  # owner only
    TEAM = "team"  # owner + named team members
    PUBLIC = "public"  # anyone


class CollectionArtifact(BaseModel):
    """The diffable content of a collection: scenarios plus discovery metadata."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    framework: str | None = None  # e.g. "langgraph", "crewai" - for discovery
    use_case: str | None = None  # e.g. "rag", "tool-use" - for discovery
    scenarios: dict[str, dict[str, Any]] = Field(default_factory=dict)


class Version(BaseModel):
    """One immutable point in a collection's history."""

    id: str
    parent_id: str | None
    content_hash: str
    artifact: CollectionArtifact
    message: str
    author: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ForkRef(BaseModel):
    """Points at the origin collection and version a fork was created from."""

    collection_id: str
    version_id: str


class Collection(BaseModel):
    """A versioned, access-controlled repository of test scenarios."""

    id: str
    owner: str
    name: str
    visibility: Visibility
    team: list[str] = Field(default_factory=list)
    framework: str | None = None
    use_case: str | None = None
    versions: list[Version]
    forked_from: ForkRef | None = None

    @property
    def head(self) -> Version:
        """The most recent version (the tip of history)."""
        return self.versions[-1]

    def version(self, version_id: str) -> Version | None:
        """Look up a version by id, or None if it is not in this collection."""
        return next((v for v in self.versions if v.id == version_id), None)


class PullRequest(BaseModel):
    """A proposal to merge one collection's changes into another.

    The proposed content is *snapshotted* at open time (the author can see their
    own fork), so merging never needs the target maintainer to read the source
    collection directly - mirroring how a git PR is against a fixed commit.
    """

    id: str
    source_id: str  # the fork proposing changes
    target_id: str  # the collection to merge into
    base_version_id: str  # common ancestor (the fork's root version)
    title: str
    author: str
    diff: dict[str, Any]
    base_artifact: CollectionArtifact  # common ancestor content
    source_artifact: CollectionArtifact  # proposed content (source head at open time)

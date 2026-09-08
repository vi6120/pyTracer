"""Registry facade — create, version, fork, propose, merge, and discover
collections, enforcing access control on every operation.
"""

from __future__ import annotations

import uuid

from .diffmerge import diff_artifacts, three_way_merge
from .hashing import content_hash
from .models import (
    Collection,
    CollectionArtifact,
    ForkRef,
    PullRequest,
    Version,
    Visibility,
)
from .rbac import can_edit, can_fork, can_view
from .store import InMemoryStore, Store


class NotFoundError(KeyError):
    """The collection or pull request does not exist (or is not visible)."""


class AccessDenied(PermissionError):
    """The acting user lacks permission for the operation."""


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class Registry:
    def __init__(self, store: Store | None = None) -> None:
        self._store = store or InMemoryStore()
        self._prs: dict[str, PullRequest] = {}

    # --- helpers ------------------------------------------------------------
    def _require_visible(self, user: str | None, collection_id: str) -> Collection:
        collection = self._store.get(collection_id)
        # Do not leak existence of collections the user cannot see.
        if collection is None or not can_view(user, collection):
            raise NotFoundError(collection_id)
        return collection

    def _new_version(self, artifact: CollectionArtifact, parent_id: str | None,
                     message: str, author: str) -> Version:
        return Version(
            id=_new_id("v"),
            parent_id=parent_id,
            content_hash=content_hash(artifact.model_dump()),
            artifact=artifact,
            message=message,
            author=author,
        )

    # --- collection lifecycle ----------------------------------------------
    def create_collection(
        self,
        owner: str,
        artifact: CollectionArtifact,
        *,
        visibility: Visibility = Visibility.PRIVATE,
        team: list[str] | None = None,
    ) -> Collection:
        version = self._new_version(artifact, None, "initial", owner)
        collection = Collection(
            id=_new_id("col"),
            owner=owner,
            name=artifact.name,
            visibility=visibility,
            team=team or [],
            framework=artifact.framework,
            use_case=artifact.use_case,
            versions=[version],
        )
        self._store.put(collection)
        return collection

    def get(self, user: str | None, collection_id: str) -> Collection:
        """Return a collection the user is allowed to see, else raise NotFound."""
        return self._require_visible(user, collection_id)

    def commit(
        self, user: str | None, collection_id: str, artifact: CollectionArtifact, message: str
    ) -> Version:
        collection = self._require_visible(user, collection_id)
        if not can_edit(user, collection):
            raise AccessDenied(f"{user!r} cannot edit {collection_id}")
        version = self._new_version(artifact, collection.head.id, message, user or "unknown")
        collection.versions.append(version)
        collection.name = artifact.name
        collection.framework = artifact.framework
        collection.use_case = artifact.use_case
        self._store.put(collection)
        return version

    # --- fork & pull requests ----------------------------------------------
    def fork(
        self, user: str, collection_id: str, *, name: str | None = None
    ) -> Collection:
        source = self._require_visible(user, collection_id)
        if not can_fork(user, source):
            raise AccessDenied(f"{user!r} cannot fork {collection_id}")
        # Deep-copy the head artifact so the fork is fully independent.
        artifact = source.head.artifact.model_copy(deep=True)
        if name is not None:
            artifact.name = name
        root = self._new_version(artifact, None, f"forked from {collection_id}", user)
        fork = Collection(
            id=_new_id("col"),
            owner=user,
            name=artifact.name,
            visibility=Visibility.PRIVATE,  # forks start private
            versions=[root],
            framework=artifact.framework,
            use_case=artifact.use_case,
            forked_from=ForkRef(collection_id=source.id, version_id=source.head.id),
        )
        self._store.put(fork)
        return fork

    def open_pull_request(
        self, user: str, source_id: str, target_id: str, title: str
    ) -> PullRequest:
        source = self._require_visible(user, source_id)
        target = self._require_visible(user, target_id)
        diff = diff_artifacts(target.head.artifact, source.head.artifact)
        pr = PullRequest(
            id=_new_id("pr"),
            source_id=source.id,
            target_id=target.id,
            base_version_id=source.versions[0].id,  # the fork root is the common ancestor
            title=title,
            author=user,
            diff=diff,
            base_artifact=source.versions[0].artifact.model_copy(deep=True),
            source_artifact=source.head.artifact.model_copy(deep=True),
        )
        self._prs[pr.id] = pr
        return pr

    def get_pull_request(self, pr_id: str) -> PullRequest:
        pr = self._prs.get(pr_id)
        if pr is None:
            raise NotFoundError(pr_id)
        return pr

    def merge_pull_request(self, user: str | None, pr_id: str) -> Version:
        """Three-way merge the PR into its target. Raises MergeConflict on clash."""
        pr = self.get_pull_request(pr_id)
        target = self._require_visible(user, pr.target_id)
        if not can_edit(user, target):
            raise AccessDenied(f"{user!r} cannot merge into {pr.target_id}")
        # Merge the PR's snapshot; no direct read of the (possibly private) source.
        merged = three_way_merge(pr.base_artifact, target.head.artifact, pr.source_artifact)
        return self.commit(user, target.id, merged, f"Merge PR {pr.id}: {pr.title}")

    # --- discovery ----------------------------------------------------------
    def discover(
        self,
        viewer: str | None = None,
        *,
        framework: str | None = None,
        use_case: str | None = None,
    ) -> list[Collection]:
        """List collections the viewer may see, filtered by framework/use case."""
        results = []
        for collection in self._store.all():
            if not can_view(viewer, collection):
                continue
            if framework is not None and collection.framework != framework:
                continue
            if use_case is not None and collection.use_case != use_case:
                continue
            results.append(collection)
        results.sort(key=lambda c: c.name)
        return results

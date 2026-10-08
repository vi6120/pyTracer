"""FastAPI application for the collaboration registry (optional ``[api]`` extra).

Exposes the registry library over HTTP so a web UI (or any client) can create,
version, fork, propose, merge, and discover collections. Every operation is the
library's, so the same access control applies here: reads may be anonymous (and
see only public collections), writes require an authenticated user.

Auth is a small seam: an API key in the ``X-API-Key`` header resolves to a user
id via ``UserAuth``. ``StaticUserAuth`` reads a key->user map for local and small
deployments; a database-backed resolver can drop in behind the same protocol.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .diffmerge import MergeConflict
from .models import Collection, CollectionArtifact, PullRequest, Version, Visibility
from .registry import AccessDenied, NotFoundError, Registry
from .store import FilesystemStore, InMemoryStore

API_KEY_HEADER = "X-API-Key"


class UserAuth(Protocol):
    """Resolve a request credential to a user id, or None for anonymous."""

    def user_for(self, api_key: str | None) -> str | None: ...


@dataclass
class StaticUserAuth:
    """Map API keys to user ids from a static dict (dev / small deployments)."""

    keys: dict[str, str]

    def user_for(self, api_key: str | None) -> str | None:
        return self.keys.get(api_key) if api_key else None


class CreateCollectionRequest(BaseModel):
    artifact: CollectionArtifact
    visibility: Visibility = Visibility.PRIVATE
    team: list[str] = Field(default_factory=list)


class CommitRequest(BaseModel):
    artifact: CollectionArtifact
    message: str


class ForkRequest(BaseModel):
    name: str | None = None


class OpenPullRequestRequest(BaseModel):
    source_id: str
    target_id: str
    title: str


def create_app(*, registry: Registry | None = None, auth: UserAuth | None = None) -> FastAPI:
    """Build a registry API app around a registry and an auth resolver."""
    reg = registry or Registry()
    resolver: UserAuth = auth or StaticUserAuth({})
    app = FastAPI(title="pyTracer Registry", version="0.1.0")

    def current_user(
        x_api_key: str | None = Header(default=None, alias=API_KEY_HEADER),
    ) -> str | None:
        return resolver.user_for(x_api_key)

    def require_user(user: str | None = Depends(current_user)) -> str:
        if user is None:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "authentication required")
        return user

    @app.exception_handler(NotFoundError)
    async def _not_found(_request: Request, _exc: NotFoundError) -> JSONResponse:
        # A single 404 for missing-or-not-visible, so existence never leaks.
        return JSONResponse({"detail": "not found"}, status_code=status.HTTP_404_NOT_FOUND)

    @app.exception_handler(AccessDenied)
    async def _forbidden(_request: Request, exc: AccessDenied) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=status.HTTP_403_FORBIDDEN)

    @app.exception_handler(MergeConflict)
    async def _conflict(_request: Request, exc: MergeConflict) -> JSONResponse:
        return JSONResponse(
            {"detail": "merge conflict", "conflicts": exc.conflicts},
            status_code=status.HTTP_409_CONFLICT,
        )

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/collections")
    def discover(
        user: str | None = Depends(current_user),
        framework: str | None = Query(default=None),
        use_case: str | None = Query(default=None),
    ) -> list[Collection]:
        return reg.discover(user, framework=framework, use_case=use_case)

    @app.post("/collections", status_code=status.HTTP_201_CREATED)
    def create_collection(
        body: CreateCollectionRequest, user: str = Depends(require_user)
    ) -> Collection:
        return reg.create_collection(
            user, body.artifact, visibility=body.visibility, team=body.team
        )

    @app.get("/collections/{collection_id}")
    def get_collection(
        collection_id: str, user: str | None = Depends(current_user)
    ) -> Collection:
        return reg.get(user, collection_id)

    @app.post("/collections/{collection_id}/versions", status_code=status.HTTP_201_CREATED)
    def commit(
        collection_id: str, body: CommitRequest, user: str = Depends(require_user)
    ) -> Version:
        return reg.commit(user, collection_id, body.artifact, body.message)

    @app.post("/collections/{collection_id}/fork", status_code=status.HTTP_201_CREATED)
    def fork(
        collection_id: str, body: ForkRequest, user: str = Depends(require_user)
    ) -> Collection:
        return reg.fork(user, collection_id, name=body.name)

    @app.post("/pull-requests", status_code=status.HTTP_201_CREATED)
    def open_pull_request(
        body: OpenPullRequestRequest, user: str = Depends(require_user)
    ) -> PullRequest:
        return reg.open_pull_request(user, body.source_id, body.target_id, body.title)

    @app.get("/pull-requests/{pr_id}")
    def get_pull_request(pr_id: str) -> PullRequest:
        return reg.get_pull_request(pr_id)

    @app.post("/pull-requests/{pr_id}/merge", status_code=status.HTTP_201_CREATED)
    def merge_pull_request(pr_id: str, user: str = Depends(require_user)) -> Version:
        return reg.merge_pull_request(user, pr_id)

    return app


def _load_keys() -> dict[str, str]:
    """Parse ``PYTRACER_REGISTRY_API_KEYS`` into a ``{api_key: user}`` mapping."""
    raw = os.getenv("PYTRACER_REGISTRY_API_KEYS", "").strip()
    keys: dict[str, str] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair or ":" not in pair:
            continue
        key, user = pair.split(":", 1)
        key, user = key.strip(), user.strip()
        if key and user:
            keys[key] = user
    return keys


def build_default_app() -> FastAPI:
    """Factory used by uvicorn: a filesystem-backed registry when configured.

    Set ``PYTRACER_REGISTRY_DIR`` to persist collections on disk; otherwise the
    store is in-memory. API keys come from ``PYTRACER_REGISTRY_API_KEYS``
    (``key:user,key:user``).
    """
    store_dir = os.getenv("PYTRACER_REGISTRY_DIR")
    store = FilesystemStore(store_dir) if store_dir else InMemoryStore()
    return create_app(registry=Registry(store), auth=StaticUserAuth(_load_keys()))

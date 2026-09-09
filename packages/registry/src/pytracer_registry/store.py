"""Storage backends for collections.

Each backend returns an independent copy of a collection on every read, so a
caller mutating what it reads can never corrupt another caller's view - the
same isolation a real database or a git repo per collection would give. The
filesystem backend writes one JSON file per collection, diffable in git.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .models import Collection


class Store(Protocol):
    """Persistence for collections. ``get``/``all`` return independent copies.

    ``put`` upserts by collection id; ``get`` returns None when absent; ``all``
    returns every stored collection.
    """

    def put(self, collection: Collection) -> None: ...
    def get(self, collection_id: str) -> Collection | None: ...
    def all(self) -> list[Collection]: ...


class InMemoryStore:
    """Keeps collections as serialized JSON so every read is an independent copy."""

    def __init__(self) -> None:
        self._data: dict[str, str] = {}

    def put(self, collection: Collection) -> None:
        self._data[collection.id] = collection.model_dump_json()

    def get(self, collection_id: str) -> Collection | None:
        raw = self._data.get(collection_id)
        return Collection.model_validate_json(raw) if raw is not None else None

    def all(self) -> list[Collection]:
        return [Collection.model_validate_json(raw) for raw in self._data.values()]


class FilesystemStore:
    """Persists each collection as ``<root>/<id>.json`` (diffable in git)."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _path(self, collection_id: str) -> Path:
        return self._root / f"{collection_id}.json"

    def put(self, collection: Collection) -> None:
        self._path(collection.id).write_text(
            collection.model_dump_json(indent=2), encoding="utf-8"
        )

    def get(self, collection_id: str) -> Collection | None:
        path = self._path(collection_id)
        if not path.exists():
            return None
        return Collection.model_validate_json(path.read_text(encoding="utf-8"))

    def all(self) -> list[Collection]:
        return [
            Collection.model_validate_json(p.read_text(encoding="utf-8"))
            for p in sorted(self._root.glob("*.json"))
        ]

"""API-key store: create, resolve, revoke, list, and secret handling.

Runs against the docker-compose PostgreSQL; skips cleanly when it is not
reachable (see conftest.py).
"""

from __future__ import annotations

from navik_stores import ApiKeyStore
from navik_stores.api_key_store import KEY_PREFIX, generate_key, hash_key


def test_create_returns_full_key_once_and_resolves(api_key_store: ApiKeyStore) -> None:
    record, full_key = api_key_store.create("web-agent", name="ci")
    assert record.project == "web-agent"
    assert record.name == "ci"
    assert record.active is True
    # The full key is a prefixed, opaque token that embeds the public key_id.
    assert full_key.startswith(f"{KEY_PREFIX}_{record.key_id}_")
    # Presenting it resolves to the project.
    assert api_key_store.resolve(full_key) == "web-agent"


def test_secret_is_never_stored_in_plaintext(api_key_store: ApiKeyStore) -> None:
    _, full_key = api_key_store.create("proj")
    # Read the raw row: the stored hash matches the key, and the plaintext is absent.
    row = api_key_store._conn.execute(
        f"SELECT key_hash FROM {api_key_store._table}"
    ).fetchone()
    assert row is not None
    assert row[0] == hash_key(full_key)
    assert row[0] != full_key


def test_unknown_key_resolves_to_none(api_key_store: ApiKeyStore) -> None:
    assert api_key_store.resolve("nvk_deadbeef_notreal") is None
    assert api_key_store.resolve(None) is None
    assert api_key_store.resolve("") is None


def test_revoked_key_stops_resolving(api_key_store: ApiKeyStore) -> None:
    record, full_key = api_key_store.create("proj")
    assert api_key_store.resolve(full_key) == "proj"

    assert api_key_store.revoke(record.key_id) is True
    assert api_key_store.resolve(full_key) is None
    # Revoking again is a no-op (already inactive).
    assert api_key_store.revoke(record.key_id) is False

    fetched = api_key_store.get(record.key_id)
    assert fetched is not None
    assert fetched.active is False
    assert fetched.revoked_at is not None


def test_resolve_stamps_last_used(api_key_store: ApiKeyStore) -> None:
    record, full_key = api_key_store.create("proj")
    assert api_key_store.get(record.key_id).last_used_at is None  # type: ignore[union-attr]
    api_key_store.resolve(full_key)
    assert api_key_store.get(record.key_id).last_used_at is not None  # type: ignore[union-attr]


def test_list_filters_by_project_and_revoked(api_key_store: ApiKeyStore) -> None:
    a, _ = api_key_store.create("alpha")
    api_key_store.create("alpha")
    api_key_store.create("beta")

    assert len(api_key_store.list()) == 3
    assert len(api_key_store.list(project="alpha")) == 2
    assert api_key_store.count(project="beta") == 1

    api_key_store.revoke(a.key_id)
    assert len(api_key_store.list(project="alpha")) == 1  # active only by default
    assert len(api_key_store.list(project="alpha", include_revoked=True)) == 2


def test_generate_key_is_unique_and_hashes_stably() -> None:
    id1, key1 = generate_key()
    id2, key2 = generate_key()
    assert id1 != id2
    assert key1 != key2
    assert hash_key(key1) == hash_key(key1)  # deterministic
    assert hash_key(key1) != hash_key(key2)

"""Registry HTTP API: auth, RBAC visibility, and the fork -> PR -> merge flow."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from pytracer_registry import Registry
from pytracer_registry.api import StaticUserAuth, create_app

ALICE = {"X-API-Key": "alice"}
BOB = {"X-API-Key": "bob"}


@pytest.fixture
def client() -> TestClient:
    auth = StaticUserAuth({"alice": "alice", "bob": "bob"})
    return TestClient(create_app(registry=Registry(), auth=auth))


def _create(
    client: TestClient, headers: dict[str, str], *, visibility: str = "private", **art: Any
) -> dict[str, Any]:
    art.setdefault("name", "suite")
    resp = client.post(
        "/collections", json={"artifact": art, "visibility": visibility}, headers=headers
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def test_create_requires_auth(client: TestClient) -> None:
    assert client.post("/collections", json={"artifact": {"name": "x"}}).status_code == 401


def test_create_and_owner_can_get(client: TestClient) -> None:
    col = _create(client, ALICE, name="mine")
    assert col["owner"] == "alice"
    got = client.get(f"/collections/{col['id']}", headers=ALICE)
    assert got.status_code == 200
    assert got.json()["name"] == "mine"


def test_private_hidden_from_others_and_anonymous(client: TestClient) -> None:
    col = _create(client, ALICE, visibility="private")
    assert client.get(f"/collections/{col['id']}", headers=BOB).status_code == 404
    assert client.get(f"/collections/{col['id']}").status_code == 404


def test_public_is_discoverable_anonymously_with_filters(client: TestClient) -> None:
    _create(client, ALICE, name="pub", visibility="public", framework="langgraph")
    listed = client.get("/collections")
    assert listed.status_code == 200
    assert "pub" in [c["name"] for c in listed.json()]
    assert client.get("/collections", params={"framework": "crewai"}).json() == []


def test_commit_requires_edit_rights(client: TestClient) -> None:
    cid = _create(client, ALICE, visibility="public")["id"]
    ok = client.post(
        f"/collections/{cid}/versions",
        headers=ALICE,
        json={"artifact": {"name": "v2"}, "message": "bump"},
    )
    assert ok.status_code == 201
    denied = client.post(
        f"/collections/{cid}/versions",
        headers=BOB,
        json={"artifact": {"name": "hack"}, "message": "no"},
    )
    assert denied.status_code == 403


def test_fork_pull_request_merge_flow(client: TestClient) -> None:
    cid = _create(client, ALICE, name="orig", visibility="public")["id"]

    fork = client.post(f"/collections/{cid}/fork", headers=BOB, json={})
    assert fork.status_code == 201
    fork_id = fork.json()["id"]
    assert fork.json()["owner"] == "bob"

    client.post(
        f"/collections/{fork_id}/versions",
        headers=BOB,
        json={"artifact": {"name": "orig", "description": "improved"}, "message": "improve"},
    )

    pr = client.post(
        "/pull-requests",
        headers=BOB,
        json={"source_id": fork_id, "target_id": cid, "title": "improve"},
    )
    assert pr.status_code == 201
    pr_id = pr.json()["id"]

    # Bob cannot merge into Alice's collection; Alice can.
    assert client.post(f"/pull-requests/{pr_id}/merge", headers=BOB).status_code == 403
    merged = client.post(f"/pull-requests/{pr_id}/merge", headers=ALICE)
    assert merged.status_code == 201


def test_missing_collection_is_404(client: TestClient) -> None:
    assert client.get("/collections/col_missing", headers=ALICE).status_code == 404

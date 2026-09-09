"""Auth: an invalid or missing API key is rejected before any processing."""

from __future__ import annotations

from fastapi.testclient import TestClient
from helpers import FakeWriter, make_span

from pytracer_gateway import create_app

API_KEYS = {"goodkey": "proj-a"}


def _client(writer: FakeWriter) -> TestClient:
    return TestClient(create_app(writer=writer, api_keys=API_KEYS))


def test_missing_key_is_rejected() -> None:
    writer = FakeWriter()
    with _client(writer) as client:
        resp = client.post("/v1/spans", json={"spans": [make_span().model_dump(mode="json")]})
    assert resp.status_code == 401
    assert writer.count == 0  # rejected before any span was processed


def test_invalid_key_is_rejected() -> None:
    writer = FakeWriter()
    with _client(writer) as client:
        resp = client.post(
            "/v1/spans",
            headers={"X-API-Key": "wrong"},
            json={"spans": [make_span().model_dump(mode="json")]},
        )
    assert resp.status_code == 401
    assert writer.count == 0


def test_valid_key_is_accepted() -> None:
    writer = FakeWriter()
    with _client(writer) as client:
        resp = client.post(
            "/v1/spans",
            headers={"X-API-Key": "goodkey"},
            json={"spans": [make_span().model_dump(mode="json")]},
        )
    assert resp.status_code == 202
    assert resp.json()["accepted"] == 1

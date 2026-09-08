"""End-to-end ingest: valid spans reach the writer; invalid ones are rejected."""

from __future__ import annotations

from collections.abc import Callable

from fastapi.testclient import TestClient
from helpers import FakeWriter, make_span

from navik_gateway import create_app

API_KEYS = {"k": "proj"}
HEADERS = {"X-API-Key": "k"}


def _written(client: TestClient) -> int:
    return int(client.get("/v1/stats").json()["stats"]["written"])


def test_batch_lands_in_storage(wait_until: Callable[..., bool]) -> None:
    writer = FakeWriter()
    spans = [make_span(f"s{i}").model_dump(mode="json") for i in range(10)]
    with TestClient(create_app(writer=writer, api_keys=API_KEYS)) as client:
        resp = client.post("/v1/spans", headers=HEADERS, json={"spans": spans})
        assert resp.status_code == 202
        assert resp.json()["accepted"] == 10
        assert wait_until(lambda: writer.count == 10)
    # Spans were tagged with the project resolved from the API key.
    assert all(project == "proj" for project, _ in writer.records)


def test_bare_list_body_is_accepted(wait_until: Callable[..., bool]) -> None:
    writer = FakeWriter()
    spans = [make_span().model_dump(mode="json") for _ in range(3)]
    with TestClient(create_app(writer=writer, api_keys=API_KEYS)) as client:
        resp = client.post("/v1/spans", headers=HEADERS, json=spans)
        assert resp.status_code == 202
        assert wait_until(lambda: writer.count == 3)


def test_partial_batch_rejects_invalid_keeps_valid(wait_until: Callable[..., bool]) -> None:
    writer = FakeWriter()
    good = make_span("good").model_dump(mode="json")
    bad = {"span_id": "tooshort", "name": "x"}  # missing/invalid required fields
    with TestClient(create_app(writer=writer, api_keys=API_KEYS)) as client:
        resp = client.post("/v1/spans", headers=HEADERS, json={"spans": [good, bad]})
        assert resp.status_code == 202
        body = resp.json()
        assert body["accepted"] == 1
        assert body["rejected"] == 1
        assert body["errors"][0]["index"] == 1
        assert wait_until(lambda: writer.count == 1)
        assert client.get("/v1/stats").json()["dead_letters"] == 1


def test_malformed_json_is_400() -> None:
    writer = FakeWriter()
    with TestClient(create_app(writer=writer, api_keys=API_KEYS)) as client:
        resp = client.post(
            "/v1/spans", headers={**HEADERS, "Content-Type": "application/json"}, content="{"
        )
    assert resp.status_code == 400


def test_healthz_and_readyz() -> None:
    with TestClient(create_app(writer=FakeWriter(), api_keys=API_KEYS)) as client:
        assert client.get("/healthz").json() == {"status": "ok"}
        assert client.get("/readyz").status_code == 200

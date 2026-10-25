"""End-to-end ingest: valid spans reach the writer; invalid ones are rejected."""

from __future__ import annotations

from collections.abc import Callable

from fastapi.testclient import TestClient
from helpers import FakeWriter, make_span

from pytracer_gateway import create_app
from pytracer_gateway.app import GATEWAY_VERSION

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
        assert client.get("/healthz").json() == {"status": "ok", "version": GATEWAY_VERSION}
        assert client.get("/readyz").status_code == 200


def test_gateway_reports_its_installed_version() -> None:
    # /healthz and the OpenAPI schema report the deployed release, not a stale
    # hardcoded 0.0.1, so an operator can confirm what a gateway runs.
    from importlib.metadata import version

    assert GATEWAY_VERSION == version("pytracer-gateway")
    with TestClient(create_app(writer=FakeWriter(), api_keys=API_KEYS)) as client:
        assert client.get("/openapi.json").json()["info"]["version"] == GATEWAY_VERSION
        assert client.get("/healthz").json()["version"] == GATEWAY_VERSION


def test_span_missing_branch_or_commit_is_rejected_at_ingest() -> None:
    # The trace store requires branch/commit; reject at ingest so the sender learns
    # now, instead of a silent accept-then-dead-letter at write time.
    from pytracer_sdk import new_span_id, new_trace_id
    from pytracer_sdk.schema import Resource, Span, SpanKind

    writer = FakeWriter()
    no_meta = Span(
        trace_id=new_trace_id(),
        span_id=new_span_id(),
        name="x",
        kind=SpanKind.TOOL,
        start_time_ns=1,
        end_time_ns=2,
        resource=Resource(service_name="t"),  # no branch/commit
    ).model_dump(mode="json")
    with TestClient(create_app(writer=writer, api_keys=API_KEYS)) as client:
        resp = client.post("/v1/spans", headers=HEADERS, json={"spans": [no_meta]})
        assert resp.status_code == 202
        body = resp.json()
        assert body["accepted"] == 0
        assert body["rejected"] == 1
        assert "branch" in str(body["errors"][0]["error"])
        assert client.get("/v1/stats").json()["stats"]["written"] == 0


def test_requeue_re_enqueues_storage_dead_letters() -> None:
    # After an outage, storage dead letters can be re-queued for another write;
    # validation dead letters are kept (not re-runnable).
    import asyncio

    from pytracer_gateway import GatewayConfig
    from pytracer_gateway.ingest import IngestPipeline
    from pytracer_gateway.queue import DeadLetter

    async def scenario() -> None:
        pipe = IngestPipeline(FakeWriter(), GatewayConfig())
        await pipe._queue.start()
        payload = make_span("replayed").model_dump(mode="json")
        await pipe._queue.push_dead_letters(
            [
                DeadLetter("storage", "proj", "outage", payload),
                DeadLetter("validation", "proj", "bad schema", None),
            ]
        )
        assert await pipe.dead_letter_count() == 2
        requeued = await pipe.requeue_dead_letters(10)
        assert requeued == 1
        assert await pipe.dead_letter_count() == 1  # the validation one is kept
        batch = await pipe._queue.reserve(10, 0.1)
        assert [r.span.name for r in batch] == ["replayed"]
        await pipe._queue.close()

    asyncio.run(scenario())


def test_ingest_never_instantiates_a_class_named_in_a_payload() -> None:
    # A client controls the payload; the gateway only stores it. A tagged value
    # naming any class (here subprocess.Popen) must be kept as data, unimported.
    from pytracer_gateway.ingest import validate_spans

    gadget = {"__pytracer__": "path", "v": ["true"], "cls": "subprocess:Popen"}
    span = make_span().model_dump(mode="json")
    span["output"] = gadget
    valid, errors = validate_spans([span])
    assert errors == []
    assert valid[0].output == gadget  # kept as the client wrote it
    stored = valid[0].model_dump(mode="json")["output"]
    assert stored == gadget  # and stored unchanged

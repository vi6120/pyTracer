# pytracer-gateway

Layer 2 of pyTracer: a stateless FastAPI service that receives batched spans from
deployed SDKs, validates them against the span schema, authenticates each
project by API key, and writes them through to the ClickHouse trace store via
an internal queue so a traffic spike never blocks ingestion.

```bash
# Run it (writes to ClickHouse from docker-compose):
PYTRACER_GATEWAY_API_KEYS="devkey:my-app" uvicorn pytracer_gateway.app:build_default_app --factory --port 8080
```

```
POST /v1/spans      body: {"spans": [ <span>, ... ]}   header: X-API-Key: <key>
GET  /healthz       liveness, and the gateway's version
GET  /readyz        readiness (ingest worker running)
GET  /v1/stats      ingest counters
```

## Compatibility

The gateway ignores span fields it does not recognise instead of rejecting the
span, so a newer SDK that adds an optional field still ingests against an older
gateway. When upgrading a release that adds a span field, upgrade the gateway
before the SDKs that send to it, so the field is understood wherever it lands.

# navik-gateway

Layer 2 of Navik: a stateless FastAPI service that receives batched spans from
deployed SDKs, validates them against the span schema, authenticates each
project by API key, and writes them through to the ClickHouse trace store via
an internal queue so a traffic spike never blocks ingestion.

```bash
# Run it (writes to ClickHouse from docker-compose):
NAVIK_GATEWAY_API_KEYS="devkey:my-app" uvicorn navik_gateway.app:build_default_app --factory --port 8080
```

```
POST /v1/spans      body: {"spans": [ <span>, ... ]}   header: X-API-Key: <key>
GET  /healthz       liveness
GET  /readyz        readiness (ingest worker running)
GET  /v1/stats      ingest counters
```

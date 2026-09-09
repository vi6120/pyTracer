# Deploying pyTracer

This runs the ingestion gateway alongside its stores (ClickHouse for traces,
PostgreSQL for mocks) and Redis (the durable ingest queue). It is a
single-instance setup suitable for a small team or a beta; see "Scaling" below
for where it grows.

## Run it

From the repository root:

```bash
docker compose -f deploy/docker-compose.yml up -d --build
```

This builds the gateway image from [packages/gateway/Dockerfile](../packages/gateway/Dockerfile),
starts ClickHouse, PostgreSQL, and Redis, waits for them to be healthy, then
starts the gateway on port 8080. The gateway migrates the trace-store schema on
startup, so there is no separate migration step.

Check it:

```bash
curl http://localhost:8080/healthz     # {"status":"ok"}
curl http://localhost:8080/v1/stats    # ingest counters
```

## Configure

Set these before a real deployment (defaults are for local trials only):

| Variable | Purpose | Default |
| --- | --- | --- |
| `PYTRACER_GATEWAY_API_KEYS` | `key:project[,key:project]` mapping clients to projects | `devkey:default` |
| `PYTRACER_CLICKHOUSE_PASSWORD` | ClickHouse password (traces) | `pytracer` |
| `PYTRACER_POSTGRES_PASSWORD` | PostgreSQL password (mocks) | `pytracer` |
| `PYTRACER_GATEWAY_QUEUE_BACKEND` | `redis` (durable) or `memory` (in-process) | `redis` in this stack |
| `PYTRACER_REDIS_URL` | Redis connection for the durable queue | `redis://redis:6379/0` |
| `PYTRACER_GATEWAY_AUTH_BACKEND` | `static` (env map) or `postgres` (managed keys) | `static` |
| `PYTRACER_GATEWAY_RATE_LIMIT_PER_SEC` | Per-key request rate; `0` disables limiting | `0` |
| `PYTRACER_GATEWAY_RATE_LIMIT_BURST` | Per-key burst ceiling; defaults to the rate | `0` |

For example:

```bash
PYTRACER_GATEWAY_API_KEYS="prod-abc123:web-agent" \
PYTRACER_CLICKHOUSE_PASSWORD="$(openssl rand -hex 16)" \
PYTRACER_POSTGRES_PASSWORD="$(openssl rand -hex 16)" \
  docker compose -f deploy/docker-compose.yml up -d --build
```

## Point the SDK at it

In your agent, ship spans to the deployed gateway:

```python
import pytracer_sdk as pytracer

pytracer.configure(
    pytracer.HTTPTransport("http://your-host:8080", api_key="prod-abc123"),
    service_name="web-agent", branch="main", commit="<sha>",
)
```

## Managing API keys

By default the gateway authenticates against the static `PYTRACER_GATEWAY_API_KEYS`
map, which is fine for a trial but cannot rotate or revoke a key without a
redeploy. For a real deployment, switch to the PostgreSQL-backed key store:

```bash
PYTRACER_GATEWAY_AUTH_BACKEND=postgres \
  docker compose -f deploy/docker-compose.yml up -d --build
```

Then create and revoke keys with the CLI (it needs the store client:
`pip install pytracer-cli[stores]`, and the same `PYTRACER_POSTGRES_*` settings the
gateway uses):

```bash
pytracer keys create web-agent --name ci   # prints the secret ONCE
pytracer keys list                         # never shows secrets
pytracer keys revoke <key_id>              # stops it authenticating
```

Keys are stored hashed (SHA-256), never in plaintext: the full key is shown only
at creation. A revoked key stops working within `PYTRACER_GATEWAY_AUTH_CACHE_TTL`
seconds (the gateway caches resolved keys to keep the request path fast).

Set a per-key rate limit with `PYTRACER_GATEWAY_RATE_LIMIT_PER_SEC` (a token bucket,
`PYTRACER_GATEWAY_RATE_LIMIT_BURST` for the burst ceiling); an over-limit client
gets `429` with a `Retry-After` header. The limit is enforced per gateway
instance.

## Durable ingestion

The gateway accepts spans on the request path and drains them to storage on a
background worker. The queue between the two runs on Redis Streams
(`PYTRACER_GATEWAY_QUEUE_BACKEND=redis`), so a gateway restart does not lose data:

- **In-flight spans** are held as unacknowledged stream entries. A span is
  acknowledged only after it is written (or moved to the dead-letter list), so a
  restart reclaims and replays anything a crashed worker had in hand.
- **The dead-letter list** (spans that failed schema validation, or failed to
  write after the retry budget) lives in a Redis stream too. Inspect it at
  `GET /v1/dead-letters`; it survives restarts instead of vanishing with the
  process.

Set `PYTRACER_GATEWAY_QUEUE_BACKEND=memory` to fall back to the in-process queue
(no Redis needed, but a restart drops whatever it holds). This is the default
only when you run the gateway outside this compose stack.

## Production notes

- **TLS**: the gateway speaks plain HTTP. Terminate TLS at a reverse proxy
  (nginx, Caddy, a cloud load balancer) in front of it, and expose only the
  proxy. Do not expose ClickHouse, PostgreSQL, or Redis publicly.
- **Secrets**: change the store passwords and use real API keys. Manage them
  through your orchestrator's secret store rather than committing them.
- **Persistence**: trace and mock data live in the `clickhouse-data` and
  `postgres-data` volumes, and the ingest backlog in `redis-data` (Redis runs
  with append-only persistence). Back them up; removing the volumes wipes the
  data.
- **Restart**: the gateway restarts automatically (`restart: unless-stopped`).

## Scaling

The gateway is stateless, so the first scaling step is running several gateway
instances behind the load balancer. They already share one durable queue (the
Redis stream), so this mostly works out of the box; when running more than one
instance, raise `PYTRACER_REDIS_RECLAIM_MIN_IDLE_MS` above zero so a healthy
instance does not reclaim a busy peer's in-flight entries. The trace store
shards by project or organization once single-node ClickHouse write throughput
becomes the bottleneck, and the single Redis node grows into a cluster (or is
swapped for NATS JetStream) at much higher ingest volumes.

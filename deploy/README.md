# Deploying Navik

This runs the ingestion gateway alongside its stores (ClickHouse for traces,
PostgreSQL for mocks). It is a single-instance setup suitable for a small team
or a beta; see "Scaling" below for where it grows.

## Run it

From the repository root:

```bash
docker compose -f deploy/docker-compose.yml up -d --build
```

This builds the gateway image from [packages/gateway/Dockerfile](../packages/gateway/Dockerfile),
starts ClickHouse and PostgreSQL, waits for them to be healthy, then starts the
gateway on port 8080. The gateway migrates the trace-store schema on startup, so
there is no separate migration step.

Check it:

```bash
curl http://localhost:8080/healthz     # {"status":"ok"}
curl http://localhost:8080/v1/stats    # ingest counters
```

## Configure

Set these before a real deployment (defaults are for local trials only):

| Variable | Purpose | Default |
| --- | --- | --- |
| `NAVIK_GATEWAY_API_KEYS` | `key:project[,key:project]` mapping clients to projects | `devkey:default` |
| `NAVIK_CLICKHOUSE_PASSWORD` | ClickHouse password (traces) | `navik` |
| `NAVIK_POSTGRES_PASSWORD` | PostgreSQL password (mocks) | `navik` |

For example:

```bash
NAVIK_GATEWAY_API_KEYS="prod-abc123:web-agent" \
NAVIK_CLICKHOUSE_PASSWORD="$(openssl rand -hex 16)" \
NAVIK_POSTGRES_PASSWORD="$(openssl rand -hex 16)" \
  docker compose -f deploy/docker-compose.yml up -d --build
```

## Point the SDK at it

In your agent, ship spans to the deployed gateway:

```python
import navik_sdk as navik

navik.configure(
    navik.HTTPTransport("http://your-host:8080", api_key="prod-abc123"),
    service_name="web-agent", branch="main", commit="<sha>",
)
```

## Production notes

- **TLS**: the gateway speaks plain HTTP. Terminate TLS at a reverse proxy
  (nginx, Caddy, a cloud load balancer) in front of it, and expose only the
  proxy. Do not expose ClickHouse or PostgreSQL publicly.
- **Secrets**: change the store passwords and use real API keys. Manage them
  through your orchestrator's secret store rather than committing them.
- **Persistence**: trace and mock data live in the `clickhouse-data` and
  `postgres-data` volumes. Back them up; removing the volumes wipes the data.
- **Restart**: the gateway restarts automatically (`restart: unless-stopped`).

## Scaling

The gateway is stateless, so the first scaling step is running several gateway
instances behind the load balancer. Its ingest queue is in-process today; a
multi-instance deployment swaps it for Redis Streams or NATS (the write-through
boundary is already isolated). The trace store shards by project or organization
once single-node ClickHouse write throughput becomes the bottleneck.

# Security and self-hosting hardening

pyTracer captures what your agents do (LLM calls, tool calls, reasoning, and
handoffs), so a deployment handles data that can be sensitive. This guide covers
how the platform protects that data and what you must configure when you run it
yourself. It is written for the single-instance deployment in [deploy/](deploy);
the same principles scale up.

## What pyTracer already does

- **Secret and PII redaction at capture time.** The SDK runs a redaction ruleset
  (API keys, emails, card numbers, and your own patterns) over span inputs and
  outputs *before anything leaves the process*, so secrets are scrubbed at the
  source rather than in storage. See `pytracer_sdk.redaction`. This is the
  hardest part of the problem and it is handled in the client.
- **Per-project authentication.** Every span batch is authenticated by an API
  key that maps to a project; an invalid or missing key is rejected before any
  span is processed.
- **Managed, hashed API keys.** With the Postgres key backend
  (`PYTRACER_GATEWAY_AUTH_BACKEND=postgres`), keys are stored as SHA-256 hashes,
  never plaintext, and can be created and revoked at runtime with
  `pytracer keys`. See [deploy/README.md](deploy/README.md#managing-api-keys).
- **Per-key rate limiting.** A token bucket caps sustained per-key throughput so
  one noisy or abusive key cannot crowd out others.
- **Durable, non-lossy ingestion.** In-flight spans and the dead-letter list
  survive a restart (Redis Streams), so data is not silently dropped.

## What you must configure

### TLS in transit

The gateway speaks plain HTTP so it can sit behind whatever TLS terminator you
already run. **Do not expose it directly.** Two supported paths:

1. **Bundled reverse proxy.** The deploy stack ships a Caddy service behind a
   `tls` profile that fetches and renews a certificate automatically:

   ```bash
   PYTRACER_DOMAIN=gateway.pytracer.com \
     docker compose -f deploy/docker-compose.yml --profile tls up -d --build
   ```

   Point the domain's DNS at the host, open ports 80 and 443, and stop
   publishing the gateway's own port 8080 to the public network.

2. **Your own proxy or load balancer.** Terminate TLS at nginx, a cloud load
   balancer, or a service mesh, and forward to the gateway on the internal
   network only.

On the client side, point the SDK at the `https://` endpoint. TLS is verified
against the system trust store by default; for an internal CA, pass its bundle:

```python
import pytracer_sdk as pytracer

pytracer.configure(
    pytracer.HTTPTransport(
        "https://gateway.pytracer.com",
        api_key="...",
        ca_bundle="/etc/ssl/internal-ca.pem",  # omit to use the system trust store
    ),
    service_name="web-agent", branch="main", commit="<sha>",
)
```

`verify=False` exists for local test servers only; never use it against a real
deployment.

### Network isolation

Only the TLS terminator should be reachable from outside. ClickHouse,
PostgreSQL, and Redis must stay on the internal network with no published ports.
In the deploy stack only the gateway (and, with the `tls` profile, Caddy) publish
ports; the stores do not. Keep it that way, and put the whole stack behind a
firewall or private network.

### Secrets management

- Change every default password (`PYTRACER_CLICKHOUSE_PASSWORD`,
  `PYTRACER_POSTGRES_PASSWORD`) and use real API keys before any real use.
- Supply secrets through your orchestrator's secret store (Docker/Swarm secrets,
  Kubernetes Secrets, a cloud secret manager) rather than committing them or
  baking them into images.
- Prefer the Postgres key backend so keys can be rotated and revoked without a
  redeploy.

### Encryption at rest

Span and mock data live in the ClickHouse and PostgreSQL volumes, and the ingest
backlog in the Redis volume. pyTracer relies on the storage layer for at-rest
encryption rather than reimplementing it:

- **Disk-level encryption** (LUKS, cloud provider volume encryption such as EBS
  or PD encryption) is the simplest and covers all three stores at once. Enable
  it on the volumes backing `clickhouse-data`, `postgres-data`, and
  `redis-data`.
- **Managed database encryption.** If you run ClickHouse Cloud or a managed
  Postgres, at-rest encryption is typically on by default; confirm it and enable
  key management to your policy.
- **Back up the encrypted volumes**, and treat backups with the same care as the
  live data.

Because the SDK redacts secrets before storage, the highest-sensitivity fields
should never reach these volumes in the first place; disk encryption protects the
remaining captured content.

### The dead-letter list

Spans that fail validation or exhaust write retries are kept in a Redis
dead-letter stream (inspectable at `GET /v1/dead-letters`) so nothing is lost
silently. It contains real span data, so secure Redis the same as the other
stores (internal network, at-rest encryption) and drain or purge the DLQ on a
schedule.

## Reporting a vulnerability

Report suspected vulnerabilities privately to the maintainers rather than opening
a public issue. Include steps to reproduce and the affected version.

# Hosting pyTracer on pytracer.com

This walks the whole path from no server to a running gateway on your own
domain, with one subdomain per concern. It assumes you have the domain
(`pytracer.com`), its DNS is on Cloudflare, and you have Docker experience but no
server yet.

## Subdomain architecture

| Subdomain | Serves | Built today | Hosted on |
| --- | --- | --- | --- |
| `pytracer.com` / `www` | Landing page | placeholder in [site/](../site) | Cloudflare Pages (static) |
| `docs.pytracer.com` | Documentation | not yet | Cloudflare Pages (static) |
| `gateway.pytracer.com` | Ingestion gateway API | yes | a VPS running Docker |
| `registry.pytracer.com` | Test-collection registry UI | engine only, no UI | later |
| `app.pytracer.com` | Dashboard | not yet | later |

Only `gateway.pytracer.com` is a live service today. It is a stateful Docker
stack (gateway + ClickHouse + PostgreSQL + Redis), so it needs a real server.
The static subdomains are just files and go on Cloudflare Pages (free). Park the
unbuilt subdomains until their UIs exist.

## Step 1: Provision a VPS

The stores (ClickHouse especially) want memory, so start around 4 GB RAM / 2
vCPU. Any provider works; examples at the time of writing:

- Hetzner Cloud CX22 (2 vCPU, 4 GB) is inexpensive and plenty for a beta.
- DigitalOcean or Linode: a 4 GB shared-CPU instance.
- AWS Lightsail: a 4 GB plan.

Create an Ubuntu 24.04 LTS instance, then SSH in and prepare it:

```bash
# As root on the fresh server:
adduser pytracer && usermod -aG sudo pytracer      # a non-root user
# Install Docker Engine + compose plugin (official convenience script):
curl -fsSL https://get.docker.com | sh
usermod -aG docker pytracer
```

### Firewall (important, and a Docker gotcha)

Expose only SSH and the web ports:

```bash
ufw allow 22/tcp && ufw allow 80/tcp && ufw allow 443/tcp && ufw enable
```

Docker publishes container ports by writing its own iptables rules that bypass
`ufw`, so a published `8080` can be reachable from the internet even with the
firewall above. In a TLS deployment nothing outside the host should reach the
gateway directly (Caddy proxies to it over the internal Docker network), so bind
the gateway's published port to localhost. Set this in your env file (Step 3), no
file edit needed:

```
PYTRACER_GATEWAY_PUBLISH=127.0.0.1:8080:8080
```

Leave the store services with no published ports at all (they already have none
in the deploy stack). Only Caddy should face the internet.

## Step 2: Get the code onto the server

The gateway image builds from source, so the server needs the repository (the
`packages/` and `deploy/` trees):

```bash
# While the repo is private, clone with your credentials (gh auth login, a
# deploy key, or a fine-grained token). Once it is public, a plain clone works.
git clone https://github.com/vi6120/pyTracer.git
cd pyTracer
```

## Step 3: Configure secrets and start the stack

Put real secrets in an env file (never the defaults). Keeping them in a file (not
edits to tracked files) means `git pull` stays clean on later updates:

```bash
cat > deploy/prod.env <<EOF
PYTRACER_DOMAIN=gateway.pytracer.com
PYTRACER_GATEWAY_API_KEYS=$(openssl rand -hex 16):web-agent
PYTRACER_CLICKHOUSE_PASSWORD=$(openssl rand -hex 24)
PYTRACER_POSTGRES_PASSWORD=$(openssl rand -hex 24)
PYTRACER_GATEWAY_PUBLISH=127.0.0.1:8080:8080
EOF
chmod 600 deploy/prod.env
cat deploy/prod.env    # note your API key (the hex before ":web-agent")
```

Then build and launch with the TLS profile:

```bash
docker compose --env-file deploy/prod.env -f deploy/docker-compose.yml --profile tls up -d --build
```

For real use manage these through your orchestrator's secret store. Consider the
Postgres-backed key store so keys can be rotated and revoked without a redeploy
(see [README.md](README.md#managing-api-keys)):

```bash
# add PYTRACER_GATEWAY_AUTH_BACKEND=postgres to the command above, then:
pytracer keys create web-agent --name prod   # prints the secret once
```

## Step 4: Point Cloudflare DNS at the gateway

In the Cloudflare dashboard for pytracer.com, add:

```
Type: A    Name: gateway    Content: <your VPS public IP>    Proxy: DNS only
```

Set the proxy to **DNS only** (grey cloud). Caddy obtains a Let's Encrypt
certificate over ports 80/443, and Cloudflare's proxy (orange cloud) would
intercept that challenge. If you want to keep Cloudflare's proxy in front,
instead set SSL/TLS mode to **Full (Strict)** and either install a Cloudflare
Origin Certificate on the server or switch Caddy to the DNS-01 challenge (needs a
Caddy build with the Cloudflare DNS plugin and a scoped Cloudflare API token).

Give DNS a minute, then verify:

```bash
curl https://gateway.pytracer.com/healthz     # {"status":"ok"}
```

## Step 5: Static subdomains (landing and docs)

These do not touch the VPS.

1. In Cloudflare, create a **Pages** project connected to this repository.
2. Set the build output directory to `site/` (the landing page lives there; no
   build step needed for the placeholder).
3. Add custom domains in the Pages project: `pytracer.com` (and `www`), and
   later `docs.pytracer.com` once docs exist. Cloudflare issues the certificates
   and serves them over its CDN automatically; the orange-cloud proxy is fine
   here.

Park `registry.pytracer.com` and `app.pytracer.com` (a Cloudflare "parked" page
or a placeholder) until those UIs are built.

## Step 6: Point your agents at it

```python
import pytracer_sdk as pytracer

pytracer.configure(
    pytracer.HTTPTransport("https://gateway.pytracer.com", api_key="<your key>"),
    service_name="web-agent", branch="main", commit="<sha>",
)
```

## Operations

- **Updates:** `git pull && docker compose -f deploy/docker-compose.yml --profile tls up -d --build`.
- **Backups:** back up the `clickhouse-data`, `postgres-data`, and `redis-data`
  volumes; removing them wipes the data.
- **Hardening:** enable at-rest disk encryption on the volumes, keep the stores
  off the public network, and follow the full checklist in
  [SECURITY.md](../SECURITY.md).

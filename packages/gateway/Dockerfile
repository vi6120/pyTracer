# Navik ingestion gateway image.
# Build from the repository root so the sibling packages are in the context:
#   docker build -f packages/gateway/Dockerfile -t navik-gateway .
FROM python:3.12-slim

WORKDIR /app

# Install the gateway and the packages it depends on, from source (sequential
# so each sibling dependency is already present when the next one installs).
COPY packages/sdk /app/packages/sdk
COPY packages/stores /app/packages/stores
COPY packages/gateway /app/packages/gateway
# The [redis] extra pulls in the durable ingest queue backend.
RUN pip install --no-cache-dir ./packages/sdk \
    && pip install --no-cache-dir ./packages/stores \
    && pip install --no-cache-dir "./packages/gateway[redis]"

# Run as a non-root user.
RUN useradd --create-home --uid 10001 navik
USER navik

EXPOSE 8080

# The factory reads its store connection, API keys, and queue backend from the
# environment (NAVIK_CLICKHOUSE_*, NAVIK_POSTGRES_*, NAVIK_GATEWAY_API_KEYS,
# NAVIK_GATEWAY_QUEUE_BACKEND, NAVIK_REDIS_*).
CMD ["uvicorn", "navik_gateway.app:build_default_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8080"]

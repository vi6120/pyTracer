# pyTracer ingestion gateway image.
# Build from the repository root so the sibling packages are in the context:
#   docker build -f packages/gateway/Dockerfile -t pytracer-gateway .
FROM python:3.12-slim

WORKDIR /app

# Install the gateway and the packages it depends on, from source (sequential
# so each sibling dependency is already present when the next one installs).
COPY packages/sdk /app/packages/sdk
COPY packages/stores /app/packages/stores
COPY packages/gateway /app/packages/gateway
# Extras installed with the gateway. Defaults to [redis] (the durable ingest
# queue backend); pass --build-arg GATEWAY_EXTRAS=redis,metrics to also expose
# the Prometheus /metrics endpoint (used by the benchmark stack).
ARG GATEWAY_EXTRAS=redis
RUN pip install --no-cache-dir ./packages/sdk \
    && pip install --no-cache-dir ./packages/stores \
    && pip install --no-cache-dir "./packages/gateway[${GATEWAY_EXTRAS}]"

# Run as a non-root user.
RUN useradd --create-home --uid 10001 pytracer
USER pytracer

EXPOSE 8080

# The factory reads its store connection, API keys, and queue backend from the
# environment (PYTRACER_CLICKHOUSE_*, PYTRACER_POSTGRES_*, PYTRACER_GATEWAY_API_KEYS,
# PYTRACER_GATEWAY_QUEUE_BACKEND, PYTRACER_REDIS_*).
CMD ["uvicorn", "pytracer_gateway.app:build_default_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8080"]

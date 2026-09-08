# Navik

An **agent test & replay platform** — capture, replay, and regression-test AI agents (LangGraph, CrewAI, AutoGen, OpenAI Agents SDK) deterministically, with cross-branch failure reproduction.

Built from [agent-test-platform-guide.md](agent-test-platform-guide.md). See [PLAN.md](PLAN.md) for the build tracker.

## Layout

```
packages/
  sdk/        Layer 1 — instrumentation SDK (navik-sdk)
  gateway/    Layer 2 — ingestion gateway
  stores/     Layer 3 — trace store (ClickHouse) + mock store (Postgres)
  replay/     Layer 4 — replay & test engine
  registry/   Layer 5 — collaboration registry
  cli/        Layer 6 — CLI + CI/CD integration
docker/       service configs (ClickHouse, Postgres)
tests/        cross-package integration tests
```

## Development

```bash
# 1. Activate the virtual environment
source .venv/bin/activate

# 2. Install the SDK (editable) with dev tooling
pip install -e "packages/sdk[dev]"

# 3. Run tests
pytest packages/sdk

# 4. Bring up backing services (needs Docker Desktop running)
docker compose up -d
```

## Status

Early scaffolding. Building layer by layer, SDK first — see [PLAN.md](PLAN.md).

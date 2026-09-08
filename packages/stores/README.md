# navik-stores

Layer 3 of Navik: the **trace store** (ClickHouse) holding every captured span,
and the **mock store** (PostgreSQL) holding recorded tool responses for
deterministic replay.

```python
from navik_stores import TraceStore, MockStore

traces = TraceStore(); traces.migrate()
traces.insert(spans, project="my-app")          # branch + commit required
failed = traces.query(failed_only=True, agent_name="planner")

mocks = MockStore(); mocks.migrate()
mocks.record("search", {"q": "hi"}, ["result"], is_failure=False)
mock = mocks.lookup("search", {"q": "hi"})       # newest recorded outcome
```

Connection defaults match the repo's `docker-compose.yml`; override via
`NAVIK_CLICKHOUSE_*` / `NAVIK_POSTGRES_*` env vars.

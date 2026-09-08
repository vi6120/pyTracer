# navik-runner

Layer 4 infrastructure for Navik: the per-commit isolated runner behind
"try on this branch". It checks out an arbitrary branch or commit into a clean
workspace, runs a command there (typically installing dependencies and running
`navik run`), and returns the result, so a captured failing trace can be
replayed against a different branch with the model version and recorded tool
responses held frozen and only the agent code varying (guide section 5).

Each run happens in its own fresh workspace, because two branches may have
different dependency versions and must not share an environment.

```python
from navik_runner import CrossBranchRunner, GitSourceProvider, DockerExecutor

runner = CrossBranchRunner(GitSourceProvider("/path/to/repo"), DockerExecutor())
result = runner.reproduce(
    ref="feature/fix-bug",
    collection_path="tests/collection.yaml",
    frozen_trace="baseline.jsonl",           # the failing trace, held frozen
    trace_dest="tests/traces/summarize.jsonl",
)
```

The source and execution backends are pluggable (`SourceProvider`,
`Executor`), so the orchestration is testable with fakes and the real git and
Docker backends are used in production.

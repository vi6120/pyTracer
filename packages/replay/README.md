# navik-replay

Layer 4 of Navik: reconstructs a past agent run deterministically, injects
recorded tool/model responses instead of live calls, runs assertions, diffs two
runs, and classifies a cross-branch reproduction as fixed / still failing /
diverged.

```python
from navik_replay import MockSet, ReplayEngine, ReplayMode, classify
import navik_sdk as navik

# 1. Build a mock set from a captured trace (its recorded tool/model outputs).
mocks = MockSet.from_trace(recorded_spans)

# 2. Replay the agent deterministically — tools and model are mocked.
engine = ReplayEngine(mode=ReplayMode.FULL)
result = engine.replay(lambda: agent_entrypoint(task), mocks)

# 3. Assert, diff, fingerprint, or classify against another branch's run.
```

Modes: `FULL` (recorded model + tools), `PARTIAL` (recorded tools, live model),
`LIVE` (both live, for drift detection).

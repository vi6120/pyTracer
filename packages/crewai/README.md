# pytracer-crewai

pyTracer adapter for CrewAI. It auto-instruments a crew through CrewAI's event bus,
so you do not add `@op` by hand: crew kickoffs, task and agent execution, tool
usage, and LLM calls are captured as pyTracer spans.

```python
import pytracer_sdk as pytracer
from pytracer_crewai import PyTracerEventListener

pytracer.configure(
    pytracer.HTTPTransport("https://gateway.internal", api_key="YOUR_KEY"),
    service_name="my-crew", branch="main", commit="abc123",
)

PyTracerEventListener()          # registers on the global bus; every crew run is now captured
crew.kickoff(inputs={...})
```

The bus stamps each event with an id, a parent id, and (on completion) the
matching start id, which the listener uses to build a correctly nested trace.

## Replay

This adapter **captures** runs. Replaying a captured run currently requires `@op`
instrumentation, because the replay engine intercepts `@op` calls; replay through
the adapter itself is on the roadmap. Attempting to replay an adapter-only trace
is flagged (the run has recordings but hits no `@op` seam) rather than silently
running live. See the main README for the replay workflow.

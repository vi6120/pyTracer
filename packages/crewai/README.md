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

## One trace per request

Each run starts its own trace. To keep a request that makes several runs, or
also uses another framework, in one trace, wrap it in a pyTracer span; the run
nests under the span that is open when it starts:

```python
with pytracer.start_span("answer-question"):
    crew.kickoff(inputs={...})
    ...  # further runs, other adapters, or @op calls join the same trace
```

## Replay

Call `install_seams(listener)` and a captured run replays without adding `@op`
to anything:

```python
from pytracer_crewai import PyTracerEventListener, install_seams

listener = PyTracerEventListener()
install_seams(listener)   # routes LLM calls and tools through pyTracer
crew.kickoff(inputs={...})
```

CrewAI's model calls (`call` / `acall` on every `BaseLLM` class, including the
native provider classes CrewAI loads lazily and custom LLMs) and tools
(`BaseTool.run` for native function calling, and the structured-tool `invoke`
used by the ReAct loop) then run through pyTracer, nested under the agent that
made them, so the replay engine can return their recorded results: in full mode
neither the model nor the tools run again, and in partial mode the tools stay
frozen while the model runs live. A model call is matched on its messages, the
tools offered and the response model. `uninstall_seams()` removes the routing
again.

CrewAI delivers events to listeners on a thread pool, so before a seam call the
listener waits for already-emitted events (normally well under a millisecond,
capped at 0.25 seconds) to nest the call under the right agent.

Without `install_seams` the adapter only captures, and replaying an
adapter-only trace is flagged (the run has recordings but hits no `@op` seam)
rather than silently running live. See the main README for the replay workflow.

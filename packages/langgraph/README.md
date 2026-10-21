# pytracer-langgraph

pyTracer adapter for LangGraph and LangChain. It auto-instruments an agent through
LangChain's callback system, so you do not add `@op` by hand: model calls, tool
calls, and graph nodes are captured as pyTracer spans automatically.

```python
import pytracer_sdk as pytracer
from pytracer_langgraph import PyTracerCallbackHandler

pytracer.configure(
    pytracer.HTTPTransport("https://gateway.internal", api_key="YOUR_KEY"),
    service_name="my-agent", branch="main", commit="abc123",
)

handler = PyTracerCallbackHandler()
app.invoke(inputs, config={"callbacks": [handler]})   # spans captured for this run
```

`on_chat_model_start` / `on_llm_start` become LLM spans, `on_tool_start` become
tool spans, and LangGraph nodes become agent spans, all linked into one trace by
the run tree.

## Replay

This adapter **captures** runs. Replaying a captured run currently requires `@op`
instrumentation, because the replay engine intercepts `@op` calls; replay through
the adapter itself is on the roadmap. Attempting to replay an adapter-only trace
is flagged (the run has recordings but hits no `@op` seam) rather than silently
running live. See the main README for the replay workflow.

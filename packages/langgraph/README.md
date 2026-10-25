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

## One trace per request

Each run starts its own trace. To keep a request that makes several runs, or
also uses another framework, in one trace, wrap it in a pyTracer span; the run
nests under the span that is open when it starts:

```python
with pytracer.start_span("answer-question"):
    app.invoke(inputs, config={"callbacks": [handler]})
    ...  # further runs, other adapters, or @op calls join the same trace
```

## Replay

Call `install_seams()` and a captured run replays without adding `@op` to
anything:

```python
from pytracer_langgraph import PyTracerCallbackHandler, install_seams

install_seams()   # routes chat models and tools through pyTracer
app.invoke(inputs, config={"callbacks": [PyTracerCallbackHandler()]})
```

LangChain chat models (`invoke` / `ainvoke` / `stream` / `astream`, including a
model bound with `bind_tools`) and tools (`BaseTool.run` / `arun`, which
`ToolNode` and agents go through) then run through pyTracer, nested under the
graph node that called them, so the replay engine can return their recorded
results: in full mode neither the model nor the tools run again, in partial mode
the tools stay frozen while the model runs live, and streamed replies replay
chunk by chunk. A model call is matched on its messages (without the per-run
message ids LangGraph assigns), its bound tools and stop words.
`uninstall_seams()` removes the routing again.

Without `install_seams()` the adapter only captures, and replaying an
adapter-only trace is flagged (the run has recordings but hits no `@op` seam)
rather than silently running live. See the main README for the replay workflow.

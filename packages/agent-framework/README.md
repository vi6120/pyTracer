# pytracer-agent-framework

pyTracer adapter for [Microsoft Agent Framework](https://github.com/microsoft/agent-framework),
the successor to AutoGen. It captures agent runs, model calls and function tool
calls as pyTracer spans with no change to the agent code, and a captured run
replays with the model and tool responses frozen.

```python
import pytracer_sdk as pytracer
from pytracer_agent_framework import install

pytracer.configure(
    pytracer.HTTPTransport("https://gateway.internal", api_key="YOUR_KEY"),
    service_name="my-agent", branch="main", commit="abc123",
)

install()   # every Agent Framework run is now captured, and replayable
```

What is recorded:

- **Model calls**: every chat client's call to its model, one LLM span per turn
  of the tool loop, with its token count. A streamed call is recorded update by
  update and replays as a stream. The seam sits on the method every Agent
  Framework chat client implements, including clients you define; CI verifies
  Agent Framework's OpenAI Chat Completions client and custom clients.
- **Function tools**: each `FunctionTool` call, as a tool span with its arguments
  and result. Hosted tools (web search, code interpreter) run inside the model
  call, so the recorded model response covers them.
- **Agent runs**: each `agent.run(...)`, streamed or not, as an agent span that its
  model and tool calls nest under. A replay still runs the agent itself.

Agent Framework generates fresh function call ids (`af-call-...`) and message ids
on every run. They are left out of a model call's match key, so a replay of the
same conversation finds its recording.

Requires Python 3.11 or later (Agent Framework's own floor) and
`agent-framework-core` 1.26.0 or later. `uninstall()` removes the seams.

## Coming from AutoGen

Microsoft has archived AutoGen and points its users to Agent Framework.
`pytracer-autogen` keeps working for AutoGen agents; once an agent is ported, use
this adapter instead. Recordings do not carry over: an AutoGen recording replays
AutoGen code, so capture the ported agent's scenarios again with this adapter.

# navik-langgraph

Navik adapter for LangGraph and LangChain. It auto-instruments an agent through
LangChain's callback system, so you do not add `@op` by hand: model calls, tool
calls, and graph nodes are captured as Navik spans automatically.

```python
import navik_sdk as navik
from navik_langgraph import NavikCallbackHandler

navik.configure(
    navik.HTTPTransport("https://gateway.internal", api_key="YOUR_KEY"),
    service_name="my-agent", branch="main", commit="abc123",
)

handler = NavikCallbackHandler()
app.invoke(inputs, config={"callbacks": [handler]})   # spans captured for this run
```

`on_chat_model_start` / `on_llm_start` become LLM spans, `on_tool_start` become
tool spans, and LangGraph nodes become agent spans, all linked into one trace by
the run tree.

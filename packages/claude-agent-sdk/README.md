# pytracer-claude-agent-sdk

pyTracer adapter for the [Claude Agent SDK](https://code.claude.com/docs/en/agent-sdk).
It captures `query()` and `ClaudeSDKClient` sessions with no change to the agent
code, and a captured session replays without starting the Claude Code CLI, so a
replay makes no model calls and runs no tools.

```python
import pytracer_sdk as pytracer
from pytracer_claude_agent_sdk import install

pytracer.configure(
    pytracer.HTTPTransport("https://gateway.internal", api_key="YOUR_KEY"),
    service_name="my-agent", branch="main", commit="abc123",
)

install()   # every Claude Agent SDK session is now captured, and replayable
```

## What is recorded

The Agent SDK runs the agent loop in the Claude Code CLI, a subprocess: the model
and the built-in tools (Bash, Read, Edit, web search) run there, and Python talks
to it over a pipe. The adapter records that conversation.

- **CLI turns**: each line Python sends the CLI (the prompt, a tool result, a
  hook's answer, a permission decision) starts a turn, recorded as an LLM span
  holding the messages the CLI sent back, with the run's token count on the
  last turn. A replay serves each turn only after Python has sent exactly what
  it sent when the session was captured (request ids and uuids aside).
- **In-process tools**: each `@tool` handler, as a tool span with its arguments
  and result. These freeze on replay like any other tool.

Hooks and `can_use_tool` callbacks are your code, so they still run on replay.
If one answers differently, or an in-process tool returns something else, the
next turn's input no longer matches the recording: the replay reports the model
input as changed (so a verdict is `cannot_judge`, naming the turn) instead of
serving what the CLI said to different input.

## Limits

- The CLI's built-in tools run inside the CLI, so they are frozen as part of its
  turns, not as tool spans of their own; partial replay (tools frozen, model
  live) therefore runs the whole CLI session live, with only in-process tools
  frozen.
- A replayed session cannot switch to live part way: with
  `evaluate_changed_prompts`, a changed turn fails with `ReplayedSessionError`
  instead of starting a CLI that has none of the session's history. Re-record the
  session instead.

CI verifies the adapter with the real SDK (`query()`, `ClaudeSDKClient`, the
control protocol, the in-process MCP bridge) talking to a scripted CLI. It has
not yet been verified against a live Claude Code CLI session.

Requires `claude-agent-sdk` 0.2.189 or later. `uninstall()` removes the seams.

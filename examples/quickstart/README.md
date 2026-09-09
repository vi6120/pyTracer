# Navik quickstart

Run the whole Navik loop in a couple of minutes, with no services and no API
keys. You will instrument an agent, capture a run, turn it into a test, and
replay it deterministically.

## Setup

From the repository root, install the SDK and CLI into a virtual environment:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e "packages/sdk[dev]" -e "packages/replay[dev]" -e "packages/cli[dev]"
cd examples/quickstart
```

## 1. The agent

[`agent.py`](agent.py) is a small "researcher" that searches for documents and
summarizes them. Each function has one `@op` decorator, which is all Navik needs
to capture it. The search and model calls are stand-ins, so nothing hits the
network.

## 2. Capture a run

```bash
python capture.py
```

This runs the agent once and writes its spans to `traces/run.jsonl`, simulating
a run captured in production. You will see the agent's output and the span count.

## 3. Replay it as a test

[`collection.yaml`](collection.yaml) defines a test: replay the agent against the
recorded trace and assert the run has no errors and returns a summary. Run it:

```bash
navik run collection.yaml
```

The replay injects the recorded tool and model responses, so it is deterministic
and touches no live services. All scenarios pass.

## 4. See it fail

Break the agent (for example, change `summarize` in `agent.py` to return
`"oops"` instead of a summary) and run `navik run collection.yaml` again. The
`contains: "Summary"` assertion fails and the command exits non-zero, exactly as
it would in CI.

## One-shot check

To run steps 2 and 3 together as a smoke test:

```bash
python verify.py
```

It prints each scenario's result and exits 0 when the replay passes.

## What's next

- **Auto-instrument a framework** instead of hand-adding `@op`: see the
  `navik-langgraph`, `navik-openai-agents`, `navik-crewai`, and `navik-autogen`
  packages.
- **Browse real captured traces** with `navik traces list` and scaffold tests
  with `navik record` (needs the trace store: `pip install navik-cli[stores]`).
- **Try a fix on another branch** with `navik reproduce --branch <ref>` (needs
  the runner: `pip install navik-cli[runner]`).

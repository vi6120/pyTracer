<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="site/assets/logo-dark.png">
    <img src="site/assets/logo.png" alt="pyTracer" width="440">
  </picture>
</p>

<p align="center">
  <a href="https://pytracer.com">Website</a> &middot;
  <a href="https://pytracer.com/docs">Docs</a> &middot;
  <a href="https://pytracer.com/docs#packages">Packages</a> &middot;
  <a href="https://pytracer.com/benchmarks">Benchmarks</a> &middot;
  <a href="https://pypi.org/project/pytracer-all/">PyPI</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/pytracer-all/"><img src="https://img.shields.io/pypi/v/pytracer-all.svg?color=2563eb" alt="PyPI"></a>
  <img src="https://img.shields.io/pypi/pyversions/pytracer-sdk.svg" alt="Python versions 3.10">
  <img src="https://img.shields.io/badge/license-Apache--2.0%20%7C%20AGPL--3.0-blue.svg" alt="License">
</p>

pyTracer is an **agent test and replay platform**. It captures what an AI agent
does (its LLM calls, tool calls, reasoning steps, and handoffs), replays a past
run deterministically by injecting the recorded responses instead of calling
live services, and regression-tests agents in CI. Its headline capability is
**cross-branch reproduction**: take a captured failing run and re-run it against
a different branch with the model version and recorded tool responses held
frozen, so you learn whether a branch actually fixed the bug.

It works with agents built on LangGraph, CrewAI, AutoGen, and the OpenAI Agents
SDK, or any Python agent you instrument by hand.

New here? Run the [quickstart](examples/quickstart) to try the whole loop in a
couple of minutes, with no services.

### A use case

A customer asks a support agent to exchange an order; the agent confirms, but the
exchange goes out with the old item IDs as the new ones, a one-line bug in the glue
code between the model and its tools. Re-running it live proves nothing, because the
model words things differently each time. With pyTracer:

1. The failing conversation is **captured**, with its branch and commit.
2. `pytracer record` **turns it into a test** that asserts the right items are
   exchanged, so CI goes red.
3. On the fix PR, `/pytracer try` replays the exact conversation with every model and
   tool response frozen: **fixed**.
4. A PR that only renamed a variable: **still failing**, not a lucky green run.
5. A PR that changes the exchange call itself: **cannot judge**, naming that call and
   what to do, instead of guessing.

No model calls and no API spend in any of those checks.

### How it works

**Capture** a run (each model call and tool result, stamped with branch and commit),
**freeze** its responses as mocks, **replay** it on any branch in full, partial or
live mode, and **classify** the outcome. A verdict is only drawn from a run that
stayed on its recording: if a branch makes a call the recording cannot serve, you get
`cannot_judge` with that call named; if the baseline itself drifts, `cannot_reproduce`;
and a branch that fails exactly as before is `still_failing`, whatever else changed.

## Why

Agents are non-deterministic and hard to test: a run depends on live model
output and live tool calls, so you cannot just re-run it. pyTracer records each run
as structured spans, turns those spans into deterministic mocks, and replays the
agent against them. A failure captured once can then be reproduced on demand,
asserted against, diffed between two runs, and re-checked on any branch.

## Architecture

Seven core packages, each an installable Python package under `packages/`:

| Package | Layer | What it does |
| --- | --- | --- |
| `pytracer-sdk` | 1. Instrumentation | `@op` decorator captures agent actions as OpenTelemetry-compatible spans, with local secret/PII redaction and non-blocking flush |
| `pytracer-gateway` | 2. Ingestion | FastAPI service that validates spans, authenticates per project by API key (static map or managed keys in Postgres, with per-key rate limiting), and writes them through to storage without blocking; an optional Redis Streams queue makes ingestion durable across restarts |
| `pytracer-stores` | 3. Storage | Trace store (ClickHouse) for captured spans, mock store (PostgreSQL) for recorded tool responses |
| `pytracer-replay` | 4. Replay engine | Deterministic replay in three modes, an assertion framework, run diffing, failure fingerprinting, an outcome classifier that abstains with the reason rather than guess, and a network sandbox; replay stays faithful on real agents (streaming responses, concurrent and repeated tool calls replayed in order, nondeterminism controls for a frozen clock and seeded RNG, retry dedupe, and side-effect detection) |
| `pytracer-runner` | 4. Isolation | Per-commit isolated runner: check out any ref into a clean workspace and replay a frozen trace against it |
| `pytracer-registry` | 5. Collaboration | Versioned, forkable, access-controlled test collections with pull requests and discovery |
| `pytracer-cli` | 6. CI/CD | `pytracer run` executes test collections locally and in CI, with a pytest plugin (`pytracer-cli[pytest]`) that runs them as pytest tests and a `pytracer reproduce --pr` "try on this branch" PR comment; ships a GitHub Action and a GitLab template |

A `pytracer-all` meta-package ties them together: installing it pulls in all seven,
so `pip install pytracer-all` gives you the whole platform in one command (and
`import pytracer` re-exports the SDK's core entrypoints for convenience).

Four framework adapter packages (`pytracer-langgraph`, `pytracer-openai-agents`,
`pytracer-crewai`, `pytracer-autogen`) auto-instrument agents built on those
frameworks, so neither capture nor replay needs `@op` in the agent code: with
replay switched on, the adapter routes the framework's own model and tool calls
through pyTracer, so a captured run replays like an `@op`-instrumented one.
Twelve packages in all, published on PyPI; the full list is on the
[docs site](https://pytracer.com/docs#packages).

What each adapter captures and replays:

| | LangGraph | OpenAI Agents | CrewAI | AutoGen |
|---|---|---|---|---|
| Agent spans | Yes | Yes | Yes | Yes |
| Model calls (model, tokens, output) | Yes | Yes | Yes | Yes, from the model client's call events (the `autogen_ext` clients log them); a client that logs none has its calls rebuilt from AutoGen's replies. Before AutoGen 0.7.2 the calls are recorded through pyTracer's routing ([AutoGen versions](packages/autogen/README.md#autogen-versions)) |
| Tool calls with their output | Yes | Yes | Yes | Yes |
| One trace per framework run | Yes | Yes | Yes | Yes |
| Nests under an enclosing `pytracer.start_span(...)` | Yes | Yes | Yes | Yes |
| Replay through the adapter, no `@op` | Yes, `install_seams()` | Yes, `install(replayable=True)` | Yes, `install_seams(listener)` | Yes, `install(replayable=True)` |

All four frameworks can share one environment, with two constraints from their own
dependencies: CrewAI caps the OpenAI Agents SDK at 0.20.0 (CrewAI requires
`openai<3`), and AutoGen's protobuf pin requires `googleapis-common-protos<1.75.1`,
which a single resolve picks for you but adding AutoGen to an existing environment
may not. Each adapter requires at least the framework version CI tests, so a
request the frameworks cannot meet together is reported by the installer rather
than settled with an untested release. See [using several frameworks in one environment](https://pytracer.com/docs#framework-combinations).

Backing services (ClickHouse, PostgreSQL, and Redis for the durable ingest
queue) run via [docker-compose.yml](docker-compose.yml).

## How a development team uses pyTracer

The workflow has three phases: instrument, turn failures into tests, and gate
your branches on them.

### 1. Instrument your agent (once)

Add `@op` to the functions that call models and tools. This is one or two lines
per function and does not change their behavior.

```python
import pytracer_sdk as pytracer
from pytracer_sdk import SpanKind

@pytracer.op(kind=SpanKind.TOOL, tool_name="search")
def search(query: str) -> dict:
    ...  # your real tool

@pytracer.op(kind=SpanKind.LLM)
def call_model(prompt: str) -> str:
    ...  # your real model call

@pytracer.op(kind=SpanKind.AGENT, agent_name="researcher")
def run(task: str) -> str:
    return call_model(str(search(task)))
```

Point the SDK at the gateway so spans flow into the trace store. Stamp each run
with the branch and commit so a captured failure is tied to exact code:

```python
pytracer.configure(
    pytracer.HTTPTransport("https://gateway.internal", api_key="YOUR_PROJECT_KEY"),
    service_name="researcher",
    branch="main",
    commit="abc123",
)
```

Capture is fire-and-forget: it never blocks your agent, and if the gateway is
unreachable the agent keeps running.

With a framework adapter, each framework run starts its own trace. To keep a
request that spans several runs (or several frameworks) in one trace, wrap it in a
pyTracer span; all four adapters nest under it:

```python
with pytracer.start_span("answer-question"):
    plan = graph.invoke(inputs, config={"callbacks": [handler]})   # LangGraph
    result = crew.kickoff(inputs={"plan": plan})                   # CrewAI
```

### 2. Turn a captured failure into a test

When a run fails, browse recent captured runs and scaffold a test from one, no
hand-writing of files required:

```bash
pytracer traces list --failed              # find the failing run
pytracer traces show <trace_id>            # inspect its span tree
pytracer record <trace_id> --agent myapp.agents:run --entry task="summarize the doc"
```

`record` writes the trace's spans to JSONL and creates a **collection**: a small
YAML file naming the agent to run, the recorded trace to mock from, and the
assertions that should hold.

```yaml
# tests/collection.yaml
name: captured-suite
scenarios:
  - name: summarize-happy-path
    agent: myapp.agents:run          # import path "module:callable"
    entry: { task: "summarize the doc" }
    trace: traces/summarize.jsonl    # the recorded spans
    mode: full                       # full | partial | live
    assertions:
      - { type: no_errors }
      - { type: contains, needle: "summary" }
```

Run it locally. pyTracer replays the agent with the recorded tool and model
responses injected, so the run is deterministic and touches no live services:

```bash
pytracer run tests/collection.yaml
```

The `traces` and `record` commands need the store client: `pip install
pytracer-cli[stores]`.

### 3. Gate your branches in CI

Add the GitHub Action (see [packages/cli/ci/github-action.yml](packages/cli/ci/github-action.yml)).
On every pull request it runs the collection and posts a pass/fail summary as a
PR comment, and fails the check if any scenario regresses. A GitLab template
([packages/cli/ci/gitlab-ci.yml](packages/cli/ci/gitlab-ci.yml)) is included too: it
runs in merge-request and branch pipelines, once per change, and shows the results
in the merge request's test report.

On a push (not a PR), it also **auto-documents failures as GitHub issues**:
`pytracer issues sync` opens an issue per failing scenario, keyed by the
failure's fingerprint so a recurring failure updates one issue (a new comment,
reopening it if closed) instead of piling up duplicates. Run it yourself with:

```bash
pytracer run tests/collection.yaml --json results.json
pytracer issues sync results.json --repo owner/name   # needs a GitHub token
```

### 4. Reproduce a failure on any branch

When a teammate opens a fix, re-run the failing collection against their branch
with the mocks frozen, so only the code changes. From the terminal:

```bash
pytracer reproduce --branch feature/fix-bug --baseline main \
  --collection tests/collection.yaml
```

It checks out each ref in its own isolated workspace, replays, and prints a
per-scenario verdict of **fixed**, **still failing**, or **diverged** (a new,
different failure), exiting non-zero unless every scenario is fixed. When it cannot
settle a scenario it says so with the reason (**cannot judge**: the branch made a call
the recording cannot serve, which it names; **cannot reproduce**: the baseline did),
exiting with code 2. Needs the runner: `pip install pytracer-cli[runner]`.

With `--isolation docker` (the default) each ref installs into a throwaway container;
with `--isolation local` it runs in your current environment and installs nothing
(`--install` / `--no-install` override either).

Add `--pr <number> --pr-repo owner/name` to post the verdict as a "try on this
branch" comment on the pull request.

The same thing is available as a library (`CrossBranchRunner`,
`classify_reproduction`, `explain_reproduction`) for building it into your own
tooling.

Each ref runs in its own clean workspace with its own dependencies, so branches
never interfere with each other.

### 5. Share and reuse test suites

The registry turns collections into shareable, forkable artifacts with version
history, pull requests, three-way merge, access control (private / team /
public), and discovery by framework and use case, the way Postman collections
worked for API testing.

## Installing

pyTracer requires **Python 3.10 or newer**.

The whole platform installs from PyPI with one command:

```bash
pip install pytracer-all        # the whole platform (the seven core packages)
# or install just what you need, e.g. the SDK in your agent:
pip install pytracer-sdk
```

Framework adapters are separate packages, one per framework:

```bash
pip install pytracer-langgraph        # LangGraph / LangChain
pip install pytracer-openai-agents    # OpenAI Agents SDK
pip install pytracer-crewai           # CrewAI
pip install pytracer-autogen          # AutoGen
```

These are standard PyPI wheels, so [uv](https://docs.astral.sh/uv/) works too
(`uv add pytracer-sdk`, or `uv pip install pytracer-all`). pyTracer is a Python
library; there is no npm package (a TypeScript SDK is on the roadmap, not yet
released).

All twelve packages are on PyPI at 0.1.2 (each with a wheel and a source
distribution); the full list is on the [docs site](https://pytracer.com/docs#packages).
To hack on pyTracer itself instead, install from this repo (see below).

## Quick start (local development)

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install every package in editable mode with dev tooling
#    (order matters: the pytracer meta-package resolves against the others)
for p in sdk stores gateway replay runner registry cli pytracer; do
  pip install -e "packages/$p[dev]"
done

# 3. Start the backing services (needs Docker running)
docker compose up -d

# 4. Run the whole test suite
pytest packages
```

Store-backed tests skip automatically when ClickHouse or PostgreSQL is not
reachable, the durable-queue tests skip when Redis is not reachable, and the
Docker-backed runner test skips when Docker is not available, so the suite runs
anywhere.

### Running the gateway

```bash
PYTRACER_GATEWAY_API_KEYS="devkey:my-app" \
  uvicorn pytracer_gateway.app:build_default_app --factory --port 8080
```

Connection defaults for both stores match [docker-compose.yml](docker-compose.yml)
(`pytracer` / `pytracer`) and are overridable via `PYTRACER_CLICKHOUSE_*` and
`PYTRACER_POSTGRES_*` environment variables.

To run the gateway and its stores together as containers, see
[deploy/](deploy/): `docker compose -f deploy/docker-compose.yml up -d --build`.

## Testing and quality

Every package is covered by tests and checked with `ruff` and `mypy --strict`:

```bash
pytest packages            # 589 tests
ruff check packages
mypy packages/*/src
```

## Status

pyTracer 0.1.4 is released: all twelve packages are on PyPI, and the hosted gateway
runs it. 0.1.4 fixes every verdict issue found in outside testing (a run that leaves
its recording now gets no verdict and the reason, never a lucky "fixed"), captures
every AutoGen release from 0.4.0, keeps the gateway writing through a Redis outage,
lets several gateways share one queue, and keeps model calls' token counts with
replay switched on. See the [changelog](CHANGELOG.md) for the full list, including
the behaviour changes and known limitations.

Every layer is in place: capture with local redaction, a durable ingestion gateway
(Redis Streams, dead-letter list, managed API keys, per-key rate limits, TLS), a
ClickHouse trace store, the replay engine with streaming, concurrency and
nondeterminism controls, cross-branch reproduction with a PR trigger, a CI gate and
pytest plugin, failure auto-documentation as GitHub issues, and the test registry.
Remaining work is the registry web UI and outbound webhooks.

## Benchmarks

Measured on 0.1.4 (laptop and cloud VM, 26 October 2026); every number with its
hardware and the command to rerun it is on the
[benchmarks page](https://pytracer.com/benchmarks):

- **Gateway soak:** 3,600 spans/s for 30 minutes, p95 5.6 ms, no failed or lost spans,
  memory flat (99 MB to 75 MB).
- **Failover:** two gateways, one killed mid-run: no spans lost, one request failed.
- **Self-hosting:** the full stack with TLS runs on a free-tier VM.
- **Verdicts on real pull requests:** fixed, still failing and cannot judge (with the
  reason), each as expected, through the shipped GitHub Action.

Next: replay versus live cost, flakiness, verdict trust on seeded bugs, and SDK
overhead. Run the load tests yourself with [benchmarks/](benchmarks/).

## Roadmap

Near-term: published benchmarks, side-effect assertions and return values in
`run --json`, replay controls (choose the redactor, reproduce a collaborator's remote
branch), reusing existing OpenTelemetry or Langfuse spans, and full retry-chain
capture. Later: a TypeScript SDK, side-effect governance on the hosted service, and
cross-run failure intelligence. The [roadmap](https://pytracer.com/#roadmap) is kept
current on the website.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, the
quality gates, and the pull-request process, and note that a one-time
[CLA](CLA.md) signature is required. All participants follow the
[Code of Conduct](CODE_OF_CONDUCT.md).

## Maintainers

- **Vikas Ramaswamy** - [@vi6120](https://github.com/vi6120) - [LinkedIn](https://www.linkedin.com/in/vikasramaswamy/)
- **Nagesha Sheshadri** - [@sfdnas-adm](https://github.com/sfdnas-adm) - [LinkedIn](https://www.linkedin.com/in/nageshasheshadri/)

## License

pyTracer is split-licensed. The client toolchain you embed and run yourself (the
SDK, framework adapters, replay engine, runner, and CLI) is **Apache-2.0**; the
hosted-server components (`pytracer-gateway` and `pytracer-registry`) are
**AGPL-3.0-or-later**. So you can embed the SDK in a proprietary agent freely,
while a competing hosted service built on the server components must share its
changes. See [LICENSING.md](LICENSING.md) for the per-package breakdown and what
it means in practice.

The license covers the code; the **pyTracer** name and logo are trademarks. See
[TRADEMARK.md](TRADEMARK.md) for how you may use them (you can say "works with
pyTracer"; you can't name a fork "pyTracer").

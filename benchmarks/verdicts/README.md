# Seeded-bug verdict benchmark

When pyTracer says a branch fixed a bug, is it right? When it cannot tell, does it
say so instead of guessing? This benchmark plants bugs in a support agent on
[tau2-bench](https://github.com/sierra-research/tau2-bench)'s retail domain (MIT,
pinned at `v2.0.0`), and scores `pytracer reproduce` against what really happened.

## What runs

- **The agent** (`support_agent/`): a customer-support agent on tau2's retail
  tools, policy and LLM user simulator. Its loop and instrumentation are fixed;
  `glue.py`, the code between the model and the tools, is where bugs live.
- **The bugs** (`bugs.py`): five edits to `glue.py`, each exercised on one tau2 task.

  | Bug | What goes wrong | Task |
  |---|---|---|
  | B1 | a write is retried after a timeout that had already committed it | 38 (cancel) |
  | B2 | order ids lose their `#`, so every order lookup fails | 0 (exchange) |
  | B3 | tool results are cut to 400 characters to save context | 1 (exchange) |
  | B4 | only the last 6 messages are sent to the model | 5 (return) |
  | B5 | a "batch limit" returns at most 2 items per call | 13 (return) |

- **The branches**: for each bug, `main` carries it; `fix` fixes it; `nofix` makes
  a cosmetic edit; `other` fixes it and adds a different bug.
- **The test** each branch is judged by: no error, exactly the task's gold writes
  (tool and arguments, once each, and no other write), and no duplicated write.

## How it is scored

1. **Ground truth, measured live.** Every branch runs live `--repeats` times (3 by
   default) and is judged by the test above and by tau2's own check (the database
   against the gold actions). A branch whose runs pass is truly fixed; one that
   fails as `main` fails is still failing; one that fails another way diverged
   (majority over the repeats). Labels are measured, not assumed.
2. **pyTracer's verdict.** `main` is captured live once; then `pytracer reproduce`
   judges every branch against that recording, under three settings:
   - `strict`: everything frozen, strict;
   - `fallback`: frozen but not strict (a call with no recording runs live), and a
     changed model call runs live (`--evaluate-changed-prompts`);
   - `partial`: tools frozen, the model live.
3. **The result** per branch: correct, an abstention (`cannot_judge` or
   `cannot_reproduce`, with pyTracer's reason), or wrong. The headline is the unsafe
   count: branches that are not fixed but reported `fixed`. It must be 0.

A verdict is drawn only from runs that stayed on their recording. A fix that
changes the calls themselves (B1's fix adds a status check before retrying) cannot
be judged from a recording made on the buggy branch, so pyTracer abstains and names
the call; the benchmark reports how often that happens.

## Run it

```bash
git clone --depth 1 --branch v2.0.0 https://github.com/sierra-research/tau2-bench.git
python3.12 -m venv .venv && . .venv/bin/activate
pip install ./tau2-bench -e packages/sdk -e packages/replay -e packages/runner -e packages/cli pyyaml
export TAU2_DATA_DIR="$PWD/tau2-bench/data"
export ANTHROPIC_API_KEY=...            # or another provider litellm supports
export BENCH_MODEL=anthropic/claude-haiku-5-5
python benchmarks/verdicts/run.py --plan   # what would run; calls no model
python benchmarks/verdicts/run.py          # all bugs, all settings, 3 repeats
```

### On an open model

Any model litellm can reach works through `BENCH_MODEL`; `BENCH_LLM_ARGS` passes
extra arguments (JSON) to every model call, the agent's and the simulated
customer's. Measured on task 38 (cancel a pending order), 26 November 2026:

| Model | Where | Outcome |
|---|---|---|
| `llama3.1:8b` | local, Ollama | changed the payment method twice instead of cancelling |
| `qwen3:8b`, thinking off | local, Ollama | never looked the customer up; invented an order id and looped |

Small local models cannot plan tau2's retail lookups, so the bugs are masked by the
model's own mistakes. A larger open-weights model through a hosted router is cheap
enough for the full run, for example:

```bash
export OPENROUTER_API_KEY=...                      # set it yourself
export BENCH_MODEL=openrouter/openai/gpt-oss-120b  # about $1.40 for the full run
```

For a local Ollama model, `BENCH_MODEL=ollama_chat/<model>` and, for a Qwen3 model,
`BENCH_LLM_ARGS='{"reasoning_effort": "none"}'` (its thinking mode is slow).

Results go to `benchmarks/verdicts/results/`: one JSON per bug (every live run,
every verdict and reason) and `summary.md`, the table per setting. A full run is
about 125 agent runs, about 4,000 model calls and 25 million tokens.

`BENCH_MODEL=scripted:benchmarks/verdicts/smoke_script_task38.json` replaces the
model with a fixed conversation for task 38, which checks the plumbing for
`--bugs B1` at no cost.

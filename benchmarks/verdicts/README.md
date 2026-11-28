# Seeded-bug verdict benchmark

When pyTracer says a branch fixed a bug, is it right? When it cannot tell, does it
say so instead of guessing? This benchmark plants bugs in a support agent on
[tau2-bench](https://github.com/sierra-research/tau2-bench)'s retail domain (MIT,
pinned at `v2.0.0`), and scores `pytracer reproduce` against what really happened.

## What runs

- **The agent** (`support_agent/`): a customer-support agent on tau2's retail
  tools, policy and LLM user simulator. Its loop and instrumentation are fixed;
  `glue.py`, the code between the model and the tools and the record the run
  returns, is where bugs live.
- **The bugs** (`bugs.py`): edits to `glue.py`, each exercised on one tau2 task.

  | Bug | What goes wrong | Task |
  |---|---|---|
  | B1 | a write is retried after a timeout that had already committed it | 88 (cancel) |
  | B2 | order ids lose their `#`, so every order lookup fails | 0 (exchange) |
  | B3 | tool results are cut to 400 characters to save context | 1 (exchange) |
  | B4 | only the last 6 messages are sent to the model | 5 (return) |
  | B5 | a "batch limit" returns at most 2 items per call | 13 (return) |
  | B6 | the run's record drops its last write | 88 (cancel) |
  | B7 | order ids lose their `#` in the run's record | 13 (return) |
  | B8 | a write counts as failed when it worked | 0 (exchange) |
  | B9 | a guard rejects valid order ids | 1 (exchange) |
  | B10 | cancellations are left out of the run's record | 88 (cancel) |
  | B11 | an upstream error from a tool ends the run ([openai-agents#879](https://github.com/openai/openai-agents-python/issues/879)) | 13 (return) |
  | B12 | text sent alongside tool calls is dropped ([llama_index#21124](https://github.com/run-llama/llama_index/issues/21124)) | 0 (exchange) |
  | B13 | the model gets an empty tool result ([openai-agents#2110](https://github.com/openai/openai-agents-python/issues/2110)) | 88 (cancel) |
  | B14 | tool arguments silently dropped ([crewAI#5275](https://github.com/crewAIInc/crewAI/issues/5275)) | 0 (exchange) |
  | B15 | the policy never reaches the model ([llama_index#20872](https://github.com/run-llama/llama_index/issues/20872)) | 13 (return) |

  B11 to B15 are ported from real, fixed issues in agent frameworks, chosen by the
  rule and seeded draw in [PREREGISTRATION.md](PREREGISTRATION.md) (amendment 1).
  B1 to B5 are fixed by changing the calls the agent makes; B6 to B10 by changing
  only `run_record`, which builds the record after the conversation, so their fix
  leaves every call unchanged and a frozen replay can judge it.

- **The branches**: for each bug, `main` carries it; `fix` fixes it; `nofix` makes
  a cosmetic edit; `other` fixes it and adds a different bug.
- **The test** each branch is judged by: no error, exactly the task's gold writes
  (tool and arguments, once each, and no other write), and no duplicated write;
  for B6 to B10, also that the returned record lists the gold writes as resolved.

## How it is scored

1. **Ground truth.** Every branch runs live `--repeats` times (3 by default) and
   is judged by the test above and by tau2's own check (the database against the
   gold actions). Whether a branch is fixed is measured: it is if most of its
   runs pass. How a failing branch fails follows from how it was built, because a
   live model fails a little differently every run, so comparing live failures
   cannot tell "the same bug" from "another bug": `nofix` behaves exactly as
   `main`, so it is still failing; a failing `other` carries a different bug, so
   it diverged.
2. **Masked bugs are not scored.** If `main` passes live anyway (the bug never
   shows) or `fix` fails live anyway (the model fails the task without the bug),
   the bug cannot be judged on that model. The summary lists it with its live
   pass counts and leaves it out of the score.
3. **pyTracer's verdict.** `main` is recorded live; the recording must show the
   bug, failing the same checks as most of `main`'s live runs. A live model
   sometimes fails another way (it never reaches the write the bug is in), and
   such a recording cannot tell a fix from no fix, so it is recorded again, up to
   3 tries; a bug no try shows is reported as masked. Then `pytracer reproduce`
   judges every branch against that recording, under three settings:
   - `strict`: everything frozen, strict;
   - `fallback`: frozen but not strict (a call with no recording runs live), and a
     changed model call runs live (`--evaluate-changed-prompts`);
   - `partial`: tools frozen, the model live. Its verdicts are not frozen
     results, and pyTracer says so.
   - `cut-point` (added after the pre-registration, reported apart, not run by
     default: `--settings strict,cut-point`): frozen up to the first call that
     leaves the recording, live from there. It lets fixes that change the calls be
     judged, from a live run.
4. **The result** per branch: correct, an abstention (`cannot_judge` or
   `cannot_reproduce`, with pyTracer's reason), or wrong. The headline is the unsafe
   count: branches that are not fixed but reported `fixed`. It must be 0. Every
   share comes with its exact (Clopper-Pearson) 95% interval, the unsafe count with
   its upper bound, `fixed` with its precision and recall, and the verdicts are
   also split by the shape of each bug's fix, since that is what decides whether a
   frozen replay can judge it.

A verdict is drawn only from runs that stayed on their recording. A fix that
changes the calls themselves (B1's fix adds a status check before retrying) cannot
be judged from a recording made on the buggy branch, so pyTracer abstains and names
the call; the benchmark reports how often that happens.

Version 2 of the benchmark (15 bugs, a baseline, and bugs whose fix leaves the
calls unchanged, so `fixed` can be tested) is pre-registered in
[PREREGISTRATION.md](PREREGISTRATION.md): its bugs, metrics and predictions were
fixed before any v2 run.

`--baseline-repeats 5` adds the rerun-live baseline: 5 more live runs of each branch,
separate from the ground truth, scored at N = 1, 3 and 5 against pyTracer's verdicts,
tokens and minutes per verdict (the "Baseline" table in `summary.md`).

`python benchmarks/verdicts/run.py --rescore <out folder>` recomputes the labels and
scores of a finished run from its saved JSON without calling a model.
`BENCH_REPRODUCE_TIMEOUT` (seconds, 3600 by default) is how long each reproduce run
may take; a slow model running live needs far longer than pyTracer's default.

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

### The published runs: one customer, three agents

The simulated customer is GPT-5.2 in every run (`BENCH_USER_MODEL`), as in Sierra's
standardized tau2 runs, so differences between runs come from the agent alone.

| Run | Agent (`BENCH_MODEL`) | Why | Scope |
|---|---|---|---|
| 1 | GPT-5.2 | has an independent standardized retail score (81.6) | full |
| 2 | DeepSeek-V3.2 | open weights: anyone can rerun it (retail 81.1, reported by DeepSeek) | full |
| 3 | Claude Opus 5.5 | a top model from a third vendor | `--settings strict --repeats 1` |

On Microsoft Foundry, an Azure OpenAI deployment is `azure/<deployment>` (with
`AZURE_API_KEY`, `AZURE_API_BASE`, `AZURE_API_VERSION`); DeepSeek and Claude are
`azure_ai/<deployment>` (with `AZURE_AI_API_KEY`, `AZURE_AI_API_BASE`). Deploy
pay-per-token (Global Standard, or Standard / Data Zone Standard): a managed
compute template rents GPU machines by the hour, used or not. Each run writes to
its own `--out` folder.

### Capping the spend

`--max-usd` charges every live model call to `<out>/usage.jsonl` at
`BENCH_PRICE_IN` / `BENCH_PRICE_OUT` (the agent's USD per million tokens) and
`BENCH_USER_PRICE_IN` / `BENCH_USER_PRICE_OUT` (the customer's; the agent's by
default, `0` when it is paid from credit), all input at the full price. The
benchmark stops before a call that would start past the cap; a call already
running finishes, so the total can exceed the cap by at most one call. The ledger
persists in the `--out` folder, so a budget spans several invocations. With prompt
caching on an Anthropic model,
`BENCH_LLM_ARGS='{"cache_control_injection_points": [{"location": "message", "role": "system"}]}'`
caches the policy and the tool list; the cap still counts that input at full price.

Results go to `benchmarks/verdicts/results/`: one JSON per bug (every live run,
every verdict and reason) and `summary.md`, the table per setting. A full run is
about 125 to 135 agent runs, about 4,000 model calls and 25 million tokens.

`BENCH_MODEL=scripted:benchmarks/verdicts/smoke_script_task88.json` (and the same
for `BENCH_USER_MODEL`) replaces both models with a fixed conversation for task 88,
which checks the plumbing for `--bugs B1,B6,B10` at no cost; `smoke_script_task13`,
`task0` and `task1` do the same for B7, B8 and B9.

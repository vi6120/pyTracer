# Seeded-bug benchmark v2: pre-registration

Written 27 November 2026, before any v2 run. It fixes the bugs, the expected
outcomes, the metrics and the analysis, so the results cannot be fitted to them
afterwards. The git tag `bench-v2-prereg` marks this text; every v2 result is
reported against it, including the ones that come out badly for pyTracer.

## 1. Why v2

v1 (5 bugs, GPT-5.2 and Opus 5.5) gave no unsafe verdict (no broken branch
reported `fixed`) and no wrong verdict, but it left four gaps:

1. **`fixed` was never tested.** Every v1 fix changes the calls the agent makes,
   so a recording of the buggy code cannot judge it and pyTracer abstains. v1
   shows pyTracer does not wrongly say `fixed`; it never shows that it can rightly
   say so.
2. **The sample is small.** About 15 broken-branch verdicts: 0 unsafe bounds the
   unsafe rate only below about 20% (95%, rule of three).
3. **We chose every bug.**
4. **There is no baseline** to compare pyTracer against.

## 2. What stays the same

- tau2-bench retail, pinned at `v2.0.0`; its tasks, policy, tools, database check
  and user simulator. The simulated customer is GPT-5.2 in every run.
- Each bug has four branches: `main` carries it; `fix` fixes it; `nofix` makes a
  cosmetic edit; `other` fixes it and adds a different bug.
- Ground truth: every branch runs live; a branch is fixed if most of its runs pass.
  `nofix` is still failing by construction; a failing `other` diverged. A bug is
  masked (not scored) if `main` passes by majority or `fix` fails by majority.
- The recording is kept only if it fails the same checks as most of `main`'s live
  runs (up to 3 tries); otherwise the bug is masked (#165).
- Settings: `strict`, `fallback`, `partial` (partial verdicts come from a live
  model and are reported apart).

## 3. The bugs

### B1 to B5 (v1, unchanged)

| Bug | Task | What goes wrong | Change shape of the fix |
|---|---|---|---|
| B1 | 88 | a write is retried after a timeout that had already committed it | adds a call (a status check) |
| B2 | 0 | order ids lose their `#` | changes a call's arguments |
| B3 | 1 | tool results cut to 400 characters | changes the model's input |
| B4 | 5 | only the last 6 messages go to the model | changes the model's input |
| B5 | 13 | at most 2 items per call | changes a call's arguments |

### B6 to B10 (new: the fix leaves every call unchanged)

The agent gains one step after the conversation ends: `glue.run_record(calls)`
builds the record it returns to the caller (a ticketing system would store it):

```python
{"status": "resolved" | "failed" | "no_action",
 "writes": [{"tool": <write tool>, "order_id": <order id as sent>}, ...]}
```

`writes` lists, in order, every call to an order-changing tool whose result was
not an error; `status` is `resolved` if there was at least one such write and no
write returned an error, `failed` if any write returned an error, and `no_action`
if there was no write. The agent loop records each call as it is made (name, the
arguments as sent, the result) and passes the list in; it is the same on every
branch.

`run_record` runs after the last model and tool call, so no bug or fix in it can
change what is sent to the model or the tools. Each of these scenarios adds one
assertion to the three v1 checks, built from the task's gold actions:
`returns: {"status": "resolved", "writes": [<each gold write's tool and order id>]}`.

| Bug | Task | What goes wrong in `run_record` | `fix` | `nofix` (cosmetic) | `other` (a different bug) |
|---|---|---|---|---|---|
| B6 | 88 | the last write is dropped (`writes[:-1]`) | `writes` | `writes[: len(writes) - 1]` | order ids lose their `#` in the record |
| B7 | 13 | order ids lose their `#` in the record (`order_id.lstrip("#")`) | the id as sent | `order_id.replace("#", "", 1)` | the last write is dropped |
| B8 | 0 | a write counts as failed when its result is *not* an error (`not result.startswith("Error")`) | `result.startswith("Error")` | `result[:5] != "Error"` (the same inverted check) | the status is right, the last write is dropped |
| B9 | 1 | a guard raises `ValueError` on valid order ids (`#W\d{6}` instead of `#W\d{7}`) | `#W\d{7}` | `#W[0-9]{6}` | the guard is right, order ids lose their `#` |
| B10 | 88 | the write tools are matched against `ORDER_WRITES[1:]`, so `cancel_pending_order` is never recorded and the status is `no_action` | `ORDER_WRITES` | `ORDER_WRITES[1:len(ORDER_WRITES)]` | the status check is inverted (as B8) |

Every `other` was chosen to fail on the same task as its bug (each task has one
gold write), so a correct verdict for `other` is `diverged`.

### B11 to B15 (new: modelled on real issues, chosen independently)

Selection rule, fixed now, before any search:

1. Closed GitHub issues labelled as bugs in LangGraph, CrewAI, the OpenAI Agents
   SDK, AutoGen or LlamaIndex, opened between 1 June 2025 and 27 November 2026.
2. The bug is in code between a model and a tool: retries, argument handling,
   history trimming, result truncation or parsing, parallel tool calls, or what is
   returned after the run. The fix is in that code.
3. It can be ported to `glue.py` on a tau2 retail task with gold writes, without
   changing tau2 or the agent loop.

The first 15 issues that meet the rule, in the order the search returns them,
are the candidates. A developer outside the project picks 5 without seeing any
pyTracer result; if no one is available by 4 December 2026, 5 are drawn with
`random.Random(20261127).sample(candidates, 5)`. The candidates, the pick, and
each port (with links to the issue and its fix) are added to this file in one
amendment commit, dated, before the first v2 run. Their `nofix` and `other`
branches follow the same rules as above.

## 4. Runs

| Run | Agent | Bugs | Settings | Live repeats per branch | Spend |
|---|---|---|---|---|---|
| A | GPT-5.2 | B1 to B15 | strict, fallback, partial | 5 | Foundry credit |
| B | Opus 5.5 | B1 to B15 | strict | 3 | card, capped at $60 |
| Baseline | GPT-5.2 | B1 to B15 | n/a | 5 more per branch, separate from run A's | Foundry credit |

If a cap stops a run, the bugs it finished are reported and the rest are listed
as not run.

## 5. The baseline

The alternative to pyTracer is to run the branch live and look. The baseline says
a branch is `fixed` if most of N live runs pass, else `still_failing` (it cannot
tell `diverged` from `still_failing`, which is scored as wrong for an `other`
branch that fails). It is computed for N = 1, 3 and 5 from the baseline runs,
which are never the runs that set the ground truth; scoring the baseline against
its own runs would make it always right. For N = 1 and 3 every subset of the 5
runs is used and the scores are averaged.

## 6. Metrics

For each run and setting, over the bugs that are not masked:

1. **Unsafe verdicts (primary):** branches that are not fixed but are reported
   `fixed`. Reported as a count with its 95% Clopper-Pearson upper bound.
2. **`fixed` precision:** of the branches reported `fixed`, the share that are
   fixed. **`fixed` recall:** of the fixed branches, the share reported `fixed`.
3. **Correct, abstained, wrong:** counts and shares, with 95% Clopper-Pearson
   intervals, overall and per change shape (adds a call, changes arguments,
   changes the model's input, leaves every call unchanged).
4. **Cost and time per verdict:** USD and minutes, pyTracer against the baseline
   at N = 1, 3, 5.
5. **Masked bugs and missed recordings:** listed with their live pass counts.

## 7. Predictions

Written now so they can be checked; a prediction that fails is reported as failed.

| # | Prediction |
|---|---|
| P1 | 0 unsafe verdicts in strict and fallback, in both runs. |
| P2 | In strict, the `fix` branch of at least 4 of B6 to B10 (of those not masked) is reported `fixed`. |
| P3 | In strict, every `nofix` branch of B6 to B10 is reported `still_failing`, and every `other` branch of B6 to B10 that fails live is reported `diverged`. |
| P4 | On bugs whose fix changes the calls (B1 to B5), pyTracer abstains on the `fix` branch in strict, naming the call, as in v1. |
| P5 | The baseline at N = 1 makes at least one wrong verdict on a branch that is not masked; pyTracer in strict costs less per verdict than the baseline at N = 3. |
| P6 | Partial mode reports more abstentions than strict. |

## 8. What may change after the tag, and what may not

- **May not change:** the bugs, tasks, branches, assertions, labelling rules,
  masking rules, repeats, metrics and predictions above. B11 to B15 are added
  only by the amendment described in section 3.
- **May change, and is disclosed:** a harness bug that crashes a run or loses its
  output (as #161 and #166 did) is fixed, and the affected bugs are rerun; the
  fix and the rerun are listed in the results. A pyTracer bug found by v2 is fixed
  in its own PR; results are reported for the version that ran, and a rerun with
  the fix is reported beside it, never instead of it.
- **Every run is reported**, including stopped, crashed and repeated ones, in
  `results/` with its `run.log`.

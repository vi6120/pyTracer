# Changelog

## Unreleased

### Security

- **The `/pytracer try` workflow template ran a pull request's code with the
  repository's token.** It triggered on any comment containing `/pytracer try` and
  ran `pytracer reproduce --isolation local`, so on a public repository anyone could
  open a pull request from a fork, comment, and run their code with a token that can
  write to pull requests. The template now runs only on comments from owners,
  members and collaborators, runs the code with `--isolation docker` (the container
  never sees the token), leaves no credentials in the checkout, passes branch names
  through environment variables, and pins its actions to commits. **If you copied the
  template into a repository, update it.** SECURITY.md now has a threat model for
  every mode.

### Added

- **Compressed traces.** A trace file ending in `.jsonl.zst` is read transparently
  by `pytracer run`, `reproduce` and `record --from-file`, and `pytracer record
  --zstd` writes one. Traces repeat the prompt and conversation on every model call,
  so zstd makes them about 16 times smaller (measured on the seeded-bug benchmark's
  recordings, `benchmarks/storage`). zstd is built into Python 3.14; before that,
  `pip install "pytracer-cli[zstd]"`. Plain `.jsonl` works as before.
- **Retry steps.** `pytracer.step(name)` groups the model calls of a retry loop,
  such as a model asked again until its output validates against a Pydantic model,
  into one chain of attempts even though each retry sends a different input:
  numbered, linked, with `step.reject(error)` recording why an attempt was not used
  and the step recording the attempts, the rejections and the accepted attempt.
  Without it, those calls were unrelated spans that all looked successful.
- **Cut-point replay (experimental).** `pytracer run --cut-point` and `pytracer
  reproduce --cut-point` (or `cut_point: true` on a scenario) serve the recording up to
  the first call that leaves it, then run that call and every later tool and model
  call live, so a fix that changes the calls gets a verdict (`fixed`, `still_failing`
  or `diverged`) from that live run instead of `cannot_judge`; the verdict says it
  came from a live run. A write served from the recording before the cut still means
  `cannot_judge`, since the live run then starts from a state the recording does not
  describe. `run --json` reports `cut_at`, `live_after_cut` and
  `frozen_writes_before_cut`. On the seeded-bug benchmark's call-changing bugs
  (scripted), it took the correct verdicts from 5 of 15 to 14 of 15, with no wrong
  verdict.
- **Token counts on your own model calls.** An `@op(kind=SpanKind.LLM)` function
  that returns a provider response (OpenAI, Anthropic, LangChain, or a dict of the
  same) now records its token count from the response's usage, and
  `@op(..., tokens=reader)` reads any other shape (`pytracer.usage_tokens` is the
  default reader). Retry waste is then a number for hand-written retry loops, not
  only for the framework adapters, which already counted tokens.
- **Retry waste.** `pytracer run --json` (`retry_steps`), the PR report and
  `pytracer traces show` report each retry step's attempts, the accepted one, why
  the others were rejected, and the tokens and cost spent on attempts that were not
  used. A frozen replay reports what each attempt cost when it was recorded, only
  when it sent the recorded input; otherwise the waste is reported as unknown.
- **Side-effect assertions.** A scenario can now check what the agent wrote, not
  only that nothing visibly failed: `writes` (each expected write call happened,
  exactly `times` when given, with `args` matching; with `only`, no other declared
  write tool was called), `no_duplicate_writes` (no write tool called twice with
  the same arguments, the duplicate refund a retry after a timeout causes) and
  `returns` (fields of the agent's return value). They judge the write calls in
  the span tree, frozen or live, so a full replay checks them too.
- **Retry chains.** The tracer numbers repeated calls of an op with the same input
  under the same parent (`pytracer.attempt`) and links a retry to the attempt
  that failed (`pytracer.retry_of`). Reports list each repeated call with how
  every attempt ended, and `writes` failures quote that evidence.
- **Retries below the call.** HTTP requests sent inside a tool or model call are
  recorded on its span (`pytracer.http`: method, URL without the query string,
  status or error; never headers or bodies), and a request repeated after a
  failure is marked a retry. That makes a model SDK's own retry loop, or a tool
  that retries its POST after a timeout, visible where the call alone looks like
  one clean success. `no_duplicate_writes` fails on a repeated non-read request
  inside a write tool, and reports list repeated requests. Covers `httpx`,
  `httpx2` and `urllib3` (so `requests`); set `PYTRACER_HTTP_CAPTURE=0` to turn
  it off. A frozen call sends no requests, so this evidence comes from calls that
  ran: the capture, and tools a replay runs live.
- **`run --json`** now includes each scenario's return value, the spans of the
  replayed run, and its repeated calls (`retries`).
- **Microsoft Agent Framework adapter** (`pytracer-agent-framework`). Microsoft
  has archived AutoGen and points its users to Agent Framework. `install()`
  captures agent runs, model calls (streamed or not, with token counts) and
  function tools with no change to the agent code, and a captured run replays
  with the model and tool responses frozen. CI runs it on
  `agent-framework-core` 1.26.0, through a scripted client, Agent Framework's own
  OpenAI Chat Completions client, and the gateway-to-store path. Requires Python
  3.11 or later.
- **Claude Agent SDK adapter** (`pytracer-claude-agent-sdk`). `install()` captures
  `query()` and `ClaudeSDKClient` sessions with no change to the agent code: each
  turn of the Claude Code CLI is an LLM span keyed on what Python sent it, and
  each in-process `@tool` is a tool span. A captured session replays without
  starting the CLI; when a tool result, hook answer or prompt differs from the
  recording, the replay reports the changed turn instead of serving it. CI runs
  it on `claude-agent-sdk` 0.2.189 with a scripted CLI and the gateway-to-store
  path, and a live session (Claude Code 2.1.332) was captured and replayed
  with no CLI started and no tool run.

### Changed

- **Every package is now Apache-2.0.** `pytracer-gateway` and `pytracer-registry`,
  released under AGPL-3.0-or-later up to 0.1.4, are Apache-2.0 from this release, so
  teams whose policies rule out AGPL can self-host them. Published versions keep
  their license. The hosted service and its enterprise features remain a separate
  commercial product.

- Redaction is about 3x faster on the agent's thread: each default rule first runs
  a cheap check that every match must pass (a key prefix, an `@`, a run of 13
  digits) and skips its regex when the text cannot hold that secret. A rule's
  optional `may_match` does the same for custom rules. What is redacted does not
  change. On the SDK overhead benchmark the SDK's added time per agent run fell from
  9.7 ms to 3.0 ms (511 to 156 µs per span).

### Fixed

- **`pytracer reproduce` quoted the collection path into its shell command.** The
  runner builds one shell command (it chains steps with `&&`, and `--install` takes a
  shell command), and pasted `--collection` into it unquoted, so a path with a space
  broke the run and a path containing shell syntax (`x.yaml; ...`) ran it, in both the
  local and the Docker executor. The path is now quoted. Found by the new Semgrep scan.

- A recording could not replay itself when the code sliced or truncated a value
  the SDK had redacted: the placeholder (`[redacted:email]`) is shorter than the
  email it replaced, so `result[:400]` kept more text on replay than when it was
  recorded, the model input changed, and the baseline read as `cannot_reproduce`
  with advice to pass `--env-file`. A placeholder now keeps the length of a value
  at least as long as itself (padded with `*`; `Redactor(preserve_length=False)`
  turns it off), and a changed input that holds a redacted value is flagged, so
  the reason names redaction as the likely cause. Found by the seeded-bug
  benchmark (B3).
- Partial replay matched a frozen tool call by position even when an id in it
  had changed, as long as the id looked alike: a branch that fixed an order id
  (`#W2378156` instead of `W2378156`) was served the bug's recorded "Order not
  found" and reported still failing. A single-token string (an id, a code, an
  email) now has to match exactly, like a number; only text with spaces in it is
  matched loosely. Found by the seeded-bug benchmark.
- A `writes` failure now lists the calls of the tool whose arguments did not
  match, by the arguments that differ. Before, a write made with the wrong
  arguments read exactly like a write never made, so a branch that failed a new
  way could be judged still failing. Found by the seeded-bug benchmark.
- `reproduce` reported a branch `still_failing` when it failed the baseline's
  assertions the same way with no error, even after part of its run went live.
  A missing write can come from the live part (a simulated customer ending the
  call early), so a real fix read as a no-op. That match now counts only when
  both runs stayed on their recordings; otherwise the verdict is `cannot_judge`
  (or `cannot_reproduce`). The same error fingerprint still counts despite drift.
  Found by the seeded-bug benchmark.
- A `reproduce` run stopped by the time limit was reported as "produced no
  results", followed by whatever the agent printed last, so it read as a crash.
  It now says the run timed out, and `reproduce --timeout` sets the limit per
  ref (600 seconds local, 900 docker by default), for a slow live model in
  partial mode. Found by the seeded-bug benchmark.
- A `reproduce` verdict in partial or live mode was presented like a frozen one,
  though the model ran live and can answer differently next time. Such a verdict
  now carries a note that it came from a live model (in the CLI output and the
  PR comment), and `run --json` records each scenario's `mode`. Found by the
  seeded-bug benchmark.
- In partial or live mode, a `reproduce` abstention blamed the settings ("pass
  the run's settings with --env-file") when the baseline left its recording,
  though the usual cause there is the live model taking another path. It now
  says so and points to full mode; a branch that left its recording gets the
  same caveat. Found by the seeded-bug benchmark.

## 0.1.4

Fixes every verdict issue an outside study on tau2-bench found in 0.1.3, captures
every AutoGen release from 0.4.0, and hardens the gateway (it keeps writing through
a Redis outage, and several gateways can share one stream). All 12 packages move to
0.1.4 together. Upgrade a self-hosted gateway before the SDKs that send to it.

### Behaviour changes

- `reproduce` verdicts are stricter: a branch whose run left its recording is
  `cannot_judge` even if it passed, a baseline that left it is `cannot_reproduce`,
  and failing exactly as the baseline did is `still_failing`. Some verdicts that
  were `fixed` or `cannot_reproduce` on 0.1.3 change.
- With `strict: false`, a model call whose input changed no longer runs live as a
  fallback; it fails the run unless `evaluate_changed_prompts` is set.
- `pytracer reproduce --isolation local` installs nothing into your environment;
  pass `--install` for the old behaviour.

### Added

- **AutoGen 0.4 and later, captured in full.** AutoGen before 0.7.2 does not
  report a tool's result, and before 0.6.2 emits no agent or tool spans at all, so
  the adapter captured those runs only partly. `install()` now records what such
  a release leaves out through pyTracer itself (model and tool calls, and before
  0.6.2 each agent turn and team run), and CI runs the adapter on 0.4.0, 0.4.9,
  0.6.2, 0.7.2 and the latest release. Model calls recorded this way, including
  with `install(replayable=True)` on any release, now carry their token counts.
- **Strict replay fails on a missing recording even if the agent catches it.**
  A framework that swallows an agent's exception (as AutoGen's runtime does before
  0.4.9) no longer turns a strict-mode miss into a passing replay.
- **Replay through every framework adapter, without `@op`.** With replay switched
  on (`install_seams()` for LangGraph and CrewAI, `install(replayable=True)` for
  OpenAI Agents and AutoGen), the adapter routes the framework's own model and
  tool calls through pyTracer, so a captured run replays in full and partial
  modes like an `@op`-instrumented one, and streamed model replies replay chunk
  by chunk for LangGraph, OpenAI Agents and AutoGen (CrewAI replays the full reply
  its model call returns). Model calls are matched on what
  was asked, not on ids the framework generates per run. Built on new seams in
  the SDK (`pytracer_sdk.seam`).
- **AutoGen captures model calls and tool output.** Tool spans get their result
  from AutoGen's result message; model calls are recorded from the call events
  the model clients log (or rebuilt from AutoGen's replies for a client that logs
  none); a GenAI-instrumented client keeps its prompt and reply, read from the
  OpenTelemetry log records it sends them as (P7). A tool span waiting for its
  result is recorded within the wait even if the process goes idle (one background
  thread runs only while spans are waiting).

### Fixed

- **`pytracer reproduce --isolation local` leaves your environment alone.** It ran
  each ref's install steps in the current environment, with whichever `python` was
  first on `PATH`: `pip install -e .` re-pointed an editable install at a temporary
  checkout, and provisioning `pytracer-cli` at the running version could downgrade
  pyTracer itself. Local runs now install nothing and use the interpreter running
  `pytracer`; Docker runs still install in their container. `--install/--no-install`
  overrides either (`--no-install` also suits environments without pip, such as uv's).
- **`cannot_judge` and `cannot_reproduce` say why.** Every non-verdict now names
  the first call that left the recording and how (different arguments, more calls
  than recorded, or a call never recorded), in the terminal and the PR comment, and
  only a changed model input gets the `--evaluate-changed-prompts` advice; a changed
  tool call is told to re-record. Before, every `cannot_judge` blamed a prompt edit,
  even when the branch had only changed a tool call. `ReplayResult.divergences` and
  `run --json` carry each miss with its reason; `explain_reproduction` turns them
  into the message.
- **Cross-branch verdicts never trust a run that left its recording.** A branch
  that passed was reported `fixed` even when its calls had run live, while one that
  failed the same way was reported as an abstention, so a broken branch could be
  called fixed. Now a branch that left its recording (a frozen call found none, or
  ran live) is `cannot_judge` whether it passed or failed, a baseline that left it is
  `cannot_reproduce`, and a branch failing exactly as the baseline did is
  `still_failing` whatever else drifted. With `--evaluate-changed-prompts`, changed
  model calls are judged as asked. A model call whose input changed no longer runs
  live as a silent fallback when strict is off; it needs `evaluate_changed_prompts`.
  A swallowed miss is now reported as the run's cause even when a later call failed
  too, and `ReplayResult` and `run --json` list the calls that left the recording
  (`unrecorded_calls`) and those that ran live (`live_fallbacks`). Found by a partner
  study on tau2-bench.
- **Verdicts are plain text.** The `/pytracer try` comment labelled verdicts with
  emoji (check mark, cross, warning, question mark, party popper); it now uses the
  words alone, and a test keeps emoji out of every package's source.
- **`/pytracer try` and `pytracer reproduce` say "still failing" for a wrong
  answer.** A scenario that fails only on its assertions (nothing raised) has no
  failure fingerprint, so a branch that had not fixed it was reported as
  "diverged", a new failure, even against identical code. The verdict now treats
  the same assertions failing the same way as the same failure. Found by running
  the GitHub Action templates on a real pull request.
- **Several gateways on one Redis stream recover a crashed peer's spans.** A
  gateway reclaimed unacknowledged entries only when it started, so spans a
  crashed peer held stayed unwritten until some gateway restarted. With
  `PYTRACER_REDIS_RECLAIM_MIN_IDLE_MS` above zero a running gateway now writes
  them within that time, and never reclaims its own in-flight entries. Reads from
  the trace store return each span once, since such a retry can store a span
  twice. `deploy/docker-compose.scale.yml` runs two gateways behind Caddy.
- **The k6 load test measures ingestion.** Its spans had no `branch` or `commit`,
  so the gateway rejected every one (answering 202, as it does for a partly
  rejected batch) and the script counted them as accepted. Spans now carry the git
  context, acceptance is counted from the gateway's answer, and any rejected span
  fails the run. A `MODE=soak` constant-rate profile was added; a 30-minute soak
  at 3,600 spans/s wrote every span exactly once (benchmarks README).
- **Model calls keep their token counts with replay switched on.** With the
  replay routing installed (`install_seams()` for LangGraph and CrewAI,
  `install(replayable=True)` for OpenAI Agents), a model call recorded through it
  had no token count: the framework's own report of the usage was dropped. It now
  reaches the call's span (LangGraph from its callback, CrewAI from its usage tally,
  OpenAI Agents from the response), for streamed calls too. Adapters can credit
  usage with the new `pytracer_sdk.seam.add_tokens`.
- **The GitLab template runs in merge-request pipelines.** Its job had no `rules:`,
  so a project using merge-request pipelines got an empty pipeline. It now runs in
  merge-request and branch pipelines, once per change (only in the merge-request
  pipeline while one is open), verified on gitlab.com.
- **Adapters require the framework versions they are tested with.** The CrewAI,
  OpenAI Agents and LangGraph adapters now require CrewAI 1.15.20, OpenAI Agents
  0.20.0 and langchain-core 1.4.7 or later (they allowed far older releases), and
  CI tests the adapters at those minimums too. Asking for CrewAI beside OpenAI
  Agents 0.21 or later is now reported as a conflict by pip and uv, instead of
  being settled by quietly installing CrewAI 1.6.1.
- **The gateway keeps writing after a Redis outage.** One error from the ingest
  queue (for example Redis restarting) ended the gateway's background writer for
  good: ingest kept answering 202 while nothing reached storage until the gateway
  was restarted (the spans waited in Redis). The writer now logs the error, backs
  off and retries, keeps a batch it already took until it is written and
  acknowledged, and recreates the Redis consumer group if Redis came back without
  its data; Redis commands retry once over a fresh connection. Shutdown no longer
  hangs on an unreachable queue (`PYTRACER_GATEWAY_SHUTDOWN_TIMEOUT`), retried
  errors are counted (`writer_errors`, `pytracer_ingest_writer_errors_total`), and
  the compose health check probes `/readyz` instead of `/healthz`.
- **Recorded values keep their type.** `Decimal`, `datetime`, `date`, `time`,
  `timedelta`, `UUID`, `bytes`, enums, tuples, sets, dataclasses and pydantic
  models in span inputs and outputs are tagged when serialized and restored on
  load, on the JSONL, gateway and store paths, so a replay computes with the same
  types the agent saw. A pydantic model or dataclass returned by a call is kept as
  that object (it was flattened to a dict before), and a rebuilt pydantic model
  keeps which fields were set. Recordings made before this release must be
  re-recorded to benefit. Only serialized data is decoded (a recording, a store
  row, the gateway queue): a live value that looks like a tag, such as an end
  user's dict, is recorded as that dict. Classes, and recorded exception types,
  are only rebuilt from packages the process has already imported, so a recording
  never imports a module; and the gateway stores payloads exactly as received,
  including when it requeues dead letters.
- **Client-side span loss is visible (L1).** A full SDK queue or a failed delivery
  is logged as a rate-limited warning and passed to an optional
  `on_drop(reason, count, error)` callback set through `configure()`. A full queue
  is warned at its first drop, even while delivery is stuck in a slow send; the
  callback runs on the background thread, never the agent's.
- **Concurrent same-input calls replay by caller (R7).** Same-input calls racing
  on different threads each get the response recorded under their own calling op;
  only calls with an identical path are still reported as ambiguous.
- **AutoGen: one trace per team run (P2).** The adapter follows the links AutoGen
  puts between its per-message traces, records a team run as one trace, and nests
  it under an open pyTracer span like the other adapters.
- **The gateway reports its real version** in `/openapi.json`, `/docs`, the
  startup log and `/healthz` (instead of `0.0.1`).

### Documentation

- Using several frameworks in one environment: all four install together, with
  the constraints their own dependencies impose (CrewAI caps the OpenAI Agents SDK
  at 0.20.0; AutoGen needs `googleapis-common-protos<1.75.1`) and how to handle
  them. CI now runs every adapter with all four frameworks installed together.
- A per-adapter capability table on the docs page and in the README, and an
  accurate description of what the AutoGen adapter captures.
- One trace per request with `pytracer.start_span(...)`, for every adapter.
- Queue sizing guidance in the deploy guide.

### Known limitations

- Replays serve the redacted recording (masked values reach the agent), and a
  replay cannot choose its redactor.
- `reproduce` checks out local branches only; fetch a collaborator's branch first.
- `run --json` does not include the agent's return value or the replayed spans.
- `fixed` means no visible failure: a side effect that never happened (a write the
  agent skipped) fails only if an assertion checks for it (`tools_used` can).
- ClickHouse's own system logs grow without limit in the deploy stack (about 2.8 GB
  in six weeks with no traffic); cap them in its server config if disk matters.

## 0.1.3

First release after the pilot re-tests. All 12 packages move to 0.1.3 together.

**Upgrade order: upgrade the gateway before the SDKs that send to it (the hosted
gateway first).** A gateway on 0.1.3 ignores span fields it does not recognise, so
a 0.1.3 SDK stays compatible with it. A gateway still on 0.1.2 rejects a failed
span that carries `error_module`; a 0.1.3 SDK resends such a span without the
field so it still lands, but the field is stored only once the gateway is on
0.1.3.

### Fixed

- **SDK / gateway wire compatibility (V1).** A failed span is no longer dropped
  against an older gateway: the SDK resends the batch without a field the gateway
  rejects as unknown, and a 0.1.3 gateway ignores unknown fields.
- **Trace store keeps `error_module` (V2 / R4).** A failure captured through the
  gateway replays with its real exception type instead of a generic error; an
  idempotent column backfill upgrades an existing table.
- **Positional replay matching (R12 / V4).** Matching compares argument values,
  not the JSON skeleton, so a different id or a genuinely different question is a
  miss, never the wrong recording served silently.
- **Run verdict by the agent's outcome (V3).** A tool error the agent caught and
  recovered from no longer fails the replay; `no_errors` gains a `scope` (`root`
  default, `any` to also flag a handled child error).
- **Failure reason in `--json` (R3).** The report carries `error_message`, so
  `issues sync` and CI can see which call failed.
- **Honest `reproduce` verdicts (X4, X2 / X5).** `cannot_judge` (a model input
  changed) and `cannot_reproduce` (recordings did not match) exit with code `2`;
  `reproduce` gains `--evaluate-changed-prompts`.
- **Backpressure is not lossy (L1).** The queue-full `429` carries `Retry-After`
  and the SDK retries the backpressured spans instead of dropping the batch.
- **Deploy settings take effect (D1).** The compose file forwards the managed-key,
  rate-limit and queue-size settings, and the gateway logs the active auth backend
  at startup.
- **Full-mode replay blocks loopback (P6).** A framework model call cannot reach a
  local model server during replay; add a loopback host to the allowlist to permit
  it.
- **Framework adapters.** AutoGen captures via global registration with `install()`
  and resolves its tracer lazily (P1 / P2 / P8); CrewAI nests model and tool calls
  under the open agent (P3); the OpenAI Agents adapter folds structural task/turn
  spans (P4); the LangGraph, CrewAI and OpenAI Agents adapters nest a framework run
  under an active pyTracer span, so wrapping a multi-framework request in
  `pytracer.start_span(...)` (or an `@op`) makes it one trace; without that wrapper
  each framework run is still its own trace, and AutoGen does not nest (P5).

### Documentation

- `HTTPTransport.rejected_spans` and the `on_rejection` callback are documented
  (G1 / G5).
- The docs page covers positional matching, the `cannot_judge` / `cannot_reproduce`
  verdicts and exit code 2, `reproduce --evaluate-changed-prompts`, `no_errors`
  `scope`, and the gateway-first upgrade order.
- The deploy guide documents reaching the published stack's stores from the CLI,
  which publish no ports (D2).

### Known limitations

- A model call captured only through a framework adapter cannot be replayed through
  `@op` on tools; replaying an adapter-only run is flagged, not run live (A1,
  roadmap).
- Some GenAI-instrumented libraries send model output to the OpenTelemetry logs
  API, which the AutoGen exporter does not read, so those outputs are not captured
  (P7).

## 0.1.2

Initial public release (12 packages on PyPI).

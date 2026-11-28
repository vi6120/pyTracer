# The pick for B11 to B15: the pre-registered seeded draw

28 November 2026. The pre-registration (tag `bench-v2-prereg`, section 3) names an
outside developer to pick 5 of the 15 candidates, and the seeded draw as the
fallback. The planned picker, Nagesha (GitHub `sfdnas-adm`), has 26 commits in
pyTracer, so he is not outside the project; vi6120 chose the fallback instead.

    random.Random(20261127).sample(candidates, 5)

`candidates` is the 15 passing issues in creation order, as in candidates.md and
screening-log.csv. Python 3.12.4.

| Bug | Issue | Kind | Port to glue.py |
|---|---|---|---|
| B11 | openai-agents-python#879 | tool-call errors | call_tool lets some tool errors raise and end the run instead of returning "Error: ..." to the model |
| B12 | llama_index#21124 | history | history_for_model drops assistant text from messages that also carry tool calls |
| B13 | openai-agents-python#2110 | tool results | call_tool hands the model an empty string in place of a tool's result |
| B14 | ~~crewAI#5460~~ redrawn: crewAI#5275 | arguments | prepare_args looks the arguments up with a truthy default, so the tool runs with none |
| B15 | llama_index#20872 | history | history_for_model leaves out the system prompt (policy) |

Bug numbers follow the draw order.

## Redraw for B14

crewAI#5460 cannot fail in tau2 (every tool result is pure JSON, where its greedy
`{.*}` pattern happens to work); passing it in screening was an error. Rule fixed
before redrawing: one replacement from the 10 unpicked candidates,
`random.Random(20261128).sample(remaining, 1)`, and the next seed if that one cannot
be ported either. Result: crewAI#5275 (tool arguments silently dropped).


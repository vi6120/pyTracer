# Determinism: live runs versus replays

Source: `v2-gpt-5.2` (live runs as recorded by the seeded-bug benchmark). Machine: arm64, Python 3.12.4.

## Live

- Branches whose live runs disagreed (some passed, some failed): 14/60 (23.3%)
- Single live runs that gave a different answer than their branch's majority: 29/480 (6.0%)

| Bug | main | fix | nofix | other |
|---|---|---|---|---|
| B1 | 0/5 pass | 10/10 pass | 0/10 pass | 0/10 pass |
| B2 | 0/5 pass | 7/10 pass (flaky) | 0/10 pass | 0/10 pass |
| B3 | 0/5 pass | 7/10 pass (flaky) | 0/10 pass | 9/10 pass (flaky) |
| B4 | 0/5 pass | 0/5 pass | 0/5 pass | 2/5 pass (flaky) |
| B5 | 0/5 pass | 6/10 pass (flaky) | 0/10 pass | 0/10 pass |
| B6 | 0/5 pass | 10/10 pass | 0/10 pass | 0/10 pass |
| B7 | 0/5 pass | 3/10 pass (flaky) | 0/10 pass | 0/10 pass |
| B8 | 0/5 pass | 9/10 pass (flaky) | 0/10 pass | 0/10 pass |
| B9 | 0/5 pass | 9/10 pass (flaky) | 0/10 pass | 0/10 pass |
| B10 | 0/5 pass | 10/10 pass | 0/10 pass | 0/10 pass |
| B11 | 0/5 pass | 6/10 pass (flaky) | 0/10 pass | 0/10 pass |
| B12 | 5/5 pass | 4/5 pass (flaky) | 5/5 pass | 3/5 pass (flaky) |
| B13 | 0/5 pass | 10/10 pass | 0/10 pass | 0/10 pass |
| B14 | 0/5 pass | 9/10 pass (flaky) | 0/10 pass | 0/10 pass |
| B15 | 3/5 pass (flaky) | 5/5 pass | 4/5 pass (flaky) | 0/5 pass |

## Replay (strict, no model key in the environment)

- Branches whose 10 replays all gave the same verdict and reason: 36/36 (100.0%)
- Branches whose replays all matched the verdict the benchmark run reported: 36/36 (100.0%)

| Bug | Branch | Verdicts over the replays | Same every time | Matches the run | Median seconds |
|---|---|---|---|---|---|
| B1 | fix | cannot_judge x10 | yes | yes | 5.5 |
| B1 | nofix | still_failing x10 | yes | yes | 5.5 |
| B1 | other | cannot_judge x10 | yes | yes | 5.5 |
| B2 | fix | cannot_judge x10 | yes | yes | 5.4 |
| B2 | nofix | still_failing x10 | yes | yes | 5.5 |
| B2 | other | cannot_judge x10 | yes | yes | 5.3 |
| B3 | fix | cannot_judge x10 | yes | yes | 5.3 |
| B3 | nofix | still_failing x10 | yes | yes | 5.5 |
| B3 | other | cannot_judge x10 | yes | yes | 5.3 |
| B5 | fix | cannot_judge x10 | yes | yes | 5.5 |
| B5 | nofix | still_failing x10 | yes | yes | 5.4 |
| B5 | other | cannot_judge x10 | yes | yes | 5.5 |
| B6 | fix | fixed x10 | yes | yes | 5.5 |
| B6 | nofix | still_failing x10 | yes | yes | 5.5 |
| B6 | other | diverged x10 | yes | yes | 5.5 |
| B7 | fix | still_failing x10 | yes | yes | 5.6 |
| B7 | nofix | still_failing x10 | yes | yes | 5.6 |
| B7 | other | still_failing x10 | yes | yes | 5.5 |
| B8 | fix | fixed x10 | yes | yes | 5.7 |
| B8 | nofix | still_failing x10 | yes | yes | 5.6 |
| B8 | other | diverged x10 | yes | yes | 5.8 |
| B9 | fix | fixed x10 | yes | yes | 5.7 |
| B9 | nofix | still_failing x10 | yes | yes | 5.8 |
| B9 | other | diverged x10 | yes | yes | 5.7 |
| B10 | fix | fixed x10 | yes | yes | 5.9 |
| B10 | nofix | still_failing x10 | yes | yes | 5.8 |
| B10 | other | diverged x10 | yes | yes | 5.9 |
| B11 | fix | cannot_judge x10 | yes | yes | 5.7 |
| B11 | nofix | still_failing x10 | yes | yes | 5.7 |
| B11 | other | cannot_judge x10 | yes | yes | 5.7 |
| B13 | fix | cannot_judge x10 | yes | yes | 5.8 |
| B13 | nofix | still_failing x10 | yes | yes | 5.8 |
| B13 | other | cannot_judge x10 | yes | yes | 5.8 |
| B14 | fix | cannot_judge x10 | yes | yes | 5.7 |
| B14 | nofix | still_failing x10 | yes | yes | 5.7 |
| B14 | other | cannot_judge x10 | yes | yes | 5.7 |

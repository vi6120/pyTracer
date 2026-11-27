# pyTracer SDK overhead

A recorded 19-span agent run (model and tools stubbed), 1000 runs per configuration after 50 warm-up runs, interleaved in a seeded random order (seed 20261128). Times are on the agent's own thread, in milliseconds per run.

| Machine | |
|---|---|
| cpu | Apple M1 Pro |
| cores | 10 |
| os | Darwin 25.2.0 |
| python | 3.12.4 |
| pytracer-sdk | 0.1.4 |

| Configuration | p50 | p95 | p99 | Added at p50 | Added per span at p50 | Share of the run's model time |
|---|---|---|---|---|---|---|
| none | 0.002 | 0.003 | 0.004 | - | - | - |
| sdk | 9.710 | 10.256 | 16.200 | 9.708 | 510.9 µs | 0.035% |
| sdk-no-redaction | 0.895 | 6.537 | 9.602 | 0.893 | 47.0 µs | 0.003% |
| sdk-http | 9.710 | 10.162 | 15.880 | 9.708 | 510.9 µs | 0.035% |

The share compares the added time with the 27.6 s the recorded run spent waiting on its model calls (GPT-5.2 agent and customer), which is what a live run adds on top of the code measured here.

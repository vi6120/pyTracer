# Trace storage

9 trace file(s), 174 spans, 1191 KiB raw. Median run: 108 KiB raw, 18 spans; 6.8 KiB per span on average.

| Stored as | Size | Smaller by |
|---|---|---|
| JSONL as written | 1191 KiB | 1.0x |
| gzip -6 | 88 KiB | 13.5x |
| zstd -3 | 74 KiB | 16.1x |
| xz | 65 KiB | 18.2x |
| repeated values stored once, within each run | 408 KiB | 2.9x |
| repeated values stored once, across all runs | 330 KiB | 3.6x |

Repeated values are JSON values of at least 256 bytes (messages, tool results, the policy) kept once by hash, each repeat replaced by a 64-byte reference; the store would then compress the blobs as well.

"""Storage: what does an agent trace cost to keep, and how much of it repeats?

Measures trace files (pyTracer JSONL, one span per line) as they are written today:
bytes per span and per run, raw and compressed (gzip, zstd, xz), and how much of
the raw size a content-addressed store would save by keeping each repeated value
once. Agents resend the same large inputs on every model call (the system prompt,
the conversation so far), so a trace repeats itself.

The dedupe estimate treats every JSON value of at least ``--blob-bytes`` (a message,
a tool result, the policy) as a blob stored once by its hash: a value seen before
costs nothing again, and is not looked inside. It is computed within each run and
across all runs given (a store shares blobs between runs too).

    python benchmarks/storage/storage.py                       # the committed fixture
    python benchmarks/storage/storage.py path/to/*.jsonl ...   # your own traces
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import lzma
import statistics
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
FIXTURE = HERE.parent / "overhead" / "fixture_run.jsonl"


def _zstd(data: bytes) -> int | None:
    try:
        import zstandard
    except ImportError:
        return None
    return len(zstandard.ZstdCompressor(level=3).compress(data))


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()


def _unique_bytes(value: Any, seen: set[str], blob_bytes: int) -> int:
    """Bytes ``value`` adds to a store that keeps each large value once."""
    data = _canonical(value)
    if len(data) >= blob_bytes:
        key = hashlib.sha256(data).hexdigest()
        if key in seen:
            return 64  # a reference to the stored blob
        seen.add(key)
    if isinstance(value, dict):
        inner = sum(_unique_bytes(v, seen, blob_bytes) + len(_canonical(k)) + 2 for k, v in value.items())
        return min(len(data), inner + 2) if len(data) >= blob_bytes else len(data)
    if isinstance(value, list):
        inner = sum(_unique_bytes(v, seen, blob_bytes) + 1 for v in value)
        return min(len(data), inner + 2) if len(data) >= blob_bytes else len(data)
    return len(data)


def measure(path: Path, seen_across: set[str], blob_bytes: int) -> dict[str, Any]:
    raw = path.read_bytes()
    spans = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    seen_within: set[str] = set()
    within = sum(_unique_bytes(s, seen_within, blob_bytes) for s in spans)
    across = sum(_unique_bytes(s, seen_across, blob_bytes) for s in spans)
    return {
        "file": str(path),
        "spans": len(spans),
        "raw": len(raw),
        "gzip": len(gzip.compress(raw, compresslevel=6)),
        "zstd": _zstd(raw),
        "xz": len(lzma.compress(raw)),
        "dedupe_within": within,
        "dedupe_across": across,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("traces", nargs="*", type=Path, default=[FIXTURE])
    parser.add_argument("--blob-bytes", type=int, default=256)
    parser.add_argument("--out", type=Path, help="write the summary here as well")
    args = parser.parse_args()
    seen: set[str] = set()
    rows = [measure(p, seen, args.blob_bytes) for p in args.traces]
    total = {k: sum(r[k] or 0 for r in rows) for k in ("spans", "raw", "gzip", "zstd", "xz",
                                                       "dedupe_within", "dedupe_across")}
    have_zstd = all(r["zstd"] is not None for r in rows)

    def ratio(k: str) -> str:
        return f"{total['raw'] / total[k]:.1f}x" if total[k] else "-"

    lines = [
        "# Trace storage", "",
        f"{len(rows)} trace file(s), {total['spans']} spans, {total['raw'] / 1024:.0f} KiB raw. "
        f"Median run: {statistics.median(r['raw'] for r in rows) / 1024:.0f} KiB raw, "
        f"{statistics.median(r['spans'] for r in rows):.0f} spans; "
        f"{total['raw'] / total['spans'] / 1024:.1f} KiB per span on average.", "",
        "| Stored as | Size | Smaller by |",
        "|---|---|---|",
        f"| JSONL as written | {total['raw'] / 1024:.0f} KiB | 1.0x |",
        f"| gzip -6 | {total['gzip'] / 1024:.0f} KiB | {ratio('gzip')} |",
    ]
    if have_zstd:
        lines.append(f"| zstd -3 | {total['zstd'] / 1024:.0f} KiB | {ratio('zstd')} |")
    lines += [
        f"| xz | {total['xz'] / 1024:.0f} KiB | {ratio('xz')} |",
        f"| repeated values stored once, within each run | {total['dedupe_within'] / 1024:.0f} KiB | {ratio('dedupe_within')} |",
        f"| repeated values stored once, across all runs | {total['dedupe_across'] / 1024:.0f} KiB | {ratio('dedupe_across')} |",
        "",
        f"Repeated values are JSON values of at least {args.blob_bytes} bytes (messages, tool "
        "results, the policy) kept once by hash, each repeat replaced by a 64-byte reference; "
        "the store would then compress the blobs as well.",
    ]
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)


if __name__ == "__main__":
    sys.exit(main())

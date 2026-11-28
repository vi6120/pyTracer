"""Reading and writing trace files, plain (``.jsonl``) or zstd-compressed (``.jsonl.zst``).

A trace repeats the agent's prompt and conversation on every model call, so zstd
makes it about 16 times smaller (``benchmarks/storage``). A path ending in ``.zst``
is compressed; any other is plain text. zstd comes from the standard library on
Python 3.14 (``compression.zstd``), else from the ``zstandard`` package
(``pip install "pytracer-cli[zstd]"``).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

ZST_SUFFIX = ".zst"
_LEVEL = 3


def _codec() -> tuple[Callable[[bytes], bytes], Callable[[bytes], bytes]]:
    """(compress, decompress) from whichever zstd is installed."""
    try:
        from compression import zstd  # type: ignore[import-not-found,unused-ignore]  # Python 3.14+
    except ImportError:
        pass
    else:
        return (lambda data: zstd.compress(data, level=_LEVEL)), zstd.decompress
    try:
        import zstandard
    except ImportError as exc:
        raise RuntimeError(
            'reading or writing a .zst trace needs zstd: pip install "pytracer-cli[zstd]"'
        ) from exc
    return (
        lambda data: zstandard.ZstdCompressor(level=_LEVEL).compress(data),
        # decompressobj also reads frames that do not record their size.
        lambda data: zstandard.ZstdDecompressor().decompressobj().decompress(data),
    )


def is_compressed(path: Path) -> bool:
    return path.name.endswith(ZST_SUFFIX)


def read_trace_text(path: Path) -> str:
    """A trace file's text, decompressing a ``.zst`` file."""
    data = path.read_bytes()
    if is_compressed(path):
        data = _codec()[1](data)
    return data.decode("utf-8")


def write_trace_text(path: Path, text: str) -> None:
    """Write a trace file, compressing it when the path ends in ``.zst``."""
    data = text.encode("utf-8")
    if is_compressed(path):
        data = _codec()[0](data)
    path.write_bytes(data)

"""What the installed AutoGen reports on its own, which decides how a run is captured.

AutoGen's OpenTelemetry output grew over its releases (measured on 0.4.0 to 0.7.5):

- 0.7.2 and later: GenAI agent and tool spans, and each tool's result in the runtime
  message that delivers it, so the exporter captures a run from AutoGen's own spans.
- 0.6.2 to 0.7.1: the GenAI spans, but no tool results in the messages, so a tool
  span would have no output.
- before 0.6.2: no GenAI spans at all (0.4.9 to 0.6.1 emit only the runtime's
  messaging spans; 0.4.0 to 0.4.8 nothing).

Where AutoGen reports less, the adapter records the calls itself through the seams
replay already uses, so every supported version is captured in full.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

# Below this, model and tool calls are recorded through the seams (with their output).
SEAMS_CAPTURE_BELOW = (0, 7, 2)
# Below this, agent turns and team runs are recorded through seams too.
AGENT_SEAMS_BELOW = (0, 6, 2)


def autogen_version() -> tuple[int, ...] | None:
    """The installed ``autogen-core`` release as numbers, or ``None`` if absent."""
    try:
        raw = version("autogen-core")
    except PackageNotFoundError:
        return None
    parts: list[int] = []
    for piece in raw.split("."):
        digits = ""
        for char in piece:
            if not char.isdigit():
                break
            digits += char
        if not digits:
            break
        parts.append(int(digits))
        if len(digits) != len(piece):
            break  # a pre-release or local suffix ends the release number
    return tuple(parts)


def captures_by_seams(installed: tuple[int, ...] | None = None) -> bool:
    """True when model and tool calls must be recorded through the seams."""
    found = installed if installed is not None else autogen_version()
    return found is not None and found < SEAMS_CAPTURE_BELOW


def seams_agent_turns(installed: tuple[int, ...] | None = None) -> bool:
    """True when agent turns and team runs must be recorded through seams."""
    found = installed if installed is not None else autogen_version()
    return found is not None and found < AGENT_SEAMS_BELOW

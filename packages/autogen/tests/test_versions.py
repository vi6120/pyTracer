"""Which capture path each AutoGen release gets (see ``pytracer_autogen.versions``)."""

from __future__ import annotations

import pytest

from pytracer_autogen import versions


@pytest.mark.parametrize(
    ("release", "by_seams", "agent_turns"),
    [
        ((0, 4, 0), True, True),
        ((0, 4, 9, 3), True, True),
        ((0, 6, 1), True, True),
        ((0, 6, 2), True, False),
        ((0, 7, 1), True, False),
        ((0, 7, 2), False, False),
        ((0, 7, 5), False, False),
        ((1, 0), False, False),
    ],
)
def test_capture_path_by_release(release: tuple[int, ...], by_seams: bool, agent_turns: bool) -> None:
    assert versions.captures_by_seams(release) is by_seams
    assert versions.seams_agent_turns(release) is agent_turns


@pytest.mark.parametrize(
    ("raw", "parsed"),
    [("0.7.5", (0, 7, 5)), ("0.4.9.3", (0, 4, 9, 3)), ("0.4.0.dev13", (0, 4, 0)), ("0.8.0rc1", (0, 8, 0))],
)
def test_release_numbers_are_parsed(monkeypatch: pytest.MonkeyPatch, raw: str, parsed: tuple[int, ...]) -> None:
    monkeypatch.setattr(versions, "version", lambda name: raw)
    assert versions.autogen_version() == parsed


def test_no_autogen_installed_keeps_the_default_path(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str) -> str:
        raise versions.PackageNotFoundError(name)

    monkeypatch.setattr(versions, "version", missing)
    assert versions.autogen_version() is None
    assert versions.captures_by_seams() is False and versions.seams_agent_turns() is False

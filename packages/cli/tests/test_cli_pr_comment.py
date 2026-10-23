"""'Try on this branch' PR comment: client comment methods + upsert orchestration.

The client's REST comment methods are exercised with an in-memory fake opener;
the render/upsert logic is exercised with a fake client, so neither touches the
network.
"""

from __future__ import annotations

import json
import urllib.parse
from typing import Any

from pytracer_cli.github import CommentRef, GitHubClient
from pytracer_cli.pr_comment import (
    COMMENT_MARKER,
    post_reproduction_comment,
    render_comment,
)
from pytracer_replay import Outcome


class _Resp:
    def __init__(self, payload: Any) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _Resp:  # noqa: PYI034
        return self

    def __exit__(self, *_: object) -> bool:
        return False


class FakeComments:
    """A tiny in-memory GitHub issue-comments API for one issue/PR."""

    def __init__(self) -> None:
        self.comments: list[dict[str, Any]] = []
        self._n = 0

    def opener(self, req: Any) -> _Resp:
        method = req.method
        parts = urllib.parse.urlparse(req.full_url)
        path = parts.path
        qs = urllib.parse.parse_qs(parts.query)
        data = json.loads(req.data.decode("utf-8")) if req.data else None

        if method == "GET" and path.endswith("/comments"):
            if int(qs.get("page", ["1"])[0]) > 1:
                return _Resp([])
            return _Resp(self.comments)
        if method == "POST" and path.endswith("/comments"):
            self._n += 1
            assert data is not None
            comment = {"id": self._n, "body": data["body"], "html_url": f"https://gh/c/{self._n}"}
            self.comments.append(comment)
            return _Resp(comment)
        if method == "PATCH" and "/issues/comments/" in path:
            cid = int(path.rsplit("/", 1)[1])
            assert data is not None
            for comment in self.comments:
                if comment["id"] == cid:
                    comment["body"] = data["body"]
            return _Resp({"id": cid})
        raise AssertionError(f"unexpected {method} {path}")


def _client(gh: FakeComments) -> GitHubClient:
    return GitHubClient("tok", "owner/name", opener=gh.opener)


# --- client comment methods -------------------------------------------------


def test_find_comment_by_marker_absent_then_present() -> None:
    gh = FakeComments()
    client = _client(gh)
    assert client.find_comment_by_marker(7, COMMENT_MARKER) is None
    ref = client.create_comment(7, f"{COMMENT_MARKER}\nhello")
    assert isinstance(ref, CommentRef)
    found = client.find_comment_by_marker(7, COMMENT_MARKER)
    assert found is not None and found.id == ref.id


def test_update_comment_patches_body() -> None:
    gh = FakeComments()
    client = _client(gh)
    ref = client.create_comment(7, f"{COMMENT_MARKER}\nold")
    client.update_comment(ref.id, f"{COMMENT_MARKER}\nnew")
    assert gh.comments[0]["body"].endswith("new")


# --- rendering --------------------------------------------------------------


def test_render_all_fixed() -> None:
    body = render_comment(
        candidate_ref="fix", baseline_ref="main", verdicts={"a": Outcome.FIXED}
    )
    assert COMMENT_MARKER in body
    assert "try on this branch" in body
    assert "fixed" in body
    assert "All 1 scenario(s) fixed" in body


def test_render_mixed_outcomes() -> None:
    body = render_comment(
        candidate_ref="fix",
        baseline_ref="main",
        verdicts={"a": Outcome.FIXED, "b": Outcome.STILL_FAILING, "c": Outcome.DIVERGED},
    )
    assert "still failing" in body
    assert "diverged" in body
    assert "All" not in body.split("| Scenario")[0]  # no all-fixed summary


def test_render_notes_changed_model_inputs() -> None:
    # When a model input changed, the comment stops claiming everything was held
    # frozen and names the scenarios that cannot be judged by a frozen replay (X4).
    body = render_comment(
        candidate_ref="fix",
        baseline_ref="main",
        verdicts={"a": Outcome.DIVERGED},
        changed_models={"a": ["call_model"]},
    )
    assert "Model inputs changed" in body
    assert "call_model" in body
    assert "held frozen" not in body  # the unconditional claim is gone when inputs changed


def test_render_without_changes_keeps_the_frozen_wording() -> None:
    body = render_comment(
        candidate_ref="fix", baseline_ref="main", verdicts={"a": Outcome.FIXED},
        changed_models={"a": []},
    )
    assert "held frozen" in body
    assert "Model inputs changed" not in body


def test_render_cannot_reproduce_hints_env_file() -> None:
    # X2/X5: a cannot-reproduce verdict explains the setup cause instead of a verdict.
    body = render_comment(
        candidate_ref="fix", baseline_ref="main", verdicts={"a": Outcome.CANNOT_REPRODUCE}
    )
    assert "cannot reproduce" in body
    assert "--env-file" in body


def test_render_advice_switches_when_evaluate_is_on() -> None:
    # X4: do not advise turning on --evaluate-changed-prompts when it is already on.
    off = render_comment(
        candidate_ref="fix", baseline_ref="main", verdicts={"a": Outcome.CANNOT_JUDGE},
        changed_models={"a": ["call_model"]},
    )
    assert "re-run with `--evaluate-changed-prompts`" in off
    on = render_comment(
        candidate_ref="fix", baseline_ref="main", verdicts={"a": Outcome.CANNOT_JUDGE},
        changed_models={"a": ["call_model"]}, evaluate_changed_prompts=True,
    )
    assert "evaluated live" in on
    assert "re-run with `--evaluate-changed-prompts`" not in on


# --- upsert orchestration ---------------------------------------------------


class FakeClient:
    def __init__(self, existing: CommentRef | None = None) -> None:
        self._existing = existing
        self.created: list[tuple[int, str]] = []
        self.updated: list[tuple[int, str]] = []

    def find_comment_by_marker(self, number: int, marker: str) -> CommentRef | None:
        return self._existing

    def create_comment(self, number: int, body: str) -> CommentRef:
        self.created.append((number, body))
        return CommentRef(11, "https://gh/c/11")

    def update_comment(self, comment_id: int, body: str) -> None:
        self.updated.append((comment_id, body))


def test_post_creates_when_absent() -> None:
    client = FakeClient(existing=None)
    url = post_reproduction_comment(
        client, number=7, candidate_ref="fix", baseline_ref="main", verdicts={"a": Outcome.FIXED}
    )
    assert client.created and not client.updated
    assert url == "https://gh/c/11"


def test_post_updates_when_present() -> None:
    client = FakeClient(existing=CommentRef(99, "https://gh/c/99"))
    url = post_reproduction_comment(
        client, number=7, candidate_ref="fix", baseline_ref="main", verdicts={"a": Outcome.FIXED}
    )
    assert client.updated and not client.created
    assert client.updated[0][0] == 99
    assert url == "https://gh/c/99"

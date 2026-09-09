"""GitHub issue client: create / update / reopen, driven by an in-memory fake.

The fake implements just enough of the REST API to exercise the real client's
URL building, JSON, and dedup logic without any network access.
"""

from __future__ import annotations

import json
import urllib.parse
from typing import Any

from pytracer_cli.github import GitHubClient, fingerprint_marker, sync_issue
from pytracer_cli.report import issue_body_from_json


class _Resp:
    def __init__(self, payload: Any) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _Resp:  # noqa: PYI034
        return self

    def __exit__(self, *_: object) -> bool:
        return False


class FakeGitHub:
    """A tiny in-memory GitHub issues API."""

    def __init__(self) -> None:
        self.issues: list[dict[str, Any]] = []
        self._n = 0

    def _issue(self, num: int) -> dict[str, Any]:
        return next(i for i in self.issues if i["number"] == num)

    def opener(self, req: Any) -> _Resp:
        method = req.method
        parts = urllib.parse.urlparse(req.full_url)
        path = parts.path
        qs = urllib.parse.parse_qs(parts.query)
        data = json.loads(req.data.decode("utf-8")) if req.data else None

        if method == "GET" and path.endswith("/issues"):
            state = qs.get("state", ["open"])[0]
            page = int(qs.get("page", ["1"])[0])
            if page > 1:
                return _Resp([])
            return _Resp([i for i in self.issues if i["state"] == state])
        if method == "POST" and path.endswith("/issues"):
            self._n += 1
            assert data is not None
            issue = {
                "number": self._n,
                "state": "open",
                "title": data["title"],
                "body": data["body"],
                "html_url": f"https://github.example/{self._n}",
                "comments": [],
            }
            self.issues.append(issue)
            return _Resp(issue)
        if method == "POST" and path.endswith("/comments"):
            num = int(path.split("/issues/")[1].split("/")[0])
            assert data is not None
            self._issue(num)["comments"].append(data["body"])
            return _Resp({"id": 1})
        if method == "PATCH" and "/issues/" in path:
            num = int(path.rsplit("/", 1)[1])
            issue = self._issue(num)
            assert data is not None
            issue["state"] = data.get("state", issue["state"])
            return _Resp(issue)
        raise AssertionError(f"unexpected {method} {path}")


def _client(gh: FakeGitHub) -> GitHubClient:
    return GitHubClient("tok", "owner/name", opener=gh.opener)


def _body(fp: str) -> str:
    return f"a failure body\n{fingerprint_marker(fp)}\n"


def test_creates_new_issue() -> None:
    gh = FakeGitHub()
    res = sync_issue(_client(gh), fingerprint="fp1", title="T", body=_body("fp1"), labels=["l"])
    assert res.action == "created"
    assert len(gh.issues) == 1
    assert gh.issues[0]["title"] == "T"


def test_recurrence_updates_the_same_issue() -> None:
    gh = FakeGitHub()
    client = _client(gh)
    r1 = sync_issue(client, fingerprint="fp1", title="T", body=_body("fp1"))
    r2 = sync_issue(client, fingerprint="fp1", title="T", body=_body("fp1"))
    assert r1.action == "created"
    assert r2.action == "updated"
    assert r2.number == r1.number
    assert len(gh.issues) == 1  # not duplicated
    assert len(gh.issues[0]["comments"]) == 1  # recurrence recorded as a comment


def test_reopens_a_closed_issue() -> None:
    gh = FakeGitHub()
    client = _client(gh)
    sync_issue(client, fingerprint="fp1", title="T", body=_body("fp1"))
    gh.issues[0]["state"] = "closed"  # the bug came back after the issue was closed
    res = sync_issue(client, fingerprint="fp1", title="T", body=_body("fp1"))
    assert res.action == "reopened"
    assert gh.issues[0]["state"] == "open"


def test_find_ignores_pull_requests() -> None:
    gh = FakeGitHub()
    gh.issues.append(
        {
            "number": 9,
            "state": "open",
            "title": "a PR",
            "body": fingerprint_marker("fp1"),
            "html_url": "u",
            "comments": [],
            "pull_request": {"url": "x"},
        }
    )
    assert _client(gh).find_issue_by_fingerprint("fp1") is None


def test_distinct_fingerprints_get_distinct_issues() -> None:
    gh = FakeGitHub()
    client = _client(gh)
    sync_issue(client, fingerprint="fp1", title="A", body=_body("fp1"))
    sync_issue(client, fingerprint="fp2", title="B", body=_body("fp2"))
    assert len(gh.issues) == 2


def test_issue_body_from_json_carries_marker() -> None:
    scenario = {
        "name": "broken",
        "passed": False,
        "status": "error",
        "error_type": "KeyError",
        "config_error": None,
        "fingerprint": "abc123",
        "assertions": [{"name": "no_errors", "passed": False, "detail": "KeyError"}],
    }
    body = issue_body_from_json(scenario, branch="feature/x", commit="deadbeef")
    assert "abc123" in body
    assert "<!-- pytracer-fingerprint: abc123 -->" in body
    assert "feature/x" in body and "deadbeef" in body
    assert "KeyError" in body

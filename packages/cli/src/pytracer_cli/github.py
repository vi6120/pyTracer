"""Open or update GitHub issues for agent-test failures (live auto-documentation).

A failure fingerprint deduplicates issues: it is embedded as a hidden marker in
the issue body, so a recurring failure updates one issue (a new comment, and a
reopen if it had been closed) instead of opening a fresh issue each run.

The client speaks the GitHub REST API over stdlib ``urllib`` (no extra
dependency). The HTTP layer is injectable via ``opener`` so the client is tested
without network access, and ``api_url`` can point at a GitHub Enterprise host.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .report import FINGERPRINT_MARKER

GITHUB_API = "https://api.github.com"

# An opener takes a urllib Request and returns a context manager yielding a
# response with .read(); urllib.request.urlopen is the default.
HttpOpener = Callable[[urllib.request.Request], Any]


def fingerprint_marker(fingerprint: str) -> str:
    """The hidden body marker CI matches on to dedup issues for one failure."""
    return FINGERPRINT_MARKER.format(fp=fingerprint)


@dataclass
class IssueRef:
    """A reference to a GitHub issue."""

    number: int
    state: str  # "open" | "closed"
    url: str


@dataclass
class CommentRef:
    """A reference to a GitHub issue or pull-request comment."""

    id: int
    url: str


@dataclass
class SyncResult:
    """Outcome of syncing one failure to an issue."""

    action: str  # "created" | "updated" | "reopened"
    number: int
    url: str


class GitHubClient:
    """Thin GitHub REST client for issue create/find/comment/reopen."""

    def __init__(
        self,
        token: str,
        repo: str,
        *,
        api_url: str = GITHUB_API,
        opener: HttpOpener | None = None,
    ) -> None:
        if "/" not in repo:
            raise ValueError(f"repo must be 'owner/name', got {repo!r}")
        self._token = token
        self._repo = repo
        self._api = api_url.rstrip("/")
        self._opener: HttpOpener = opener or urllib.request.urlopen

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        url = f"{self._api}{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Authorization", f"Bearer {self._token}")
        req.add_header("Accept", "application/vnd.github+json")
        req.add_header("X-GitHub-Api-Version", "2022-11-28")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        with self._opener(req) as resp:
            body = resp.read()
        return json.loads(body) if body else {}

    def find_issue_by_fingerprint(self, fingerprint: str) -> IssueRef | None:
        """Return the open-or-closed issue carrying this fingerprint, or None.

        Lists issues and scans bodies for the marker, which is deterministic and
        avoids the search index's short delay for freshly created issues.
        """
        marker = fingerprint_marker(fingerprint)
        for state in ("open", "closed"):
            page = 1
            while True:
                issues = self._request(
                    "GET",
                    f"/repos/{self._repo}/issues",
                    params={"state": state, "per_page": 100, "page": page},
                )
                if not issues:
                    break
                for issue in issues:
                    # The issues endpoint also returns pull requests; skip them.
                    if "pull_request" in issue:
                        continue
                    if marker in (issue.get("body") or ""):
                        return IssueRef(issue["number"], issue["state"], issue["html_url"])
                if len(issues) < 100:
                    break
                page += 1
        return None

    def create_issue(
        self, title: str, body: str, labels: list[str] | None = None
    ) -> IssueRef:
        payload: dict[str, Any] = {"title": title, "body": body}
        if labels:
            payload["labels"] = labels
        issue = self._request("POST", f"/repos/{self._repo}/issues", payload)
        return IssueRef(issue["number"], issue["state"], issue["html_url"])

    def add_comment(self, number: int, body: str) -> None:
        self._request("POST", f"/repos/{self._repo}/issues/{number}/comments", {"body": body})

    def reopen_issue(self, number: int) -> None:
        self._request("PATCH", f"/repos/{self._repo}/issues/{number}", {"state": "open"})

    def find_comment_by_marker(self, number: int, marker: str) -> CommentRef | None:
        """Return the issue/PR comment on ``number`` carrying ``marker``, or None.

        Comments live under the issues endpoint (a pull request is an issue for
        comment purposes), so this works for both.
        """
        page = 1
        while True:
            comments = self._request(
                "GET",
                f"/repos/{self._repo}/issues/{number}/comments",
                params={"per_page": 100, "page": page},
            )
            if not comments:
                return None
            for comment in comments:
                if marker in (comment.get("body") or ""):
                    return CommentRef(comment["id"], comment["html_url"])
            if len(comments) < 100:
                return None
            page += 1

    def create_comment(self, number: int, body: str) -> CommentRef:
        comment = self._request(
            "POST", f"/repos/{self._repo}/issues/{number}/comments", {"body": body}
        )
        return CommentRef(comment["id"], comment["html_url"])

    def update_comment(self, comment_id: int, body: str) -> None:
        self._request(
            "PATCH", f"/repos/{self._repo}/issues/comments/{comment_id}", {"body": body}
        )


def sync_issue(
    client: GitHubClient,
    *,
    fingerprint: str,
    title: str,
    body: str,
    labels: list[str] | None = None,
) -> SyncResult:
    """Create the issue for a failure, or update the existing one for its fingerprint.

    First time: open a new issue. Recurrence: comment with the latest run's body,
    reopening the issue first if it had been closed.
    """
    existing = client.find_issue_by_fingerprint(fingerprint)
    if existing is None:
        ref = client.create_issue(title, body, labels)
        return SyncResult("created", ref.number, ref.url)
    action = "updated"
    if existing.state == "closed":
        client.reopen_issue(existing.number)
        action = "reopened"
    client.add_comment(existing.number, f"Failure seen again.\n\n{body}")
    return SyncResult(action, existing.number, existing.url)

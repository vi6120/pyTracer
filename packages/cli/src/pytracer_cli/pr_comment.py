"""Post a cross-branch reproduction verdict as a PR comment ("try on this branch").

Renders the fixed / still-failing / diverged verdicts as a Markdown comment that
carries a hidden marker, then upserts it on the pull request: update the existing
pyTracer comment if one is present (so a re-push edits one comment instead of
stacking new ones), otherwise create it. The GitHub calls reuse ``GitHubClient``,
and the orchestration takes any object with the same comment methods so it is
tested without the network.
"""

from __future__ import annotations

from typing import Any, Protocol

from .github import CommentRef

# Hidden HTML comment used to find and update our own comment on re-runs.
COMMENT_MARKER = "<!-- pytracer:try-on-branch -->"

_LABELS = {
    "fixed": ":white_check_mark: fixed",
    "still_failing": ":x: still failing",
    "diverged": ":warning: diverged",
    "cannot_judge": ":grey_question: cannot judge (a model input changed)",
    "cannot_reproduce": ":grey_question: cannot reproduce (recordings did not match)",
}


class _CommentClient(Protocol):
    def find_comment_by_marker(self, number: int, marker: str) -> CommentRef | None: ...
    def create_comment(self, number: int, body: str) -> CommentRef: ...
    def update_comment(self, comment_id: int, body: str) -> None: ...


def _value(outcome: Any) -> str:
    return getattr(outcome, "value", str(outcome))


def render_comment(
    *,
    candidate_ref: str,
    baseline_ref: str,
    verdicts: dict[str, Any],
    changed_models: dict[str, list[str]] | None = None,
    evaluate_changed_prompts: bool = False,
) -> str:
    """Render the verdicts as a Markdown PR comment carrying the dedup marker.

    ``changed_models`` maps a scenario to the model ops whose input changed on the
    candidate (a prompt edit). Those cannot be judged by a frozen replay, so the
    comment says so instead of claiming everything was held frozen.
    ``evaluate_changed_prompts`` reflects whether the run already evaluated those
    live, so the note does not advise turning on an option that is already on.
    """
    changed = {n: m for n, m in (changed_models or {}).items() if m}
    counts: dict[str, int] = {}
    rows: list[str] = []
    for name in sorted(verdicts):
        value = _value(verdicts[name])
        counts[value] = counts.get(value, 0) + 1
        rows.append(f"| `{name}` | {_LABELS.get(value, value)} |")

    total = len(verdicts)
    fixed = counts.get("fixed", 0)
    if total == 0:
        summary = "No scenarios ran."
    elif fixed == total:
        summary = f"**All {total} scenario(s) fixed on this branch.** :tada:"
    else:
        parts: list[str] = []
        if counts.get("still_failing"):
            parts.append(f"{counts['still_failing']} still failing")
        if counts.get("diverged"):
            parts.append(f"{counts['diverged']} diverged")
        if counts.get("cannot_reproduce"):
            parts.append(f"{counts['cannot_reproduce']} cannot reproduce")
        if counts.get("cannot_judge"):
            parts.append(f"{counts['cannot_judge']} cannot judge")
        parts.append(f"{fixed} fixed")
        summary = "**" + ", ".join(parts) + " on this branch.**"

    if changed:
        intro = (
            f"Reproduced the captured failure on `{candidate_ref}` (baseline "
            f"`{baseline_ref}`) with the recorded responses frozen - except where a model "
            f"input changed (a prompt edit), which a frozen replay cannot judge (see below)."
        )
    else:
        intro = (
            f"Reproduced the captured failure on `{candidate_ref}` (baseline "
            f"`{baseline_ref}`), with the recorded model and tool responses held frozen "
            f"so only the code differs."
        )

    lines = [
        COMMENT_MARKER,
        "### pyTracer: try on this branch",
        "",
        intro,
        "",
        "| Scenario | Outcome |",
        "| --- | --- |",
        *rows,
        "",
        summary,
    ]
    if changed:
        if evaluate_changed_prompts:
            heading = (
                "**Model inputs changed (a prompt edit) and were evaluated live. The "
                "verdict only holds when nothing after the changed call also changed; "
                "a step past it that was not in the recording still cannot be judged:**"
            )
        else:
            heading = (
                "**Model inputs changed (a prompt edit) - re-run with "
                "`--evaluate-changed-prompts` to evaluate them live:**"
            )
        lines += [
            "",
            heading,
            *(f"- `{name}`: {', '.join(changed[name])}" for name in sorted(changed)),
        ]
    if any(_value(v) == "cannot_reproduce" for v in verdicts.values()):
        lines += [
            "",
            "**Could not reproduce the baseline failure (the recordings did not match, "
            "so every call drifted). Re-run with `--env-file` so both refs reach the bug.**",
        ]
    lines += [
        "",
        "<sub>fixed = now passes &middot; still failing = same failure &middot; "
        "diverged = a new, different failure</sub>",
    ]
    return "\n".join(lines)


def post_reproduction_comment(
    client: _CommentClient,
    *,
    number: int,
    candidate_ref: str,
    baseline_ref: str,
    verdicts: dict[str, Any],
    changed_models: dict[str, list[str]] | None = None,
    evaluate_changed_prompts: bool = False,
) -> str:
    """Upsert the verdict comment on PR ``number``; return the comment URL."""
    body = render_comment(
        candidate_ref=candidate_ref, baseline_ref=baseline_ref, verdicts=verdicts,
        changed_models=changed_models, evaluate_changed_prompts=evaluate_changed_prompts,
    )
    existing = client.find_comment_by_marker(number, COMMENT_MARKER)
    if existing is None:
        return client.create_comment(number, body).url
    client.update_comment(existing.id, body)
    return existing.url

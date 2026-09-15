"""``pytracer`` command-line interface.

The GitHub Action and GitLab CI templates are thin wrappers around this, so a
run behaves identically locally and in CI.
"""

from __future__ import annotations

import json
import os
import urllib.error
from pathlib import Path
from typing import Any

import typer

from . import __version__
from .github import GitHubClient, sync_issue
from .keys import format_key_table, format_new_key
from .report import dumps_json, issue_body_from_json, issue_title, to_junit, to_markdown
from .runner import CollectionResult, run_path
from .traces import format_summary_table, format_trace_tree, parse_entry, record_trace

app = typer.Typer(add_completion=False, help="Run and manage agent tests.")
traces_app = typer.Typer(add_completion=False, help="Browse captured traces.")
app.add_typer(traces_app, name="traces")
keys_app = typer.Typer(add_completion=False, help="Manage gateway API keys.")
app.add_typer(keys_app, name="keys")
issues_app = typer.Typer(add_completion=False, help="Open or update GitHub issues for failures.")
app.add_typer(issues_app, name="issues")


def _open_trace_store() -> Any:
    """Open the trace store, or exit with guidance if the extra is not installed."""
    try:
        from pytracer_stores import TraceStore
    except ImportError as exc:
        typer.echo(
            "error: trace commands need the store client. Install with "
            "`pip install pytracer-cli[stores]`.",
            err=True,
        )
        raise typer.Exit(code=2) from exc
    return TraceStore()


def _open_key_store() -> Any:
    """Open the API-key store (migrating it), or exit if the extra is missing."""
    try:
        from pytracer_stores import ApiKeyStore
    except ImportError as exc:
        typer.echo(
            "error: key commands need the store client. Install with "
            "`pip install pytracer-cli[stores]`.",
            err=True,
        )
        raise typer.Exit(code=2) from exc
    store = ApiKeyStore()
    store.migrate()
    return store


def _emit(result: CollectionResult) -> None:
    c = result.counts
    for s in result.scenarios:
        mark = "PASS" if s.passed else "FAIL"
        typer.echo(f"  [{mark}] {s.name}")
    summary = f"{c['passed']}/{c['total']} passed"
    typer.echo(typer.style(summary, fg=typer.colors.GREEN if result.passed else typer.colors.RED))


@app.command()
def run(
    collection: Path = typer.Argument(..., help="Path to a collection YAML/JSON file."),
    scenario: str | None = typer.Option(None, help="Run only this scenario by name."),
    json_out: Path | None = typer.Option(None, "--json", help="Write JSON results to a file."),
    junit_out: Path | None = typer.Option(None, "--junit", help="Write JUnit XML to a file."),
    comment_out: Path | None = typer.Option(
        None, "--comment", help="Write a Markdown PR-comment summary to a file."
    ),
) -> None:
    """Run a collection; exit non-zero if any scenario fails."""
    try:
        result = run_path(collection)
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    if scenario is not None:
        result.scenarios = [s for s in result.scenarios if s.name == scenario]
        if not result.scenarios:
            typer.echo(f"error: no scenario named {scenario!r}", err=True)
            raise typer.Exit(code=2)

    _emit(result)

    if json_out is not None:
        json_out.write_text(dumps_json(result), encoding="utf-8")
    if junit_out is not None:
        junit_out.write_text(to_junit(result), encoding="utf-8")
    if comment_out is not None:
        comment_out.write_text(to_markdown(result), encoding="utf-8")

    raise typer.Exit(code=0 if result.passed else 1)


@traces_app.command("list")
def traces_list(
    project: str | None = typer.Option(None, help="Only this project."),
    failed: bool = typer.Option(False, "--failed", help="Only failed traces."),
    limit: int = typer.Option(20, help="Maximum traces to list."),
) -> None:
    """List recent captured traces, newest first."""
    store = _open_trace_store()
    rows = store.list_traces(project=project, failed_only=failed, limit=limit)
    if not rows:
        typer.echo("no traces found")
        return
    typer.echo(format_summary_table(rows))


@traces_app.command("show")
def traces_show(
    trace_id: str = typer.Argument(..., help="The trace id to inspect."),
    project: str | None = typer.Option(None, help="Project the trace belongs to."),
) -> None:
    """Show a single trace as a span tree."""
    store = _open_trace_store()
    spans = store.get_trace(trace_id, project=project)
    if not spans:
        typer.echo(f"error: no spans for trace {trace_id!r}", err=True)
        raise typer.Exit(code=2)
    typer.echo(format_trace_tree(spans))


@app.command()
def record(
    trace_id: str = typer.Argument(..., help="The captured trace to turn into a test."),
    agent: str = typer.Option(..., help="Agent entrypoint, 'module:callable'."),
    entry: list[str] = typer.Option([], "--entry", help="Agent kwargs as key=value."),
    project: str | None = typer.Option(None, help="Project the trace belongs to."),
    out: Path = typer.Option(Path("tests"), help="Directory to write the collection into."),
    name: str | None = typer.Option(None, help="Scenario name."),
    mode: str = typer.Option("full", help="Replay mode: full | partial | live."),
) -> None:
    """Scaffold a test collection from a captured trace."""
    store = _open_trace_store()
    spans = store.get_trace(trace_id, project=project)
    if not spans:
        typer.echo(f"error: no spans for trace {trace_id!r}", err=True)
        raise typer.Exit(code=2)
    jsonl, collection = record_trace(
        spans, agent=agent, entry=parse_entry(entry), out_dir=out, name=name, mode=mode
    )
    typer.echo(f"wrote {jsonl}")
    typer.echo(f"wrote {collection}")
    typer.echo(f"run it with: pytracer run {collection}")


@app.command()
def reproduce(
    branch: str = typer.Option(..., help="Candidate branch or commit to try the failure on."),
    collection: Path = typer.Option(
        Path("tests/collection.yaml"), help="Collection path, relative to the repo."
    ),
    repo: Path = typer.Option(Path("."), help="Path to the git repository."),
    baseline: str = typer.Option("main", help="Baseline ref to compare against."),
    isolation: str = typer.Option("docker", help="Isolation backend: docker | local."),
    freeze_trace: Path | None = typer.Option(
        None, help="A recorded trace to freeze into the run (holds the mocks fixed)."
    ),
    trace_dest: str | None = typer.Option(
        None, help="Repo-relative path to inject the frozen trace at."
    ),
    pr: int | None = typer.Option(
        None, help="Pull request number to post the verdict on ('try on this branch')."
    ),
    pr_repo: str | None = typer.Option(
        None, help="GitHub 'owner/name' for the PR comment (default: $GITHUB_REPOSITORY)."
    ),
    token: str | None = typer.Option(
        None, help="GitHub token (default: $GITHUB_TOKEN or $PYTRACER_GITHUB_TOKEN)."
    ),
    api_url: str = typer.Option(
        "https://api.github.com", help="GitHub API base URL (set for GitHub Enterprise)."
    ),
    pr_dry_run: bool = typer.Option(
        False, "--pr-dry-run", help="Print the PR comment instead of posting it."
    ),
) -> None:
    """Reproduce a captured failure on another branch (fixed / still failing / diverged)."""
    try:
        from pytracer_runner import (
            CrossBranchRunner,
            DockerExecutor,
            GitSourceProvider,
            LocalExecutor,
            classify_reproduction,
        )
    except ImportError as exc:
        typer.echo(
            "error: reproduce needs the runner. Install with `pip install pytracer-cli[runner]`.",
            err=True,
        )
        raise typer.Exit(code=2) from exc

    from .reproduce import all_fixed, format_verdicts, run_reproduction

    executor = DockerExecutor() if isolation == "docker" else LocalExecutor()
    runner = CrossBranchRunner(GitSourceProvider(repo), executor)
    try:
        _, _, verdicts = run_reproduction(
            runner,
            collection_path=str(collection),
            baseline_ref=baseline,
            candidate_ref=branch,
            classify=classify_reproduction,
            frozen_trace=str(freeze_trace) if freeze_trace else None,
            trace_dest=trace_dest,
        )
    except (RuntimeError, OSError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    typer.echo(format_verdicts(verdicts))

    if pr is not None:
        from .pr_comment import post_reproduction_comment, render_comment

        if pr_dry_run:
            typer.echo(
                render_comment(candidate_ref=branch, baseline_ref=baseline, verdicts=verdicts)
            )
        else:
            slug = pr_repo or os.getenv("GITHUB_REPOSITORY")
            gh_token = token or os.getenv("GITHUB_TOKEN") or os.getenv("PYTRACER_GITHUB_TOKEN")
            if not slug or not gh_token:
                typer.echo(
                    "error: --pr needs a repo (--pr-repo or $GITHUB_REPOSITORY) and a "
                    "token (--token or $GITHUB_TOKEN)",
                    err=True,
                )
                raise typer.Exit(code=2)
            client = GitHubClient(gh_token, slug, api_url=api_url)
            try:
                url = post_reproduction_comment(
                    client,
                    number=pr,
                    candidate_ref=branch,
                    baseline_ref=baseline,
                    verdicts=verdicts,
                )
            except (urllib.error.HTTPError, urllib.error.URLError) as exc:
                typer.echo(f"error: could not post the PR comment: {exc}", err=True)
                raise typer.Exit(code=1) from exc
            typer.echo(f"posted verdict to {url}")

    raise typer.Exit(code=0 if all_fixed(verdicts) else 1)


@keys_app.command("create")
def keys_create(
    project: str = typer.Argument(..., help="Project the key authorizes."),
    name: str = typer.Option("", help="A human label for the key (e.g. 'ci')."),
) -> None:
    """Mint an API key for a project. The secret is printed once."""
    store = _open_key_store()
    record, full_key = store.create(project, name=name)
    typer.echo(format_new_key(record, full_key))


@keys_app.command("revoke")
def keys_revoke(
    key_id: str = typer.Argument(..., help="The public key id to revoke."),
) -> None:
    """Revoke a key by its id so it stops authenticating."""
    store = _open_key_store()
    if store.revoke(key_id):
        typer.echo(f"revoked {key_id}")
    else:
        typer.echo(f"error: no active key with id {key_id!r}", err=True)
        raise typer.Exit(code=2)


@keys_app.command("list")
def keys_list(
    project: str | None = typer.Option(None, help="Only keys for this project."),
    show_revoked: bool = typer.Option(False, "--all", help="Include revoked keys."),
) -> None:
    """List API keys (never shows the secret)."""
    store = _open_key_store()
    rows = store.list(project=project, include_revoked=show_revoked)
    if not rows:
        typer.echo("no keys found")
        return
    typer.echo(format_key_table(rows))


@issues_app.command("sync")
def issues_sync(
    results: Path = typer.Argument(..., help="Results JSON from `pytracer run --json`."),
    repo: str | None = typer.Option(None, help="Target repo 'owner/name' (default: $GITHUB_REPOSITORY)."),
    token: str | None = typer.Option(
        None, help="GitHub token (default: $GITHUB_TOKEN or $PYTRACER_GITHUB_TOKEN)."
    ),
    branch: str | None = typer.Option(None, help="Branch to record (default: $GITHUB_REF_NAME)."),
    commit: str | None = typer.Option(None, help="Commit to record (default: $GITHUB_SHA)."),
    label: list[str] = typer.Option(
        ["pytracer-failure"], "--label", help="Label to add to new issues (repeatable)."
    ),
    api_url: str = typer.Option(
        "https://api.github.com", help="GitHub API base URL (set for GitHub Enterprise)."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print planned actions; call no API."),
) -> None:
    """Open or update a deduplicated GitHub issue for each failed scenario.

    Reads the JSON that `pytracer run --json` writes. A failure's fingerprint is
    embedded in the issue body, so a recurring failure updates one issue (a
    comment, reopening it if closed) instead of opening duplicates.
    """
    try:
        data = json.loads(results.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        typer.echo(f"error: cannot read results {results}: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    failures = [s for s in data.get("scenarios", []) if not s.get("passed")]
    if not failures:
        typer.echo("no failures to document")
        return

    repo = repo or os.getenv("GITHUB_REPOSITORY")
    branch = branch or os.getenv("GITHUB_REF_NAME")
    commit = commit or os.getenv("GITHUB_SHA")

    if dry_run:
        for s in failures:
            fp = s.get("fingerprint") or "none"
            typer.echo(f"would sync issue for {s['name']!r} (fingerprint {fp})")
        return

    token = token or os.getenv("GITHUB_TOKEN") or os.getenv("PYTRACER_GITHUB_TOKEN")
    if not repo or not token:
        typer.echo(
            "error: a repo (--repo or $GITHUB_REPOSITORY) and a token "
            "(--token or $GITHUB_TOKEN) are required",
            err=True,
        )
        raise typer.Exit(code=2)

    client = GitHubClient(token, repo, api_url=api_url)
    for s in failures:
        # A fingerprint dedups by failure; fall back to the scenario name so
        # distinct un-fingerprinted failures still get distinct issues.
        fingerprint = s.get("fingerprint") or f"name:{s['name']}"
        body = issue_body_from_json(s, branch=branch, commit=commit)
        try:
            outcome = sync_issue(
                client,
                fingerprint=fingerprint,
                title=issue_title(s["name"]),
                body=body,
                labels=list(label),
            )
        except (urllib.error.HTTPError, urllib.error.URLError) as exc:
            typer.echo(f"error: GitHub API call failed for {s['name']!r}: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        typer.echo(f"{outcome.action} #{outcome.number}: {outcome.url}")


@app.command()
def version() -> None:
    """Print the CLI version."""
    typer.echo(__version__)


if __name__ == "__main__":
    app()

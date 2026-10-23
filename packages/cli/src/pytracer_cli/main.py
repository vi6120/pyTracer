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
from pytracer_sdk.schema import Span

from . import __version__
from .github import GitHubClient, sync_issue
from .keys import format_key_table, format_new_key
from .report import (
    changed_model_note,
    dedup_key,
    dumps_json,
    issue_body_from_json,
    issue_title,
    scenario_reason,
    to_junit,
    to_markdown,
)
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
        if not s.passed:
            reason = scenario_reason(s)
            if reason:
                typer.echo(typer.style(f"      {reason}", fg=typer.colors.RED))
        note = changed_model_note(s)
        if note:
            typer.echo(typer.style(f"      {note}", fg=typer.colors.CYAN))
        if s.replay.concurrency_ambiguities:
            names = ", ".join(s.replay.concurrency_ambiguities)
            typer.echo(
                typer.style(
                    f"      concurrent same-input calls may be out of order: {names}",
                    fg=typer.colors.YELLOW,
                )
            )
        # Non-exact mock matches (fuzzy / positional) are substitutions, not an
        # exact replay; name them so a drifted input is visible, not just on a span.
        substitutions: dict[str, int] = {}
        for span in s.replay.spans:
            match = span.attributes.get("pytracer.mock_match")
            if isinstance(match, str) and match not in ("", "exact"):
                substitutions[match] = substitutions.get(match, 0) + 1
        if substitutions:
            kinds = ", ".join(f"{v}×{k}" for k, v in sorted(substitutions.items()))
            typer.echo(f"      non-exact matches: {kinds}")
        log = s.replay.side_effects
        if len(log):
            kinds = ", ".join(f"{v}×{k}" for k, v in sorted(log.summary().items()))
            typer.echo(f"      live side effects: {kinds}")
            repeated = log.non_idempotent()
            if repeated:
                targets = ", ".join(sorted({e.target for e in repeated}))
                typer.echo(
                    typer.style(
                        f"      non-idempotent write replayed: {targets}",
                        fg=typer.colors.YELLOW,
                    )
                )
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
    strict: bool | None = typer.Option(
        None,
        "--strict/--no-strict",
        help="Fail a scenario on an unrecorded call instead of running it live "
        "(default: on in full mode).",
    ),
    sandbox: bool | None = typer.Option(
        None,
        "--sandbox/--no-sandbox",
        help="Block outbound network during replay (default: on in full mode).",
    ),
    evaluate_changed_prompts: bool | None = typer.Option(
        None,
        "--evaluate-changed-prompts/--no-evaluate-changed-prompts",
        help="Run a frozen model op live when its input changed (a prompt edit), so "
        "the change is evaluated instead of reported as diverged. Makes a live call.",
    ),
) -> None:
    """Run a collection; exit non-zero if any scenario fails."""
    try:
        result = run_path(
            collection, strict=strict, sandbox=sandbox,
            evaluate_changed_prompts=evaluate_changed_prompts,
        )
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
    trace_id: str | None = typer.Argument(
        None, help="The captured trace to turn into a test (omit when using --from-file)."
    ),
    agent: str = typer.Option(..., help="Agent entrypoint, 'module:callable'."),
    entry: list[str] = typer.Option([], "--entry", help="Agent kwargs as key=value."),
    project: str | None = typer.Option(None, help="Project the trace belongs to."),
    out: Path = typer.Option(Path("tests"), help="Directory to write the collection into."),
    name: str | None = typer.Option(None, help="Scenario name."),
    mode: str = typer.Option("full", help="Replay mode: full | partial | live."),
    from_file: Path | None = typer.Option(
        None, "--from-file", help="Read the trace from a local JSONL file instead of the store."
    ),
) -> None:
    """Scaffold a test collection from a captured trace (trace store or local file)."""
    if from_file is not None:
        if not from_file.exists():
            typer.echo(f"error: trace file not found: {from_file}", err=True)
            raise typer.Exit(code=2)
        spans = [
            Span.from_json(line)
            for line in from_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        source = str(from_file)
    else:
        if not trace_id:
            typer.echo(
                "error: give a trace id, or --from-file to read a locally captured trace", err=True
            )
            raise typer.Exit(code=2)
        store = _open_trace_store()
        spans = store.get_trace(trace_id, project=project)
        source = repr(trace_id)
    if not spans:
        typer.echo(f"error: no spans for trace {source}", err=True)
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
    env_file: Path | None = typer.Option(
        None,
        "--env-file",
        help="A settings file (e.g. .env) copied into each ref's clean checkout, so "
        "both reach the bug instead of failing on missing configuration.",
    ),
    env_dest: str = typer.Option(
        ".env", help="Repo-relative path to place --env-file at in each checkout."
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
        _, candidate, verdicts = run_reproduction(
            runner,
            collection_path=str(collection),
            baseline_ref=baseline,
            candidate_ref=branch,
            classify=classify_reproduction,
            frozen_trace=str(freeze_trace) if freeze_trace else None,
            trace_dest=trace_dest,
            env_file=str(env_file) if env_file else None,
            env_dest=env_dest,
        )
    except (RuntimeError, OSError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    typer.echo(format_verdicts(verdicts))

    if pr is not None:
        from .pr_comment import post_reproduction_comment, render_comment

        # Model ops whose input changed on the candidate (a prompt edit): those
        # verdicts are not a frozen-replay result, and the comment says so.
        changed_models = {
            s["name"]: s.get("changed_models", [])
            for s in (getattr(candidate, "results", None) or {}).get("scenarios", [])
        }
        if pr_dry_run:
            typer.echo(
                render_comment(
                    candidate_ref=branch, baseline_ref=baseline, verdicts=verdicts,
                    changed_models=changed_models,
                )
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
                    changed_models=changed_models,
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
            typer.echo(
                f"would sync issue for {s['name']!r} (key {dedup_key(s['name'], s.get('fingerprint'))})"
            )
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
        # Search by the same key the body marker embeds (see report.dedup_key),
        # so a recurring un-fingerprinted failure updates one issue, not a new one.
        fingerprint = dedup_key(s["name"], s.get("fingerprint"))
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

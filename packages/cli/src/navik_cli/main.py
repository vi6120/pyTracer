"""``navik`` command-line interface.

The GitHub Action and GitLab CI templates are thin wrappers around this, so a
run behaves identically locally and in CI.
"""

from __future__ import annotations

from pathlib import Path

import typer

from . import __version__
from .report import dumps_json, to_junit, to_markdown
from .runner import CollectionResult, run_path

app = typer.Typer(add_completion=False, help="Run agent test collections.")


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


@app.command()
def version() -> None:
    """Print the CLI version."""
    typer.echo(__version__)


if __name__ == "__main__":
    app()

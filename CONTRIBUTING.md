# Contributing to pyTracer

Thanks for your interest in improving pyTracer. This guide covers how to get set
up, the standards a change needs to meet, and how to open a pull request.

By participating you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## Before you start

- **Sign the CLA.** The first time you open a pull request, a bot will ask you to
  sign the [Contributor License Agreement](CLA.md) by leaving a one-line comment.
  You only sign once. We cannot merge contributions without it.
- **Know the license split.** Most of the project is Apache-2.0, but the
  `pytracer-gateway` and `pytracer-registry` packages are AGPL-3.0. A change to
  those packages is contributed under AGPL-3.0. See [LICENSING.md](LICENSING.md).

## Ways to contribute

- Report a bug or request a feature via the issue templates.
- Improve documentation.
- Fix a bug or implement a feature. For anything substantial, open an issue first
  so we can agree on the approach before you invest time.

## Development setup

You need Python 3.12 and, for the store-backed tests, Docker.

```bash
python3 -m venv .venv && source .venv/bin/activate

# install every package in editable mode with dev tooling
for p in sdk stores gateway replay runner registry cli pytracer; do
  pip install -e "packages/$p[dev]"
done

# start the backing services (ClickHouse, Postgres, Redis)
docker compose up -d
```

## Standards a change must meet

Run these before opening a pull request; CI runs them too.

```bash
pytest packages            # tests must pass
ruff check packages        # lint must be clean
mypy packages/*/src        # types must be clean (strict)
```

- **Tests.** Add tests for new behavior and bug fixes. Store- and Redis-backed
  tests skip automatically when those services are not running, so the suite runs
  anywhere; if you touch that code, run it with the services up.
- **Types and lint.** `ruff` and `mypy --strict` must pass with no new errors.
- **ASCII source.** Keep source files ASCII (no smart quotes, em dashes, or
  emoji in code), which keeps diffs and terminals clean.
- **Comments.** Explain the why, not the what, in the style of the surrounding
  code.

## Opening a pull request

1. Fork the repo and create a branch from `main`.
2. Make your change with tests, and keep commits focused with clear, one-line
   messages.
3. Make sure the three checks above pass locally.
4. Open the pull request and fill in the template.
5. Sign the CLA when the bot asks (first PR only).

A maintainer will review it. Thanks for contributing.

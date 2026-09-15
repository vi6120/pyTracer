#!/usr/bin/env bash
#
# pyTracer cross-branch reproduction demo.
#
# Builds a throwaway git repo with a buggy "main" branch and a "fix" branch,
# then reproduces a captured failure on the fix branch with the recorded mocks
# frozen, so the only thing that changes between branches is the agent's code.
#
# Requires pytracer with the runner extra on PATH:
#   python3 -m venv .venv && source .venv/bin/activate
#   pip install "pytracer-cli[runner]"
#   ./run_demo.sh
#
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

# Isolation backend: "local" (no Docker, runs in the current venv) is the
# default so the demo works anywhere; set ISOLATION=docker for full container
# isolation (each ref in its own python:3.12-slim container).
ISOLATION="${ISOLATION:-local}"

command -v pytracer >/dev/null 2>&1 || {
  echo "pytracer not found on PATH. Install it first:  pip install 'pytracer-cli[runner]'" >&2
  exit 1
}

REPO="$(mktemp -d -t pytracer-repro-demo-XXXXXX)"
trap 'rm -rf "$REPO"' EXIT

echo "==> Building a throwaway agent repo at $REPO"
cp "$HERE/pyproject.toml" "$HERE/collection.yaml" "$REPO/"
mkdir -p "$REPO/demo_agent"

cd "$REPO"
git init -q
git config user.email demo@pytracer.local
git config user.name "pyTracer demo"

# Baseline branch "main": the buggy agent that undercounts its sources.
cp "$HERE/variants/analyst_buggy.py" demo_agent/__init__.py
git add -A
git commit -q -m "analyst agent (buggy: undercounts sources)"

# Candidate branch "fix": the corrected agent.
git checkout -q -b fix
cp "$HERE/variants/analyst_fixed.py" demo_agent/__init__.py
git commit -q -am "fix: count every source in the headline"
git checkout -q main

echo
echo "==> The only difference between main and fix:"
git --no-pager diff --unified=0 main fix -- demo_agent/__init__.py \
  | grep -E '^[+-]' | grep -vE '^[+-][+-]' || true

echo
echo "==> Reproducing the captured failure on the fix branch (mocks frozen)..."
echo
if pytracer reproduce \
    --repo "$REPO" \
    --baseline main \
    --branch fix \
    --collection collection.yaml \
    --isolation "$ISOLATION" \
    --freeze-trace "$HERE/traces/run.jsonl" \
    --trace-dest traces/run.jsonl; then
  echo
  echo "==> Every scenario is FIXED on the fix branch. With the model and tool"
  echo "    responses held frozen, the code change is what fixed it, not luck."
else
  echo
  echo "==> Not every scenario was fixed (see the verdicts above)."
fi

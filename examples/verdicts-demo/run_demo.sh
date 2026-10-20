#!/usr/bin/env bash
#
# pyTracer three-verdict cross-branch reproduction demo.
#
# Builds a throwaway git repo whose "main" branch holds a buggy agent and three
# candidate branches (fix, still, diverged), then reproduces the same captured
# failure on each with the recorded mocks frozen, so only the agent code varies.
# The point: pyTracer returns fixed / still-failing / diverged, not just
# pass/fail, so a team can tell a real fix from a no-op from a branch that broke
# things a new way.
#
# Requires pytracer with the runner extra on PATH:
#   python3 -m venv .venv && source .venv/bin/activate
#   pip install "pytracer-cli[runner]"
#   ./run_demo.sh
#
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

# Isolation backend: "local" (no Docker, runs in the current venv) is the default
# so the demo works anywhere; set ISOLATION=docker for full container isolation.
ISOLATION="${ISOLATION:-local}"

command -v pytracer >/dev/null 2>&1 || {
  echo "pytracer not found on PATH. Install it first:  pip install 'pytracer-cli[runner]'" >&2
  exit 1
}

# Regenerate the frozen trace if it is missing (needs pytracer-sdk importable).
[ -f "$HERE/traces/run.jsonl" ] || python "$HERE/capture.py"

REPO="$(mktemp -d -t pytracer-verdicts-demo-XXXXXX)"
trap 'rm -rf "$REPO"' EXIT

echo "==> Building a throwaway agent repo at $REPO"
cp "$HERE/pyproject.toml" "$HERE/collection.yaml" "$REPO/"
mkdir -p "$REPO/demo_agent"

cd "$REPO"
git init -q
git config user.email demo@pytracer.local
git config user.name "pyTracer demo"

# Baseline branch "main": the buggy agent that raises IndexError.
cp "$HERE/variants/agent_buggy.py" demo_agent/__init__.py
git add -A
git commit -q -m "analyst agent (buggy: indexes a missing source)"

make_branch () {  # name variant_file commit_message
  git checkout -q main
  git checkout -q -b "$1"
  cp "$HERE/variants/$2" demo_agent/__init__.py
  git commit -q -am "$3"
}
make_branch fix      agent_fixed.py        "fix: count the sources instead of indexing a missing one"
make_branch still    agent_stillfailing.py "reword the headline (bug untouched)"
make_branch diverged agent_diverged.py     "rework the headline (introduces a different bug)"
git checkout -q main

printf '\n%-10s %-14s %s\n' BRANCH VERDICT MEANING
printf '%-10s %-14s %s\n' "--------" "------------" "-------"
for b in fix still diverged; do
  verdict="$(pytracer reproduce \
      --repo "$REPO" --baseline main --branch "$b" \
      --collection collection.yaml --isolation "$ISOLATION" \
      --freeze-trace "$HERE/traces/run.jsonl" --trace-dest traces/run.jsonl 2>/dev/null \
    | awk '$1=="analyst-answers"{print $2}')"
  case "$b" in
    fix)      meaning="the branch actually fixed it" ;;
    still)    meaning="same failure: the change was a no-op" ;;
    diverged) meaning="fails a new, different way" ;;
  esac
  printf '%-10s %-14s %s\n' "$b" "${verdict:-<none>}" "$meaning"
done

echo
echo "==> A plain pass/fail tool reports 'still' and 'diverged' identically (both fail)."
echo "    pyTracer distinguishes them, which is the signal a team acts on."

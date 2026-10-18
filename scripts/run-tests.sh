#!/usr/bin/env bash
# Run the full pyTracer test suite in an isolated virtualenv.
#
# Creates ./.venv-test, installs every package editable (so the cross-package
# "==0.1.0" pins resolve to the local checkout) plus the test tools, and runs
# each package's suite. Your base/global environment is left untouched.
#
# Usage:
#   scripts/run-tests.sh            # create the venv if needed, then test
#   PYTHON=python3.12 scripts/run-tests.sh
#   scripts/run-tests.sh --recreate # rebuild the venv from scratch

set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python3}"
VENV=".venv-test"

if [ "${1:-}" = "--recreate" ]; then
    rm -rf "$VENV"
fi
if [ ! -d "$VENV" ]; then
    echo "Creating virtualenv in $VENV ..."
    "$PY" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

python -m pip install --quiet --upgrade pip

echo "Installing all packages (editable) + test tools ..."
editable=()
for d in packages/*/; do
    editable+=("-e" "$d")
done
# One install so the local editable packages satisfy the "==0.1.0" pins between them.
python -m pip install --quiet "${editable[@]}"
python -m pip install --quiet "pytest>=8" "pytest-asyncio>=0.23" "hypothesis>=6.100"

echo "Running the full suite ..."
failed=0
for d in packages/*/; do
    [ -d "${d}tests" ] || continue
    name=$(basename "$d")
    printf '\n### %s ###\n' "$name"
    ( cd "$d" && python -m pytest -q ) || failed=1
done

if [ "$failed" -eq 0 ]; then
    echo ""
    echo "All package suites passed."
else
    echo ""
    echo "Some package suites failed." >&2
fi
exit "$failed"

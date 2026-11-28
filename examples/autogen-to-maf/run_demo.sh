#!/usr/bin/env bash
#
# Port an agent from AutoGen to Microsoft Agent Framework and keep its regression
# test. Builds a throwaway git repo:
#
#   main        the agent on AutoGen, with a bug in its receipt (drops the cents)
#   port-maf    the same agent ported to Agent Framework; the bug comes along
#   fix-on-maf  the bug fixed on the ported agent
#
# and shows what pyTracer can and cannot judge across the port. Runs with no API
# key: both agents use a scripted model client.
#
# Needs pyTracer with both adapters and both frameworks:
#   pip install "pytracer-cli[runner]" pytracer-autogen pytracer-agent-framework \
#     "autogen-agentchat==0.7.5" "autogen-ext==0.7.5" "agent-framework-core>=1.26.0"
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-python}"

REPO="$(mktemp -d -t pytracer-autogen-to-maf-XXXXXX)"
trap 'rm -rf "$REPO"' EXIT
cd "$REPO"
git init -q -b main
git config user.email demo@pytracer.local
git config user.name "pyTracer demo"
mkdir -p support_agent
cp "$HERE/collection.yaml" .

commit () { git add -A && git commit -q -m "$1"; }

cp "$HERE/variants/agent_autogen.py" support_agent/__init__.py
cp "$HERE/variants/refunds_buggy.py" support_agent/refunds.py
commit "support agent on AutoGen"

git checkout -q -b port-maf
cp "$HERE/variants/agent_maf.py" support_agent/__init__.py
commit "port the support agent to Microsoft Agent Framework"

git checkout -q -b fix-on-maf
cp "$HERE/variants/refunds_fixed.py" support_agent/refunds.py
commit "fix: keep the cents in the receipt"
git checkout -q main

reproduce () {  # baseline, branch, recording
  pytracer reproduce --repo "$REPO" --baseline "$1" --branch "$2" \
    --collection collection.yaml --isolation local --no-install \
    --freeze-trace "$3" --trace-dest traces/run.jsonl || true
}

echo "==> 1. Record the failing run on AutoGen (main)"
"$PYTHON" "$HERE/capture.py" "$REPO/.recordings/autogen.jsonl"

echo
echo "==> 2. Replay the AutoGen recording on the Agent Framework port"
reproduce main port-maf "$REPO/.recordings/autogen.jsonl"

echo
echo "==> 3. Record the port once, and run the unchanged test on it"
git checkout -q port-maf
"$PYTHON" "$HERE/capture.py" "$REPO/.recordings/maf.jsonl"
mkdir -p traces && cp .recordings/maf.jsonl traces/run.jsonl
pytracer run collection.yaml || true   # the bug came along with the port: same failed check
rm -rf traces
git checkout -q main

echo
echo "==> 4. Fix the bug on the port, judged against the port's own recording"
reproduce port-maf fix-on-maf "$REPO/.recordings/maf.jsonl"

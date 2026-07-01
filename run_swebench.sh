#!/usr/bin/env bash
#
# Run the Scout-vs-frontier comparison on real SWE-bench Verified instances.
# This is the large-context test that the Cloud/Web sandbox could not run
# (HuggingFace + arbitrary GitHub clones are blocked there). Locally it works.
#
#   git clone <this repo> && cd experiments
#   export ANTHROPIC_API_KEY=sk-ant-...
#   ./run_swebench.sh                 # defaults: 6 instances, $15 budget
#   ./run_swebench.sh 8 20            # 8 instances, $20 budget
#
# Requires: python3, git, and Docker-free (per-instance venvs are built for you).
set -euo pipefail

LIMIT="${1:-6}"
BUDGET="${2:-15}"

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  echo "ERROR: set ANTHROPIC_API_KEY first  (export ANTHROPIC_API_KEY=sk-ant-...)" >&2
  exit 1
fi

cd "$(dirname "$0")"

echo "==> Setting up harness virtualenv (.venv)"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --quiet -U pip
pip install --quiet -r requirements.txt pytest

echo "==> Running SWE-bench slice: ${LIMIT} instances, \$${BUDGET} budget cap"
echo "    (first run clones repos + builds a venv per instance; that part is slow)"
python -m eval.run_eval --source swebench --limit "$LIMIT" --budget "$BUDGET"

echo "==> Report"
python -m eval.report

echo
echo "Done. See results/summary.csv and results/pareto.png"
echo "Headline to read: does 'scout' have a lower main\$ per task than 'frontier_only'"
echo "at a comparable resolve rate? That is the Fusion 'match quality, cut cost' claim."

#!/usr/bin/env bash
#
# THE FALSIFYING RUN — frontier_only vs scout on real SWE-bench Verified tasks.
#
# Decides whether the Scout architecture is worth pursuing:
#   * scout main$ meaningfully LOWER than frontier_only at ~equal resolve  -> thesis lives
#   * otherwise                                                            -> stop, rethink
#
# Resolution is scored the SWE-bench way (this is NOT a bare `pytest -q`):
# each instance's gold test patch is applied and its FAIL_TO_PASS tests must
# pass, with sampled PASS_TO_PASS staying green. Instances whose env/tests
# don't validate are skipped BEFORE any LLM spend, so `resolved` means something.
#
# Usage (from a machine with open network — not the Claude Code web sandbox):
#   export ANTHROPIC_API_KEY=sk-ant-...
#   ./run_swebench.sh                 # 15 instances, $25 budget cap
#   ./run_swebench.sh 10 15           # 10 instances, $15 cap
#
# Notes:
#   * python3.11 or 3.10 recommended on PATH (old 2019-23 repos break on 3.12+).
#   * First run is slow: clones repos + builds a venv per instance (cached in
#     ~/.cache/fusion_swebench for reruns).
set -euo pipefail

LIMIT="${1:-15}"
BUDGET="${2:-25}"

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

echo "==> Falsifying run: ${LIMIT} instances, \$${BUDGET} cap, variants: frontier_only scout"
echo "    (env build + validation gate happens first; skipped instances are printed with reasons)"
python -m eval.run_eval --source swebench --limit "$LIMIT" --budget "$BUDGET" \
  --per-task 1.5 --max-steps 20 --variants frontier_only scout

echo "==> Report"
python -m eval.report

cat <<'EOF'

================================ DECISION RULE ================================
Read results/summary.csv (or the table above). Compare, per task:

    scout.main$   vs   frontier_only.main$      at comparable resolve rate

* scout main$ meaningfully lower AND resolve within ~1 task of frontier
    -> the Scout thesis holds on real repos; build Idea C (routing) next.
* scout main$ not lower, or resolve clearly worse
    -> the architecture doesn't pay for itself; STOP and rethink before
       building anything else.
Artifacts: results/summary.csv, results/summary.json, results/pareto.png
===============================================================================
EOF

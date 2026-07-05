#!/usr/bin/env bash
#
# CHEAP CONFIRMATION — "did we truly fix the scoring, and do I need OpenCode?"
#
# Runs in two phases on a SMALL slice (default 5 instances):
#
#   Phase 1  ($0 LLM):  gold-patch self-test via the OFFICIAL SWE-bench Docker
#            harness. Scores each instance's known-correct human fix in the
#            pinned image; every one MUST resolve. If any fails, scoring is still
#            broken and no agent — yours or OpenCode's — could ever score here.
#            The run STOPS if phase 1 fails, so you never spend on a bad scorer.
#
#   Phase 2  (CHEAP, hard $ cap): a real frontier_only + scout run on the same
#            slice, scored authoritatively in Docker. Now that scoring is trusted,
#            an "unresolved" here is genuinely the agent — exactly the signal you
#            need to decide whether the harness is the problem.
#
# Requires Docker Desktop running. First run pulls per-instance images from
# Docker Hub (slow, several GB, then cached).
#
# Usage (machine with open network + Docker — not the web sandbox):
#   cp .env.example .env   # put your key in .env
#   ./confirm_scoring.sh                 # 5 instances, $6 cap on phase 2
#   ./confirm_scoring.sh 5 6             # same, explicit
#   ./confirm_scoring.sh 3 4             # 3 instances, $4 cap
#   ./confirm_scoring.sh 5 0             # phase 1 only (Docker gold check), no agent run
#
set -euo pipefail
cd "$(dirname "$0")"

if [ -f .env ]; then
  echo "==> Loading .env"
  set -a; . ./.env; set +a
fi

LIMIT="${1:-${LIMIT:-5}}"
BUDGET="${2:-${BUDGET:-6}}"

echo "==> Setting up harness virtualenv (.venv)"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --quiet -U pip
pip install --quiet -r requirements.txt pytest

VERIFIED_FILE="results/verified_instances.txt"

echo ""
echo "############################################################"
echo "# PHASE 1 — Docker gold self-test (no LLM calls)           #"
echo "############################################################"
# Phase 1 needs no API key: it runs the official harness on the GOLD patches and
# writes the instances that scored cleanly to $VERIFIED_FILE. Phase 2 runs the
# agent on exactly that verified set (an instance whose own gold patch can't
# score is excluded, so the agent is never judged unfairly).
if python -m eval.selftest_scoring --backend docker --limit "$LIMIT" --out "$VERIFIED_FILE"; then
  echo "==> Phase 1 PASSED: scoring pipeline is trustworthy."
else
  echo ""
  echo "==> Phase 1 FAILED: the scorer still can't resolve known-correct fixes."
  echo "    Not spending money on an agent run against a broken scorer. Stopping."
  exit 1
fi

# Allow a free-only invocation: second arg 0 means "phase 1 only".
if [ "${BUDGET}" = "0" ]; then
  echo ""
  echo "==> BUDGET=0: skipping the agent run (phase 1 only). Done."
  exit 0
fi

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  echo "ERROR: phase 2 needs ANTHROPIC_API_KEY (put it in .env). Skipping agent run." >&2
  exit 1
fi

echo ""
echo "############################################################"
echo "# PHASE 2 — cheap real agent run (hard \$${BUDGET} cap)        #"
echo "############################################################"
# Pinned to the gold-verified set from phase 1, scored in Docker. Because those
# instances' own gold patches resolve, every 'unresolved' below is the agent's
# doing — read the 'why:' line + per-run log.
python -m eval.run_eval --source swebench --backend docker \
  --instance-ids-file "$VERIFIED_FILE" \
  --budget "$BUDGET" --per-task 2.5 --max-steps 30 --variants frontier_only scout

echo ""
echo "==> Report"
python -m eval.report || true

cat <<'EOF'

================================ HOW TO READ THIS =============================
Phase 1 green => scoring is correct; unresolved runs in phase 2 are the AGENT,
not the harness plumbing. For each phase-2 run, the 'why:' line attributes the
outcome (no diff / edited tests / wrong fix / budget cap / env error), and the
per-run .log file has the full trace, diff, and actual pytest output.

Decide from phase 2:
* frontier_only resolves a fair share, scout close behind at lower main$
    -> harness is fine; the Scout thesis is worth the mature-harness A/B next.
* frontier_only itself resolves ~nothing despite real, non-test diffs that look
    plausible -> THEN a stronger harness (OpenCode / Claude Agent SDK) is the
    lever, and this is your evidence for it.
==============================================================================
EOF

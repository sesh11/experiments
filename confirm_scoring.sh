#!/usr/bin/env bash
#
# CHEAP CONFIRMATION — "did we truly fix the scoring, and do I need OpenCode?"
#
# Runs in two phases on a SMALL slice (default 5 instances):
#
#   Phase 1  (FREE, $0 LLM):  gold-patch self-test.
#            Applies each instance's known-correct human fix and checks the
#            scorer marks it RESOLVED. If this fails, the pipeline is still
#            broken and no agent — yours or OpenCode's — could ever score here.
#            The run STOPS if phase 1 fails, so you never spend on a bad scorer.
#
#   Phase 2  (CHEAP, hard $ cap): a real frontier_only + scout run on the same
#            slice with a tight per-task cap. Now that scoring is trusted, an
#            "unresolved" here is genuinely the agent — exactly the signal you
#            need to decide whether the harness is the problem.
#
# Usage (machine with open network — not the web sandbox):
#   cp .env.example .env   # put your key in .env
#   ./confirm_scoring.sh                 # 5 instances, $6 cap on phase 2
#   ./confirm_scoring.sh 5 6             # same, explicit
#   ./confirm_scoring.sh 3 4             # 3 instances, $4 cap
#   ./confirm_scoring.sh 5 0             # phase 1 only (free), skip agent run
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

echo ""
echo "############################################################"
echo "# PHASE 1 — FREE scoring self-test (no LLM calls)          #"
echo "############################################################"
# Phase 1 needs no API key: it only builds envs and runs pytest with gold patches.
if python -m eval.selftest_scoring --limit "$LIMIT"; then
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
# Small slice, tight per-task cap. Because phase 1 verified scoring, every
# 'unresolved' below is the agent's doing — read the 'why:' line + per-run log.
python -m eval.run_eval --source swebench --limit "$LIMIT" --budget "$BUDGET" \
  --per-task 2.5 --max-steps 30 --variants frontier_only scout

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

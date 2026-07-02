"""Scoring self-test — proves the SWE-bench pipeline is correct, with $0 of LLM spend.

For each validated instance it applies the dataset's GOLD SOLUTION patch (the
known-correct human fix) and runs the exact scorer used for agent runs. A correct
pipeline MUST mark every one of these `resolved=True`: the fix is real by
construction. Any FAIL here is a bug in the scorer/env, not the agent — and tells
you a real run's "unresolved" verdicts can't be trusted yet.

This is the check that answers "did we actually fix the scoring, or am I guessing?"
without spending money on a frontier model. Only after this is green does a real
agent run's resolve rate mean anything.

Usage (from repo root, network open):
    python -m eval.selftest_scoring --limit 5
"""

from __future__ import annotations

import argparse

from . import swebench_env


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=5, help="instances to check")
    args = ap.parse_args()

    print(f"== Scoring self-test: gold solution patch must score RESOLVED "
          f"on {args.limit} instance(s) ==\n")
    tasks = swebench_env.load(args.limit)
    if not tasks:
        print("No scoreable instances loaded — nothing to self-test.")
        return 2

    passed = 0
    failed: list[tuple[str, str]] = []
    for task in tasks:
        iid = task["instance_id"]
        resolved, detail = swebench_env.score_with_gold_patch(task)
        mark = "✅ PASS" if resolved else "❌ FAIL"
        print(f"  {mark}  {iid}: {detail}")
        if resolved:
            passed += 1
        else:
            failed.append((iid, detail))

    n = len(tasks)
    print(f"\n== {passed}/{n} gold patches scored RESOLVED ==")
    if passed == n:
        print("SCORING PIPELINE VERIFIED: the known-correct fix scores resolved on every")
        print("instance. Any 'unresolved' in a real agent run is now the AGENT's doing,")
        print("not the scorer. You can trust the harness and rule out a scoring artifact.")
        return 0
    print("SCORING STILL BROKEN on these instances — a real run cannot resolve them")
    print("no matter how good the agent is. Fix these before spending on agent runs:")
    for iid, detail in failed:
        print(f"    - {iid}: {detail}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

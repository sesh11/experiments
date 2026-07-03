"""Scoring self-test — proves the SWE-bench scoring pipeline is correct BEFORE
you spend anything on an agent run. It scores the dataset's GOLD patches (the
known-correct human fixes); a correct pipeline MUST mark every one resolved.
Any FAIL here is a scorer/env bug, not the agent — so a real run's verdicts
can't be trusted until this is green.

Two backends:
  * docker (recommended): runs the OFFICIAL SWE-bench harness with `-p gold`
    in the pinned per-instance images. This is the authoritative check and needs
    no local venv — just Docker. This is what makes 'resolved' trustworthy.
  * local: applies each gold solution patch in a local venv and runs the local
    pytest scorer. Fast, but can't reproduce pinned deps, so it will (correctly)
    skip most Verified instances as unscoreable — that's the whole reason the
    docker backend exists.

Usage (from repo root, network open):
    python -m eval.selftest_scoring --backend docker --limit 5
    python -m eval.selftest_scoring --backend local  --limit 5
"""

from __future__ import annotations

import argparse
from datetime import datetime


def _docker(limit: int) -> int:
    from . import docker_score, swebench_env
    ok, reason = docker_score.preflight()
    if not ok:
        print(f"Docker backend unavailable: {reason}")
        return 2
    ids = swebench_env.list_instance_ids(limit)
    if not ids:
        print("No instances found in the allowlist.")
        return 2
    print(f"== Docker scoring self-test: GOLD patches must resolve on "
          f"{len(ids)} instance(s) ==")
    print("   First run builds/pulls a Docker image PER INSTANCE — minutes each, then")
    print("   cached. The harness prints live progress below; it is NOT frozen. To watch")
    print("   from another terminal:  docker ps   and   docker images | grep sweb\n")
    for iid in ids:
        print(f"   - {iid}")
    run_id = f"gold_selftest_{datetime.now():%Y%m%d_%H%M%S}"
    res = docker_score.gold_selftest(ids, dataset_name=swebench_env.DATASET_NAME,
                                     split=swebench_env.SPLIT, run_id=run_id)
    resolved, unresolved = res["resolved_ids"], res["unresolved_ids"]
    print(f"\n   gold resolved: {len(resolved)}/{len(ids)}")
    for iid in resolved:
        print(f"   ✅ {iid}")
    for iid in unresolved:
        print(f"   ❌ {iid}")
    if len(resolved) == len(ids):
        print("\nSCORING PIPELINE VERIFIED (Docker): every known-correct fix scores")
        print("resolved. Any 'unresolved' in a real agent run is now the AGENT's doing,")
        print("not the scorer. You can trust the harness and rule out a scoring artifact.")
        return 0
    print("\nSCORING NOT YET TRUSTWORTHY: some gold patches did not resolve. This is a")
    print("Docker/harness/image problem, not the agent. Inspect the harness output above")
    print("(often: image pull failed, disk full, or daemon not running). Tail:\n")
    print(res.get("harness_tail", "")[-1500:])
    return 1


def _local(limit: int) -> int:
    from . import swebench_env
    print(f"== Local scoring self-test: gold solution patch must score RESOLVED "
          f"on up to {limit} instance(s) ==\n")
    tasks = swebench_env.load(limit, backend="local")
    if not tasks:
        print("No locally-scoreable instances (expected on Verified — use --backend docker).")
        return 2
    passed = 0
    failed: list[tuple[str, str]] = []
    for task in tasks:
        resolved, detail = swebench_env.score_with_gold_patch(task)
        print(f"  {'✅ PASS' if resolved else '❌ FAIL'}  {task['instance_id']}: {detail}")
        if resolved:
            passed += 1
        else:
            failed.append((task["instance_id"], detail))
    n = len(tasks)
    print(f"\n== {passed}/{n} gold patches scored RESOLVED (local) ==")
    if passed == n:
        print("Local scoring is self-consistent on the instances it could reproduce.")
        return 0
    for iid, detail in failed:
        print(f"    - {iid}: {detail}")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="docker", choices=["docker", "local"])
    ap.add_argument("--limit", type=int, default=5, help="instances to check")
    args = ap.parse_args()
    return _docker(args.limit) if args.backend == "docker" else _local(args.limit)


if __name__ == "__main__":
    raise SystemExit(main())

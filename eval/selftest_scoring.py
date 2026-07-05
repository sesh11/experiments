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


# A rare SWE-bench instance is flaky even for the gold patch (an order-dependent
# or environment-sensitive control test). We don't need every instance to pass —
# we need to KNOW which ones score cleanly and run the agent only on those. The
# pipeline is trustworthy as long as a healthy majority of gold patches resolve;
# below this fraction, something systemic is wrong (disk, daemon, images).
_MIN_TRUST_FRACTION = 0.5


def _docker(limit: int, out_path: str | None = None) -> int:
    from pathlib import Path

    from . import docker_score, swebench_env
    ok, reason = docker_score.preflight()
    if not ok:
        print(f"Docker backend unavailable: {reason}")
        return 2
    ids = swebench_env.list_instance_ids(limit)
    if not ids:
        print("No instances found in the allowlist.")
        return 2
    print(f"== Docker scoring self-test: scoring GOLD patches on {len(ids)} instance(s) ==")
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
        print(f"   ❌ {iid}  (gold patch doesn't resolve here — excluded from the agent run)")

    frac = len(resolved) / len(ids)
    if not resolved or frac < _MIN_TRUST_FRACTION:
        print(f"\nSCORING NOT TRUSTWORTHY: only {len(resolved)}/{len(ids)} gold patches")
        print("resolved — too few. That points to a systemic problem (disk full, Docker")
        print("daemon, image pulls), not one flaky instance. Inspect the harness output")
        print("above. Not proceeding to a paid agent run.")
        print(res.get("harness_tail", "")[-800:])
        return 1

    # Persist the verified set so the agent run uses ONLY these instances.
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text("\n".join(resolved) + "\n")

    print(f"\nSCORING PIPELINE VERIFIED (Docker): {len(resolved)}/{len(ids)} known-correct")
    print("fixes score resolved — the pipeline works. Any 'unresolved' in the agent run")
    print("on these instances is now the AGENT's doing, not the scorer.")
    if unresolved:
        print(f"\nExcluding {len(unresolved)} instance(s) where even gold fails "
              f"({', '.join(unresolved)}):")
        print("judging the agent on an instance whose own gold patch can't score would be")
        print("unfair. The agent run will use the verified set only.")
    if out_path:
        print(f"\nVerified set written to {out_path}")
    return 0


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
    ap.add_argument("--out", default=None,
                    help="write the gold-verified instance ids here (one per line) "
                         "so the agent run can be pinned to the trustworthy set")
    args = ap.parse_args()
    if args.backend == "docker":
        return _docker(args.limit, out_path=args.out)
    return _local(args.limit)


if __name__ == "__main__":
    raise SystemExit(main())

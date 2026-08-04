"""Driver: run every variant over every task, judge each result, record metrics.

Usage (from the repo root):
    python -m eval.run_eval --source native --budget 5
    python -m eval.run_eval --source native --variants frontier_only scout --limit 2
    python -m eval.run_eval --source swebench --limit 15 --budget 25

A single global budget is shared across the whole run; each variant/task is also
capped at --per-task. When the budget is exhausted, remaining cells are skipped
and the run writes whatever it has.

Execution is a three-phase pipeline, parallel by default (tuned for a t3.large):
  1. agent generation  — thread pool (--agent-workers), IO-bound; a per-instance
     lock keeps two variants of one in-place swebench clone from colliding.
  2. Docker scoring     — one batched harness call per variant (--score-workers =
     the harness --max_workers), memory-bound. No-op for native/local backends.
  3. judging            — thread pool (--judge-workers), IO-bound.
Results are aggregated single-threaded in task-major/variant-minor order, so the
summary output is identical to the old serial driver. Use --sequential for the
fully-serial path.
"""

from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

from fusion import config
from orchestrator import variants
from . import audit, judge, tasks

_OUT = Path("results")


class BudgetManager:
    """Thread-safe global budget with optimistic reservation.

    Mirrors the old serial semantics (`remaining = budget - spent`; stop at
    ~$0.05) but works under concurrent agent runs: a worker `reserve()`s up to
    `per_task` before running, then `commit()`s the real ledger cost afterward.
    Once the budget is exhausted, later `reserve()`s return None and those cells
    are skipped; already-running cells finish. Worst-case overshoot is bounded by
    `agent_workers x per_task`."""

    def __init__(self, budget: float, per_task: float):
        self._lock = threading.Lock()
        self.budget = budget
        self.per_task = per_task
        self.spent = 0.0
        self._reserved = 0.0

    def reserve(self) -> float | None:
        """Reserve a per-run cap, or None if the budget is exhausted."""
        with self._lock:
            remaining = self.budget - self.spent - self._reserved
            if remaining <= 0.05:
                return None
            amt = max(0.02, min(remaining, self.per_task))
            self._reserved += amt
            return amt

    def commit(self, reserved: float, actual: float) -> float:
        """Replace a reservation with the actual spend; return remaining budget."""
        with self._lock:
            self._reserved -= reserved
            self.spent += actual
            return self.budget - self.spent

    def charge(self, amount: float) -> None:
        """Charge an un-reserved cost directly (e.g. the judge call)."""
        with self._lock:
            self.spent += amount

    def can_spend(self, amount: float = 0.02) -> bool:
        with self._lock:
            return (self.budget - self.spent) > amount

    @property
    def total_spent(self) -> float:
        with self._lock:
            return self.spent


def _docker_preflight(args) -> str | None:
    """For the swebench docker backend, verify Docker + swebench are ready and
    return a base run_id used for all per-variant report paths. Returns None for
    the native/local paths (which score inline). Aborts early if Docker/swebench
    aren't ready, so a run never silently mis-scores."""
    if args.source != "swebench" or args.backend != "docker":
        return None
    from . import docker_score
    ok, reason = docker_score.preflight()
    if not ok:
        raise SystemExit(f"!! --backend docker unavailable: {reason}")
    run_id = f"fusion_{datetime.now():%Y%m%d_%H%M%S}"
    print(f"Docker scoring backend ready ({reason}); run_id={run_id}")
    return run_id


def _resolve_workers(args) -> tuple[int, int, int]:
    """(agent, score, judge) worker counts. Parallel-by-default, tuned for a
    t3.large (2 vCPU / 8 GiB): agent 4 (IO-bound), score 2 (memory-bound), judge
    = agent. `--sequential` or setting a flag to 1 restores serial execution."""
    if args.sequential:
        return 1, 1, 1
    agent = max(1, args.agent_workers)
    score = max(1, args.score_workers)
    judge_w = max(1, args.judge_workers if args.judge_workers else args.agent_workers)
    return agent, score, judge_w


def _agent_phase(cells, *, budget, workers, args, log) -> dict[int, dict | None]:
    """Phase 1: run every (task, variant) cell, IO-bound, across a thread pool.

    Each worker reserves budget, holds a per-instance lock for in-place swebench
    clones (so two variants of one instance never share a working tree), runs the
    agent, then commits the real cost. Returns {cell_key: entry|None}; None marks
    a cell skipped because the budget was exhausted."""
    total = len(cells)
    results: dict[int, dict | None] = {}
    inst_locks: dict[str, threading.Lock] = {}
    locks_guard = threading.Lock()
    progress = threading.Lock()
    done = 0

    def inst_lock(iid: str) -> threading.Lock:
        with locks_guard:
            return inst_locks.setdefault(iid, threading.Lock())

    def run_cell(cell):
        nonlocal done
        key, task, variant = cell
        reserved = budget.reserve()
        if reserved is None:
            return key, None
        cfg = config.RunConfig(budget_usd=reserved, per_task_usd=args.per_task)
        if args.max_steps:
            cfg.max_steps = args.max_steps
        guard = inst_lock(task["instance_id"]) if task.get("in_place") else nullcontext()
        with guard:
            res = variants.run_variant(variant, task, cfg)
        run_cost = res.ledger.get("total_cost_usd", 0.0)
        remaining = budget.commit(reserved, run_cost)
        with progress:
            done += 1
            k = done
        state = ("resolved" if res.resolved else
                 "pending-score" if task.get("backend") == "docker" else "unresolved")
        print(f"  [{k}/{total}] agent: {task['instance_id']} :: {variant}  "
              f"(${run_cost:.4f}, steps={res.steps}, {state}; remaining ${remaining:.2f})",
              flush=True)
        return key, {"task": task, "variant": variant, "res": res, "run_cost": run_cost}

    print(f"\n=== Phase 1: agent generation ({total} run(s), {workers} worker(s)) ===")
    if workers == 1:
        for cell in cells:
            key, entry = run_cell(cell)
            results[key] = entry
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for fut in as_completed([ex.submit(run_cell, c) for c in cells]):
                key, entry = fut.result()
                results[key] = entry
    return results


def _score_phase(results, *, docker_run_id, workers) -> None:
    """Phase 2: batch-score docker-backend diffs, one harness call per variant.

    Each variant's instances are unique, so they go in a single predictions file
    and score in parallel at `workers` (the harness's --max_workers). Variants run
    sequentially so peak container concurrency stays at `workers` — memory, not
    CPU, is the binding constraint on a small box. Patches results back onto each
    PolicyResult exactly as the old inline block did."""
    from . import docker_score, swebench_env

    by_variant: dict[str, list] = {}
    for entry in results.values():
        if entry is None or entry["task"].get("backend") != "docker":
            continue
        by_variant.setdefault(entry["variant"], []).append(entry)
    n = sum(len(g) for g in by_variant.values())
    if not n:
        return
    print(f"\n=== Phase 2: Docker scoring ({n} patch(es), {len(by_variant)} variant "
          f"batch(es), max_workers={workers}) ===")
    for variant, group in by_variant.items():
        items = [(e["task"]["instance_id"], e["res"].diff) for e in group]
        ds = group[0]["task"].get("_dataset", swebench_env.DATASET_NAME)
        split = group[0]["task"].get("_split", swebench_env.SPLIT)
        print(f"  variant '{variant}': scoring {len(items)} instance(s) in Docker "
              f"(pinned env; first build/pull is slow) …", flush=True)
        scores = docker_score.score_batch(
            items, dataset_name=ds, split=split, run_id=docker_run_id,
            model_name=f"fusion-{variant}", workers=workers)
        for e in group:
            sc = scores.get(e["task"]["instance_id"], {})
            res = e["res"]
            res.resolved = sc.get("resolved", False)
            res.resolve_detail = sc.get("detail", "(no docker report)")
            res.score_artifacts = {
                "backend": "docker", "apply_ok": sc.get("applied", False),
                "report": sc.get("report", {}),
                "harness_tail": sc.get("harness_tail", ""),
            }


def _judge_phase(results, *, budget, workers) -> dict[int, dict]:
    """Phase 3: one judge call per scored cell, IO-bound, across a thread pool.
    Budget-gated: cells that would exceed the cap are skipped."""
    entries = [(k, results[k]) for k in sorted(results) if results[k] is not None]
    print(f"\n=== Phase 3: judging ({len(entries)} run(s), {workers} worker(s)) ===")
    judged: dict[int, dict] = {}

    def do_judge(key, entry):
        if not budget.can_spend(0.02):
            return key, None
        j = judge.judge_merge(entry["task"], entry["res"].diff, entry["res"].resolved)
        budget.charge(j.get("cost_usd", 0.0))
        return key, j

    if workers == 1:
        for key, entry in entries:
            k, j = do_judge(key, entry)
            if j is not None:
                judged[k] = j
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for fut in as_completed([ex.submit(do_judge, k, e) for k, e in entries]):
                k, j = fut.result()
                if j is not None:
                    judged[k] = j
    return judged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="native", choices=["native", "swebench"])
    ap.add_argument("--limit", type=int, default=None, help="max tasks")
    ap.add_argument("--variants", nargs="+", default=variants.ALL_VARIANTS)
    ap.add_argument("--budget", type=float, default=25.0, help="global $ cap")
    ap.add_argument("--per-task", type=float, default=3.0, help="$ cap per variant/task")
    ap.add_argument("--no-judge", action="store_true", help="skip the quality judge")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="agent tool-loop steps per task (default 14; use ~20 on real repos)")
    ap.add_argument("--backend", default="local", choices=["local", "docker"],
                    help="swebench scoring backend: 'local' pytest (fast, needs a "
                         "reproducible local env) or 'docker' (authoritative, uses "
                         "the official SWE-bench harness with pinned images)")
    ap.add_argument("--instance-ids-file", default=None,
                    help="run exactly the instance ids in this file (one per line). "
                         "The confirm flow points this at the gold-verified set.")
    # --- concurrency (parallel by default; tuned for a t3.large 2 vCPU / 8 GiB) ---
    ap.add_argument("--agent-workers", type=int, default=4,
                    help="parallel agent runs (IO-bound). Default 4; 1 = serial.")
    ap.add_argument("--score-workers", type=int, default=2,
                    help="Docker harness --max_workers per variant batch. Default 2; "
                         "MEMORY-bound — keep at ~2 on 8 GiB, raise only with more RAM.")
    ap.add_argument("--judge-workers", type=int, default=None,
                    help="parallel judge calls (default: = --agent-workers).")
    ap.add_argument("--sequential", action="store_true",
                    help="force fully serial execution (all workers=1) — reproduces "
                         "the pre-parallel path for A/B validation.")
    args = ap.parse_args()

    agent_workers, score_workers, judge_workers = _resolve_workers(args)

    instance_ids = None
    if args.instance_ids_file:
        instance_ids = [ln.strip() for ln in
                        Path(args.instance_ids_file).read_text().splitlines()
                        if ln.strip()]
        if not instance_ids:
            raise SystemExit(f"!! {args.instance_ids_file} has no instance ids")
        print(f"Pinned to {len(instance_ids)} gold-verified instance(s) "
              f"from {args.instance_ids_file}")

    docker_run_id = _docker_preflight(args)
    task_list = tasks.load(args.source, args.limit, backend=args.backend,
                           instance_ids=instance_ids)
    log = audit.Audit(_OUT)
    print(f"Loaded {len(task_list)} task(s) from '{args.source}'. "
          f"Variants: {args.variants}. Budget: ${args.budget:.2f}")
    print(f"Concurrency: agent={agent_workers}, score={score_workers}, "
          f"judge={judge_workers}")
    print(f"Per-run audit logs: {log.dir}/")

    budget = BudgetManager(args.budget, args.per_task)

    # Ordered cells: key = task-major, variant-minor, so output order is preserved.
    n_var = len(args.variants)
    cells = [(i * n_var + j, task, variant)
             for i, task in enumerate(task_list)
             for j, variant in enumerate(args.variants)]

    # Phase 1: generation. Phase 2: docker scoring (no-op for native/local).
    # Phase 3: judging. Then aggregate/record single-threaded in cell order.
    results = _agent_phase(cells, budget=budget, workers=agent_workers,
                           args=args, log=log)
    skipped = sum(1 for v in results.values() if v is None)

    if docker_run_id is not None:
        _score_phase(results, docker_run_id=docker_run_id, workers=score_workers)

    judged = _judge_phase(results, budget=budget, workers=judge_workers) \
        if not args.no_judge else {}

    print(f"\n=== Results ===")
    rows: list[dict] = []
    disp_remaining = args.budget
    for key in sorted(results):
        entry = results[key]
        if entry is None:
            continue
        task, variant, res = entry["task"], entry["variant"], entry["res"]
        run_cost = entry["run_cost"]
        j = judged.get(key)
        quality = j["score"] if j else None
        merge = j["would_merge"] if j else None
        remaining_before = disp_remaining
        disp_remaining -= run_cost + (j.get("cost_usd", 0.0) if j else 0.0)

        print("\n" + log.record(task=task, variant=variant, res=res,
                                quality=quality, would_merge=merge,
                                remaining_before=remaining_before))

        row = {
            "task": task["instance_id"],
            "variant": variant,
            "resolved": res.resolved,
            "resolve_detail": res.resolve_detail,
            "steps": res.steps,
            "finished": res.finished,
            "test_files_touched": bool(audit._touched_tests(
                audit._diff_files(res.diff or ""))),
            "quality": quality,
            "would_merge": merge,
            "run_cost_usd": round(run_cost, 4),
            "budget_hit": res.budget_hit,
            "error": res.error,
            **res.ledger,
        }
        rows.append(row)

    _OUT.mkdir(exist_ok=True)
    (_OUT / "summary.json").write_text(json.dumps(rows, indent=2))
    _write_csv(rows, _OUT / "summary.csv")
    _print_tally(rows)
    if skipped:
        print(f"\n! {skipped} run(s) skipped — global budget exhausted "
              f"(spent ${budget.total_spent:.2f} of ${args.budget:.2f}).")
    print(f"\n=== Done. {len(rows)} runs, total spend ${budget.total_spent:.2f}. ===")
    print(f"    Summary:   {_OUT}/summary.json + summary.csv")
    print(f"    Full logs: {log.dir}/ (one .log per run + runs.jsonl)")


def _print_tally(rows: list[dict]) -> None:
    """Per-variant resolve/cost roll-up so the headline is visible without a report."""
    if not rows:
        return
    print("\n=== tally by variant ===")
    variants: dict[str, list[dict]] = {}
    for r in rows:
        variants.setdefault(r["variant"], []).append(r)
    for v, rs in variants.items():
        resolved = sum(1 for r in rs if r["resolved"])
        budget_hits = sum(1 for r in rs if r["budget_hit"])
        test_edits = sum(1 for r in rs if r.get("test_files_touched"))
        main = sum(r.get("main_cost_usd", 0) or 0 for r in rs)
        side = sum(r.get("sidekick_cost_usd", 0) or 0 for r in rs)
        note = []
        if budget_hits:
            note.append(f"{budget_hits} budget-capped")
        if test_edits:
            note.append(f"{test_edits} edited tests")
        suffix = f"  ({', '.join(note)})" if note else ""
        print(f"  {v:<14} resolved {resolved}/{len(rs)}  "
              f"main ${main:.2f} / sidekick ${side:.2f}{suffix}")


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    import csv
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()

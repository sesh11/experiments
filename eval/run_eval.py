"""Driver: run every variant over every task, judge each result, record metrics.

Usage (from the repo root):
    python -m eval.run_eval --source native --budget 5
    python -m eval.run_eval --source native --variants frontier_only scout --limit 2
    python -m eval.run_eval --source swebench --limit 15 --budget 25

A single global budget is shared across the whole run; each variant/task is also
capped at --per-task. When the budget is exhausted the run stops and writes
whatever it has.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fusion import config, policies
from . import judge, tasks

_OUT = Path("results")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="native", choices=["native", "swebench"])
    ap.add_argument("--limit", type=int, default=None, help="max tasks")
    ap.add_argument("--variants", nargs="+", default=policies.ALL_VARIANTS)
    ap.add_argument("--budget", type=float, default=25.0, help="global $ cap")
    ap.add_argument("--per-task", type=float, default=3.0, help="$ cap per variant/task")
    ap.add_argument("--no-judge", action="store_true", help="skip the quality judge")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="agent tool-loop steps per task (default 14; use ~20 on real repos)")
    args = ap.parse_args()

    task_list = tasks.load(args.source, args.limit)
    print(f"Loaded {len(task_list)} task(s) from '{args.source}'. "
          f"Variants: {args.variants}. Budget: ${args.budget:.2f}")

    rows: list[dict] = []
    spent = 0.0
    stop = False

    for task in task_list:
        if stop:
            break
        for variant in args.variants:
            remaining = args.budget - spent
            if remaining <= 0.05:
                print(f"! Global budget exhausted (spent ${spent:.2f}). Stopping.")
                stop = True
                break

            cfg = config.RunConfig(
                budget_usd=max(0.02, min(remaining, args.per_task)),
                per_task_usd=args.per_task,
            )
            if args.max_steps:
                cfg.max_steps = args.max_steps
            print(f"\n▶ {task['instance_id']} :: {variant} "
                  f"(remaining ${remaining:.2f})")
            res = policies.run_variant(variant, task, cfg)
            run_cost = res.ledger.get("total_cost_usd", 0.0)
            spent += run_cost

            quality = None
            merge = None
            if not args.no_judge and (args.budget - spent) > 0.02:
                j = judge.judge_merge(task, res.diff, res.resolved)
                quality, merge = j["score"], j["would_merge"]
                spent += j.get("cost_usd", 0.0)

            row = {
                "task": task["instance_id"],
                "variant": variant,
                "resolved": res.resolved,
                "resolve_detail": res.resolve_detail,
                "quality": quality,
                "would_merge": merge,
                "run_cost_usd": round(run_cost, 4),
                "budget_hit": res.budget_hit,
                "error": res.error,
                **res.ledger,
            }
            rows.append(row)
            print(f"  resolved={res.resolved} ({res.resolve_detail}) quality={quality} "
                  f"cost=${run_cost:.4f} "
                  f"main=${res.ledger.get('main_cost_usd', 0)} "
                  f"sidekick=${res.ledger.get('sidekick_cost_usd', 0)}")

    _OUT.mkdir(exist_ok=True)
    (_OUT / "summary.json").write_text(json.dumps(rows, indent=2))
    _write_csv(rows, _OUT / "summary.csv")
    print(f"\n=== Done. {len(rows)} runs, total spend ${spent:.2f}. "
          f"Wrote {_OUT}/summary.json and summary.csv ===")


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

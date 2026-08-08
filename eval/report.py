"""Aggregate results/summary.json into a cost-vs-quality scatter and a table.

    python -m eval.report

Produces results/pareto.png and prints a per-variant/configuration summary:
resolve rate, mean quality, mean total cell cost, and main/sidekick split.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

_OUT = Path("results")


def load_rows() -> list[dict]:
    p = _OUT / "summary.json"
    if not p.exists():
        raise SystemExit("No results/summary.json — run `python -m eval.run_eval` first.")
    return json.loads(p.read_text())


def aggregate(rows: list[dict]) -> dict[str, dict]:
    by = defaultdict(list)
    for r in rows:
        label = r["variant"]
        if r.get("config", "default") != "default":
            label = f"{label}/{r['config']}"
        by[label].append(r)
    agg = {}
    for variant, rs in by.items():
        n = len(rs)
        resolved = sum(1 for r in rs if r.get("resolved")) / n
        quals = [r["quality"] for r in rs if r.get("quality") is not None]
        cost = sum(r.get("cell_cost_usd", r.get("run_cost_usd", 0)) for r in rs) / n
        main = sum(r.get("main_cost_usd", 0) for r in rs)
        side = sum(r.get("sidekick_cost_usd", 0) for r in rs)
        agg[variant] = {
            "n": n,
            "resolve_rate": resolved,
            "mean_quality": (sum(quals) / len(quals)) if quals else 0.0,
            "mean_cost": cost,
            "main_cost": main,
            "sidekick_cost": side,
        }
    return agg


def print_table(agg: dict[str, dict]) -> None:
    width = max(16, *(len(label) + 2 for label in agg)) if agg else 16
    print(f"\n{'variant/config':<{width}}{'n':>3}{'resolve':>9}{'quality':>9}"
          f"{'$/task':>9}{'main$':>9}{'side$':>9}")
    print("-" * (width + 48))
    for variant, a in agg.items():
        print(f"{variant:<{width}}{a['n']:>3}{a['resolve_rate']:>9.0%}"
              f"{a['mean_quality']:>9.1f}{a['mean_cost']:>9.4f}"
              f"{a['main_cost']:>9.4f}{a['sidekick_cost']:>9.4f}")


def plot(agg: dict[str, dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # matplotlib optional
        print(f"(skipping plot: {exc})")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for variant, a in agg.items():
        ax.scatter(a["mean_cost"], a["mean_quality"],
                   s=120 + 400 * a["resolve_rate"], alpha=0.75)
        ax.annotate(f"{variant}\n(resolve {a['resolve_rate']:.0%})",
                    (a["mean_cost"], a["mean_quality"]),
                    textcoords="offset points", xytext=(8, 6), fontsize=9)
    ax.set_xlabel("mean cost per task ($)")
    ax.set_ylabel("mean quality (would-you-merge, 0-100)")
    ax.set_title("Opinionated Sidekicks: cost vs quality\n(marker size ∝ resolve rate)")
    ax.grid(True, alpha=0.3)
    _OUT.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(_OUT / "pareto.png", dpi=130)
    print(f"Wrote {_OUT}/pareto.png")


def main() -> None:
    rows = load_rows()
    agg = aggregate(rows)
    print_table(agg)
    plot(agg)


if __name__ == "__main__":
    main()

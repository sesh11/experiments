"""Resource-aware, resumable evaluation driver.

Each (instance, variant, configuration, repetition) is an independent cell.
Agent work and Docker scoring flow through separate bounded worker pools, while
the coordinator exclusively owns budgets and persistent result state.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from fusion import config
from orchestrator import variants

from . import audit, docker_score, tasks
from .parallel import run_cells
from .resources import choose_workers
from .run_state import (RunStore, atomic_write_json, atomic_write_text,
                        build_cells, new_run_id, resolve_manifest, run_lease)
from .timing import format_duration, summarize_timing

_OUT = Path(os.environ.get("EVAL_RESULTS_DIR", "results")).expanduser()
_CONFIG_FIELDS = set(asdict(config.RunConfig())) - {"budget_usd", "per_task_usd"}


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["native", "swebench"], default=None)
    ap.add_argument("--limit", type=int, default=None, help="max SWE-bench instances")
    ap.add_argument("--variants", nargs="+", default=None)
    ap.add_argument("--budget", type=float, default=None, help="global $ cap for the run")
    ap.add_argument("--per-task", type=float, default=None,
                    help="$ cap per experiment cell's agent execution")
    ap.add_argument("--no-judge", action="store_true", default=None,
                    help="skip the quality judge")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="override tool-loop steps in every configuration")
    ap.add_argument("--backend", choices=["local", "docker"], default=None,
                    help="SWE-bench scoring backend")
    ap.add_argument("--instance-ids-file", default=None,
                    help="run exactly these instance ids, one per line")
    ap.add_argument("--configs-file", default=None,
                    help="JSON configuration matrix; each entry has name/run_config")
    ap.add_argument("--repetitions", type=int, default=None,
                    help="repeat every instance/variant/configuration cell")
    ap.add_argument("--seed", type=int, default=None,
                    help="base seed recorded for repetitions (seed + repetition)")
    ap.add_argument("--workers", default="auto",
                    help="parallel agent cells: auto or a positive integer")
    ap.add_argument("--docker-workers", default="auto",
                    help="parallel Docker scorers: auto or a positive integer")
    ap.add_argument("--progress-interval", type=float, default=10.0,
                    help="seconds between elapsed-time heartbeats; 0 disables console heartbeats")
    ap.add_argument("--run-id", default=None,
                    help="optional new run id (otherwise generated)")
    ap.add_argument("--resume", default=None,
                    help="resume a run id, run directory, or manifest.json")
    return ap


def _read_instance_ids(path: str | None) -> list[str] | None:
    if not path:
        return None
    ids = [line.strip() for line in Path(path).read_text().splitlines()
           if line.strip()]
    if not ids:
        raise SystemExit(f"!! {path} has no instance ids")
    if len(ids) != len(set(ids)):
        raise SystemExit(f"!! {path} contains duplicate instance ids")
    return ids


def _base_run_config() -> dict:
    raw = asdict(config.RunConfig())
    return {key: raw[key] for key in sorted(_CONFIG_FIELDS)}


def _load_configurations(path: str | None, max_steps: int | None) -> list[dict]:
    base = _base_run_config()
    if not path:
        if max_steps is not None:
            base["max_steps"] = max_steps
        return [{"name": "default", "run_config": base}]
    raw = json.loads(Path(path).read_text())
    entries = raw.get("configurations") if isinstance(raw, dict) else raw
    if not isinstance(entries, list) or not entries:
        raise SystemExit("!! --configs-file must contain a non-empty JSON list")
    configurations = []
    names = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise SystemExit(f"!! configuration {index + 1} is not an object")
        name = str(entry.get("name") or f"config-{index + 1}")
        overrides = entry.get("run_config")
        if overrides is None:
            overrides = {k: v for k, v in entry.items() if k != "name"}
        if not isinstance(overrides, dict):
            raise SystemExit(f"!! configuration '{name}' run_config is not an object")
        unknown = set(overrides) - _CONFIG_FIELDS
        if unknown:
            raise SystemExit(
                f"!! configuration '{name}' has unknown RunConfig fields: "
                f"{', '.join(sorted(unknown))}")
        if name in names:
            raise SystemExit(f"!! duplicate configuration name: {name}")
        names.add(name)
        merged = {**base, **overrides}
        if max_steps is not None:
            merged["max_steps"] = max_steps
        # Let the dataclass validate constructor compatibility now, not after
        # paid work has started.
        config.RunConfig(**merged)
        configurations.append({"name": name, "run_config": merged})
    return configurations


def _load_exact_tasks(source: str, backend: str, task_ids: list[str]) -> list[dict]:
    loaded = tasks.load(source, len(task_ids), backend=backend,
                        instance_ids=task_ids if source == "swebench" else None)
    by_id = {task["instance_id"]: task for task in loaded}
    missing = [iid for iid in task_ids if iid not in by_id]
    if missing:
        raise SystemExit(
            f"!! could not reconstruct {len(missing)} task(s) for resume: {missing[:5]}")
    return [by_id[iid] for iid in task_ids]


def _validate_resume_args(args, experiment: dict) -> None:
    checks = {
        "source": args.source,
        "backend": args.backend,
        "variants": args.variants,
        "repetitions": args.repetitions,
        "base_seed": args.seed,
        "per_task_usd": args.per_task,
        "requested_limit": args.limit,
    }
    for key, supplied in checks.items():
        if supplied is not None and supplied != experiment[key]:
            raise SystemExit(
                f"!! --resume setting mismatch for {key}: manifest has "
                f"{experiment[key]!r}, command supplied {supplied!r}. Start a new run instead.")
    if args.no_judge is not None and args.no_judge != experiment["no_judge"]:
        raise SystemExit("!! --resume judge setting differs from the manifest")
    ids = _read_instance_ids(args.instance_ids_file)
    if ids is not None and ids != experiment["task_ids"]:
        raise SystemExit("!! --resume instance id file differs from the manifest")
    if args.configs_file or args.max_steps is not None:
        supplied = _load_configurations(args.configs_file, args.max_steps)
        if supplied != experiment["configurations"]:
            raise SystemExit("!! --resume configurations differ from the manifest")


def _new_run(args) -> tuple[RunStore, list[dict], dict, float]:
    source = args.source or "native"
    backend = args.backend or "local"
    variants_list = args.variants or list(variants.ALL_VARIANTS)
    unknown_variants = sorted(set(variants_list) - set(variants.registered_variants()))
    if unknown_variants:
        raise SystemExit(f"!! unknown variant(s): {', '.join(unknown_variants)}")
    budget = 25.0 if args.budget is None else args.budget
    per_task = 3.0 if args.per_task is None else args.per_task
    repetitions = 1 if args.repetitions is None else args.repetitions
    base_seed = 0 if args.seed is None else args.seed
    no_judge = False if args.no_judge is None else args.no_judge
    requested_limit = args.limit
    if budget <= 0:
        raise SystemExit("!! --budget must be positive")
    if per_task <= 0:
        raise SystemExit("!! --per-task must be positive")
    if repetitions < 1:
        raise SystemExit("!! --repetitions must be positive")
    ids = _read_instance_ids(args.instance_ids_file)
    if ids:
        print(f"Pinned to {len(ids)} instance(s) from {args.instance_ids_file}")
    if backend == "docker" and source != "swebench":
        raise SystemExit("!! --backend docker is only valid with --source swebench")
    if backend == "docker":
        ok, reason = docker_score.preflight()
        if not ok:
            raise SystemExit(f"!! --backend docker unavailable: {reason}")
        print(f"Docker scoring backend ready ({reason})")
    task_list = tasks.load(source, requested_limit, backend=backend, instance_ids=ids)
    configurations = _load_configurations(args.configs_file, args.max_steps)
    cells = build_cells(task_list, variants_list, configurations, repetitions, base_seed)
    if len({c.cell_id for c in cells}) != len(cells):
        raise SystemExit("!! experiment matrix produced duplicate cell identities")
    experiment = {
        "source": source,
        "backend": backend,
        "task_ids": [task["instance_id"] for task in task_list],
        "variants": variants_list,
        "configurations": configurations,
        "repetitions": repetitions,
        "base_seed": base_seed,
        "no_judge": no_judge,
        "per_task_usd": per_task,
        "requested_limit": requested_limit,
    }
    run_id = args.run_id or new_run_id()
    # Worker selection is saved as run metadata but does not define cell identity.
    worker_count, docker_count, snap = choose_workers(
        workers=args.workers, docker_workers=args.docker_workers,
        queued_cells=len(cells), docker_enabled=backend == "docker", path=_OUT.parent,
    )
    resources = {
        "initial": snap.summary(),
        "workers": worker_count,
        "docker_workers": docker_count,
        "workers_requested": args.workers,
        "docker_workers_requested": args.docker_workers,
        "resume_history": [],
    }
    try:
        store = RunStore.create(out_dir=_OUT, run_id=run_id, experiment=experiment,
                                cells=cells, budget_usd=budget, resources=resources)
    except (ValueError, FileExistsError) as exc:
        raise SystemExit(f"!! {exc}") from exc
    return store, task_list, experiment, budget


def _resume_run(args) -> tuple[RunStore, list[dict], dict, float]:
    if args.run_id:
        raise SystemExit("!! --run-id cannot be combined with --resume")
    store = RunStore.load(resolve_manifest(_OUT, args.resume))
    experiment = store.manifest["experiment"]
    _validate_resume_args(args, experiment)
    budget = float(store.manifest["budget_usd"] if args.budget is None else args.budget)
    if budget + 1e-9 < store.spent_usd:
        raise SystemExit(
            f"!! resumed budget ${budget:.2f} is below already-recorded spend "
            f"${store.spent_usd:.2f}")
    if experiment["backend"] == "docker":
        ok, reason = docker_score.preflight()
        if not ok:
            raise SystemExit(f"!! --backend docker unavailable: {reason}")
    task_list = _load_exact_tasks(
        experiment["source"], experiment["backend"], experiment["task_ids"])
    store.manifest["budget_usd"] = budget
    store.reopen_budget_skips()
    pending = sum(1 for c in store.specs() if store.status(c.cell_id) == "pending")
    worker_count, docker_count, snap = choose_workers(
        workers=args.workers, docker_workers=args.docker_workers,
        queued_cells=pending, docker_enabled=experiment["backend"] == "docker",
        path=_OUT.parent,
    )
    store.manifest["resources"]["workers"] = worker_count
    store.manifest["resources"]["docker_workers"] = docker_count
    store.manifest["resources"].setdefault("resume_history", []).append({
        "snapshot": snap.summary(), "workers": worker_count,
        "docker_workers": docker_count,
    })
    store.save()
    return store, task_list, experiment, budget


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        atomic_write_text(path, "")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buf.getvalue())


def _persist_aggregates(store: RunStore, audit_log: audit.Audit) -> None:
    payloads = store.completed_payloads()
    payloads.sort(key=lambda payload: payload["index"])
    rows = [payload["row"] for payload in payloads]
    timing = summarize_timing(
        rows,
        execution_history=store.manifest.get("resources", {}).get(
            "execution_history", []),
        created_at=store.manifest.get("created_at"),
    )
    for directory in (store.run_dir, _OUT):
        atomic_write_json(directory / "summary.json", rows)
        _write_csv(rows, directory / "summary.csv")
        atomic_write_json(directory / "timing_summary.json", timing)
    audit_log.rebuild_jsonl(payloads)


def _print_tally(rows: list[dict]) -> None:
    if not rows:
        return
    print("\n=== tally by variant/configuration ===")
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["variant"], row.get("config", "default")), []).append(row)
    for (variant, cfg_name), group in groups.items():
        resolved = sum(1 for row in group if row["resolved"])
        budget_hits = sum(1 for row in group if row["budget_hit"])
        test_edits = sum(1 for row in group if row.get("test_files_touched"))
        main = sum(row.get("main_cost_usd", 0) or 0 for row in group)
        side = sum(row.get("sidekick_cost_usd", 0) or 0 for row in group)
        elapsed = [float(row.get("cell_elapsed_seconds", 0) or 0)
                   for row in group]
        mean_elapsed = sum(elapsed) / len(elapsed) if elapsed else 0.0
        notes = []
        if budget_hits:
            notes.append(f"{budget_hits} budget-capped")
        if test_edits:
            notes.append(f"{test_edits} edited tests")
        suffix = f"  ({', '.join(notes)})" if notes else ""
        print(f"  {variant}/{cfg_name:<22} resolved {resolved}/{len(group)}  "
              f"mean time {format_duration(mean_elapsed)}  "
              f"main ${main:.2f} / sidekick ${side:.2f}{suffix}")


def _execute(args, store: RunStore, task_list: list[dict],
             experiment: dict, budget: float) -> None:
    resources = store.manifest["resources"]
    audit_log = audit.Audit(_OUT, run_id=store.manifest["run_id"])
    _persist_aggregates(store, audit_log)
    counts = store.summary_counts()
    print(
        f"Run {store.manifest['run_id']}: {len(store.manifest['cells'])} cell(s), "
        f"states={counts}, budget=${budget:.2f}, already spent=${store.spent_usd:.2f}\n"
        f"Workers: agent={resources['workers']}, scoring={resources['docker_workers']} "
        f"(requested {args.workers}/{args.docker_workers})\n"
        f"Manifest: {store.manifest_path}",
        flush=True,
    )
    summary = run_cells(
        store=store, tasks=task_list,
        workers=int(resources["workers"]),
        docker_workers=int(resources["docker_workers"]),
        budget_usd=budget, per_task_usd=float(experiment["per_task_usd"]),
        no_judge=bool(experiment["no_judge"]), audit_log=audit_log,
        on_persist=lambda: _persist_aggregates(store, audit_log),
        progress_interval=args.progress_interval,
    )
    invocation = {
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "workers": int(resources["workers"]),
        "docker_workers": int(resources["docker_workers"]),
        "completed_now": summary.completed_now,
        "interrupted": summary.interrupted,
        "fatal_error": summary.fatal_error,
        **summary.telemetry,
    }
    store.manifest["resources"].setdefault("execution_history", []).append(invocation)
    store.save()
    # Refresh timing_summary.json now that this invocation's active wall time is
    # durable in execution_history.
    _persist_aggregates(store, audit_log)
    rows = [payload["row"] for payload in store.completed_payloads()]
    _print_tally(rows)
    outcome = "Failed" if summary.fatal_error else (
        "Interrupted" if summary.interrupted else "Done")
    print(
        f"\n=== {outcome}. "
        f"{len(rows)} completed cell(s), total spend ${summary.spent_usd:.2f}. ===\n"
        f"    Resume:    python -m eval.run_eval --resume {store.manifest['run_id']}\n"
        f"    Summary:   {store.run_dir}/summary.json + summary.csv\n"
        f"    Timing:    {store.run_dir}/timing_summary.json\n"
        f"    Progress:  {store.run_dir}/progress.json\n"
        f"    Full logs: {store.run_dir}/",
        flush=True,
    )
    print(
        f"    Concurrency: agent effective {summary.telemetry['effective_agent_concurrency']:.2f} "
        f"(peak {summary.telemetry['peak_agent_cells']}); scoring effective "
        f"{summary.telemetry['effective_scoring_concurrency']:.2f} "
        f"(peak {summary.telemetry['peak_scoring_cells']})",
        flush=True,
    )
    print(
        f"    Elapsed: {format_duration(summary.telemetry['wall_seconds'])}; "
        f"throughput {summary.telemetry['completed_cells_per_hour']:.1f} cells/hour",
        flush=True,
    )
    if summary.fatal_error:
        raise RuntimeError(
            "systemic provider failure; no further cells were dispatched. "
            f"Fix the provider configuration and resume with: "
            f"python -m eval.run_eval --resume {store.manifest['run_id']}")


def main() -> None:
    args = _parser().parse_args()
    if args.progress_interval < 0:
        raise SystemExit("!! --progress-interval must be zero or positive")
    if args.resume:
        manifest_path = resolve_manifest(_OUT, args.resume)
        try:
            with run_lease(manifest_path.parent):
                _execute(args, *_resume_run(args))
        except RuntimeError as exc:
            raise SystemExit(f"!! {exc}") from exc
        return
    store, task_list, experiment, budget = _new_run(args)
    try:
        with run_lease(store.run_dir):
            _execute(args, store, task_list, experiment, budget)
    except RuntimeError as exc:
        raise SystemExit(f"!! {exc}") from exc


if __name__ == "__main__":
    main()

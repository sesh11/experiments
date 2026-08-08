#!/usr/bin/env python3
"""Run the same real evaluation serially and in parallel, then compare.

This intentionally requires an explicit invocation because both arms make paid
model calls. Use a gold-verified instance-id file on EC2 for a trustworthy
SWE-bench/Docker benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


# Direct execution (`python scripts/benchmark_parallel.py`) puts scripts/, not
# the repository root, on sys.path. Add the root before the parity phase imports
# the eval package. This also makes invocation from outside the checkout work.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_local_env() -> None:
    """Load simple KEY=VALUE entries from the repository .env when unset."""
    path = _REPO_ROOT / ".env"
    if not path.exists():
        return
    loaded = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value
        loaded.append(key)
    if loaded:
        print(f"Loaded {', '.join(loaded)} from {_REPO_ROOT / '.env'}", flush=True)


def _preflight_anthropic() -> None:
    """Validate credentials with a free Models API request before paid arms."""
    import anthropic

    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if (not key or key in {"sk-ant-...", "sk-ant-your-key-here"}
            or "your-key" in key.lower()):
        raise SystemExit(
            "Anthropic preflight failed: ANTHROPIC_API_KEY is missing or still a "
            "placeholder. Update .env or export a valid key before benchmarking.")
    try:
        # Listing one model validates authentication without generating tokens.
        client = anthropic.Anthropic(
            api_key=key, timeout=10.0, max_retries=0)
        next(iter(client.models.list(limit=1)), None)
    except anthropic.AuthenticationError:
        raise SystemExit(
            "Anthropic preflight failed: the API rejected ANTHROPIC_API_KEY (401). "
            "Update .env, then run `set -a; . ./.env; set +a` and retry.") from None
    except anthropic.APIError as exc:
        raise SystemExit(
            f"Anthropic preflight failed before any benchmark work: "
            f"{type(exc).__name__}: {exc}") from None
    print("Anthropic credential preflight passed (no token-generating call).", flush=True)


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance-ids-file", required=True)
    ap.add_argument("--variants", nargs="+", default=["frontier_only", "scout"])
    ap.add_argument("--budget", type=float, required=True,
                    help="separate dollar cap for EACH serial/parallel arm")
    ap.add_argument("--per-task", type=float, default=2.5)
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--configs-file")
    ap.add_argument("--repetitions", type=int, default=1)
    ap.add_argument("--parallel-workers", default="auto")
    ap.add_argument("--parallel-docker-workers", default="auto")
    ap.add_argument("--with-judge", action="store_true")
    ap.add_argument("--skip-scoring-parity", action="store_true",
                    help="skip no-LLM replay of serial patches through parallel Docker scoring")
    return ap


def _invoke(run_id: str, args, *, workers: str, docker_workers: str) -> tuple[float, Path]:
    cmd = [
        sys.executable, "-m", "eval.run_eval",
        "--source", "swebench", "--backend", "docker",
        "--instance-ids-file", args.instance_ids_file,
        "--variants", *args.variants,
        "--budget", str(args.budget), "--per-task", str(args.per_task),
        "--max-steps", str(args.max_steps),
        "--repetitions", str(args.repetitions),
        "--workers", workers, "--docker-workers", docker_workers,
        "--run-id", run_id,
    ]
    if args.configs_file:
        cmd += ["--configs-file", args.configs_file]
    if not args.with_judge:
        cmd.append("--no-judge")
    started = time.monotonic()
    proc = subprocess.run(cmd, cwd=_REPO_ROOT)
    elapsed = time.monotonic() - started
    if proc.returncode:
        raise SystemExit(f"benchmark arm '{run_id}' failed with exit {proc.returncode}")
    return elapsed, Path("results") / "runs" / run_id


def _rows(run_dir: Path) -> list[dict]:
    return json.loads((run_dir / "summary.json").read_text())


def _manifest(run_dir: Path) -> dict:
    return json.loads((run_dir / "manifest.json").read_text())


def _timing_summary(run_dir: Path) -> dict:
    path = run_dir / "timing_summary.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _latest_telemetry(manifest: dict) -> dict:
    history = manifest.get("resources", {}).get("execution_history", [])
    return history[-1] if history else {}


def _bottleneck(manifest: dict, telemetry: dict) -> str:
    counts: dict[str, int] = {}
    for cell in manifest.get("cells", {}).values():
        status = cell.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1
    if counts.get("skipped_budget"):
        return "global_budget"
    if telemetry.get("resource_pressure_pauses", 0):
        return "host_memory_or_disk_pressure"
    workers = int(telemetry.get("workers", 1) or 1)
    scorers = int(telemetry.get("docker_workers", 1) or 1)
    agent_saturated = telemetry.get("peak_agent_cells", 0) >= workers
    scorer_saturated = telemetry.get("peak_scoring_cells", 0) >= scorers
    agent_eff = float(telemetry.get("effective_agent_concurrency", 0) or 0)
    scorer_eff = float(telemetry.get("effective_scoring_concurrency", 0) or 0)
    if scorer_saturated and scorer_eff >= agent_eff:
        return "docker_scoring_pool"
    if agent_saturated:
        return "agent_pool_or_provider_latency"
    return "insufficient_cells_or_external_setup"


def _resource_report(manifest: dict, telemetry: dict) -> dict:
    initial = manifest.get("resources", {}).get("initial", {})
    min_mem = telemetry.get("minimum_memory_available_bytes")
    min_disk = telemetry.get("minimum_disk_available_bytes")
    gib = 1024 ** 3
    return {
        "cpus": initial.get("cpus"),
        "initial_memory_available_gib": initial.get("memory_available_gib"),
        "minimum_memory_available_gib": (
            round(min_mem / gib, 2) if min_mem is not None else None),
        "initial_disk_available_gib": initial.get("disk_available_gib"),
        "minimum_disk_available_gib": (
            round(min_disk / gib, 2) if min_disk is not None else None),
        "maximum_load_1m": telemetry.get("maximum_load_1m"),
        "resource_pressure_pauses": telemetry.get("resource_pressure_pauses"),
    }


def _key(row: dict) -> tuple:
    return (row["task"], row["variant"], row.get("config", "default"),
            row.get("repetition", 0))


def _cell_payloads(run_dir: Path) -> list[dict]:
    payloads = []
    for path in (run_dir / "cells").glob("*/result.json"):
        payloads.append(json.loads(path.read_text()))
    return sorted(payloads, key=lambda payload: payload["index"])


def _verify_scoring_parity(serial_dir: Path, *, stamp: str,
                           docker_workers: int) -> dict:
    """Replay identical serial-arm patches with bounded parallel Docker scoring."""
    from eval import docker_score, swebench_env

    payloads = _cell_payloads(serial_dir)
    out_dir = Path("results") / f"benchmark_{stamp}_scoring_parity"
    out_dir.mkdir(parents=True, exist_ok=True)
    locks_guard = threading.Lock()
    locks: dict[str, threading.Lock] = {}

    def score(payload: dict) -> dict:
        row = payload["row"]
        cell_id = row["cell_id"]
        instance_id = row["task"]
        with locks_guard:
            instance_lock = locks.setdefault(instance_id, threading.Lock())
        started = time.monotonic()
        cell_dir = out_dir / cell_id
        cell_dir.mkdir(parents=True, exist_ok=True)
        try:
            with instance_lock:
                result = docker_score.score_patch(
                    instance_id, payload.get("model_patch", ""),
                    dataset_name=swebench_env.DATASET_NAME,
                    split=swebench_env.SPLIT,
                    run_id=f"parity-{stamp}-{cell_id[-12:]}",
                    model_name=f"parity-{cell_id[-12:]}",
                    workers=1, cwd=cell_dir, stream=False,
                )
            return {
                "cell_id": cell_id,
                "task": instance_id,
                "expected_resolved": bool(row["resolved"]),
                "replay_resolved": bool(result["resolved"]),
                "replay_detail": result["detail"],
                "wall_seconds": round(time.monotonic() - started, 3),
                "error": "",
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "cell_id": cell_id, "task": instance_id,
                "expected_resolved": bool(row["resolved"]),
                "replay_resolved": None, "replay_detail": "",
                "wall_seconds": round(time.monotonic() - started, 3),
                "error": f"{type(exc).__name__}: {exc}",
            }

    started = time.monotonic()
    results = []
    with ThreadPoolExecutor(max_workers=max(1, docker_workers)) as pool:
        futures = [pool.submit(score, payload) for payload in payloads]
        for future in as_completed(futures):
            results.append(future.result())
    wall = time.monotonic() - started
    results.sort(key=lambda result: result["cell_id"])
    differences = [result for result in results
                   if result["error"] or
                   result["expected_resolved"] != result["replay_resolved"]]
    work = sum(result["wall_seconds"] for result in results)
    return {
        "cells": len(results),
        "docker_workers": docker_workers,
        "wall_seconds": round(wall, 3),
        "effective_scoring_concurrency": round(work / wall, 3) if wall else 0.0,
        "differences": differences,
        "passed": not differences and len(results) == len(payloads),
        "artifact_dir": str(out_dir),
    }


def main() -> None:
    args = _parser().parse_args()
    # Resolve user-supplied paths before anchoring all run artifacts to the repo.
    args.instance_ids_file = str(Path(args.instance_ids_file).expanduser().resolve())
    if args.configs_file:
        args.configs_file = str(Path(args.configs_file).expanduser().resolve())
    os.chdir(_REPO_ROOT)
    _load_local_env()
    _preflight_anthropic()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    serial_id = f"benchmark_{stamp}_serial"
    parallel_id = f"benchmark_{stamp}_parallel"
    print("\n=== Serial benchmark arm ===", flush=True)
    serial_s, serial_dir = _invoke(serial_id, args, workers="1", docker_workers="1")
    print("\n=== Parallel benchmark arm ===", flush=True)
    parallel_s, parallel_dir = _invoke(
        parallel_id, args, workers=args.parallel_workers,
        docker_workers=args.parallel_docker_workers)

    serial_rows = _rows(serial_dir)
    parallel_rows = _rows(parallel_dir)
    serial_manifest = _manifest(serial_dir)
    parallel_manifest = _manifest(parallel_dir)
    serial_timing = _timing_summary(serial_dir)
    parallel_timing = _timing_summary(parallel_dir)
    serial_telemetry = _latest_telemetry(serial_manifest)
    parallel_telemetry = _latest_telemetry(parallel_manifest)
    serial_by_key = {_key(row): row for row in serial_rows}
    parallel_by_key = {_key(row): row for row in parallel_rows}
    common = sorted(set(serial_by_key) & set(parallel_by_key))
    missing_from_parallel = sorted(set(serial_by_key) - set(parallel_by_key))
    missing_from_serial = sorted(set(parallel_by_key) - set(serial_by_key))
    scoring_differences = [
        {"cell": key, "serial_resolved": serial_by_key[key]["resolved"],
         "parallel_resolved": parallel_by_key[key]["resolved"]}
        for key in common
        if serial_by_key[key]["resolved"] != parallel_by_key[key]["resolved"]
    ]
    serial_scheduler_s = float(serial_telemetry.get("wall_seconds", 0) or 0)
    parallel_scheduler_s = float(parallel_telemetry.get("wall_seconds", 0) or 0)
    scoring_parity = None
    if not args.skip_scoring_parity:
        print("\n=== Parallel Docker scoring parity replay (no LLM spend) ===", flush=True)
        scoring_parity = _verify_scoring_parity(
            serial_dir, stamp=stamp,
            docker_workers=int(parallel_telemetry.get("docker_workers", 1) or 1),
        )
    report = {
        "serial_run": serial_id,
        "parallel_run": parallel_id,
        "serial_wall_seconds": round(serial_s, 2),
        "parallel_wall_seconds": round(parallel_s, 2),
        "end_to_end_speedup": round(serial_s / parallel_s, 3) if parallel_s else None,
        "scheduler_speedup": (
            round(serial_scheduler_s / parallel_scheduler_s, 3)
            if parallel_scheduler_s else None),
        "serial_cells": len(serial_rows),
        "parallel_cells": len(parallel_rows),
        "serial_duplicate_cells": len(serial_rows) - len(serial_by_key),
        "parallel_duplicate_cells": len(parallel_rows) - len(parallel_by_key),
        "common_cells": len(common),
        "missing_from_parallel": missing_from_parallel,
        "missing_from_serial": missing_from_serial,
        "serial_spend_usd": round(sum(r.get("cell_cost_usd", 0) for r in serial_rows), 4),
        "parallel_spend_usd": round(sum(r.get("cell_cost_usd", 0) for r in parallel_rows), 4),
        "serial_resolved": sum(bool(r["resolved"]) for r in serial_rows),
        "parallel_resolved": sum(bool(r["resolved"]) for r in parallel_rows),
        "scoring_differences": scoring_differences,
        "serial_scheduler_telemetry": serial_telemetry,
        "parallel_scheduler_telemetry": parallel_telemetry,
        "serial_timing_summary": serial_timing,
        "parallel_timing_summary": parallel_timing,
        "serial_resource_use": _resource_report(serial_manifest, serial_telemetry),
        "parallel_resource_use": _resource_report(parallel_manifest, parallel_telemetry),
        "parallel_bottleneck": _bottleneck(parallel_manifest, parallel_telemetry),
        "identical_patch_scoring_parity": scoring_parity,
        "serial_setup_overhead_seconds": round(
            max(0.0, serial_s - serial_scheduler_s), 2),
        "parallel_setup_overhead_seconds": round(
            max(0.0, parallel_s - parallel_scheduler_s), 2),
    }
    report_path = Path("results") / f"benchmark_{stamp}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"\nBenchmark report: {report_path}")


if __name__ == "__main__":
    main()

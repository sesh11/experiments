"""Two-stage parallel experiment scheduler.

Agent execution and Docker scoring have deliberately separate bounded pools.
Only the coordinator mutates the manifest, audit index, or aggregate outputs,
so completion order cannot corrupt persistent state.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from fusion import config
from fusion.llm import Ledger
from orchestrator import variants
from orchestrator.variants import PolicyResult

from . import audit, docker_score, judge
from .isolation import isolated_task
from .resources import dispatch_headroom, snapshot as resource_snapshot
from .run_state import CellSpec, RunStore, atomic_write_json
from .timing import format_duration, utc_now_iso


_DOCKER_LOCKS_GUARD = threading.Lock()
_DOCKER_INSTANCE_LOCKS: dict[str, threading.Lock] = {}


def _docker_instance_lock(instance_id: str) -> threading.Lock:
    # The official harness names containers by instance+run-id and may also
    # pull/remove the same instance image. Unique run ids solve container-name
    # collisions; this keyed lock prevents image lifecycle races.
    with _DOCKER_LOCKS_GUARD:
        return _DOCKER_INSTANCE_LOCKS.setdefault(instance_id, threading.Lock())


@dataclass
class AgentOutput:
    cell: CellSpec
    result: PolicyResult
    remaining_before: float
    reserved_usd: float
    judge_reserved_usd: float
    queue_wait_seconds: float
    dispatched_monotonic: float
    dispatched_at: str
    agent_finished_monotonic: float
    agent_finished_at: str
    workspace_setup_seconds: float
    runtime_wall_seconds: float
    workspace_cleanup_seconds: float
    agent_wall_seconds: float


@dataclass
class FinalOutput:
    agent: AgentOutput
    quality: int | None
    would_merge: bool | None
    judge_cost_usd: float
    scoring_started_at: str
    completed_at: str
    scoring_queue_wait_seconds: float
    docker_lock_wait_seconds: float
    docker_scoring_wall_seconds: float
    judge_wall_seconds: float
    scoring_wall_seconds: float
    completed_monotonic: float


@dataclass(frozen=True)
class ActiveAgent:
    cell: CellSpec
    dispatched_monotonic: float
    dispatched_at: str
    queue_wait_seconds: float


@dataclass
class RunSummary:
    interrupted: bool
    fatal_error: str | None
    completed_now: int
    spent_usd: float
    counts: dict[str, int]
    telemetry: dict


def _error_result(variant: str, exc: BaseException) -> PolicyResult:
    return PolicyResult(
        variant=variant, resolved=False, diff="", summary="",
        ledger=Ledger(cap_usd=0).summary(),
        error=f"{type(exc).__name__}: {exc}",
    )


def _agent_stage(cell: CellSpec, base_task: dict, workspace_dir: Path,
                 *, agent_cap_usd: float, per_task_usd: float,
                 remaining_before: float, reserved_usd: float,
                 judge_reserved_usd: float, queue_wait_seconds: float,
                 dispatched_monotonic: float, dispatched_at: str) -> AgentOutput:
    stage_started = time.monotonic()
    setup_finished: float | None = None
    runtime_started: float | None = None
    runtime_finished: float | None = None
    try:
        with isolated_task(base_task, workspace_dir, seed=cell.seed) as task:
            setup_finished = time.monotonic()
            runtime_started = time.monotonic()
            try:
                cfg = config.RunConfig(
                    **cell.config,
                    budget_usd=agent_cap_usd,
                    per_task_usd=per_task_usd,
                )
                result = variants.run_variant(cell.variant, task, cfg)
            finally:
                runtime_finished = time.monotonic()
    except BaseException as exc:  # one broken cell must not stop its siblings
        result = _error_result(cell.variant, exc)
    stage_finished = time.monotonic()
    setup_seconds = max(0.0, (setup_finished or stage_finished) - stage_started)
    runtime_seconds = (
        max(0.0, runtime_finished - runtime_started)
        if runtime_started is not None and runtime_finished is not None else 0.0)
    cleanup_anchor = runtime_finished or setup_finished or stage_finished
    cleanup_seconds = max(0.0, stage_finished - cleanup_anchor)
    return AgentOutput(
        cell=cell, result=result, remaining_before=remaining_before,
        reserved_usd=reserved_usd, judge_reserved_usd=judge_reserved_usd,
        queue_wait_seconds=queue_wait_seconds,
        dispatched_monotonic=dispatched_monotonic,
        dispatched_at=dispatched_at,
        agent_finished_monotonic=stage_finished,
        agent_finished_at=utc_now_iso(),
        workspace_setup_seconds=setup_seconds,
        runtime_wall_seconds=runtime_seconds,
        workspace_cleanup_seconds=cleanup_seconds,
        agent_wall_seconds=stage_finished - stage_started,
    )


def _score_stage(agent: AgentOutput, base_task: dict, *, run_id: str,
                 run_dir: Path, no_judge: bool) -> FinalOutput:
    started = time.monotonic()
    scoring_started_at = utc_now_iso()
    scoring_queue_wait = max(0.0, started - agent.agent_finished_monotonic)
    docker_lock_wait = 0.0
    docker_scoring_seconds = 0.0
    judge_seconds = 0.0
    res = agent.result
    if base_task.get("backend") == "docker":
        score_dir = run_dir / "cells" / agent.cell.cell_id / "scoring"
        score_dir.mkdir(parents=True, exist_ok=True)
        try:
            lock_started = time.monotonic()
            with _docker_instance_lock(base_task["instance_id"]):
                docker_lock_wait = time.monotonic() - lock_started
                docker_started = time.monotonic()
                try:
                    sc = docker_score.score_patch(
                        base_task["instance_id"], res.diff,
                        dataset_name=base_task.get(
                            "_dataset", "princeton-nlp/SWE-bench_Verified"),
                        split=base_task.get("_split", "test"),
                        # SWE-bench container names are instance+run_id (not model),
                        # so same-instance variants need a cell-unique run id.
                        run_id=f"{run_id}-{agent.cell.cell_id[-12:]}",
                        model_name=(f"fusion-{agent.cell.variant}-"
                                    f"{agent.cell.cell_id[-12:]}"),
                        workers=1,
                        cwd=score_dir,
                        # Concurrent inherited subprocess output interleaves. Capture
                        # it per cell and surface the tail through the audit log.
                        stream=False,
                    )
                finally:
                    docker_scoring_seconds = time.monotonic() - docker_started
            res.resolved = sc["resolved"]
            res.resolve_detail = sc["detail"]
            res.score_artifacts = {
                "backend": "docker", "apply_ok": sc["applied"],
                "report": sc.get("report", {}),
                "harness_tail": sc.get("harness_tail", ""),
            }
        except BaseException as exc:
            res.resolved = False
            res.resolve_detail = "Docker scoring failed"
            res.error = _join_error(res.error, f"scorer {type(exc).__name__}: {exc}")

    quality = None
    would_merge = None
    judge_cost = 0.0
    if not no_judge:
        judge_started = time.monotonic()
        try:
            verdict = judge.judge_merge(base_task, res.diff, res.resolved)
            quality = verdict["score"]
            would_merge = verdict["would_merge"]
            judge_cost = float(verdict.get("cost_usd", 0.0) or 0.0)
        except BaseException as exc:
            res.error = _join_error(res.error, f"judge {type(exc).__name__}: {exc}")
        finally:
            judge_seconds = time.monotonic() - judge_started
    completed = time.monotonic()
    return FinalOutput(agent=agent, quality=quality,
                       would_merge=would_merge, judge_cost_usd=judge_cost,
                       scoring_started_at=scoring_started_at,
                       completed_at=utc_now_iso(),
                       scoring_queue_wait_seconds=scoring_queue_wait,
                       docker_lock_wait_seconds=docker_lock_wait,
                       docker_scoring_wall_seconds=docker_scoring_seconds,
                       judge_wall_seconds=judge_seconds,
                       scoring_wall_seconds=completed - started,
                       completed_monotonic=completed)


def _join_error(first: str, second: str) -> str:
    return f"{first}; {second}" if first else second


def _systemic_provider_error(error: str) -> bool:
    """Return whether retrying other cells with the same provider must fail."""
    lowered = (error or "").lower()
    markers = (
        "authenticationerror:",
        "permissiondeniederror:",
        "ratelimiterror:",
        "apiconnectionerror:",
        "apitimeouterror:",
        "invalid x-api-key",
    )
    return any(marker in lowered for marker in markers)


def result_row(final: FinalOutput) -> dict:
    cell = final.agent.cell
    res = final.agent.result
    run_cost = float(res.ledger.get("total_cost_usd", 0.0) or 0.0)
    return {
        "cell_id": cell.cell_id,
        "cell_index": cell.index,
        "task": cell.task_id,
        "variant": cell.variant,
        "config": cell.config_name,
        "repetition": cell.repetition,
        "seed": cell.seed,
        "dispatched_at": final.agent.dispatched_at,
        "agent_finished_at": final.agent.agent_finished_at,
        "scoring_started_at": final.scoring_started_at,
        "completed_at": final.completed_at,
        "queue_wait_seconds": round(final.agent.queue_wait_seconds, 3),
        "workspace_setup_seconds": round(
            final.agent.workspace_setup_seconds, 3),
        "runtime_wall_seconds": round(final.agent.runtime_wall_seconds, 3),
        "workspace_cleanup_seconds": round(
            final.agent.workspace_cleanup_seconds, 3),
        "agent_wall_seconds": round(final.agent.agent_wall_seconds, 3),
        "scoring_queue_wait_seconds": round(
            final.scoring_queue_wait_seconds, 3),
        "docker_lock_wait_seconds": round(final.docker_lock_wait_seconds, 3),
        "docker_scoring_wall_seconds": round(
            final.docker_scoring_wall_seconds, 3),
        "judge_wall_seconds": round(final.judge_wall_seconds, 3),
        "scoring_wall_seconds": round(final.scoring_wall_seconds, 3),
        # Kept for compatibility: active agent work plus active scoring work.
        "cell_wall_seconds": round(
            final.agent.agent_wall_seconds + final.scoring_wall_seconds, 3),
        # End-to-end cell latency from dispatch through scoring completion.
        "cell_elapsed_seconds": round(
            final.completed_monotonic - final.agent.dispatched_monotonic, 3),
        "invocation_elapsed_at_completion_seconds": round(
            final.agent.queue_wait_seconds
            + final.completed_monotonic - final.agent.dispatched_monotonic, 3),
        "resolved": res.resolved,
        "resolve_detail": res.resolve_detail,
        "steps": res.steps,
        "finished": res.finished,
        "test_files_touched": bool(audit._touched_tests(
            audit._diff_files(res.diff or ""))),
        "quality": final.quality,
        "would_merge": final.would_merge,
        "run_cost_usd": round(run_cost, 4),
        "judge_cost_usd": round(final.judge_cost_usd, 4),
        "cell_cost_usd": round(run_cost + final.judge_cost_usd, 4),
        "budget_hit": res.budget_hit,
        "error": res.error,
        **res.ledger,
    }


def _actual_cost(final: FinalOutput) -> float:
    return (float(final.agent.result.ledger.get("total_cost_usd", 0.0) or 0.0)
            + final.judge_cost_usd)


class _Reservations:
    def __init__(self, total: float, spent: float) -> None:
        self.total = total
        self.spent = spent
        self.active: dict[str, float] = {}

    @property
    def available(self) -> float:
        return max(0.0, self.total - self.spent - sum(self.active.values()))

    def reserve(self, cell_id: str, amount: float) -> None:
        if amount > self.available + 1e-9:
            raise RuntimeError("internal budget reservation exceeded available dollars")
        self.active[cell_id] = amount

    def settle(self, cell_id: str, actual: float) -> None:
        reserved = self.active.pop(cell_id)
        # Never hide a runtime/provider metering violation; fail loudly.
        if actual > reserved + 0.00011:
            raise RuntimeError(
                f"cell spent ${actual:.4f}, exceeding its ${reserved:.4f} reservation")
        self.spent += actual

    def release(self, cell_id: str) -> None:
        self.active.pop(cell_id, None)


def run_cells(*, store: RunStore, tasks: list[dict], workers: int,
              docker_workers: int, budget_usd: float, per_task_usd: float,
              no_judge: bool, audit_log: audit.Audit,
              on_persist: Callable[[], None],
              progress_interval: float = 10.0) -> RunSummary:
    """Run pending cells through agent and scoring pools with bounded dispatch."""
    pending = deque(c for c in store.specs() if store.status(c.cell_id) == "pending")
    reservations = _Reservations(budget_usd, store.spent_usd)
    agent_futures: dict[Future, ActiveAgent] = {}
    score_futures: dict[Future, AgentOutput] = {}
    ready_to_score: deque[AgentOutput] = deque()
    interrupted = False
    fatal_error: str | None = None
    completed_now = 0
    pressure_note = ""
    serial_mode = workers == 1 and docker_workers == 1
    started = time.monotonic()
    invocation_started_at = utc_now_iso()
    previous_active_seconds = sum(
        float(item.get("wall_seconds", 0) or 0)
        for item in store.manifest.get("resources", {}).get("execution_history", []))
    agent_work_seconds = 0.0
    scoring_work_seconds = 0.0
    peak_agents = 0
    peak_scorers = 0
    pressure_pauses = 0
    min_memory_bytes: int | None = None
    min_disk_bytes: int | None = None
    max_load_1m = 0.0
    last_progress_print = started

    def _calendar_elapsed() -> float | None:
        try:
            created = datetime.fromisoformat(store.manifest["created_at"])
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            return max(0.0, (datetime.now(timezone.utc) - created).total_seconds())
        except (KeyError, TypeError, ValueError):
            return None

    def write_progress(*, status: str = "running", console: bool = False) -> None:
        now = time.monotonic()
        elapsed = max(0.0, now - started)
        counts = store.summary_counts()
        completed_total = counts.get("completed", 0)
        remaining = sum(counts.get(name, 0)
                        for name in ("pending", "running", "scoring"))
        rate_per_second = completed_now / elapsed if elapsed and completed_now else 0.0
        eta_seconds = remaining / rate_per_second if rate_per_second else None
        agent_cells = [
            {
                "cell_id": active.cell.cell_id,
                "task": active.cell.task_id,
                "variant": active.cell.variant,
                "config": active.cell.config_name,
                "elapsed_seconds": round(now - active.dispatched_monotonic, 3),
            }
            for active in sorted(agent_futures.values(), key=lambda item: item.cell.index)
        ]
        scoring_cells = [
            {
                "cell_id": agent.cell.cell_id,
                "task": agent.cell.task_id,
                "variant": agent.cell.variant,
                "config": agent.cell.config_name,
                "phase": phase,
                "elapsed_seconds": round(now - agent.agent_finished_monotonic, 3),
            }
            for phase, agents in (("waiting_for_scorer", list(ready_to_score)),
                                  ("scoring", list(score_futures.values())))
            for agent in sorted(agents, key=lambda item: item.cell.index)
        ]
        calendar_elapsed = _calendar_elapsed()
        snapshot = {
            "run_id": store.manifest["run_id"],
            "status": status,
            "updated_at": utc_now_iso(),
            "invocation": {
                "started_at": invocation_started_at,
                "elapsed_seconds": round(elapsed, 3),
                "completed_cells": completed_now,
                "throughput_cells_per_hour": (
                    round(rate_per_second * 3600, 3) if rate_per_second else None),
                "eta_seconds": round(eta_seconds, 3) if eta_seconds is not None else None,
            },
            "run": {
                "active_elapsed_seconds": round(previous_active_seconds + elapsed, 3),
                "calendar_elapsed_seconds": (
                    round(calendar_elapsed, 3) if calendar_elapsed is not None else None),
            },
            "cells": {"total": len(store.manifest["cells"]), **counts},
            "active": {"agents": agent_cells, "scoring": scoring_cells},
            "budget": {
                "cap_usd": round(budget_usd, 4),
                "spent_usd": round(reservations.spent, 4),
                "reserved_usd": round(sum(reservations.active.values()), 4),
                "available_usd": round(reservations.available, 4),
            },
        }
        atomic_write_json(store.run_dir / "progress.json", snapshot)
        if console:
            eta = format_duration(eta_seconds) if eta_seconds is not None else "--:--"
            print(
                f"⏱ {format_duration(elapsed)} elapsed | "
                f"{completed_total}/{len(store.manifest['cells'])} done | "
                f"agent {len(agent_cells)} | scoring {len(scoring_cells)} | "
                f"queued {counts.get('pending', 0)} | "
                f"{rate_per_second * 3600:.1f} cells/hour | ETA {eta}",
                flush=True,
            )

    def observe_resource(resource) -> None:
        nonlocal min_memory_bytes, min_disk_bytes, max_load_1m
        min_memory_bytes = (resource.memory_available_bytes
                            if min_memory_bytes is None else
                            min(min_memory_bytes, resource.memory_available_bytes))
        min_disk_bytes = (resource.disk_available_bytes
                          if min_disk_bytes is None else
                          min(min_disk_bytes, resource.disk_available_bytes))
        max_load_1m = max(max_load_1m, resource.load_1m)

    agent_pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="eval-agent")
    score_pool = ThreadPoolExecutor(max_workers=docker_workers, thread_name_prefix="eval-score")
    write_progress()
    try:
        while pending or agent_futures or ready_to_score or score_futures:
            made_progress = False
            # Sample throughout long-running Docker/provider stages, not only
            # at dispatch, so benchmark low-water marks are meaningful.
            observe_resource(resource_snapshot(store.run_dir))

            # Complete scoring first, freeing both a scoring slot and budget.
            for future in [f for f in score_futures if f.done()]:
                agent = score_futures.pop(future)
                try:
                    final = future.result()
                except BaseException as exc:
                    agent.result.error = _join_error(
                        agent.result.error, f"finalizer {type(exc).__name__}: {exc}")
                    completed = time.monotonic()
                    final = FinalOutput(
                        agent=agent, quality=None, would_merge=None,
                        judge_cost_usd=0.0, scoring_started_at=utc_now_iso(),
                        completed_at=utc_now_iso(),
                        scoring_queue_wait_seconds=max(
                            0.0, completed - agent.agent_finished_monotonic),
                        docker_lock_wait_seconds=0.0,
                        docker_scoring_wall_seconds=0.0,
                        judge_wall_seconds=0.0, scoring_wall_seconds=0.0,
                        completed_monotonic=completed,
                    )
                scoring_work_seconds += final.scoring_wall_seconds
                row = result_row(final)
                actual = _actual_cost(final)
                try:
                    reservations.settle(agent.cell.cell_id, actual)
                except RuntimeError as exc:
                    row["error"] = _join_error(row["error"], str(exc))
                    # Account truthfully if any runtime violates its reservation;
                    # stop further dispatch below.
                    reservations.spent += actual
                    reservations.active.pop(agent.cell.cell_id, None)
                    interrupted = True
                terminal, audit_row = audit_log.record(
                    task=tasks[agent.cell.task_index], variant=agent.cell.variant,
                    res=agent.result, quality=final.quality,
                    would_merge=final.would_merge,
                    remaining_before=agent.remaining_before,
                    cell_id=agent.cell.cell_id,
                    cell_meta={
                        "cell_id": agent.cell.cell_id,
                        "cell_index": agent.cell.index,
                        "config": agent.cell.config_name,
                        "repetition": agent.cell.repetition,
                        "seed": agent.cell.seed,
                        "judge_cost_usd": round(final.judge_cost_usd, 4),
                        "cell_cost_usd": actual,
                        "timing": {
                            key: value for key, value in row.items()
                            if key.endswith("_seconds") or key.endswith("_at")
                        },
                    },
                )
                payload = {"index": agent.cell.index, "row": row,
                           "audit_row": audit_row,
                           # Structured prediction enables deterministic Docker
                           # replay when proving serial/parallel scoring parity.
                           "model_patch": agent.result.diff}
                store.complete(
                    agent.cell.cell_id, payload, actual,
                    completed_at=final.completed_at,
                    timing={
                        key: value for key, value in row.items()
                        if key.endswith("_seconds") or key.endswith("_at")
                    },
                )
                on_persist()
                completed_now += 1
                print(f"\n[{completed_now} completed this invocation] {terminal}", flush=True)
                print(
                    f"    time: cell {format_duration(row['cell_elapsed_seconds'])} | "
                    f"setup {format_duration(row['workspace_setup_seconds'])} | "
                    f"runtime {format_duration(row['runtime_wall_seconds'])} | "
                    f"score wait {format_duration(row['scoring_queue_wait_seconds'])} | "
                    f"docker {format_duration(row['docker_scoring_wall_seconds'])} | "
                    f"judge {format_duration(row['judge_wall_seconds'])}",
                    flush=True,
                )
                made_progress = True

            # Move completed agents into the bounded scoring queue.
            for future in [f for f in agent_futures if f.done()]:
                active = agent_futures.pop(future)
                spec = active.cell
                try:
                    agent = future.result()
                except BaseException as exc:
                    # _agent_stage already contains failures; this is defensive.
                    reservation = reservations.active[spec.cell_id]
                    finished = time.monotonic()
                    agent = AgentOutput(
                        cell=spec, result=_error_result(spec.variant, exc),
                        remaining_before=budget_usd - reservations.spent,
                        reserved_usd=reservation, judge_reserved_usd=0.0,
                        queue_wait_seconds=active.queue_wait_seconds,
                        dispatched_monotonic=active.dispatched_monotonic,
                        dispatched_at=active.dispatched_at,
                        agent_finished_monotonic=finished,
                        agent_finished_at=utc_now_iso(),
                        workspace_setup_seconds=0.0,
                        runtime_wall_seconds=0.0,
                        workspace_cleanup_seconds=0.0,
                        agent_wall_seconds=max(
                            0.0, finished - active.dispatched_monotonic),
                    )
                agent_work_seconds += agent.agent_wall_seconds
                if _systemic_provider_error(agent.result.error):
                    reservations.release(spec.cell_id)
                    store.mark_pending(spec.cell_id)
                    interrupted = True
                    if fatal_error is None:
                        fatal_error = agent.result.error
                        print(
                            "\n!! Systemic provider failure detected; stopping new "
                            "dispatch and returning affected cells to pending. Fix "
                            "credentials/connectivity, then resume this run.\n"
                            f"   {fatal_error}",
                            flush=True,
                        )
                    made_progress = True
                    continue
                ready_to_score.append(agent)
                store.mark_scoring(
                    spec.cell_id, agent_finished_at=agent.agent_finished_at,
                    timing={
                        "queue_wait_seconds": round(agent.queue_wait_seconds, 3),
                        "workspace_setup_seconds": round(
                            agent.workspace_setup_seconds, 3),
                        "runtime_wall_seconds": round(agent.runtime_wall_seconds, 3),
                        "workspace_cleanup_seconds": round(
                            agent.workspace_cleanup_seconds, 3),
                        "agent_wall_seconds": round(agent.agent_wall_seconds, 3),
                    },
                )
                made_progress = True

            while ready_to_score and len(score_futures) < docker_workers:
                agent = ready_to_score.popleft()
                future = score_pool.submit(
                    _score_stage, agent, tasks[agent.cell.task_index],
                    run_id=store.manifest["run_id"], run_dir=store.run_dir,
                    no_judge=no_judge,
                )
                score_futures[future] = agent
                peak_scorers = max(peak_scorers, len(score_futures))
                made_progress = True

            # Dispatch agents only while the scorer backlog is bounded. Finished
            # agent results are small, but this keeps expensive stages balanced.
            backlog_limit = workers + docker_workers
            while (not interrupted and pending and len(agent_futures) < workers
                   and len(ready_to_score) + len(score_futures) < backlog_limit
                   and not (serial_mode and (ready_to_score or score_futures))):
                headroom, reason, resource = dispatch_headroom(store.run_dir)
                observe_resource(resource)
                if not headroom:
                    pressure_note = reason
                    pressure_pauses += 1
                    break
                pressure_note = ""
                cell = pending[0]
                task = tasks[cell.task_index]
                judge_reserve = (0.0 if no_judge else
                                 judge.max_cost_upper_bound(task, worst_case=True))
                desired = per_task_usd + judge_reserve
                available = reservations.available
                # Wait for active reservations to settle before shrinking a
                # later cell's cap. This retains serial budget semantics.
                if available + 1e-9 < desired and reservations.active:
                    break
                agent_cap = per_task_usd
                reservation = desired
                if available + 1e-9 < desired:
                    reason = (f"global budget exhausted: ${available:.4f} unreserved, "
                              f"needs ${desired:.4f} for a fully reserved cell")
                    while pending:
                        store.skip_budget(pending.popleft().cell_id, reason)
                    on_persist()
                    break
                pending.popleft()
                reservations.reserve(cell.cell_id, reservation)
                dispatched_monotonic = time.monotonic()
                dispatched_at = utc_now_iso()
                queue_wait_seconds = max(0.0, dispatched_monotonic - started)
                store.mark_running(
                    cell.cell_id, reservation, started_at=dispatched_at,
                    queue_wait_seconds=queue_wait_seconds,
                )
                remaining_before = budget_usd - reservations.spent
                future = agent_pool.submit(
                    _agent_stage, cell, task,
                    store.workspaces_dir / cell.cell_id,
                    agent_cap_usd=agent_cap, per_task_usd=per_task_usd,
                    remaining_before=remaining_before,
                    reserved_usd=reservation,
                    judge_reserved_usd=judge_reserve,
                    queue_wait_seconds=queue_wait_seconds,
                    dispatched_monotonic=dispatched_monotonic,
                    dispatched_at=dispatched_at,
                )
                agent_futures[future] = ActiveAgent(
                    cell=cell, dispatched_monotonic=dispatched_monotonic,
                    dispatched_at=dispatched_at,
                    queue_wait_seconds=queue_wait_seconds,
                )
                peak_agents = max(peak_agents, len(agent_futures))
                print(
                    f"▶ dispatch {cell.index + 1}/{len(store.manifest['cells'])}: "
                    f"{cell.task_id} :: {cell.variant} :: {cell.config_name} "
                    f"r{cell.repetition + 1} (reserve ${reservation:.2f})",
                    flush=True,
                )
                made_progress = True

            now = time.monotonic()
            if made_progress:
                write_progress()
            if progress_interval > 0 and now - last_progress_print >= progress_interval:
                write_progress(console=True)
                last_progress_print = now

            if not (pending or agent_futures or ready_to_score or score_futures):
                break
            if interrupted and not (agent_futures or ready_to_score or score_futures):
                # Leave undispatched cells pending for --resume.
                break
            futures = list(agent_futures) + list(score_futures)
            if futures:
                try:
                    wait(futures, timeout=1.0, return_when=FIRST_COMPLETED)
                except KeyboardInterrupt:
                    if not interrupted:
                        interrupted = True
                        print("\n! Interrupt received: stopping new dispatch and "
                              "finishing in-flight cells so the run is resumable. "
                              "Further interrupts will not discard metered work.",
                              flush=True)
                    else:
                        print("\n! Already draining in-flight cells; waiting so "
                              "their spend and results remain resumable.", flush=True)
            elif pending and not made_progress:
                if pressure_note:
                    print(f"! dispatch paused: {pressure_note}", flush=True)
                time.sleep(1.0)
    finally:
        # Never release the run lease while a worker could still mutate its
        # private workspace or incur unrecorded spend.
        agent_pool.shutdown(wait=True, cancel_futures=True)
        score_pool.shutdown(wait=True, cancel_futures=True)
        for active in agent_futures.values():
            store.mark_pending(active.cell.cell_id)
        for agent in list(ready_to_score) + list(score_futures.values()):
            store.mark_pending(agent.cell.cell_id)
        on_persist()

    wall_seconds = time.monotonic() - started
    telemetry = {
        "started_at": invocation_started_at,
        "finished_at": utc_now_iso(),
        "wall_seconds": round(wall_seconds, 3),
        "agent_work_seconds": round(agent_work_seconds, 3),
        "scoring_work_seconds": round(scoring_work_seconds, 3),
        "effective_agent_concurrency": round(
            agent_work_seconds / wall_seconds, 3) if wall_seconds else 0.0,
        "effective_scoring_concurrency": round(
            scoring_work_seconds / wall_seconds, 3) if wall_seconds else 0.0,
        "peak_agent_cells": peak_agents,
        "peak_scoring_cells": peak_scorers,
        "resource_pressure_pauses": pressure_pauses,
        "minimum_memory_available_bytes": min_memory_bytes,
        "minimum_disk_available_bytes": min_disk_bytes,
        "maximum_load_1m": round(max_load_1m, 3),
        "completed_cells_per_hour": round(
            completed_now * 3600 / wall_seconds, 3)
        if wall_seconds and completed_now else 0.0,
    }
    progress_status = "failed" if fatal_error else (
        "interrupted" if interrupted else "completed")
    write_progress(status=progress_status)
    return RunSummary(interrupted=interrupted, fatal_error=fatal_error,
                      completed_now=completed_now,
                      spent_usd=store.spent_usd, counts=store.summary_counts(),
                      telemetry=telemetry)

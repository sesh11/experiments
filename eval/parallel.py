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
from pathlib import Path
from typing import Callable

from fusion import config
from fusion.llm import Ledger
from orchestrator import variants
from orchestrator.variants import PolicyResult

from . import audit, docker_score, judge
from .isolation import isolated_task
from .resources import dispatch_headroom, snapshot as resource_snapshot
from .run_state import CellSpec, RunStore


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
    agent_wall_seconds: float


@dataclass
class FinalOutput:
    agent: AgentOutput
    quality: int | None
    would_merge: bool | None
    judge_cost_usd: float
    scoring_wall_seconds: float


@dataclass
class RunSummary:
    interrupted: bool
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
                 judge_reserved_usd: float) -> AgentOutput:
    started = time.monotonic()
    try:
        with isolated_task(base_task, workspace_dir, seed=cell.seed) as task:
            cfg = config.RunConfig(
                **cell.config,
                budget_usd=agent_cap_usd,
                per_task_usd=per_task_usd,
            )
            result = variants.run_variant(cell.variant, task, cfg)
    except BaseException as exc:  # one broken cell must not stop its siblings
        result = _error_result(cell.variant, exc)
    return AgentOutput(
        cell=cell, result=result, remaining_before=remaining_before,
        reserved_usd=reserved_usd, judge_reserved_usd=judge_reserved_usd,
        agent_wall_seconds=time.monotonic() - started,
    )


def _score_stage(agent: AgentOutput, base_task: dict, *, run_id: str,
                 run_dir: Path, no_judge: bool) -> FinalOutput:
    started = time.monotonic()
    res = agent.result
    if base_task.get("backend") == "docker":
        score_dir = run_dir / "cells" / agent.cell.cell_id / "scoring"
        score_dir.mkdir(parents=True, exist_ok=True)
        try:
            with _docker_instance_lock(base_task["instance_id"]):
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
        try:
            verdict = judge.judge_merge(base_task, res.diff, res.resolved)
            quality = verdict["score"]
            would_merge = verdict["would_merge"]
            judge_cost = float(verdict.get("cost_usd", 0.0) or 0.0)
        except BaseException as exc:
            res.error = _join_error(res.error, f"judge {type(exc).__name__}: {exc}")
    return FinalOutput(agent=agent, quality=quality,
                       would_merge=would_merge, judge_cost_usd=judge_cost,
                       scoring_wall_seconds=time.monotonic() - started)


def _join_error(first: str, second: str) -> str:
    return f"{first}; {second}" if first else second


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
        "agent_wall_seconds": round(final.agent.agent_wall_seconds, 3),
        "scoring_wall_seconds": round(final.scoring_wall_seconds, 3),
        "cell_wall_seconds": round(
            final.agent.agent_wall_seconds + final.scoring_wall_seconds, 3),
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


def run_cells(*, store: RunStore, tasks: list[dict], workers: int,
              docker_workers: int, budget_usd: float, per_task_usd: float,
              no_judge: bool, audit_log: audit.Audit,
              on_persist: Callable[[], None]) -> RunSummary:
    """Run pending cells through agent and scoring pools with bounded dispatch."""
    pending = deque(c for c in store.specs() if store.status(c.cell_id) == "pending")
    reservations = _Reservations(budget_usd, store.spent_usd)
    agent_futures: dict[Future, AgentOutput | CellSpec] = {}
    score_futures: dict[Future, AgentOutput] = {}
    ready_to_score: deque[AgentOutput] = deque()
    interrupted = False
    completed_now = 0
    pressure_note = ""
    serial_mode = workers == 1 and docker_workers == 1
    started = time.monotonic()
    agent_work_seconds = 0.0
    scoring_work_seconds = 0.0
    peak_agents = 0
    peak_scorers = 0
    pressure_pauses = 0
    min_memory_bytes: int | None = None
    min_disk_bytes: int | None = None
    max_load_1m = 0.0

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
                    final = FinalOutput(agent, None, None, 0.0, 0.0)
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
                    },
                )
                payload = {"index": agent.cell.index, "row": row,
                           "audit_row": audit_row,
                           # Structured prediction enables deterministic Docker
                           # replay when proving serial/parallel scoring parity.
                           "model_patch": agent.result.diff}
                store.complete(agent.cell.cell_id, payload, actual)
                on_persist()
                completed_now += 1
                print(f"\n[{completed_now} completed this invocation] {terminal}", flush=True)
                made_progress = True

            # Move completed agents into the bounded scoring queue.
            for future in [f for f in agent_futures if f.done()]:
                spec = agent_futures.pop(future)
                assert isinstance(spec, CellSpec)
                try:
                    agent = future.result()
                except BaseException as exc:
                    # _agent_stage already contains failures; this is defensive.
                    reservation = reservations.active[spec.cell_id]
                    agent = AgentOutput(
                        spec, _error_result(spec.variant, exc),
                        budget_usd - reservations.spent, reservation, 0.0, 0.0)
                agent_work_seconds += agent.agent_wall_seconds
                ready_to_score.append(agent)
                store.mark_scoring(spec.cell_id)
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
                store.mark_running(cell.cell_id, reservation)
                remaining_before = budget_usd - reservations.spent
                future = agent_pool.submit(
                    _agent_stage, cell, task,
                    store.workspaces_dir / cell.cell_id,
                    agent_cap_usd=agent_cap, per_task_usd=per_task_usd,
                    remaining_before=remaining_before,
                    reserved_usd=reservation,
                    judge_reserved_usd=judge_reserve,
                )
                agent_futures[future] = cell
                peak_agents = max(peak_agents, len(agent_futures))
                print(
                    f"▶ dispatch {cell.index + 1}/{len(store.manifest['cells'])}: "
                    f"{cell.task_id} :: {cell.variant} :: {cell.config_name} "
                    f"r{cell.repetition + 1} (reserve ${reservation:.2f})",
                    flush=True,
                )
                made_progress = True

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
        for spec in agent_futures.values():
            if isinstance(spec, CellSpec):
                store.mark_pending(spec.cell_id)
        for agent in list(ready_to_score) + list(score_futures.values()):
            store.mark_pending(agent.cell.cell_id)
        on_persist()

    wall_seconds = time.monotonic() - started
    telemetry = {
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
    }
    return RunSummary(interrupted=interrupted, completed_now=completed_now,
                      spent_usd=store.spent_usd, counts=store.summary_counts(),
                      telemetry=telemetry)

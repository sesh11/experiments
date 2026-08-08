from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import concurrent.futures
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import pytest

from eval import audit
from eval.isolation import isolated_task
from eval.parallel import _systemic_provider_error, run_cells
from eval.run_eval import (_load_configurations, _persist_aggregates,
                           _validate_resume_args)
from eval.resources import GIB, ResourceSnapshot, choose_workers
from eval.run_state import RunStore, build_cells, run_lease
from eval.timing import format_duration, summarize_timing
from fusion import config as fusion_config
from fusion.llm import BudgetExceeded, LLMClient, Ledger
from orchestrator.variants import PolicyResult
from runtimes.pi_rt import PiRuntime
from fusion.workspace import Workspace
from scripts import benchmark_parallel
from scripts.benchmark_parallel import (_bottleneck, _preflight_anthropic,
                                        _resource_report, _verify_scoring_parity)


RUN_CONFIG = {
    "max_tokens": 256,
    "max_steps": 2,
    "scout_max_steps": 1,
    "main_thinking": {"type": "disabled"},
    "sidekick_thinking": None,
    "runtime_timeout_s": 10,
}


def _tasks(count: int, *, backend: str = "local") -> list[dict]:
    return [{
        "instance_id": f"task-{i}",
        "problem_statement": f"fix task {i}",
        "template_dir": ".",
        "test_cmd": "true",
        "backend": backend,
    } for i in range(count)]


def _store(tmp_path: Path, tasks: list[dict], *, variants=("fake",),
           configs=None, repetitions=1, budget=10.0):
    configs = configs or [{"name": "default", "run_config": RUN_CONFIG}]
    cells = build_cells(tasks, list(variants), configs, repetitions, 100)
    store = RunStore.create(
        out_dir=tmp_path / "results", run_id="test-run",
        experiment={"test": True}, cells=cells, budget_usd=budget,
        resources={},
    )
    log = audit.Audit(tmp_path / "results", run_id="test-run")
    return store, log, cells


def _result(variant: str, cost: float = 0.1, *, resolved=True) -> PolicyResult:
    return PolicyResult(
        variant=variant, resolved=resolved, diff="(no changes)", summary="ok",
        ledger={"total_cost_usd": cost, "main_cost_usd": cost},
        resolve_detail="fake", steps=1, finished=True,
    )


def test_cell_identity_covers_config_and_repetition():
    tasks = _tasks(1)
    configs = [
        {"name": "short", "run_config": {**RUN_CONFIG, "max_steps": 1}},
        {"name": "long", "run_config": {**RUN_CONFIG, "max_steps": 9}},
    ]
    first = build_cells(tasks, ["fake"], configs, 2, 7)
    second = build_cells(tasks, ["fake"], configs, 2, 7)
    assert [c.cell_id for c in first] == [c.cell_id for c in second]
    assert len({c.cell_id for c in first}) == 4
    assert [c.seed for c in first] == [7, 8, 7, 8]


def test_manifest_resume_recovers_running_but_keeps_completed(tmp_path):
    store, _, cells = _store(tmp_path, _tasks(2))
    store.mark_running(cells[0].cell_id, 1.0)
    store.mark_running(cells[1].cell_id, 1.0)
    store.complete(cells[1].cell_id, {"index": 1, "row": {}, "audit_row": {}}, 0.2)

    resumed = RunStore.load(store.manifest_path)
    assert resumed.status(cells[0].cell_id) == "pending"
    assert resumed.status(cells[1].cell_id) == "completed"
    assert resumed.spent_usd == pytest.approx(0.2)
    assert resumed.manifest["cells"][cells[0].cell_id]["attempt_history"][0][
        "outcome"] == "interrupted"


def test_run_lease_refuses_a_second_coordinator(tmp_path):
    with run_lease(tmp_path):
        with pytest.raises(RuntimeError, match="already owned"):
            with run_lease(tmp_path):
                pass


def test_resume_executes_only_incomplete_cells(tmp_path):
    tasks = _tasks(3)
    store, log, cells = _store(tmp_path, tasks)
    completed_payload = {
        "index": 0,
        "row": {"task": "task-0", "variant": "fake", "resolved": True,
                "budget_hit": False, "cell_cost_usd": 0.1},
        "audit_row": {"task": "task-0"},
    }
    store.complete(cells[0].cell_id, completed_payload, 0.1)
    resumed = RunStore.load(store.manifest_path)
    called = []

    def fake_run(variant, task, cfg):
        called.append(task["instance_id"])
        return _result(variant)

    with patch("eval.parallel.variants.run_variant", side_effect=fake_run):
        run_cells(
            store=resumed, tasks=tasks, workers=2, docker_workers=1,
            budget_usd=10, per_task_usd=1, no_judge=True, audit_log=log,
            on_persist=lambda: None,
        )
    assert called == ["task-1", "task-2"]
    assert [payload["index"] for payload in resumed.completed_payloads()] == [0, 1, 2]


def test_graceful_interrupt_drains_inflight_and_leaves_rest_resumable(tmp_path):
    tasks = _tasks(5)
    store, log, _ = _store(tmp_path, tasks)
    calls = 0

    def fake_run(variant, task, cfg):
        time.sleep(0.04)
        return _result(variant)

    def interrupt_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise KeyboardInterrupt
        return concurrent.futures.wait(*args, **kwargs)

    with (patch("eval.parallel.variants.run_variant", side_effect=fake_run),
          patch("eval.parallel.wait", side_effect=interrupt_once)):
        summary = run_cells(
            store=store, tasks=tasks, workers=2, docker_workers=1,
            budget_usd=10, per_task_usd=1, no_judge=True,
            audit_log=log, on_persist=lambda: None,
        )
    assert summary.interrupted
    assert summary.counts == {"completed": 2, "pending": 3}

    resumed = RunStore.load(store.manifest_path)
    with patch("eval.parallel.variants.run_variant", side_effect=fake_run):
        final = run_cells(
            store=resumed, tasks=tasks, workers=2, docker_workers=1,
            budget_usd=10, per_task_usd=1, no_judge=True,
            audit_log=log, on_persist=lambda: None,
        )
    assert final.counts == {"completed": 5}


def test_auto_workers_scale_and_explicit_values_win(tmp_path):
    snap = ResourceSnapshot(cpus=16, memory_available_bytes=64 * GIB,
                            disk_available_bytes=200 * GIB)
    with patch("eval.resources.snapshot", return_value=snap):
        agents, scorers, _ = choose_workers(
            workers="auto", docker_workers="auto", queued_cells=20,
            docker_enabled=True, path=tmp_path)
        assert agents > 1
        assert scorers > 1
        explicit = choose_workers(
            workers="11", docker_workers="3", queued_cells=20,
            docker_enabled=True, path=tmp_path)
        assert explicit[:2] == (11, 3)


def test_builtin_llm_rejects_unfunded_request_before_provider_call():
    ledger = Ledger(cap_usd=0.01)
    provider = type("Provider", (), {})()
    provider.messages = type("Messages", (), {"create": lambda self, **kwargs: None})()
    with patch("fusion.llm.anthropic.Anthropic", return_value=provider):
        client = LLMClient(ledger, max_tokens=8192)
        with pytest.raises(BudgetExceeded, match="next call could exceed"):
            client.complete(role="main", model=fusion_config.MODEL_MAIN,
                            system="system", messages=[{"role": "user", "content": "hi"}])


def test_pi_live_meter_stops_before_an_unfunded_next_turn(tmp_path, monkeypatch):
    fake_pi = tmp_path / "fake-pi"
    fake_pi.write_text(
        "#!/usr/bin/env python3\n"
        "import json, time\n"
        "event={'type':'message_end','message':{'role':'assistant',"
        "'stopReason':'toolUse','usage':{'input':20000,'output':10,"
        "'cacheRead':0,'cacheWrite':0},'content':[]}}\n"
        "print(json.dumps(event), flush=True)\n"
        "time.sleep(5)\n"
    )
    fake_pi.chmod(0o755)
    monkeypatch.setenv("PI_BIN", str(fake_pi))
    repo = tmp_path / "repo"
    repo.mkdir()
    ws = Workspace(repo, "true")
    ledger = Ledger(cap_usd=0.80)
    cfg = fusion_config.RunConfig(budget_usd=0.80, max_tokens=256,
                                  runtime_timeout_s=10)
    with pytest.raises(BudgetExceeded, match="stopped before its next turn"):
        PiRuntime().run(
            {"problem_statement": "fake"}, ws,
            model=fusion_config.MODEL_MAIN, ledger=ledger, cfg=cfg)
    assert 0 < ledger.total_cost < ledger.cap_usd


def test_parallel_cells_are_faster_and_failures_are_isolated(tmp_path):
    tasks = _tasks(6)
    store, log, _ = _store(tmp_path, tasks)
    lock = threading.Lock()
    active = 0
    max_active = 0

    def fake_run(variant, task, cfg):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            time.sleep(0.12)
            if task["instance_id"] == "task-2":
                raise RuntimeError("intentional cell failure")
            return _result(variant)
        finally:
            with lock:
                active -= 1

    started = time.monotonic()
    with patch("eval.parallel.variants.run_variant", side_effect=fake_run):
        summary = run_cells(
            store=store, tasks=tasks, workers=3, docker_workers=2,
            budget_usd=10, per_task_usd=1, no_judge=True, audit_log=log,
            on_persist=lambda: None,
        )
    parallel_elapsed = time.monotonic() - started

    serial_store, serial_log, _ = _store(tmp_path / "serial", tasks)
    serial_started = time.monotonic()
    with patch("eval.parallel.variants.run_variant", side_effect=fake_run):
        run_cells(
            store=serial_store, tasks=tasks, workers=1, docker_workers=1,
            budget_usd=10, per_task_usd=1, no_judge=True,
            audit_log=serial_log, on_persist=lambda: None,
        )
    serial_elapsed = time.monotonic() - serial_started

    payloads = store.completed_payloads()
    assert summary.completed_now == 6
    assert max_active >= 2
    assert parallel_elapsed < serial_elapsed * 0.75
    assert summary.telemetry["peak_agent_cells"] == 3
    assert summary.telemetry["effective_agent_concurrency"] > 1.5
    assert summary.telemetry["minimum_memory_available_bytes"] is not None
    assert summary.telemetry["minimum_disk_available_bytes"] is not None
    assert [p["index"] for p in payloads] == list(range(6))
    assert all("model_patch" in payload for payload in payloads)
    assert all(payload["row"]["runtime_wall_seconds"] >= 0.1
               for payload in payloads)
    assert all(payload["row"]["cell_elapsed_seconds"]
               >= payload["row"]["agent_wall_seconds"]
               for payload in payloads)
    progress = json.loads((store.run_dir / "progress.json").read_text())
    assert progress["status"] == "completed"
    assert progress["cells"]["completed"] == 6
    assert progress["active"] == {"agents": [], "scoring": []}
    failed = next(p["row"] for p in payloads if p["row"]["task"] == "task-2")
    assert "intentional cell failure" in failed["error"]


def test_phase_timing_is_live_and_aggregated(tmp_path):
    tasks = _tasks(2, backend="docker")
    store, log, cells = _store(tmp_path, tasks)

    def fake_run(variant, task, cfg):
        time.sleep(0.03)
        return _result(variant)

    def fake_score(instance_id, diff, **kwargs):
        time.sleep(0.02)
        return {"resolved": True, "detail": "ok", "applied": True,
                "report": {}, "harness_tail": ""}

    with (patch("eval.parallel.variants.run_variant", side_effect=fake_run),
          patch("eval.parallel.docker_score.score_patch", side_effect=fake_score),
          patch("eval.parallel.judge.max_cost_upper_bound", return_value=0.1),
          patch("eval.parallel.judge.judge_merge", side_effect=lambda *args: (
              time.sleep(0.01) or {"score": 90, "would_merge": True,
                                   "rationale": "ok", "cost_usd": 0.0}))):
        summary = run_cells(
            store=store, tasks=tasks, workers=2, docker_workers=2,
            budget_usd=10, per_task_usd=1, no_judge=False,
            audit_log=log, on_persist=lambda: None, progress_interval=0,
        )

    rows = [payload["row"] for payload in store.completed_payloads()]
    assert len(rows) == 2
    for row in rows:
        assert row["runtime_wall_seconds"] >= 0.02
        assert row["docker_scoring_wall_seconds"] >= 0.01
        assert row["judge_wall_seconds"] >= 0.005
        assert row["cell_elapsed_seconds"] >= row["agent_wall_seconds"]
        assert row["dispatched_at"]
        assert row["completed_at"]
    entry = store.manifest["cells"][cells[0].cell_id]
    assert entry["timing"]["cell_elapsed_seconds"] > 0

    progress = json.loads((store.run_dir / "progress.json").read_text())
    assert progress["status"] == "completed"
    assert progress["invocation"]["elapsed_seconds"] > 0
    assert progress["run"]["active_elapsed_seconds"] > 0
    assert progress["budget"]["reserved_usd"] == 0

    timing = summarize_timing(
        rows, execution_history=[summary.telemetry],
        created_at=store.manifest["created_at"])
    assert timing["overall"]["phases"]["runtime_wall_seconds"]["count"] == 2
    assert timing["overall"]["phases"]["cell_elapsed_seconds"]["p95"] > 0
    assert timing["run"]["active_wall_seconds"] == summary.telemetry["wall_seconds"]
    assert timing["by_variant_config"][0]["variant"] == "fake"
    assert format_duration(0.25) == "0.25s"
    assert format_duration(65) == "01:05"
    assert format_duration(3661) == "1:01:01"


def test_systemic_provider_failure_stops_dispatch_and_stays_resumable(tmp_path):
    tasks = _tasks(4)
    store, log, cells = _store(tmp_path, tasks)

    def auth_failure(variant, task, cfg):
        result = _result(variant, cost=0.0, resolved=False)
        result.error = "AuthenticationError: invalid x-api-key"
        return result

    with patch("eval.parallel.variants.run_variant", side_effect=auth_failure):
        summary = run_cells(
            store=store, tasks=tasks, workers=4, docker_workers=2,
            budget_usd=10, per_task_usd=1, no_judge=True,
            audit_log=log, on_persist=lambda: None, progress_interval=0,
        )

    assert summary.interrupted
    assert "AuthenticationError" in summary.fatal_error
    assert summary.completed_now == 0
    assert summary.spent_usd == 0
    assert summary.counts == {"pending": 4}
    assert all(store.manifest["cells"][cell.cell_id]["attempt_history"]
               for cell in cells)
    progress = json.loads((store.run_dir / "progress.json").read_text())
    assert progress["status"] == "failed"
    assert progress["budget"]["reserved_usd"] == 0
    assert _systemic_provider_error("RateLimitError: retry exhausted")
    assert not _systemic_provider_error("RuntimeError: one task failed")


def test_benchmark_direct_script_adds_repo_root_to_import_path(tmp_path):
    script = Path(benchmark_parallel.__file__).resolve()
    command = (
        "import runpy; "
        f"runpy.run_path({str(script)!r}, run_name='benchmark_import_test'); "
        "import eval; print(eval.__file__)"
    )
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    proc = subprocess.run(
        [sys.executable, "-c", command], cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    assert "eval" in proc.stdout


def test_benchmark_anthropic_preflight_rejects_placeholder_and_401(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-your-key-here")
    with pytest.raises(SystemExit, match="missing or still a placeholder"):
        _preflight_anthropic()

    import anthropic
    import httpx

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-invalid")
    response = httpx.Response(
        401, request=httpx.Request("GET", "https://api.anthropic.com/v1/models"))
    error = anthropic.AuthenticationError(
        "invalid key", response=response,
        body={"error": {"message": "invalid x-api-key"}})
    def reject(**kwargs):
        raise error

    client = SimpleNamespace(models=SimpleNamespace(list=reject))
    with patch("anthropic.Anthropic", return_value=client):
        with pytest.raises(SystemExit, match="rejected ANTHROPIC_API_KEY"):
            _preflight_anthropic()


def test_global_budget_reservations_prevent_overdispatch(tmp_path):
    tasks = _tasks(8)
    store, log, _ = _store(tmp_path, tasks, budget=1.0)
    lock = threading.Lock()
    active = 0
    max_active = 0

    def fake_run(variant, task, cfg):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.03)
        with lock:
            active -= 1
        return _result(variant, 0.3)

    with patch("eval.parallel.variants.run_variant", side_effect=fake_run):
        summary = run_cells(
            store=store, tasks=tasks, workers=8, docker_workers=2,
            budget_usd=1.0, per_task_usd=0.4, no_judge=True,
            audit_log=log, on_persist=lambda: None,
        )
    assert summary.spent_usd <= 1.0
    assert summary.counts["completed"] == 3
    assert summary.counts["skipped_budget"] == 5
    # At most floor(1.0 / 0.4) cells may be reserved initially.
    assert max_active <= 2


def test_judge_cost_is_reserved_and_counted_in_global_budget(tmp_path):
    tasks = _tasks(4)
    store, log, _ = _store(tmp_path, tasks, budget=1.0)

    with (patch("eval.parallel.variants.run_variant",
                side_effect=lambda variant, task, cfg: _result(variant, 0.25)),
          patch("eval.parallel.judge.max_cost_upper_bound", return_value=0.2),
          patch("eval.parallel.judge.judge_merge",
                return_value={"score": 90, "would_merge": True,
                              "rationale": "ok", "cost_usd": 0.1})):
        summary = run_cells(
            store=store, tasks=tasks, workers=4, docker_workers=2,
            budget_usd=1.0, per_task_usd=0.3, no_judge=False,
            audit_log=log, on_persist=lambda: None,
        )
    assert summary.counts == {"completed": 2, "skipped_budget": 2}
    assert summary.spent_usd == pytest.approx(0.7)
    assert all(p["row"]["judge_cost_usd"] == 0.1
               for p in store.completed_payloads())


def test_docker_scoring_has_its_own_concurrency_bound_and_unique_artifacts(tmp_path):
    tasks = _tasks(6, backend="docker")
    for task in tasks:
        task.update({"_dataset": "dataset", "_split": "test"})
    store, log, _ = _store(tmp_path, tasks)
    lock = threading.Lock()
    active = 0
    max_active = 0
    score_dirs = set()
    run_ids = set()

    def fake_score(instance_id, diff, **kwargs):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
            score_dirs.add(str(kwargs["cwd"]))
            run_ids.add(kwargs["run_id"])
        time.sleep(0.08)
        with lock:
            active -= 1
        return {"resolved": True, "detail": "ok", "applied": True,
                "report": {}, "harness_tail": ""}

    with (patch("eval.parallel.variants.run_variant",
                side_effect=lambda variant, task, cfg: _result(variant)),
          patch("eval.parallel.docker_score.score_patch", side_effect=fake_score)):
        run_cells(
            store=store, tasks=tasks, workers=6, docker_workers=2,
            budget_usd=10, per_task_usd=1, no_judge=True, audit_log=log,
            on_persist=lambda: None,
        )
    assert max_active == 2
    assert len(score_dirs) == 6
    assert len(run_ids) == 6


def test_one_by_one_worker_settings_preserve_fully_serial_execution(tmp_path):
    tasks = _tasks(3, backend="docker")
    store, log, _ = _store(tmp_path, tasks)
    lock = threading.Lock()
    agent_active = False
    scorer_active = False
    overlaps = []

    def fake_run(variant, task, cfg):
        nonlocal agent_active
        with lock:
            overlaps.append(scorer_active)
            agent_active = True
        time.sleep(0.02)
        with lock:
            agent_active = False
        return _result(variant)

    def fake_score(instance_id, diff, **kwargs):
        nonlocal scorer_active
        with lock:
            overlaps.append(agent_active)
            scorer_active = True
        time.sleep(0.02)
        with lock:
            scorer_active = False
        return {"resolved": True, "detail": "ok", "applied": True,
                "report": {}, "harness_tail": ""}

    with (patch("eval.parallel.variants.run_variant", side_effect=fake_run),
          patch("eval.parallel.docker_score.score_patch", side_effect=fake_score)):
        run_cells(
            store=store, tasks=tasks, workers=1, docker_workers=1,
            budget_usd=10, per_task_usd=1, no_judge=True,
            audit_log=log, on_persist=lambda: None,
        )
    assert not any(overlaps)


def test_same_instance_docker_scores_are_serialized_to_avoid_image_races(tmp_path):
    tasks = _tasks(1, backend="docker")
    store, log, _ = _store(tmp_path, tasks, variants=("fake-a", "fake-b"))
    lock = threading.Lock()
    active = 0
    max_active = 0

    def fake_score(instance_id, diff, **kwargs):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.04)
        with lock:
            active -= 1
        return {"resolved": True, "detail": "ok", "applied": True,
                "report": {}, "harness_tail": ""}

    with (patch("eval.parallel.variants.run_variant",
                side_effect=lambda variant, task, cfg: _result(variant)),
          patch("eval.parallel.docker_score.score_patch", side_effect=fake_score)):
        run_cells(
            store=store, tasks=tasks, workers=2, docker_workers=2,
            budget_usd=10, per_task_usd=1, no_judge=True,
            audit_log=log, on_persist=lambda: None,
        )
    assert max_active == 1


def test_same_instance_cells_use_independent_git_clones(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=source, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=source, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=source, check=True)
    (source / "value.py").write_text("VALUE = 0\n")
    subprocess.run(["git", "add", "value.py"], cwd=source, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=source, check=True)
    base = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source,
                          check=True, capture_output=True, text=True).stdout.strip()
    task = {
        "instance_id": "same", "problem_statement": "change it",
        "template_dir": str(source), "_dir": str(source),
        "_base_commit": base, "in_place": True,
        "test_cmd": "python -m pytest", "backend": "docker",
    }
    roots = []

    def edit(index):
        with isolated_task(task, tmp_path / f"ws-{index}", seed=index) as isolated:
            root = Path(isolated["template_dir"])
            roots.append(root)
            (root / "value.py").write_text(f"VALUE = {index}\n")
            time.sleep(0.05)
            return (root / "value.py").read_text()

    with ThreadPoolExecutor(max_workers=2) as pool:
        values = list(pool.map(edit, [1, 2]))
    assert set(values) == {"VALUE = 1\n", "VALUE = 2\n"}
    assert len(set(roots)) == 2
    assert (source / "value.py").read_text() == "VALUE = 0\n"
    assert all(not root.exists() for root in roots)


def test_configuration_matrix_inherits_defaults_and_rejects_unknown(tmp_path):
    path = tmp_path / "configs.json"
    path.write_text(json.dumps({"configurations": [
        {"name": "quick", "run_config": {"max_steps": 3}},
        {"name": "deep", "run_config": {"max_steps": 30}},
    ]}))
    configs = _load_configurations(str(path), None)
    assert [cfg["name"] for cfg in configs] == ["quick", "deep"]
    assert configs[0]["run_config"]["max_steps"] == 3
    assert configs[0]["run_config"]["max_tokens"] > 0

    path.write_text(json.dumps([{"name": "bad", "unknown": 1}]))
    with pytest.raises(SystemExit, match="unknown RunConfig fields"):
        _load_configurations(str(path), None)


def test_resume_rejects_incompatible_experiment_settings():
    experiment = {
        "source": "native", "backend": "local", "variants": ["frontier_only"],
        "repetitions": 1, "base_seed": 0, "per_task_usd": 3.0,
        "requested_limit": 1, "no_judge": True,
        "task_ids": ["native__slugify"],
        "configurations": [{"name": "default", "run_config": RUN_CONFIG}],
    }
    args = SimpleNamespace(
        source="swebench", backend=None, variants=None, repetitions=None,
        seed=None, per_task=None, limit=None, no_judge=None,
        instance_ids_file=None, configs_file=None, max_steps=None,
    )
    with pytest.raises(SystemExit, match="setting mismatch for source"):
        _validate_resume_args(args, experiment)


def test_aggregate_outputs_are_deterministic_after_out_of_order_completion(tmp_path,
                                                                           monkeypatch):
    tasks = _tasks(2)
    store, log, cells = _store(tmp_path, tasks)
    for index in (1, 0):
        store.complete(cells[index].cell_id, {
            "index": index,
            "row": {"task": f"task-{index}", "variant": "fake",
                    "cell_index": index},
            "audit_row": {"task": f"task-{index}", "cell_index": index},
        }, 0.0)
    monkeypatch.setattr("eval.run_eval._OUT", tmp_path / "compat")
    _persist_aggregates(store, log)
    rows = json.loads((store.run_dir / "summary.json").read_text())
    assert [row["cell_index"] for row in rows] == [0, 1]
    jsonl = [json.loads(line) for line in log.jsonl.read_text().splitlines()]
    assert [row["cell_index"] for row in jsonl] == [0, 1]


def test_benchmark_diagnoses_budget_resource_and_pool_bottlenecks():
    manifest = {
        "cells": {"a": {"status": "skipped_budget"}},
        "resources": {"initial": {
            "cpus": 8, "memory_available_gib": 16,
            "disk_available_gib": 100,
        }},
    }
    assert _bottleneck(manifest, {}) == "global_budget"
    manifest["cells"]["a"]["status"] = "completed"
    telemetry = {
        "workers": 3, "docker_workers": 1,
        "peak_agent_cells": 3, "peak_scoring_cells": 1,
        "effective_agent_concurrency": 1.2,
        "effective_scoring_concurrency": 1.5,
        "resource_pressure_pauses": 0,
        "minimum_memory_available_bytes": 8 * GIB,
        "minimum_disk_available_bytes": 80 * GIB,
        "maximum_load_1m": 6.0,
    }
    assert _bottleneck(manifest, telemetry) == "docker_scoring_pool"
    resources = _resource_report(manifest, telemetry)
    assert resources["minimum_memory_available_gib"] == 8
    assert resources["minimum_disk_available_gib"] == 80


def test_identical_patch_scoring_replay_proves_parallel_parity(tmp_path, monkeypatch):
    serial_dir = tmp_path / "serial"
    for index in range(3):
        cell_dir = serial_dir / "cells" / f"cell-{index}"
        cell_dir.mkdir(parents=True)
        (cell_dir / "result.json").write_text(json.dumps({
            "index": index,
            "row": {"cell_id": f"cell-{index}", "task": f"task-{index}",
                    "resolved": index % 2 == 0},
            "model_patch": f"patch-{index}",
        }))

    def fake_score(instance_id, patch_text, **kwargs):
        time.sleep(0.03)
        index = int(instance_id.rsplit("-", 1)[1])
        return {"resolved": index % 2 == 0, "detail": "same"}

    monkeypatch.chdir(tmp_path)
    with patch("eval.docker_score.score_patch", side_effect=fake_score):
        report = _verify_scoring_parity(
            serial_dir, stamp="test", docker_workers=3)
    assert report["passed"]
    assert report["differences"] == []
    assert report["effective_scoring_concurrency"] > 1.5

from __future__ import annotations

import concurrent.futures
import csv
import json
import os
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import pytest

from eval import audit
from eval.parallel import run_cells
from eval.run_eval import _persist_aggregates
from eval.run_state import RunStore, build_cells
from fusion import config
from orchestrator import variants
from orchestrator.variants import PolicyResult, VariantSpec
from runtimes.base import RuntimeResult


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
        "instance_id": f"integration-{index}",
        "problem_statement": f"fix integration task {index}",
        "template_dir": ".",
        "test_cmd": "true",
        "backend": backend,
    } for index in range(count)]


def _store(tmp_path: Path, tasks: list[dict], *, variant: str = "fake",
           configurations: list[dict] | None = None,
           repetitions: int = 1, base_seed: int = 100) -> tuple[RunStore, audit.Audit]:
    configurations = configurations or [{
        "name": "default", "run_config": RUN_CONFIG}]
    cells = build_cells(
        tasks, [variant], configurations, repetitions, base_seed)
    store = RunStore.create(
        out_dir=tmp_path / "results", run_id="integration-run",
        experiment={"integration": True}, cells=cells,
        budget_usd=20, resources={},
    )
    return store, audit.Audit(tmp_path / "results", run_id="integration-run")


def _result(variant: str, task_id: str, cost: float = 0.1) -> PolicyResult:
    return PolicyResult(
        variant=variant, resolved=True,
        diff=("diff --git a/fix.py b/fix.py\n"
              "--- a/fix.py\n+++ b/fix.py\n@@ -1 +1 @@\n-old\n+new\n"),
        summary=f"fixed {task_id}",
        ledger={"total_cost_usd": cost, "main_cost_usd": cost},
        resolve_detail="fake integration success", steps=1, finished=True,
    )


def _write_subprocess_fakes(path: Path) -> None:
    """Install fake provider/task/scorer boundaries in every Python subprocess."""
    path.mkdir()
    (path / "sitecustomize.py").write_text(
        """
import os
import time
from types import SimpleNamespace

if os.environ.get("FUSION_FAKE_INTEGRATION") == "1":
    import anthropic
    from eval import docker_score, tasks
    from orchestrator import variants
    from orchestrator.variants import PolicyResult

    class FakeAnthropic:
        def __init__(self, *args, **kwargs):
            self.models = SimpleNamespace(
                list=lambda **kwargs: [SimpleNamespace(id="fake-model")])

    def fake_load(source, limit=None, *, backend="local", instance_ids=None):
        ids = list(instance_ids or ["fake-1", "fake-2"])
        if limit is not None:
            ids = ids[:limit]
        return [{
            "instance_id": instance_id,
            "problem_statement": f"fix {instance_id}",
            "template_dir": ".",
            "test_cmd": "true",
            "backend": backend,
            "_dataset": "fake-dataset",
            "_split": "test",
        } for instance_id in ids]

    def fake_variant(name, task, cfg):
        time.sleep(0.03)
        cost = 0.05
        return PolicyResult(
            variant=name, resolved=False,
            diff=("diff --git a/fix.py b/fix.py\\n"
                  "--- a/fix.py\\n+++ b/fix.py\\n@@ -1 +1 @@\\n-old\\n+new\\n"),
            summary="fake benchmark patch",
            ledger={"total_cost_usd": cost, "main_cost_usd": cost},
            resolve_detail="pending fake Docker score", steps=1, finished=True,
        )

    def fake_score(instance_id, patch_text, **kwargs):
        time.sleep(0.02)
        return {
            "resolved": bool(patch_text), "detail": "fake Docker resolved",
            "applied": bool(patch_text), "report": {}, "harness_tail": "fake",
        }

    anthropic.Anthropic = FakeAnthropic
    tasks.load = fake_load
    docker_score.preflight = lambda: (True, "fake Docker ready")
    docker_score.score_patch = fake_score
    variants.run_variant = fake_variant
""".lstrip())


def test_benchmark_cli_full_subprocess_with_fake_external_boundaries(tmp_path):
    """Exercise the real benchmark script, both nested evals, parity, and report."""
    repo = Path(__file__).resolve().parents[1]
    hooks = tmp_path / "python-hooks"
    _write_subprocess_fakes(hooks)
    ids = tmp_path / "instances.txt"
    ids.write_text("fake-1\nfake-2\n")
    results = tmp_path / "benchmark-results"
    env = os.environ.copy()
    env.update({
        "ANTHROPIC_API_KEY": "sk-ant-fake-integration",
        "EVAL_RESULTS_DIR": str(results),
        "FUSION_FAKE_INTEGRATION": "1",
        "PYTHONPATH": os.pathsep.join((str(hooks), str(repo))),
    })

    proc = subprocess.run(
        [
            sys.executable, str(repo / "scripts" / "benchmark_parallel.py"),
            "--instance-ids-file", str(ids),
            "--variants", "frontier_only", "scout",
            "--budget", "2", "--per-task", "0.25",
            "--parallel-workers", "2",
            "--parallel-docker-workers", "2",
        ],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    assert "=== Serial benchmark arm ===" in proc.stdout
    assert "=== Parallel benchmark arm ===" in proc.stdout
    assert "Parallel Docker scoring parity replay" in proc.stdout

    reports = list(results.glob("benchmark_*.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text())
    assert report["serial_cells"] == 4
    assert report["parallel_cells"] == 4
    assert report["common_cells"] == 4
    assert report["serial_duplicate_cells"] == 0
    assert report["parallel_duplicate_cells"] == 0
    assert report["missing_from_parallel"] == []
    assert report["missing_from_serial"] == []
    assert report["serial_spend_usd"] == pytest.approx(0.2)
    assert report["parallel_spend_usd"] == pytest.approx(0.2)
    assert report["identical_patch_scoring_parity"]["passed"]
    assert report["identical_patch_scoring_parity"]["cells"] == 4
    assert report["parallel_scheduler_telemetry"]["peak_agent_cells"] == 2
    for run_id in (report["serial_run"], report["parallel_run"]):
        run_dir = results / "runs" / run_id
        assert (run_dir / "manifest.json").exists()
        assert len(json.loads((run_dir / "summary.json").read_text())) == 4
        assert (run_dir / "timing_summary.json").exists()


def test_interruption_resume_lifecycle_is_exactly_once(tmp_path, monkeypatch):
    tasks = _tasks(5)
    store, log = _store(tmp_path, tasks)
    compatibility = tmp_path / "compatibility-results"
    monkeypatch.setattr("eval.run_eval._OUT", compatibility)
    calls: Counter[str] = Counter()
    calls_lock = threading.Lock()
    waits = 0

    def fake_run(variant, task, cfg):
        with calls_lock:
            calls[task["instance_id"]] += 1
        time.sleep(0.03)
        return _result(variant, task["instance_id"])

    def interrupt_once(*args, **kwargs):
        nonlocal waits
        waits += 1
        if waits == 1:
            raise KeyboardInterrupt
        return concurrent.futures.wait(*args, **kwargs)

    persist = lambda: _persist_aggregates(store, log)
    with (patch("eval.parallel.variants.run_variant", side_effect=fake_run),
          patch("eval.parallel.wait", side_effect=interrupt_once)):
        first = run_cells(
            store=store, tasks=tasks, workers=2, docker_workers=1,
            budget_usd=20, per_task_usd=1, no_judge=True,
            audit_log=log, on_persist=persist, progress_interval=0,
        )
    assert first.interrupted
    assert 0 < first.counts["completed"] < len(tasks)
    assert first.counts["pending"] + first.counts["completed"] == len(tasks)

    store = RunStore.load(store.manifest_path)
    persist = lambda: _persist_aggregates(store, log)
    with patch("eval.parallel.variants.run_variant", side_effect=fake_run):
        second = run_cells(
            store=store, tasks=tasks, workers=3, docker_workers=2,
            budget_usd=20, per_task_usd=1, no_judge=True,
            audit_log=log, on_persist=persist, progress_interval=0,
        )
    persist()
    assert not second.interrupted
    assert second.counts == {"completed": 5}
    assert store.spent_usd == pytest.approx(0.5)
    assert calls == Counter({task["instance_id"]: 1 for task in tasks})

    rows = json.loads((store.run_dir / "summary.json").read_text())
    cell_ids = [row["cell_id"] for row in rows]
    assert len(rows) == len(tasks)
    assert len(cell_ids) == len(set(cell_ids))
    assert all(row["cell_elapsed_seconds"] > 0 for row in rows)
    with (store.run_dir / "summary.csv").open(newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == len(tasks)
    audit_rows = [json.loads(line) for line in log.jsonl.read_text().splitlines()]
    assert len(audit_rows) == len(tasks)
    assert len({row["cell_id"] for row in audit_rows}) == len(tasks)
    timing = json.loads((store.run_dir / "timing_summary.json").read_text())
    assert timing["overall"]["cells"] == len(tasks)

    resumed_again = RunStore.load(store.manifest_path)
    with patch("eval.parallel.variants.run_variant", side_effect=fake_run):
        third = run_cells(
            store=resumed_again, tasks=tasks, workers=4, docker_workers=2,
            budget_usd=20, per_task_usd=1, no_judge=True,
            audit_log=log, on_persist=lambda: None, progress_interval=0,
        )
    assert third.completed_now == 0
    assert third.counts == {"completed": 5}
    assert calls == Counter({task["instance_id"]: 1 for task in tasks})


@pytest.mark.parametrize("error", [
    "AuthenticationError: invalid x-api-key",
    "PermissionDeniedError: workspace forbidden",
    "RateLimitError: retries exhausted",
    "APIConnectionError: connection refused",
    "APITimeoutError: request timed out",
])
def test_concurrent_systemic_provider_failures_release_every_cell(tmp_path, error):
    tasks = _tasks(6)
    store, log = _store(tmp_path, tasks)

    def fail(variant, task, cfg):
        result = _result(variant, task["instance_id"], cost=0.0)
        result.resolved = False
        result.error = error
        return result

    with patch("eval.parallel.variants.run_variant", side_effect=fail):
        summary = run_cells(
            store=store, tasks=tasks, workers=6, docker_workers=2,
            budget_usd=20, per_task_usd=1, no_judge=True,
            audit_log=log, on_persist=lambda: None, progress_interval=0,
        )
    assert summary.fatal_error == error
    assert summary.completed_now == 0
    assert summary.spent_usd == 0
    assert summary.counts == {"pending": len(tasks)}
    progress = json.loads((store.run_dir / "progress.json").read_text())
    assert progress["status"] == "failed"
    assert progress["budget"]["reserved_usd"] == 0


def _git_source(path: Path) -> tuple[Path, str]:
    path.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    (path / "value.py").write_text("VALUE = 0\n")
    subprocess.run(["git", "add", "value.py"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=path, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True,
        capture_output=True, text=True).stdout.strip()
    return path, commit


def test_registered_runtime_inherits_full_scheduler_contract(tmp_path):
    name = "integration-runtime"
    source, commit = _git_source(tmp_path / "source")
    task = {
        "instance_id": "registered-runtime-task",
        "problem_statement": "change VALUE",
        "template_dir": str(source), "_dir": str(source),
        "_base_commit": commit, "in_place": True,
        "test_cmd": "true", "backend": "local",
    }
    configurations = [
        {"name": "short", "run_config": {**RUN_CONFIG, "max_steps": 1}},
        {"name": "deep", "run_config": {**RUN_CONFIG, "max_steps": 3}},
    ]
    store, log = _store(
        tmp_path / "run", [task], variant=name,
        configurations=configurations, repetitions=2, base_seed=7)
    lock = threading.Lock()
    active = 0
    peak = 0
    observations: list[dict] = []

    class ContractRuntime:
        name = "contract-runtime"

        def preflight(self):
            return True, "test runtime ready"

        def run(self, task, ws, *, model, ledger, cfg):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                observations.append({
                    "root": str(ws.root), "seed": task["seed"],
                    "max_steps": cfg.max_steps,
                    "budget_usd": cfg.budget_usd,
                    "per_task_usd": cfg.per_task_usd,
                })
                ws.str_replace(
                    "value.py", "VALUE = 0",
                    f"VALUE = {task['seed'] + cfg.max_steps}")
                ledger.record_tokens(
                    "main", model, input_tokens=10, output_tokens=2)
                time.sleep(0.04)
                return RuntimeResult(
                    summary="contract complete", steps=cfg.max_steps,
                    finished=True, trace=[])
            finally:
                with lock:
                    active -= 1

    variants.register_variant(
        name, VariantSpec(
            runtime_factory=ContractRuntime,
            main_model=config.MODEL_MAIN))
    try:
        summary = run_cells(
            store=store, tasks=[task], workers=4, docker_workers=1,
            budget_usd=20, per_task_usd=0.5, no_judge=True,
            audit_log=log, on_persist=lambda: None, progress_interval=0,
        )
        assert name in variants.registered_variants()
        assert summary.counts == {"completed": 4}
        assert summary.telemetry["peak_agent_cells"] == 4
        assert peak >= 2
        assert len(observations) == 4
        assert {item["seed"] for item in observations} == {7, 8}
        assert {item["max_steps"] for item in observations} == {1, 3}
        assert {item["budget_usd"] for item in observations} == {0.5}
        assert {item["per_task_usd"] for item in observations} == {0.5}
        roots = [item["root"] for item in observations]
        assert len(set(roots)) == 4
        assert all(Path(root) != source for root in roots)
        assert all(not Path(root).exists() for root in roots)
        assert (source / "value.py").read_text() == "VALUE = 0\n"

        rows = [payload["row"] for payload in store.completed_payloads()]
        assert {row["config"] for row in rows} == {"short", "deep"}
        assert {row["repetition"] for row in rows} == {0, 1}
        assert all(row["variant"] == name for row in rows)
        assert all(row["runtime_wall_seconds"] > 0 for row in rows)
        assert all(row["cell_cost_usd"] > 0 for row in rows)

        resumed = RunStore.load(store.manifest_path)
        before = len(observations)
        resumed_summary = run_cells(
            store=resumed, tasks=[task], workers=4, docker_workers=1,
            budget_usd=20, per_task_usd=0.5, no_judge=True,
            audit_log=log, on_persist=lambda: None, progress_interval=0,
        )
        assert resumed_summary.completed_now == 0
        assert len(observations) == before
    finally:
        variants._REGISTRY.pop(name, None)


def test_variant_registration_rejects_collisions_and_invalid_specs():
    spec = VariantSpec(runtime_factory=lambda: object(), main_model=config.MODEL_MAIN)
    with pytest.raises(ValueError, match="legacy"):
        variants.register_variant("frontier_only", spec)
    with pytest.raises(ValueError, match="no whitespace"):
        variants.register_variant("bad name", spec)
    with pytest.raises(TypeError, match="VariantSpec"):
        variants.register_variant("bad-spec", object())
    name = "registration-collision"
    variants.register_variant(name, spec)
    try:
        with pytest.raises(ValueError, match="already registered"):
            variants.register_variant(name, spec)
        variants.register_variant(name, spec, replace=True)
    finally:
        variants._REGISTRY.pop(name, None)

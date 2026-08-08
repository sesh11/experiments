# Parallel evaluation completion audit

**Audit date:** 2026-08-08
**Implementation status:** complete locally; representative EC2 benchmark pending
**Local host:** arm64, Docker unavailable, `ANTHROPIC_API_KEY` unavailable

This audit maps every goal requirement to authoritative implementation and test
evidence. The goal must not be marked complete until the final EC2 gate passes.

| Requirement | Evidence | Status |
|---|---|---|
| Cell = instance × variant × normalized configuration × repetition/seed | `eval/run_state.py::CellSpec`, `build_cells`; identity determinism/uniqueness tests | Proven locally |
| Scheduler remains independent of runtime names | `eval/parallel.py` calls only `orchestrator.variants.run_variant`; public `register_variant()` contract test crosses configs/repetitions and proves parallelism, budgets, isolation, timing, and resume | Proven locally |
| `--workers auto\|N` and `--docker-workers auto\|N` | CLI parser plus `eval/resources.py::choose_workers`; auto/override tests | Proven locally |
| `1/1` is fully serial, including no agent/scorer overlap | `serial_mode` dispatch gate; explicit overlap test | Proven locally |
| CPU/RAM/disk-aware sizing and dispatch backpressure | resource snapshot, joint pool sizing, headroom checks, pressure telemetry | Proven locally; EC2 tuning pending |
| Separate bounded agent and Docker stages | two executors and bounded submissions; peak-scoring test | Proven locally |
| Private workspace per concurrent SWE-bench cell | `eval/isolation.py` local shared-object clones; same-instance concurrent edit test | Proven locally |
| Unique logs, results, temporary predictions, Docker reports/containers | stable cell IDs, per-cell artifact roots, cell-specific harness run/model names, same-instance Docker lock | Proven locally |
| Preserve scorer and audit behavior | existing `score_patch` and audit evidence reused; result compatibility fields retained; identical-patch replay implemented | Proven in unit/integration tests; real Docker parity pending |
| Deterministic final summaries despite completion order | coordinator-only persistence and index sorting; out-of-order aggregate test | Proven locally |
| Strict global reservation including agent and judge | `_Reservations`, full cell admission, conservative judge ceiling, pre-call ledger guards | Proven locally |
| Preserve per-cell cap for Fusion, Stirrup, and pi | pre-call bounds for Fusion/Stirrup; pi live event metering and stop-before-next-turn test | Proven locally |
| Atomic manifest and result persistence | same-directory temp + fsync + `os.replace`; recovery tests | Proven locally |
| Resume skips completed and reruns incomplete cells | `RunStore.load`, stable results, and full interrupt → drain → resume → aggregate → second-resume lifecycle test with exactly-once spend/cells | Proven locally |
| Reject incompatible resume settings | manifest-driven resume validation in `eval/run_eval.py`; mismatch test | Proven locally |
| Prevent simultaneous duplicate coordinators | advisory run lease and contention test | Proven locally |
| Failure isolation and graceful interruption | per-cell exception packaging, dispatch stop/drain, resumable pending states; concurrent systemic provider failures release reservations and remain pending | Proven locally |
| Non-interleaved useful progress | coordinator-only terminal/audit writes; Docker output captured per cell | Proven by design/tests |
| Configuration matrices and repetitions | `--configs-file`, `--repetitions`, `--seed`; inheritance/validation tests | Proven locally |
| Documentation | `docs/PARALLEL_EVAL.md`, README, deployment guide | Complete |
| Automated coverage | 33 tests across unit and integration suites; no-LLM workspace smoke | Passing |
| Report wall time, effective concurrency, resources, bottlenecks | manifest execution telemetry plus a real benchmark subprocess test covering nested serial/parallel runs, parity replay, and final report | Proven locally with fake external boundaries |
| Serial/parallel representative SWE-bench speedup and tuning | manual `parallel-eval-benchmark` workflow and `scripts/benchmark_parallel.py` are ready | **Missing external run** |
| No lost/duplicate cells, budget overrun, artifact collision, or scoring difference on target EC2 | benchmark report includes explicit checks and exact-patch scoring replay | **Missing external run** |

## Verified commands

```bash
env PATH="$PWD/.venv/bin:$PATH" \
  .venv/bin/python -m pytest -q --ignore=eval/tasks_data
# 33 passed

env PATH="$PWD/.venv/bin:$PATH" \
  .venv/bin/python scripts/smoke_workspace.py
# SMOKE OK

.venv/bin/python -m compileall -q eval fusion runtimes orchestrator scripts tests
git diff --check
# both clean
```

An end-to-end, no-provider-spend benchmark integration test launches the script
as a real subprocess, which launches both evaluator arms as nested subprocesses,
then performs identical-patch scoring parity and validates the final report.
Provider and Docker boundaries are fake; orchestration, persistence, timing,
resume, aggregation, and report code are real.

## Final EC2 gate

On the registered x86_64 EC2 runner, run **Actions →
parallel-eval-benchmark → Run workflow**. Choose a per-arm budget large enough
to complete every selected cell. The uploaded benchmark JSON must show:

- `scheduler_speedup > 1` by a meaningful margin;
- zero `serial_duplicate_cells` and `parallel_duplicate_cells`;
- empty `missing_from_parallel` and `missing_from_serial`;
- total spend in each arm at or below its cap;
- `identical_patch_scoring_parity.passed == true` with no differences;
- no resource-pressure pauses or artifact/workspace errors;
- effective concurrency above 1 in the parallel arm when enough cells exist.

Use its resource low-water marks and `parallel_bottleneck` classification to
tune the auto-sizing constants if the first run shows memory/disk pressure or a
persistently underfilled agent/scoring pool. Re-run after any tuning change.

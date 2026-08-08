# Parallel and resumable evaluation

The evaluation unit is a **cell**:

```
(SWE-bench instance, variant, configuration, repetition/seed)
```

One variant/configuration over 10 instances is therefore 10 cells. Cells pass
through two independent bounded stages: agent execution and scoring. The
scheduler is unaware of runtime names; it calls the variant registry for every
cell, so newly registered runtimes and architecture patterns inherit parallel
execution automatically.

## Typical EC2 run

Automatic sizing is the default:

```bash
python -m eval.run_eval --source swebench --backend docker \
  --instance-ids-file results/verified_instances.txt \
  --variants frontier_only scout \
  --budget 25 --per-task 2.5 --max-steps 30 --no-judge
```

Equivalent explicit controls:

```bash
python -m eval.run_eval ... --workers 6 --docker-workers 2
```

Use `--workers 1 --docker-workers 1` for the serial compatibility path.

`--workers auto` budgets approximately one vCPU and 2 GiB per agent cell after
reserving resources for scoring. `--docker-workers auto` admits approximately
two vCPUs, 8 GiB RAM, and 8 GiB free disk per scorer (steady-state accounting
uses 6 GiB). Both leave OS/daemon
headroom. A dispatch-time pressure check pauses new cells below 1 GiB available
RAM or 5 GiB free disk. Explicit counts always win, so overly large explicit
values can overload the host.

Agent calls are often network-bound, while Docker scoring is CPU/RAM/disk-heavy;
separate pools let model work continue without launching an unsafe number of
containers. `manifest.json` records the detected resources and chosen counts.
Each invocation also records agent/scoring work-seconds, effective and peak
concurrency, minimum available RAM/disk, maximum one-minute load, and resource
pressure pauses under `resources.execution_history`.

## Live timing and performance capture

Elapsed-time heartbeats print every 10 seconds by default:

```text
⏱ 04:12 elapsed | 4/10 done | agent 4 | scoring 2 | queued 4 | 57.1 cells/hour | ETA 06:18
```

Change the cadence with `--progress-interval 5`, or pass `0` to silence only
the periodic console heartbeat. Timing artifacts are still written when cell
state changes. During a run, inspect the atomic live snapshot from another SSH
session:

```bash
watch -n 2 'jq "{status, invocation, cells, active, budget}" results/runs/<run-id>/progress.json'
```

The run directory contains three complementary views:

- `progress.json`: current overall elapsed time, throughput, ETA, cell counts,
  active cell elapsed times, and live budget state.
- `summary.csv` / `summary.json`: raw timing columns for every completed cell,
  including wall-clock timestamps.
- `timing_summary.json`: overall and per-variant/configuration count, total,
  mean, p50, p95, and maximum for every phase.

Per-cell phases are `queue_wait_seconds`, `workspace_setup_seconds`,
`runtime_wall_seconds`, `workspace_cleanup_seconds`, `agent_wall_seconds`,
`scoring_queue_wait_seconds`, `docker_lock_wait_seconds`,
`docker_scoring_wall_seconds`, `judge_wall_seconds`, and
`cell_elapsed_seconds`. `cell_wall_seconds` remains the compatibility measure
of active agent plus scoring work; `cell_elapsed_seconds` is the true latency
from dispatch through completed scoring.

Use the breakdown to tune the right constraint:

- High queue wait with saturated agent workers: increase `--workers` if EC2
  memory and provider limits allow it.
- High workspace setup/cleanup: optimize checkout reuse or disk performance.
- High runtime time: compare harness/model/configuration choices; more workers
  improve throughput but do not shorten one cell.
- High scoring queue wait with saturated scoring workers: increase
  `--docker-workers` if RAM, CPU, and disk headroom allow it.
- High Docker lock wait: repeated cells for the same SWE-bench instance are
  intentionally serialized to avoid harness image races; adding scorers will
  not remove that per-instance lock.
- High Docker scoring time: use warm image caches or a stronger CPU/disk host.
- High judge time: disable the optional judge when it is not part of the
  experiment, or account for it when sizing the scoring pool.

`active_wall_seconds` in the timing summary sums scheduler invocation time and
excludes pauses between resumptions. `calendar_elapsed_seconds` measures from
run creation and includes that downtime. Phase totals are work-seconds summed
across cells, so under parallel execution they can exceed overall wall time.

## Budget behavior

`--budget` is one global cap for the invocation and all resumptions. Before a
cell starts, the coordinator reserves its full `--per-task` agent allowance plus
a conservative upper bound for its optional judge call. A cell is not started
unless that complete reservation fits. Unused dollars are released on
completion and can fund later cells.

This can leave a small unused remainder; the scheduler does not launch a
partially funded cell. Built-in and Stirrup model calls check the worst-case
request cost before billing. pi is metered from its live JSON event stream and
is stopped before another full-context request could cross the cell cap.

Docker scoring itself does not call a paid model. Judge cost is recorded as
`judge_cost_usd`; `cell_cost_usd` is agent plus judge cost. The manifest's spend
is the sum of completed cell costs.

## Resume and interruption

Every run prints a generated run id and resume command:

```bash
python -m eval.run_eval --resume 20260807_153012_a1b2c3
```

The manifest and each cell result are written with atomic replacement. On
resume, completed cells are skipped, interrupted `running`/`scoring` cells are
returned to `pending`, and budget-skipped cells are reconsidered (use a larger
`--budget` to continue them). Experiment-defining settings come from the
manifest. Supplying conflicting task, variant, config, repetition, seed,
backend, judge, or per-cell settings is rejected; worker counts and the global
budget may be changed.

Ctrl-C stops new dispatch and drains in-flight cells so their spend and results
are recorded before the coordinator releases its run lease. Further Ctrl-C
presses do not discard already-metered work. An operating-system hard kill will
leave those cells pending so they are rerun on resume.

Artifacts live under:

```
results/runs/<run-id>/
  manifest.json
  progress.json
  summary.json
  summary.csv
  timing_summary.json
  runs.jsonl
  <cell-id>.log
  cells/<cell-id>/result.json
  cells/<cell-id>/scoring/...     # isolated Docker harness output
```

Root-level `results/summary.json` and `.csv` remain compatibility copies for
the existing report command. Summaries and `runs.jsonl` are always rebuilt in
cell-index order, independent of completion order.

## Configuration matrices and repetitions

Use `--configs-file` to cross every task/variant with multiple `RunConfig`
settings. Missing fields inherit current defaults:

```json
{
  "configurations": [
    {
      "name": "short",
      "run_config": {"max_steps": 20, "scout_max_steps": 6}
    },
    {
      "name": "deep",
      "run_config": {"max_steps": 40, "scout_max_steps": 12}
    }
  ]
}
```

```bash
python -m eval.run_eval --source swebench --backend docker \
  --configs-file configs.json --repetitions 2 --seed 100 \
  --variants frontier_only scout --limit 10 --budget 80 --no-judge
```

Repetition `n` records seed `--seed + n`. The seed is included in cell identity
and exposed to runtimes through `task["seed"]`; provider determinism still
depends on the runtime/model API.

## Real serial-versus-parallel benchmark

On the EC2 runner, benchmark the same gold-verified cells in both modes:

```bash
python scripts/benchmark_parallel.py \
  --instance-ids-file results/verified_instances.txt \
  --variants frontier_only scout --budget 25
```

The same operation is available as the manual GitHub Actions workflow
`parallel-eval-benchmark`; it first gold-verifies and pins the selected slice,
then uploads both run directories and the report. Its budget input is **per
arm**, so total possible model spend is twice the entered amount.

This is intentionally opt-in because it performs two paid evaluation runs. It
writes `results/benchmark_<timestamp>.json` with wall time, speedup, cell counts,
spend, resolutions, duplicate/missing cell checks, both scheduler-only and
end-to-end speedup, effective concurrency,
resource low-water marks, setup overhead, a likely bottleneck classification,
and any serial/parallel scoring differences. By default it also replays the
serial arm's exact patches through parallel Docker scoring (no model spend) to
separate scoring-pipeline parity from model nondeterminism; use
`--skip-scoring-parity` only when that extra Docker pass is undesirable. Compare warm-cache runs when
measuring scheduler speed; first-time image pulls and repository environment
builds otherwise dominate both arms.

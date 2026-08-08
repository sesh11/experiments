# Deployment & GitHub automation

Two workflows ship in `.github/workflows/`:

| Workflow | Trigger | Runs on | Cost | What it does |
|----------|---------|---------|------|--------------|
| `ci.yml`   | every push / PR | GitHub-hosted | free | byte-compile + no-LLM smoke test |
| `eval.yml` | manual button   | **your EC2** (self-hosted) | LLM + EC2 time | the real gold check / agent comparison |
| `benchmark.yml` | manual button | **your EC2** (self-hosted) | **2× entered per-arm cap** | gold-verifies one slice, then compares serial and auto-parallel execution |

`ci` needs no setup — it runs automatically and keeps the harness honest.
`eval` needs the three one-time steps below.

---

## Why the eval runs on your own EC2 (self-hosted runner)

SWE-bench only scores reliably on **x86_64 Linux + Docker**. GitHub's hosted
runners are x86_64 but ship with little free disk (~14 GB), and SWE-bench images
are large — a multi-instance run fills the disk and dies. Your EC2 box has the
disk, caches images across runs, and has no 6-hour job cap. So we point the
`eval` workflow at it.

(If you only ever want a 1–2 instance confirmation, a hosted runner *can* work
with an added disk-cleanup step — but the EC2 path is the robust one.)

---

## One-time setup

### 1. Add your API key as a repo secret
Repo → **Settings → Secrets and variables → Actions → New repository secret**
- Name: `ANTHROPIC_API_KEY`
- Value: your `sk-ant-...` key

The workflow injects it as an env var; it is never written to the repo.

### 2. Register the EC2 box as a self-hosted runner
First run the EC2 bootstrap once (installs Docker + Python and adds you to the
`docker` group):
```bash
sudo bash scripts/ec2_bootstrap.sh && exit   # re-login so docker perms apply
```
Then, in the repo: **Settings → Actions → Runners → New self-hosted runner →
Linux / x64**, and run the commands GitHub shows you on the EC2 instance. They
look like:
```bash
mkdir actions-runner && cd actions-runner
curl -o actions-runner.tar.gz -L <url-github-gives-you>
tar xzf actions-runner.tar.gz
./config.sh --url https://github.com/<you>/experiments --token <token-github-gives-you>
```
Install it as a service so it survives reboots and stays online:
```bash
sudo ./svc.sh install
sudo ./svc.sh start
```
The runner will register with labels `self-hosted`, `Linux`, `X64` — which is
what `eval.yml` targets. Verify it shows **Idle** under Settings → Actions → Runners.

> Keep the runner's checkout and the harness on the **same EC2 disk** that has
> your 100–160 GB volume, so Docker images and the runner workspace share it.

### 3. Get the workflow onto the default branch
GitHub only shows the **Run workflow** button for `workflow_dispatch` workflows
that exist on the repo's **default branch**. Merge this branch (or cherry-pick
`.github/workflows/eval.yml`) to `main`, then the button appears under the
**Actions → eval** tab.

---

## Running it
Actions → **eval** → **Run workflow**, choose:
- `limit` — how many instances (start with 5)
- `budget` — `0` for the free gold self-test; e.g. `6` for the real comparison

When it finishes:
- the **Summary** shows the per-run table (for budget > 0),
- **Artifacts** has `eval-results-<run_id>.zip` with `results/summary.csv`,
  `results/runs/<stamp>/` (including `progress.json`, raw per-cell timings, and
  `timing_summary.json`), plus the harness reports.

The eval driver automatically sizes separate agent and Docker-scoring pools for
the current EC2 CPU, available RAM, and disk. Runs persist an atomic manifest;
if a job or instance is interrupted, copy the printed command
`python -m eval.run_eval --resume <run-id>` and run it from the same checkout and
results volume. See [PARALLEL_EVAL.md](PARALLEL_EVAL.md) for manual worker
overrides, live elapsed-time monitoring, phase-based tuning, configuration
matrices, and the paid serial/parallel benchmark.

For an archived performance comparison, use **Actions →
parallel-eval-benchmark → Run workflow**. `budget_per_arm` applies separately to
the serial and parallel arms, so the maximum agent spend is twice that input.
The workflow uploads both run manifests/log sets and the JSON report, and places
the report in the GitHub Actions job summary.

## Cost hygiene
- The self-hosted runner only consumes EC2 time while a job runs, but the
  **instance bills whenever it's on**. Stop it when idle; the runner reconnects
  automatically on start.
- `eval` never runs on push — it is manual-only (`workflow_dispatch`), so you
  never trigger LLM/EC2 spend by accident.

# Opinionated Sidekicks

A small, cost-conscious harness for studying **Devin-Fusion-style main + sidekick
agents**: a frontier "main" agent and a cheap "sidekick" agent that each keep their
own warm context, so delegation passes compact briefs instead of swapping models on
one transcript.

## Status: MVP

This is an **MVP build** — a working architecture and evaluation harness, validated on
self-authored toy tasks. Read this section before reading the numbers.

- **What it is.** A runnable main+sidekick harness implementing **Idea A (read-only
  Scout)**, with per-role token/cost accounting, a hard budget guard, a bundled native
  task set plus a SWE-bench Verified loader, and a "would-you-merge" quality judge.
- **What the MVP proves (verified live).** The machinery works end-to-end: two agents
  with separate warm-cache contexts, the Scout delegation loop, the cost split, and the
  budget guard all run against real models. All three variants resolve the native tasks;
  total spend for the live validation was **~$0.30**.
- **What it does NOT yet show.** The Fusion *cost win*. The two native tasks
  (`eval/tasks_data/slugify`, `eval/tasks_data/median`) are **deliberately trivial
  bugs I hand-wrote** to exercise the loop cheaply — not a real benchmark. Because the
  files are tiny, reading them directly is nearly free, so on this set the Scout is
  *more* expensive than the frontier baseline (delegation overhead with no offsetting
  saving). The Scout only pays off when file-reading tokens dominate the frontier
  context — i.e. real repositories.
- **Next (to actually test the thesis).** Run the **SWE-bench Verified slice**
  (`--source swebench`) where a frontier agent burns large token counts reading files,
  then layer on **Idea C (confidence-gated routing)**.

So "MVP" here means *proven machinery on toy tasks*, **not** *proven cost savings*.

This first build implements **Idea A — the read-only "Scout" sidekick**: the main
agent (Sonnet 5) never reads files directly. Instead it asks a Haiku Scout to explore
and return a compact, cited `file:line` map. The frontier model pays frontier-token
rates only for the distilled map, not for thousands of tokens of raw file content —
the biggest token sink in a coding agent.

> Idea C (confidence-gated routing) and Idea B (self-verifying task sidekicks) are
> deferred; the harness leaves seams for them (see `fusion/policies.py`).

## What it measures

We can't run FrontierCode (Cognition withholds the tasks). So we validate on a public
proxy — a bundled native mini-set by default, or a SWE-bench Verified slice — and add a
lightweight LLM **"would you merge this?"** judge to echo FrontierCode's quality axis.

The deliverable is a **cost-vs-quality** comparison across three variants:

| variant          | main        | sidekick    | idea under test |
|------------------|-------------|-------------|-----------------|
| `frontier_only`  | Sonnet 5    | –           | high-cost baseline |
| `sidekick_only`  | –           | Haiku 4.5   | low-cost baseline |
| `scout`          | Sonnet 5    | Haiku 4.5   | **A — read-only Scout** |

Success = `scout` sits at lower cost than `frontier_only` for comparable quality, and
clearly above `sidekick_only` on quality.

## Layout

```
fusion/
  config.py        model IDs + pinned pricing + budget guard settings
  llm.py           Anthropic wrapper: prompt caching + per-role token/cost ledger
  workspace.py     per-task working copy: read/search/edit/run-tests + unified diff
  tools.py         tool schemas + dispatcher over a Workspace
  agent.py         generic tool-use loop; one instance = one warm-cache context
  orchestrator.py  solver/scout prompts + Scout delegation wiring
  policies.py      the three variants (frontier_only / sidekick_only / scout)
eval/
  tasks.py         native mini-set loader + optional SWE-bench Verified slice
  judge.py         "would you merge?" rubric (0-100 + would_merge)
  run_eval.py      driver: variants × tasks, global + per-task budget caps
  report.py        results/pareto.png + per-variant table
scripts/
  smoke_workspace.py   no-LLM check of the workspace/test loop
```

## Setup

```bash
pip install -r requirements.txt        # anthropic, datasets, matplotlib, (pytest)
export ANTHROPIC_API_KEY=sk-ant-...     # REQUIRED for any run that calls the API
```

> **Prerequisite:** the eval run needs an API key. Building/inspecting the harness and
> the no-LLM smoke test do not.

## Run

```bash
# 0) No-LLM: prove the sandbox loop works (free)
python scripts/smoke_workspace.py

# 1) Tier-0 smoke (~$1-2): a couple of tasks, two variants
python -m eval.run_eval --source native --variants frontier_only scout --budget 3

# 2) Full native run, all three variants
python -m eval.run_eval --source native --budget 5

# 3) Large-context test on real repos — clone locally and run one script
#    (Cloud/Web sandbox can't: HuggingFace + arbitrary GitHub are network-blocked)
export ANTHROPIC_API_KEY=sk-ant-...
./run_swebench.sh            # 6 SWE-bench Verified instances, $15 cap
./run_swebench.sh 8 20       # 8 instances, $20 cap

# Report
python -m eval.report          # -> results/pareto.png + table
```

Budget is enforced two ways: a **global** `--budget` cap across the whole run and a
**per-task** cap (`--per-task`, default $3). When the budget is exhausted the run stops
and writes whatever it has to `results/`.

## Pricing (pinned, standard rates per 1M tokens)

| model            | input | output | cache write (5m) | cache read |
|------------------|------:|-------:|-----------------:|-----------:|
| `claude-sonnet-5`  | $3.00 | $15.00 | $3.75 | $0.30 |
| `claude-haiku-4-5` | $1.00 |  $5.00 | $1.25 | $0.10 |

See `fusion/config.py` to adjust models, budgets, or thinking settings.

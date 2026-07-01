# Opinionated Sidekicks

A small, cost-conscious harness for studying **Devin-Fusion-style main + sidekick
agents**: a frontier "main" agent and a cheap "sidekick" agent that each keep their
own warm context, so delegation passes compact briefs instead of swapping models on
one transcript.

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

# 3) Scale up to a SWE-bench Verified slice (needs network + git)
python -m eval.run_eval --source swebench --limit 15 --budget 25

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

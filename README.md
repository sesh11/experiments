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

The deliverable is a **cost-vs-quality** comparison across the orchestration variants:

| variant          | plans / locates | authors edit | idea under test |
|------------------|-----------------|--------------|-----------------|
| `frontier_only`  | Sonnet 5        | Sonnet 5     | high-cost baseline |
| `sidekick_only`  | Haiku 4.5       | Haiku 4.5    | low-cost baseline |
| `scout`          | Haiku 4.5 (reads) | Sonnet 5   | **A — read-only Scout** (Sonnet delegates reads) |
| `inverted`       | Haiku 4.5       | Sonnet 5     | **Inverted** — Haiku plans+locates, Sonnet authors from a brief |

Success for `scout` = lower cost than `frontier_only` at comparable quality, clearly above
`sidekick_only` on quality. `inverted` tests the complementary bet: push the token-heavy
exploration *and* the fix-planning onto Haiku, so Sonnet only pays to author the edit from a
compact brief (targeted `read_file` only, no repo-wide search). Toggle any variant with
`--variants <name>`.

## Layout

```
orchestrator/
  variants.py      variant registry + orchestration patterns; owns PolicyResult
                   and the shared diff/score/cleanup packaging
runtimes/
  base.py          AgentRuntime protocol: solve a task in a Workspace, record
                   usage in the shared Ledger, honor the budget guard
  fusion_rt.py     the in-repo fusion loop as a runtime (parity baseline)
  stirrup_rt.py    Stirrup (Artificial Analysis) embedded as a library
  pi_rt.py         pi (badlogic) driven as a subprocess in JSON mode
fusion/
  config.py        model IDs + pinned pricing + budget guard settings
  llm.py           Anthropic wrapper: prompt caching + per-role token/cost ledger
  workspace.py     per-task working copy: read/search/edit/run-tests + unified diff
  tools.py         tool schemas + dispatcher over a Workspace
  agent.py         generic tool-use loop; one instance = one warm-cache context
  orchestrator.py  solver/scout prompts + Scout delegation wiring
  policies.py      the fusion-loop patterns (frontier_only / sidekick_only / scout / inverted)
eval/
  tasks.py         native mini-set loader + optional SWE-bench Verified slice
  judge.py         "would you merge?" rubric (0-100 + would_merge)
  run_eval.py      driver: variants × tasks, global + per-task budget caps
  report.py        results/pareto.png + per-variant table
scripts/
  smoke_workspace.py   no-LLM check of the workspace/test loop
```

## Agent runtimes

The agent loop is pluggable. A *runtime* is one harness behind the
`AgentRuntime` protocol (`runtimes/base.py`); the orchestrator picks a runtime
and model per variant, and every runtime records usage into the same Ledger at
the same pinned pricing, so cost numbers stay comparable.

| variant            | harness                                    | notes |
|--------------------|--------------------------------------------|-------|
| `baseline-fusion`  | in-repo fusion loop                        | parity baseline; same loop as `frontier_only` |
| `baseline-stirrup` | [Stirrup](https://github.com/ArtificialAnalysis/Stirrup), embedded | `pip install -r requirements.txt` covers it |
| `baseline-pi`      | [pi](https://www.npmjs.com/package/@mariozechner/pi-coding-agent), subprocess | optional: `npm i -g @mariozechner/pi-coding-agent`; `PI_BIN` overrides discovery. Skipped with an error row if absent. |
| `baseline-openrouter` | Stirrup's litellm loop via [OpenRouter](https://openrouter.ai) | needs `OPENROUTER_API_KEY`; `OPENROUTER_MODEL` selects the model (default: Sonnet 5 through OpenRouter). Skipped with an error row if the key is absent. |

```bash
# Parity check: same tasks, same model, three harnesses
python -m eval.run_eval --source native \
  --variants baseline-fusion baseline-stirrup baseline-pi --no-judge --budget 3
```

Cost-comparability caveats:
* **Stirrup** reports no cache split, so all its input tokens are billed at the
  full input rate (conservative overestimate), and its litellm path does no
  prompt caching — expect ~2x fusion's cost on small tasks.
* **pi** self-reports a session cost, but from its bundled model registry,
  which may not know newer model ids. The ledger recomputes from token counts
  at pinned prices; pi's own figure is kept in
  `score_artifacts.runtime_extra.pi_reported_cost_usd` as a cross-check, and a
  warning is printed when the two deviate >10%.
* **pi** has no turn-limit flag; the wall-clock cap
  (`RunConfig.runtime_timeout_s`, default 1200s) is the in-flight guard and its
  usage is accounted post-hoc at session end.

## Setup

```bash
pip install -r requirements.txt        # anthropic, datasets, matplotlib, stirrup[litellm], (pytest)
export ANTHROPIC_API_KEY=sk-ant-...     # REQUIRED for any run that calls the API

# optional, only for the baseline-pi variant:
npm i -g @mariozechner/pi-coding-agent

# optional, only for the baseline-openrouter variant:
export OPENROUTER_API_KEY=sk-or-...      # https://openrouter.ai/keys
export OPENROUTER_MODEL=anthropic/claude-sonnet-5   # any OpenRouter model slug
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

# 3) CONFIRM SCORING FIRST (cheap) — before spending on a big run, prove the
#    scoring pipeline is correct and see whether you even need a better harness.
#    Phase 1 is FREE (applies each gold solution patch, checks the scorer marks
#    it resolved — $0 LLM). Phase 2 is a small, hard-capped real agent run.
./confirm_scoring.sh          # 5 instances, $6 cap on the agent phase
./confirm_scoring.sh 5 0      # phase 1 only (free): just verify scoring

#    Or run the free scoring self-test directly:
python -m eval.selftest_scoring --limit 5

# 4) THE FALSIFYING RUN — frontier_only vs scout on real SWE-bench Verified
#    tasks, with real FAIL_TO_PASS scoring. Clone locally and run one script
#    (Cloud/Web sandbox can't: HuggingFace + arbitrary GitHub are network-blocked).
export ANTHROPIC_API_KEY=sk-ant-...
./run_swebench.sh            # 15 instances, $25 cap
./run_swebench.sh 10 15      # 10 instances, $15 cap

# Report
python -m eval.report          # -> results/pareto.png + table
```

### Running on AWS EC2 (fast, trustworthy Docker scoring)

SWE-bench images are **x86_64**. On Apple Silicon scoring runs under emulation —
it works, but it's slow; an Intel/AMD box is the fast path for anything bigger
than a few instances. The whole stack (orchestrator + all three runtimes) runs
bare on the box; Docker is only used by the scoring step.

**Launch an instance:**
- **AMI:** Ubuntu 22.04 or 24.04 (x86_64).
- **Instance type:** an **Intel/AMD** type — `c6i.2xlarge` or `m6i.2xlarge` (8 vCPU,
  16–32 GB). **Not** a Graviton (`*g.*`) type — those are ARM, same emulation
  problem as the Mac (the bootstrap refuses to run there).
- **Storage:** root EBS **100–160 GB** (Docker images are large; the 8 GB default
  fills up fast).
- **Security group / network:** default outbound is fine — it needs to reach
  Docker Hub, HuggingFace, npm, and the Anthropic API.

**Set it up and run:**
```bash
# on the instance, after cloning this repo and checking out your branch:
sudo bash scripts/ec2_bootstrap.sh      # docker + python venv + node/pi, one time
exit                                     # re-login so the docker group applies
ssh ...                                  # reconnect
docker run --rm hello-world              # sanity check (no sudo needed)

cp .env.example .env && nano .env        # paste ANTHROPIC_API_KEY
source .venv/bin/activate
python scripts/smoke_workspace.py                          # $0 sandbox check
python -m eval.selftest_scoring --backend docker --limit 3 # $0 gold check — 3/3
set -a; . ./.env; set +a                                   # load the key
python -m eval.run_eval --source swebench --backend docker \
  --limit 3 --variants baseline-fusion baseline-stirrup baseline-pi \
  --per-task 1.5 --budget 12 --max-steps 30 --no-judge     # ~$7 parity slice
```
The bootstrap refuses to run on ARM and warns on low disk, so you can't
accidentally recreate the Mac problem.

### Reading a run: the audit log

Every `(task, variant)` run prints a compact block and writes a full-detail log
to `results/runs/<stamp>/<task>__<variant>.log` (plus a machine-readable
`runs.jsonl`). The terminal block's **`why:`** line attributes each outcome so a
failure is never a guess — it distinguishes *agent* problems (no diff, edited a
test file, wrong/incomplete fix, broke PASS_TO_PASS) from *scoring/env* problems
(gold patch won't apply, pytest crashed collecting) from *budget* cutoffs. The
per-run `.log` has the complete tool-call trace (main **and** scout), the diff
with test-file flagging, and the **actual pytest output** from scoring. Start
with `why:`, open the `.log` when you need the evidence.

Budget is enforced two ways: a **global** `--budget` cap across the whole run and a
**per-task** cap (`--per-task`, default $3). When the budget is exhausted the run stops
and writes whatever it has to `results/`.

## Pricing (pinned, standard rates per 1M tokens)

| model            | input | output | cache write (5m) | cache read |
|------------------|------:|-------:|-----------------:|-----------:|
| `claude-sonnet-5`  | $3.00 | $15.00 | $3.75 | $0.30 |
| `claude-haiku-4-5` | $1.00 |  $5.00 | $1.25 | $0.10 |

See `fusion/config.py` to adjust models, budgets, or thinking settings.

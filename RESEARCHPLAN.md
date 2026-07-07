# Opinionated Sidekicks — Research Plan

**Date:** 2026-07-06
**Status:** Draft for review
**Scope:** Architecture arms to benchmark, routing-policy research program, harness decision.

---

## 1. Background and current state

The harness (main + sidekick agents, separate warm contexts, delegation via compact briefs) is
proven end-to-end on toy tasks. Three variants exist today: `frontier_only` (Sonnet 5),
`sidekick_only` (Haiku 4.5), and `scout` (Idea A, read-only Scout). Eval framework includes an
LLM merge-judge, per-task/global budgets, and EC2 automation for SWE-bench Verified scoring.

**Core finding so far:** the scout costs *more* than the frontier baseline on toy tasks.
Savings appear only past a critical mass of file-reading tokens. This "context footprint"
variable becomes first-class in the routing program (§4, E0).

**External validation of the thesis:** [Devin Fusion](https://cognition.com/blog/devin-fusion)
reports 35% cost reduction at frontier parity with the same dual-persistent-context pattern.
The open question is *which control flow between the two contexts is optimal, and when* —
that is this project's contribution surface.

Pricing basis (per 1M tokens): Sonnet 5 $3 in / $15 out / $0.30 cache-read;
Haiku 4.5 $1 in / $5 out / $0.10 cache-read.

---

## 2. Track A — Research-grounded arms

Five arms, each a different answer to "who drives, and what does the frontier model pay for?"

### A1. Plan–Delegate–Review (delegation by default)
Invert the scout: main agent only plans, resolves ambiguity, reviews checkpoints, and does the
final merge-gate. Sidekick executes everything (reads, edits, tests) in its own context; main
sees only briefs, diffs, test output.
**Grounding:** Devin Fusion's stated design ("main should take minimal actions... delegate and
monitor"); Plan-and-Act (arXiv:2503.09572); LLMCompiler (arXiv:2312.04511).
**Hypothesis:** frontier spend becomes a small near-constant per task; cost → `sidekick_only`,
quality → `frontier_only`.

### A2. MinionS Decomposition Firewall
Frontier never ingests repo content. It decomposes the task into parallel micro-questions over
chunks; sidekick answers them (structured JSON, line cites); frontier aggregates answers and
specifies the edit; sidekick applies it.
**Grounding:** MinionS (arXiv:2502.15964) — 5.7× cost cut retaining 97.9% of frontier quality;
naive chat variant only kept 87%, so structured parallel decomposition is load-bearing.
**Hypothesis:** frontier quality survives total blindness to raw code; lowest critical-mass
threshold of any arm since frontier file tokens are eliminated, not reduced.

### A3. Speculative Trajectories (draft–verify)
Sidekick drafts next action or whole patch; frontier verifies in batches (approve / edit /
reject-and-author). Speculative decoding lifted to agent actions.
**Grounding:** Speculative Actions (arXiv:2510.04371); DualSpec (arXiv:2603.07416); Google
speculative cascades.
**Economics:** shifts frontier spend from output tokens ($15/M) to input tokens ($3/M, $0.30
cached); generation moves to Haiku output ($5/M). Lossless by construction — frontier gates
every commit. Estimate acceptance rate first by replaying `frontier_only` logs against Haiku.

### A4. Sidekick-First Cascade (confidence-gated escalation)
Haiku owns the task. Sonnet is invoked only when a gate fires: low self-verification
confidence, N failed test runs/edits, budget fraction exceeded, plan divergence. Handoff is a
compact brief, not the transcript.
**Grounding:** AutoMix (NeurIPS 2024, arXiv:2310.12963) — few-shot self-verification + POMDP
router, >50% cost cut at parity; FrugalGPT (arXiv:2305.05176); C3PO (arXiv:2511.07396).
**Economics:** E[cost] = haiku_attempt + P(escalate) × sonnet_takeover. `sidekick_only`
baseline already measures the key unknown.

### A5. Best-of-N Sidekick Generation + Frontier Selection
Sample N candidate patches from Haiku in parallel (shared cached prefix), filter by running
repo + sidekick-generated tests, frontier selects/reviews survivors only.
**Grounding:** Large Language Monkeys (arXiv:2407.21787) — coverage scales log-linearly with
samples; CodeMonkeys (arXiv:2501.14723) — 57.4% SWE-bench Verified; their selection over an
ensemble beat the best individual member (66.2%). Agentless (arXiv:2407.01489) as pipeline
precedent.
**Honest math:** N=5 Haiku output ≈ $25/M vs Sonnet $15/M single-shot — the win is on
$/resolved via coverage, not per-attempt cost. Composable as a knob into A1, A2, A4.

### Scout v2 (upgrade the existing arm before rerunning)
Apply SWE-grep protocol constraints to the current scout: ≤4 turns, up to 8 parallel searches
per turn, mandatory file:line citations, hard output-token budget. Cognition observed >60% of
first-turn time on retrieval; most of the win is protocol, not the trained model.
**Grounding:** SWE-grep (cognition.com/blog/swe-grep); Morph WarpGrep replication.

---

## 3. Track B — Novel arms (no direct published precedent)

Five architectures designed from gaps in the literature. Each has a distinct cost mechanism
and a falsifier. All published sidekick work (Fusion, SWE-grep, MinionS, cascades) is
per-task and stateless; B1 and B4 deliberately break that assumption.

### B1. Persistent Repo Cortex — cross-task amortized, cache-resident repo digest
**Control flow:** Sidekick builds and incrementally maintains a compact multi-resolution repo
digest (file tree → module summaries → API signatures → hot-spot notes, with line cites).
The digest lives as the *cached prefix* of the frontier context. Frontier "knows" the repo at
cache-read prices ($0.30/M) from token one; digest construction is Haiku-priced and amortized
across every task in the same repo. After each task, sidekick applies incremental updates
(diff-driven) rather than rebuilding.
**Gap:** No published system amortizes sidekick work across tasks. SWE-bench Verified has many
tasks per repo (django alone has 100+), so amortization is directly measurable.
**Cost mechanism:** amortization — marginal cost of task k in a repo falls as k grows.
**Hypothesis:** $/resolved for task k declines with k; break-even vs `scout` within ~3 tasks
per repo.
**Falsifier:** digest staleness or lossy summaries drop resolve rate below scout — a digest is
not a substitute for targeted retrieval.
**Metrics:** marginal cost vs task index; break-even k; digest update cost; resolve-rate delta
vs scout on same tasks.
**Adjacent work (not precedent):** prompt-caching economics; RAPTOR hierarchical
summarization; Aider repo-map.

### B2. Author-Once, Iterate-Cheap — inverted speculation
**Control flow:** Frontier writes the plan and the *first* patch draft (peak-priced tokens
spent exactly once, where insight density is highest). Sidekick then owns the entire
red-green loop: run tests, fix mechanical failures (imports, syntax, fixtures, assertion
details), reformat, retry. Escalate to frontier only on *semantic* failure signals: same
assertion failing after M mechanical attempts, new regressions introduced, or test output the
sidekick classifies as "design-level."
**Gap:** Draft–verify literature puts the cheap model upstream (drafting) and the expensive
model downstream (verifying). But in coding agents most post-draft tokens are mechanical
iteration churn, not insight. No published arm assigns the iteration loop wholesale to the
cheap model.
**Cost mechanism:** iteration tokens (often the majority of a trajectory after the first
draft) move from Sonnet to Haiku prices.
**Hypothesis:** ≥40% cost reduction vs `frontier_only` at ≤2pt resolve-rate loss on tasks
with a working test harness.
**Falsifier:** failures are mostly semantic → high escalation rate → cost ≈ frontier_only +
overhead. (Measurable pre-build from `frontier_only` logs: classify iteration steps as
mechanical vs semantic.)
**Adjacent:** execution-cost analyses (arXiv:2606.26978); CodeMonkeys serial iteration.

### B3. Adversarial Sidekick Duo — debate as verification compression
**Control flow:** Two Haiku instances with opposed objectives. *Proposer* writes the patch.
*Attacker* tries to break it: writes tests intended to fail it, argues misreadings of the
issue, hunts edge cases. If the Attacker cannot break the patch → auto-accept, frontier never
invoked. If they disagree → frontier adjudicates reading *only the disagreement transcript*
(patch + attack + defense + test results), never the repo.
**Gap:** Debate is an alignment/scalable-oversight technique (Irving et al. 2018,
arXiv:1805.00899; OpenAI prover-verifier games, arXiv:2407.13692) — it has not been used as a
*cost* mechanism. The insight: the disagreement surface is tiny relative to the codebase, so
frontier input is compressed to the contested points, and the Attacker's tests provide free
mechanical verification signal.
**Cost mechanism:** verification compression — frontier reads KB of argument instead of MB of
repo, and only with probability P(disagree).
**Hypothesis:** ≥ scout-level resolve rate at lower cost, with frontier invoked on <50% of
tasks.
**Falsifier:** collusion/shared blind spots — same-model-family Proposer and Attacker miss the
same bug class. Measure the auto-accepted-but-wrong rate against frontier-reviewed baselines;
if it exceeds the judge's tolerance, the auto-accept path is unsound. (Mitigation to test:
cross-provider Attacker — ties into E6.)
**Metrics:** P(disagree); adjudication transcript size vs repo footprint;
auto-accept precision.

### B4. Apprentice Loop — playbook distillation with a learning cost curve
**Control flow:** Every frontier intervention (escalation, correction, review rejection) is
distilled by the sidekick into a repo-specific playbook entry ("tests run via `tox -e py311`";
"settings live in `conf/global_settings.py`"; "issues about X usually resolve in module Y").
The playbook is a cached prefix of the sidekick context. The router (§4) shifts traffic
toward the sidekick as its *measured* per-repo success rate rises.
**Gap:** Skill/memory libraries exist for capability (Voyager, arXiv:2305.16291; Agent
Workflow Memory, arXiv:2409.07429; Reflexion, arXiv:2303.11366) but none are framed as a
*priced-down delegation* mechanism, none measure a cost decay curve, and none couple memory
to a router. This makes the arm non-stationary — its quality at task k depends on tasks 1..k-1
— which is exactly what makes it interesting and what static routing benchmarks cannot see.
**Cost mechanism:** learning — escalation rate and frontier involvement decline with
cumulative tasks per repo; cost decays toward `sidekick_only`.
**Hypothesis:** negative slope of $/resolved vs task index within a repo; escalation rate at
task 10 < half of escalation rate at task 1.
**Falsifier:** playbook bloat or overfitting to early tasks → flat or worsening curve.
**Metrics:** cost/escalation slope vs task index; playbook size vs benefit; transfer test
(does a django playbook help on a fresh django task slice?).

### B5. Foveated Context — sidekick as attention controller for a bounded frontier viewport
**Control flow:** The frontier context is a *fixed-budget viewport* with resolution tiers:
repo tree (always present), module summaries, function signatures, and full source only for
the current focus region (e.g., ≤8k tokens). The frontier never reads files; it issues
attention requests ("show me the auth middleware in detail"). The sidekick services them with
promote/demote operations — demoted content collapses back to its summary. The prefix stays
stable for cache friendliness; churn is localized to the viewport region.
**Gap:** MemGPT (arXiv:2310.08560) does OS-style context paging, but self-managed by the same
model. Here paging is delegated to a cheap controller with an explicit token-budget objective,
making the paging policy itself a cost lever. Differs from the scout in kind: continuous
viewport management vs a one-shot map, and frontier context size is *bounded by construction*.
**Cost mechanism:** bounding — frontier per-step input cost is capped and therefore roughly
constant *regardless of repo size*. This is the only arm whose cost is designed to be
independent of repo scale, i.e., the critical-mass problem is engineered away rather than
outgrown.
**Hypothesis:** flat cost-vs-repo-size curve where all other arms grow; competitive resolve
rate on large-footprint tasks.
**Falsifier:** viewport thrash — frequent swaps invalidate the cache suffix and add controller
chatter; measure swap rate and cache-hit ratio. Also risk: frontier reasoning degrades without
persistent full-file grounding.
**Metrics:** frontier tokens/step (should be ~constant); cache-hit ratio; swap frequency;
resolve rate stratified by footprint.

### Why these five
Each isolates one distinct cost mechanism, so results compose into theory rather than a
leaderboard: **B1 amortization**, **B2 iteration-shifting**, **B3 verification compression**,
**B4 learning**, **B5 context bounding**. Track A arms mostly exploit a sixth mechanism
(delegating retrieval/execution). A routing policy is ultimately a map from task features to
the cheapest *mechanism* that suffices — running these arms is what learns that map.

B1 and B4 also produce something no published system or routing benchmark has: cost curves
over *sequences* of tasks in the same repo, which the EC2 pipeline is already set up to
generate (see E1b).

### Sequencing recommendation
- **First E1 campaign (alongside Track A):** B2 and B3 — cheapest to validate. B2 has a free
  pre-check (replay `frontier_only` logs and classify iteration steps as mechanical vs
  semantic before building anything); B3 is two Haiku prompts and a router branch.
- **Second campaign:** B1, B4, B5 — these need the E1b sequential-within-repo protocol (B1,
  B4) and the viewport/paging machinery (B5), so build them after the first campaign's
  instrumentation has been proven out.

---

## 4. Track C — Smart routing policy program

### 4.1 Objective
Cost-only routing is wrong. The unit is **$/resolved task** (quality enters the denominator:
E[cost] = attempt_cost / P(success)), optimized as *maximize resolve rate s.t. budget*,
reported as a cost–quality Pareto frontier. Headline metric, RouteLLM-style: "X% of frontier
quality at Y% of frontier cost" (RouteLLM, ICLR 2025, arXiv:2406.18665). Routing can also
*raise* quality above any single arm (RouterEval, arXiv:2503.10657; survey arXiv:2603.04445)
— evals must be able to detect that. Cost is not scalar: cache reads vs fresh output differ
30–50×, so the router must see cache state (Devin Fusion's persistent contexts exist for
exactly this reason).

### 4.2 Policy inputs
- **Pre-task:** issue length; repo size; *estimated context footprint*; test-harness presence
  and quality; language; historical per-arm performance on similar repos.
- **Mid-trajectory:** steps vs plan; consecutive failed edits/tests; sidekick
  self-verification confidence (AutoMix signal); budget fraction burned; retrieval hit rate.
- **System state:** cache warmth per context; provider latency; remaining global budget.
- **Decision space:** at task start, choose arm; per step, {continue, delegate, escalate,
  sample-more, stop}.

### 4.3 Experiment ladder
Design principle: **the arm-comparison matrix and the router training set are the same
dataset** (RouterBench methodology, arXiv:2403.12031 — precompute outcomes, evaluate routers
offline for free).

| ID | Experiment | Spend | Deliverable / decision gate |
|----|-----------|-------|------------------------------|
| E0 | Instrument + stratify. Define context footprint = tokens `frontier_only` spends on file reads. Select 50-task SWE-bench Verified slice stratified by footprint. Full per-step telemetry. | ~$0 | H1 null model: one-feature threshold router ("footprint < θ → frontier_only, else delegate") that every fancier router must beat. |
| E1 | Arm matrix: all arms × 50 tasks × 2 seeds on EC2. Paired per-task stats (McNemar), not aggregate rates. | ~$800–1,200 (first pass: 25×1 ≈ $250) | Pareto frontier of fixed arms + **oracle router** ceiling. Oracle-minus-best-fixed-arm gap = the entire value of routing. |
| E1b | **Sequential protocol** for non-stationary arms (B1, B4): tasks ordered within repo, no reset between tasks; measure cost vs task index. | included in E1 | Amortization/learning curves. Static matrices (RouterBench-style) structurally cannot see these effects — this protocol is itself a contribution. |
| E2 | Static pre-router: logistic/GBM P(resolve \| features, arm) + cost regressor; route to argmin expected $/resolved. Leave-one-task-out vs oracle, best fixed arm, H1. | ~$0 (offline) | **Gate: if within ~10–15% of oracle $/resolved — stop; do not build RL.** |
| E3 | Escalation gates (arm A4): AutoMix self-verification vs mechanical counters vs budget triggers vs combos. Precision/recall against oracle escalation points from E1 logs; rerun A4 with best gate. | ~$150–300 | Priced gate-error analysis (late escalation burns a doomed trajectory; eager escalation forfeits savings). |
| E4 | Cache economics: (a) dual persistent contexts + briefs, (b) single transcript with mid-session model toggling, (c) no switching. True cost incl. cache writes/re-reads. | ~$100 | **Answers the pi question with data.** Caches are model-specific; each toggle pays a full context re-write. Predicts (a) wins on long trajectories. |
| E5 | Online contextual bandit (Thompson sampling over arms, E2 model as prior) on fresh task stream; regret vs oracle. Must handle B4's non-stationarity (drifting arm quality). | ~$300+ | Only if E2 left a gap. Also the deployment story. |
| E6 | Pool diversity: add one non-Anthropic sidekick. Router value grows with pool diversity (RouterEval scaling); also tests B3's collusion mitigation (cross-provider Attacker). | varies | Requires provider abstraction (§5). |

### 4.4 Metrics (all experiments)
Resolve rate (SWE-bench harness); $/resolved; tokens by category (read/reason/edit/verify);
cache-hit ratio; wall-clock latency; oracle-gap; judge merge-score. Two seeds minimum for any
headline claim.

---

## 5. Harness decision: pi vs custom

**Decision: keep the custom harness as orchestrator. Do not re-platform on pi.**

1. The research object *is* the control flow between two persistent contexts (briefs,
   gates, budgets, telemetry). pi is deliberately a single minimal loop; orchestrating via
   model-toggling collapses the dual-context design and destroys model-specific prompt cache.
2. Mid-experiment platform migration confounds every comparison against existing runs.
3. **Do adopt pi's one key property as a library concern:** put a provider abstraction
   (pi's provider layer or LiteLLM) behind the existing LLM wrapper so E6's non-Anthropic
   sidekicks are a config change.
4. "pi-style mid-transcript toggling" is a *hypothesis*, tested as E4 condition (b), not
   infrastructure. If it wins, revisit with evidence in hand.

Industry direction (Fusion, SWE-grep, MinionS) is converging on heterogeneous specialist
contexts coordinated by structured protocols — this harness is on the right side of that.
What nobody publishes is the routing policy learned from measured arm outcomes; E1/E1b's
dataset is the novel asset.

---

## 6. Budget summary

| Phase | Spend |
|-------|-------|
| E0 instrumentation | ~$0 |
| E1/E1b first pass (25 tasks × 1 seed, all arms) | ~$250–400 |
| E1/E1b full (50 × 2) | ~$800–1,200 |
| E2 offline routing | ~$0 |
| E3 gates | ~$150–300 |
| E4 cache economics | ~$100 |
| E5 bandit (conditional) | ~$300+ |
| EC2 scoring | ~$50–100/campaign |
| **Total core program** | **~$1,500–2,000** |

---

## 7. References

**Production systems:** [Devin Fusion](https://cognition.com/blog/devin-fusion) ·
[SWE-grep](https://cognition.com/blog/swe-grep) ·
[Windsurf Fast Context](https://docs.windsurf.com/context-awareness/fast-context)

**Collaboration protocols:** [Minions/MinionS, arXiv:2502.15964](https://arxiv.org/abs/2502.15964) ·
[Speculative Actions, arXiv:2510.04371](https://arxiv.org/abs/2510.04371) ·
[DualSpec, arXiv:2603.07416](https://arxiv.org/pdf/2603.07416) ·
[Speculative cascades (Google Research)](https://research.google/blog/speculative-cascades-a-hybrid-approach-for-smarter-faster-llm-inference/)

**Cascades & routing:** [AutoMix, arXiv:2310.12963](https://arxiv.org/abs/2310.12963) ·
[FrugalGPT, arXiv:2305.05176](https://arxiv.org/abs/2305.05176) ·
[C3PO, arXiv:2511.07396](https://arxiv.org/html/2511.07396) ·
[RouteLLM, arXiv:2406.18665](https://arxiv.org/abs/2406.18665) ·
[RouterBench, arXiv:2403.12031](https://arxiv.org/abs/2403.12031) ·
[LLMRouterBench, arXiv:2601.07206](https://arxiv.org/html/2601.07206v1) ·
[Routing/cascading survey, arXiv:2603.04445](https://arxiv.org/pdf/2603.04445) ·
[SeqRoute, arXiv:2605.25424](https://arxiv.org/pdf/2605.25424)

**Test-time compute:** [Large Language Monkeys, arXiv:2407.21787](https://arxiv.org/abs/2407.21787) ·
[CodeMonkeys, arXiv:2501.14723](https://arxiv.org/abs/2501.14723) ·
[Agentless, arXiv:2407.01489](https://arxiv.org/abs/2407.01489)

**Adjacent (Track B design inputs):** [Debate, arXiv:1805.00899](https://arxiv.org/abs/1805.00899) ·
[Prover-Verifier Games, arXiv:2407.13692](https://arxiv.org/abs/2407.13692) ·
[MemGPT, arXiv:2310.08560](https://arxiv.org/abs/2310.08560) ·
[Voyager, arXiv:2305.16291](https://arxiv.org/abs/2305.16291) ·
[Agent Workflow Memory, arXiv:2409.07429](https://arxiv.org/abs/2409.07429) ·
[Reflexion, arXiv:2303.11366](https://arxiv.org/abs/2303.11366) ·
[RAPTOR, arXiv:2401.18059](https://arxiv.org/abs/2401.18059)

**Harness landscape:** [pi (Mario Zechner)](https://github.com/badlogic/pi-mono) ·
[Building Pi — Pragmatic Engineer](https://newsletter.pragmaticengineer.com/p/building-pi-and-what-makes-self-modifying)

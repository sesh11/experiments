# Runtime parity on SWE-bench Verified (2026-07-13)

**One line:** three agent runtimes — the in-repo fusion loop, Stirrup, and pi —
produced the *same* fix for the same task and all scored resolved by the
official Docker harness, at a **4× cost spread driven entirely by prompt
caching**.

## Setup

- **Where:** fresh x86_64 EC2 box (`c6i`-class), bootstrapped with
  `scripts/ec2_bootstrap.sh`. SWE-bench images run natively (no Mac emulation).
- **Task:** `pallets__flask-5014` (1 instance), SWE-bench Verified.
- **Model:** `claude-sonnet-5`, thinking off, for all three runtimes.
- **Scoring:** official `swebench.harness.run_evaluation` in pinned Docker images.
- **Gate:** the $0 gold self-test passed 3/3 first, so any unresolved here would
  be the agent's doing, not the scorer's.
- **Command:** `--limit 1 --variants baseline-fusion baseline-stirrup baseline-pi
  --per-task 0.6 --budget 2 --max-steps 30 --no-judge`. Total spend **$1.38**.
- **Raw evidence:** `2026-07-13-runtime-parity.summary.csv` (this folder).

## Results

| runtime          | resolved | F2P | P2P   | cost   | input tok | cache reads | steps |
|------------------|----------|-----|-------|--------|-----------|-------------|-------|
| baseline-fusion  | ✅        | 1/1 | 59/59 | $0.582 | 178,055   | 23,980      | 21    |
| baseline-stirrup | ✅        | 1/1 | 59/59 | $0.650 | 195,136   | 0           | —¹    |
| baseline-pi      | ✅        | 1/1 | 59/59 | $0.147 | 42        | 142,086     | 20²   |

¹ Stirrup hit its $0.60 per-task cap ($0.65 actual — normal one-call overshoot);
the budget abort meant its tool trace wasn't returned.
² pi's ledger records `calls=1` because it accounts post-hoc (one aggregate at
session end); its trace shows ~20 real tool steps.

## Findings

1. **Parity is strong, not just "all green."** All three emitted a byte-identical
   source fix (same git blob hash `1aa82562`):

   ```python
   if not name:
       raise ValueError("'name' may not be empty.")
   ```

   Same model, same task, same answer — the orchestrator, all three harnesses,
   and Docker scoring agree end to end.

2. **The "touched tests" warning was a false alarm.** Stirrup and pi each *added*
   a new `test_empty_name_not_allowed` (mirroring the existing dotted-name test)
   before fixing — test-first behavior, not gaming. They never touched the graded
   tests, and it's moot anyway: the harness resets test files to gold before
   scoring. Fusion simply didn't write a test. A behavioral difference between
   harnesses, not a problem.

3. **Cost is dominated by prompt caching, which is a harness property, not a
   model property.** Identical work, identical result, 4× cost spread:
   - **pi ($0.15)** caches aggressively — nearly the whole context is cache reads.
   - **fusion ($0.58)** caches only its system prompt — partial.
   - **Stirrup ($0.65)** sends no cache hints on its litellm path — it re-bills
     the full growing context at full input price every turn, so it is the most
     expensive despite doing the same work.

   Stirrup's cost is not inherent to the framework; it's that its client never
   sets `cache_control`. This is the first concrete, measurable lever for the
   orchestrator: same quality, big cost swing, decided by the harness.

## Caveats to carry forward

- pi's absolute cost has more uncertainty than the others (post-hoc summing, plus
  its bundled price registry doesn't know `claude-sonnet-5`) — but it is
  unambiguously the cheapest.
- A budget-capped Stirrup run loses its tool trace; worth a small fix if traces
  on capped runs matter.
- One instance is a smoke-level parity check, not a benchmark. It proves the
  machinery and surfaces the caching lever; it does not rank the harnesses.

## Next

The infrastructure question is settled. The first real orchestration experiment
follows directly from finding 3: caching strategy yields a ~4× cost swing at
equal quality, so runtime/model selection is a measurable cost lever before any
cleverness. The `VariantSpec` seams (runtime + model + pattern) are already in
place for it. (Fixing Stirrup's caching is deferred — it deserves its own change
with clean before/after numbers.)

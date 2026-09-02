# Fusion transcript caching, live check (2026-09-02)

**One line:** with rolling `cache_control` breakpoints on the transcript, the
fusion loop bills **185 fresh input tokens** instead of 178,055 on the same task
and still resolves it — closing the 4× cost gap against pi from the runtime
parity finding.

## Setup

- **Task:** `pallets__flask-5014` (1 instance), SWE-bench Verified.
- **Model:** `claude-sonnet-5`, thinking off, `baseline-fusion`.
- **Scoring:** official `swebench.harness.run_evaluation` in pinned Docker images.
- **Command:** `--variants baseline-fusion --no-judge --budget 3 --per-task 2
  --backend docker`. Total spend **$0.04**.
- **Raw evidence:** `2026-09-02-fusion-transcript-cache.summary.csv` (this folder).

## Results

| run                   | resolved | F2P | P2P   | cost   | input tok | cache reads | cache writes | steps |
|-----------------------|----------|-----|-------|--------|-----------|-------------|--------------|-------|
| fusion, before (7/13) | ✅        | 1/1 | 59/59 | $0.582 | 178,055   | 23,980      | —            | 21    |
| fusion, after         | ✅        | 1/1 | 59/59 | $0.040 | 185       | 21,548      | 4,871        | 7     |
| pi (7/13, reference)  | ✅        | 1/1 | 59/59 | $0.147 | 42        | 142,086     | —            | 20    |

## Findings

1. **The mechanism works as intended.** Nearly all re-read context is now billed
   at cache-read prices; fresh input is only the newest user/tool block.
2. **Cost per step, not just per run, is the honest comparison.** This run took 7
   steps where the 7/13 run took 21, so the headline 14× is partly step count.
   Per step it is roughly $0.006 vs $0.028 — still a ~5× reduction at identical
   quality.
3. **Cost comparisons before this change measured the harness.** Any arm running
   on the fusion loop was re-billing its whole transcript each turn, which both
   inflated frontier baselines and blunted the context-footprint mechanism that
   `scout` is supposed to exploit. E1 arm results collected before this change
   are not comparable to results after it.

## Caveat

`swebench` 5.x removes the `run_evaluation` flags the docker backend passes
(`--cache_level`, `--force_rebuild`, `--namespace`), so scoring silently reports
"harness produced no report" on a fresh install. `requirements.txt` now pins
`<5`.

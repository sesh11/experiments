# OpenRouter path, live cache check (2026-09-02)

**One line:** the same task through `--provider openrouter` bills **24 fresh
input tokens against 60,021 cache reads** and resolves, so OpenRouter's
automatic caching is a working substitute for the Anthropic breakpoints — the
provider axis is not confounded with the caching artifact.

## Setup

- **Task:** `pallets__flask-5014` (1 instance), SWE-bench Verified.
- **Model:** `anthropic/claude-sonnet-5` via OpenRouter, thinking off,
  `baseline-fusion`.
- **Scoring:** official `swebench.harness.run_evaluation` in pinned Docker images.
- **Command:** `--provider openrouter --main-model anthropic/claude-sonnet-5
  --variants baseline-fusion --no-judge --budget 3 --per-task 2 --backend docker`.
  Total spend **$0.05**.
- **Raw evidence:** `2026-09-02-openrouter-cache.summary.csv` (this folder).

## Results

| path                 | resolved | F2P | P2P   | cost   | input tok | cache reads | cache writes | steps |
|----------------------|----------|-----|-------|--------|-----------|-------------|--------------|-------|
| anthropic direct     | ✅        | 1/1 | 59/59 | $0.040 | 185       | 21,548      | 4,871        | 7     |
| openrouter           | ✅        | 1/1 | 59/59 | $0.052 | 24        | 60,021      | 9,758        | 12    |

## Findings

1. **Both provider paths cache.** OpenRouter carries a single moving breakpoint
   from the top-level `cache_control` field rather than the two explicit block
   breakpoints Anthropic gets, and the effect is the same: fresh input is only
   the newest turn.
2. **Per-step cost is comparable.** $0.0043/step on OpenRouter vs $0.006/step
   direct; the higher run total is step count (12 vs 7), not a pricing gap.
   Runs differ in trajectory, so treat this as an order-of-magnitude check, not
   a paired measurement.
3. **The non-Claude gate holds live.** `openai/gpt-4o-mini` through the same
   client returns normally with no `cache_control` in the request and zero cache
   tokens, and costs are taken from OpenRouter's reported `cost` rather than
   pinned pricing.

## Why it matters

E6 (non-Anthropic sidekick) and any OpenRouter-routed arm can now be compared
against Anthropic-direct arms without the cost difference being an artifact of
which provider path the loop happened to take.

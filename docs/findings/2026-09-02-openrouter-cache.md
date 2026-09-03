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
- **Raw evidence:** `2026-09-02-openrouter-cache.summary.csv` and
  `2026-09-02-openrouter-glm.summary.csv` (this folder).
- **Open-weight arm:** the same cell re-run on `z-ai/glm-5.3`, an open-weight
  frontier model, to check the provider path is not Anthropic-only in practice.

## Results

| path                        | resolved | F2P | P2P   | cost   | input tok | cache reads | cache writes | output tok | steps |
|-----------------------------|----------|-----|-------|--------|-----------|-------------|--------------|------------|-------|
| sonnet-5, anthropic direct  | ✅        | 1/1 | 59/59 | $0.040 | 185       | 21,548      | 4,871        | 949        | 7     |
| sonnet-5, openrouter        | ✅        | 1/1 | 59/59 | $0.052 | 24        | 60,021      | 9,758        | 1,523      | 12    |
| glm-5.3, openrouter         | ✅        | 1/1 | 59/59 | $0.085 | 37,533    | 102,208     | 0            | 3,811      | 14    |

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
4. **An open-weight frontier model resolves the task, and costs more.** glm-5.3
   is ~4× cheaper per input token than sonnet-5 but billed $0.085 vs $0.052,
   because reasoning cannot be disabled on that endpoint (3,811 output tokens vs
   1,523) and its implicit cache left 37,533 tokens billed as fresh input where
   the explicit breakpoints left 24. Cheap tokens do not imply a cheap arm; only
   $/resolved is comparable.

## Adapter fix this run forced

The first glm-5.3 attempt died on step 0 with `400 Reasoning is mandatory for
this endpoint and cannot be disabled` — the harness sends `reasoning:
{effort: "none"}` for `--thinking off`, which reasoning-only endpoints reject
outright rather than ignore. `OpenRouterProvider.complete` now retries once
without the field:

```python
try:
    response = self._client.chat.completions.create(**kwargs)
except Exception as exc:
    if not (reasoning == {"effort": "none"} and _mandatory_reasoning(exc)):
        raise
    del extra_body["reasoning"]   # run at the endpoint's own default
    response = self._client.chat.completions.create(**kwargs)
```

The consequence is a measurement caveat, not just a compatibility win: any arm
on such a model is *not* a thinking-off arm, so its output tokens are not
comparable to `--thinking off` arms on models that honor the flag.

## Why it matters

E6 (non-Anthropic sidekick) and any OpenRouter-routed arm can now be compared
against Anthropic-direct arms without the cost difference being an artifact of
which provider path the loop happened to take. glm-5.3 is a viable E6 pool
member on quality; its arm cost has to be measured, not extrapolated from its
sticker rate.

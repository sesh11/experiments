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
| glm-5.3, or, endpoint default | ✅      | 1/1 | 59/59 | $0.085 | 37,533    | 102,208     | 0            | 3,811      | 14    |
| glm-5.3, or, effort minimal | ✅        | 1/1 | 59/59 | $0.012 | 6,482     | 7,602       | 0            | 486        | 7     |

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
4. **An open-weight frontier model resolves the task, and reasoning effort
   dominates its arm cost.** At the endpoint's default reasoning glm-5.3 billed
   $0.085 — more than sonnet-5's $0.052 despite ~4× cheaper input tokens — on
   3,811 output tokens over 14 steps. At `effort: minimal` the same cell resolves
   in 7 steps on 486 output tokens for **$0.012**, the cheapest arm measured. The
   reasoning setting moves cost ~7× on one model, more than the model choice
   does; it belongs in the arm definition.
5. **Implicit caching still leaves fresh input on the table.** Both glm runs show
   zero cache writes — nothing we send pins a prefix — and the minimal run still
   billed 6,482 fresh input tokens against 7,602 cache reads, where the explicit
   Anthropic breakpoints billed 24 against 60,021. It is cheap enough not to
   matter here, but it scales with trajectory length.

## Adapter fix this run forced

The first glm-5.3 attempt died on step 0 with `400 Reasoning is mandatory for
this endpoint and cannot be disabled` — the harness sends `reasoning:
{effort: "none"}` for `--thinking off`, which reasoning-native models reject.

That rejection comes from OpenRouter's own layer, not from whichever host is
serving the weights. Probing all 11 hosts that serve `z-ai/glm-5.3` with
`provider.only` returns the byte-identical error with `provider_name: null`, and
all 11 return reasoning tokens when the field is omitted — so picking a host is
not a workaround. `reasoning: {effort: "minimal"}` *is* accepted and returns
zero reasoning tokens, so `--thinking off` now degrades to minimal effort rather
than to the endpoint default:

```python
except Exception as exc:
    if not (reasoning == {"effort": "none"} and _mandatory_reasoning(exc)):
        raise
    extra_body["reasoning"] = dict(MINIMAL_REASONING)
    response = self._client.chat.completions.create(**kwargs)
```

A minimal-effort arm is still not identical to a thinking-off arm, but it is
close enough to compare, and 7× cheaper than letting the endpoint choose.

## Why it matters

E6 (non-Anthropic sidekick) and any OpenRouter-routed arm can now be compared
against Anthropic-direct arms without the cost difference being an artifact of
which provider path the loop happened to take. glm-5.3 is a viable E6 pool
member on quality; its arm cost has to be measured, not extrapolated from its
sticker rate. Reasoning effort has to be recorded per arm — at n=1 it is the
largest cost lever observed so far.

Endpoint probe scripts and their raw output are not committed; the counts above
come from `results/runs/glmcheck2` and `glmcheck3`.

"""Model IDs, pricing, and budget configuration.

Pricing is pinned to Anthropic's published standard rates (per 1M tokens) as of
2026-07. We intentionally use the *standard* rates (not the Sonnet 5 introductory
discount) so the budget guard errs on the side of over-estimating spend.

Prompt-caching multipliers follow the documented economics: a 5-minute cache
write costs ~1.25x base input and a cache read costs ~0.1x base input.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --- Model identifiers ------------------------------------------------------
MODEL_MAIN = "claude-sonnet-5"          # the "frontier" main agent
MODEL_SIDEKICK = "claude-haiku-4-5"     # the cheap sidekick / scout
JUDGE_MODEL = MODEL_MAIN                 # "would you merge this?" rubric grader


# --- Pricing ($ per 1M tokens) ---------------------------------------------
@dataclass(frozen=True)
class Price:
    input: float
    output: float
    cache_write_5m: float   # 1.25x input
    cache_read: float       # ~0.1x input


PRICING: dict[str, Price] = {
    "claude-sonnet-5": Price(input=3.00, output=15.00, cache_write_5m=3.75, cache_read=0.30),
    "claude-haiku-4-5": Price(input=1.00, output=5.00, cache_write_5m=1.25, cache_read=0.10),
}


def cost_for(model: str, *, input_tokens: int, output_tokens: int,
             cache_write_tokens: int = 0, cache_read_tokens: int = 0) -> float:
    """Dollar cost of a single call given a token breakdown."""
    p = PRICING[_normalize(model)]
    return (
        input_tokens / 1e6 * p.input
        + output_tokens / 1e6 * p.output
        + cache_write_tokens / 1e6 * p.cache_write_5m
        + cache_read_tokens / 1e6 * p.cache_read
    )


def _normalize(model: str) -> str:
    """Map dated/aliased/provider-prefixed IDs onto a pricing key.

    External runtimes address models as e.g. "anthropic/claude-sonnet-5"
    (LiteLLM/pi provider syntax); pricing keys are bare Anthropic ids.
    """
    if "/" in model:
        model = model.rsplit("/", 1)[-1]
    if model.startswith("claude-sonnet-5"):
        return "claude-sonnet-5"
    if model.startswith("claude-haiku-4-5"):
        return "claude-haiku-4-5"
    return model


# --- Run configuration ------------------------------------------------------
@dataclass
class RunConfig:
    budget_usd: float = 25.0            # hard ceiling for a whole run
    per_task_usd: float = 3.0           # soft ceiling per (variant, task)
    max_tokens: int = 8192              # per model response (non-streaming safe)
    max_steps: int = 14                 # tool-use loop steps per agent
    scout_max_steps: int = 10           # steps for a read-only scout delegation
    # Suppress Sonnet 5's adaptive-thinking-by-default to keep cost predictable.
    # (Haiku 4.5 has no adaptive mode; we omit the field there.)
    main_thinking: dict | None = field(default_factory=lambda: {"type": "disabled"})
    sidekick_thinking: dict | None = None
    # Wall-clock cap for subprocess runtimes (pi); their budget is only
    # accounted post-hoc, so time is the in-flight guard.
    runtime_timeout_s: int = 1200

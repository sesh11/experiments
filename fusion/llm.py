"""Anthropic client wrapper with per-role token/cost accounting and a budget guard.

Each `Agent` (main or sidekick) drives its own message history through one shared
`LLMClient`, but usage is tagged by *role* so we can report the main/sidekick cost
split that is the whole point of the Fusion architecture.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import anthropic

from . import config


class BudgetExceeded(RuntimeError):
    """Raised when cumulative estimated spend crosses the configured cap."""


@dataclass
class RoleUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0


@dataclass
class Ledger:
    """Accumulates token/cost usage across all calls in a single task run."""
    cap_usd: float = 25.0
    by_role: dict[str, RoleUsage] = field(default_factory=lambda: defaultdict(RoleUsage))

    def record(self, role: str, model: str, usage) -> None:
        """Record one Anthropic-SDK usage object (main fusion-loop path)."""
        self.record_tokens(
            role, model,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        )

    def record_tokens(self, role: str, model: str, *, input_tokens: int,
                      output_tokens: int, cache_write_tokens: int = 0,
                      cache_read_tokens: int = 0) -> None:
        """Record explicit token counts — the entry point for external runtimes
        (Stirrup, pi) whose usage objects aren't Anthropic-SDK shaped. Cost is
        always recomputed from pinned pricing so numbers stay comparable."""
        cost = config.cost_for(
            model, input_tokens=input_tokens, output_tokens=output_tokens,
            cache_write_tokens=cache_write_tokens, cache_read_tokens=cache_read_tokens,
        )
        r = self.by_role[role]
        r.input_tokens += input_tokens
        r.output_tokens += output_tokens
        r.cache_write_tokens += cache_write_tokens
        r.cache_read_tokens += cache_read_tokens
        r.cost_usd += cost
        r.calls += 1
        if self.total_cost > self.cap_usd:
            raise BudgetExceeded(
                f"budget cap ${self.cap_usd:.2f} exceeded (spent ${self.total_cost:.2f})"
            )

    def ensure_capacity(self, maximum_cost_usd: float) -> None:
        """Reject a request before billing if its worst case exceeds the cap."""
        if self.total_cost + maximum_cost_usd > self.cap_usd:
            raise BudgetExceeded(
                f"next call could exceed budget cap ${self.cap_usd:.2f} "
                f"(spent ${self.total_cost:.2f}, needs up to ${maximum_cost_usd:.2f})"
            )

    @property
    def total_cost(self) -> float:
        return sum(r.cost_usd for r in self.by_role.values())

    def summary(self) -> dict:
        # Keep exact floats for cross-cell budget settlement; presentation
        # layers round independently.
        out = {"total_cost_usd": self.total_cost}
        for role, r in self.by_role.items():
            out[f"{role}_cost_usd"] = r.cost_usd
            out[f"{role}_input_tokens"] = r.input_tokens
            out[f"{role}_output_tokens"] = r.output_tokens
            out[f"{role}_cache_read_tokens"] = r.cache_read_tokens
            out[f"{role}_calls"] = r.calls
        return out


class LLMClient:
    """Thin wrapper over the Anthropic SDK that records usage into a Ledger."""

    def __init__(self, ledger: Ledger, max_tokens: int = 8192):
        # Zero-arg client resolves credentials from the environment / ant profile.
        self._client = anthropic.Anthropic()
        self.ledger = ledger
        self.max_tokens = max_tokens

    def complete(self, *, role: str, model: str, system: str, messages: list,
                 tools: list | None = None, thinking: dict | None = None):
        """One Messages API call. `system` is cached so each role keeps a warm prefix."""
        kwargs = dict(
            model=model,
            max_tokens=self.max_tokens,
            system=[{"type": "text", "text": system,
                     "cache_control": {"type": "ephemeral"}}],
            messages=messages,
        )
        if tools:
            kwargs["tools"] = tools
        if thinking is not None:
            kwargs["thinking"] = thinking
        self.ledger.ensure_capacity(config.request_cost_upper_bound(
            model, kwargs, max_output_tokens=self.max_tokens))
        resp = self._client.messages.create(**kwargs)
        self.ledger.record(role, model, resp.usage)
        return resp

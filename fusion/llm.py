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
        inp = getattr(usage, "input_tokens", 0) or 0
        out = getattr(usage, "output_tokens", 0) or 0
        cw = getattr(usage, "cache_creation_input_tokens", 0) or 0
        cr = getattr(usage, "cache_read_input_tokens", 0) or 0
        cost = config.cost_for(
            model, input_tokens=inp, output_tokens=out,
            cache_write_tokens=cw, cache_read_tokens=cr,
        )
        r = self.by_role[role]
        r.input_tokens += inp
        r.output_tokens += out
        r.cache_write_tokens += cw
        r.cache_read_tokens += cr
        r.cost_usd += cost
        r.calls += 1
        if self.total_cost > self.cap_usd:
            raise BudgetExceeded(
                f"budget cap ${self.cap_usd:.2f} exceeded (spent ${self.total_cost:.2f})"
            )

    @property
    def total_cost(self) -> float:
        return sum(r.cost_usd for r in self.by_role.values())

    def summary(self) -> dict:
        out = {"total_cost_usd": round(self.total_cost, 4)}
        for role, r in self.by_role.items():
            out[f"{role}_cost_usd"] = round(r.cost_usd, 4)
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
        resp = self._client.messages.create(**kwargs)
        self.ledger.record(role, model, resp.usage)
        return resp

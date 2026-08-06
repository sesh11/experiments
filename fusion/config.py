"""Provider, model, pricing, and budget configuration.

Anthropic remains the default provider.  OpenRouter uses provider-prefixed
model ids (for example ``anthropic/claude-sonnet-5``) and reports the billed
cost for every completion; direct OpenRouter calls use that reported cost.
Pinned prices remain the fallback for Anthropic and external runtimes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

PROVIDERS = ("anthropic", "openrouter")


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else default


# --- Provider/model defaults ------------------------------------------------
DEFAULT_PROVIDER = _env("LLM_PROVIDER", "anthropic").lower()
MODEL_MAIN = _env("MODEL_MAIN", "claude-sonnet-5")
MODEL_SIDEKICK = _env("MODEL_SIDEKICK", "claude-haiku-4-5")
JUDGE_PROVIDER = _env("JUDGE_PROVIDER", DEFAULT_PROVIDER).lower()
JUDGE_MODEL = _env("JUDGE_MODEL", MODEL_MAIN)

OPENROUTER_BASE_URL = _env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_SITE_URL = os.environ.get("OPENROUTER_SITE_URL", "").strip()
OPENROUTER_APP_NAME = os.environ.get("OPENROUTER_APP_NAME", "").strip()


class ConfigurationError(ValueError):
    """Raised when a provider/model combination cannot be used safely."""


class PricingUnavailable(ConfigurationError):
    """Raised when token counts cannot be mapped to a pinned model price."""


def normalize_provider(provider: str) -> str:
    value = (provider or "").strip().lower()
    if value not in PROVIDERS:
        raise ConfigurationError(
            f"unsupported provider {provider!r}; choose one of: {', '.join(PROVIDERS)}"
        )
    return value


def api_model(provider: str, model: str) -> str:
    """Return the model id expected by a provider's native API.

    Bare Claude ids are accepted for both providers so switching only
    ``--provider`` keeps the historical defaults useful.  Other OpenRouter
    models must include their organization prefix to avoid ambiguous routing.
    """
    provider = normalize_provider(provider)
    value = (model or "").strip()
    if not value:
        raise ConfigurationError("model id cannot be empty")
    if provider == "anthropic":
        if value.startswith("anthropic/"):
            return value.split("/", 1)[1]
        if "/" in value:
            raise ConfigurationError(
                f"Anthropic provider cannot route model {value!r}; use --provider openrouter"
            )
        return value
    if "/" in value:
        return value
    if value.startswith("claude-"):
        return f"anthropic/{value}"
    raise ConfigurationError(
        f"OpenRouter model {value!r} needs a provider prefix, e.g. 'openai/{value}'"
    )


def litellm_model(provider: str, model: str) -> str:
    """Return the provider-qualified model id expected by LiteLLM."""
    provider = normalize_provider(provider)
    routed = api_model(provider, model)
    return f"{provider}/{routed}"


# --- Pricing ($ per 1M tokens) ---------------------------------------------
@dataclass(frozen=True)
class Price:
    input: float
    output: float
    cache_write_5m: float
    cache_read: float


# Provider-prefixed keys prevent an unrelated provider from accidentally being
# billed as a same-named Anthropic model.
PRICING: dict[str, Price] = {
    "anthropic/claude-sonnet-5": Price(
        input=3.00, output=15.00, cache_write_5m=3.75, cache_read=0.30),
    "anthropic/claude-haiku-4-5": Price(
        input=1.00, output=5.00, cache_write_5m=1.25, cache_read=0.10),
}


def pricing_key(model: str) -> str:
    """Map supported aliases/snapshots onto an exact pinned pricing key."""
    value = (model or "").strip().lstrip("~")
    if "/" not in value and value.startswith("claude-"):
        value = f"anthropic/{value}"
    sonnet = "anthropic/claude-sonnet-5"
    haiku = "anthropic/claude-haiku-4-5"
    if value == sonnet or value.startswith((f"{sonnet}-", f"{sonnet}:")):
        return "anthropic/claude-sonnet-5"
    if value == haiku or value.startswith((f"{haiku}-", f"{haiku}:")):
        return "anthropic/claude-haiku-4-5"
    return value


def has_pinned_price(model: str) -> bool:
    return pricing_key(model) in PRICING


def price_for(model: str) -> Price:
    key = pricing_key(model)
    try:
        return PRICING[key]
    except KeyError as exc:
        raise PricingUnavailable(
            f"no pinned pricing for model {model!r}; use direct OpenRouter calls "
            "(which report billed cost) or add an explicit PRICING entry"
        ) from exc


def cost_for(model: str, *, input_tokens: int, output_tokens: int,
             cache_write_tokens: int = 0, cache_read_tokens: int = 0) -> float:
    """Dollar cost of a call from token counts and pinned model pricing."""
    p = price_for(model)
    return (
        input_tokens / 1e6 * p.input
        + output_tokens / 1e6 * p.output
        + cache_write_tokens / 1e6 * p.cache_write_5m
        + cache_read_tokens / 1e6 * p.cache_read
    )


# Backwards-compatible private name used by older callers/tests.
_normalize = pricing_key


# --- Run configuration ------------------------------------------------------
@dataclass
class RunConfig:
    provider: str = DEFAULT_PROVIDER
    main_model: str = MODEL_MAIN
    sidekick_model: str = MODEL_SIDEKICK
    judge_provider: str = JUDGE_PROVIDER
    judge_model: str = JUDGE_MODEL
    openrouter_base_url: str = OPENROUTER_BASE_URL
    openrouter_site_url: str = OPENROUTER_SITE_URL
    openrouter_app_name: str = OPENROUTER_APP_NAME

    budget_usd: float = 25.0
    per_task_usd: float = 3.0
    max_tokens: int = 8192
    max_steps: int = 14
    scout_max_steps: int = 10
    # Suppress adaptive thinking by default to keep cost predictable.
    main_thinking: dict | None = field(default_factory=lambda: {"type": "disabled"})
    sidekick_thinking: dict | None = None
    runtime_timeout_s: int = 1200

    def __post_init__(self) -> None:
        self.provider = normalize_provider(self.provider)
        self.judge_provider = normalize_provider(self.judge_provider)
        # Validate combinations now; adapters perform the actual canonicalization.
        api_model(self.provider, self.main_model)
        api_model(self.provider, self.sidekick_model)
        api_model(self.judge_provider, self.judge_model)
        self.openrouter_base_url = self.openrouter_base_url.rstrip("/")
        if not self.openrouter_base_url:
            raise ConfigurationError("OPENROUTER_BASE_URL cannot be empty")

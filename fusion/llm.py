"""Provider-neutral completions with normalized tool calls and cost accounting."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Protocol

import anthropic
from openai import OpenAI

from . import config


class BudgetExceeded(RuntimeError):
    """Raised when cumulative spend crosses the configured cap."""


class ProviderResponseError(RuntimeError):
    """Raised when a provider returns a malformed or embedded error response."""


@dataclass
class RoleUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0


@dataclass(frozen=True)
class CompletionUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    # OpenRouter returns its exact billed cost.  Anthropic leaves this unset so
    # the ledger uses the repository's pinned rates.
    cost_usd: float | None = None


@dataclass
class ContentBlock:
    """Provider-neutral assistant content.

    ``raw`` preserves provider-only fields (notably reasoning signatures) that
    must be echoed back on the next turn even though the agent loop ignores them.
    """

    type: str
    text: str = ""
    id: str = ""
    name: str = ""
    input: dict = field(default_factory=dict)
    raw: dict | None = None

    def as_dict(self) -> dict:
        if self.raw is not None:
            return dict(self.raw)
        if self.type == "text":
            return {"type": "text", "text": self.text}
        if self.type == "tool_use":
            return {
                "type": "tool_use", "id": self.id,
                "name": self.name, "input": self.input,
            }
        return {"type": self.type}


@dataclass
class CompletionResponse:
    content: list[ContentBlock]
    stop_reason: str
    usage: CompletionUsage


@dataclass
class Ledger:
    """Accumulate token/cost usage across all calls in one task run."""

    cap_usd: float = 25.0
    by_role: dict[str, RoleUsage] = field(default_factory=lambda: defaultdict(RoleUsage))

    def record(self, role: str, model: str, usage: Any) -> None:
        """Record an Anthropic-shaped SDK usage object (compatibility entry point)."""
        self.record_tokens(
            role, model,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        )

    def record_usage(self, role: str, model: str, usage: CompletionUsage) -> None:
        self.record_tokens(
            role, model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_write_tokens=usage.cache_write_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cost_usd=usage.cost_usd,
        )

    def record_tokens(self, role: str, model: str, *, input_tokens: int,
                      output_tokens: int, cache_write_tokens: int = 0,
                      cache_read_tokens: int = 0,
                      cost_usd: float | None = None) -> None:
        """Record explicit usage, optionally with a provider-reported cost."""
        cost = (
            config.cost_for(
                model, input_tokens=input_tokens, output_tokens=output_tokens,
                cache_write_tokens=cache_write_tokens,
                cache_read_tokens=cache_read_tokens,
            )
            if cost_usd is None else float(cost_usd)
        )
        if cost < 0:
            raise ValueError(f"provider reported a negative cost: {cost}")
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
            out[f"{role}_cache_write_tokens"] = r.cache_write_tokens
            out[f"{role}_cache_read_tokens"] = r.cache_read_tokens
            out[f"{role}_calls"] = r.calls
        return out


CACHE_CONTROL: dict[str, str] = {"type": "ephemeral"}

# Anthropic reads its prompt cache only up to the newest breakpoint, so a
# transcript with a marker on the system block alone re-bills every tool result
# at the full input rate on each turn. Four explicit breakpoints are allowed;
# the system block takes one and two rolling markers on the newest blocks keep
# the previous prefix warm while the newest turn is being written.
TRANSCRIPT_BREAKPOINTS = 2

# Blocks Anthropic accepts a breakpoint on; notably excludes thinking blocks.
_CACHEABLE_BLOCKS = frozenset({"text", "tool_use", "tool_result", "image",
                               "document"})


def _cached_transcript(messages: list, limit: int = TRANSCRIPT_BREAKPOINTS) -> list:
    """Copy of ``messages`` with cache breakpoints on its newest blocks.

    The caller's provider-neutral transcript is never mutated: only the touched
    messages and their last content block are copied.
    """
    out = list(messages)
    marked = 0
    for index in range(len(out) - 1, -1, -1):
        if marked >= limit:
            break
        message = out[index]
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list) or not content:
            continue
        block = content[-1]
        if not isinstance(block, dict) or block.get("type") not in _CACHEABLE_BLOCKS:
            continue
        marked += 1
        if "cache_control" in block:
            continue
        blocks = list(content)
        blocks[-1] = {**block, "cache_control": dict(CACHE_CONTROL)}
        out[index] = {**message, "content": blocks}
    return out


class ProviderAdapter(Protocol):
    def complete(self, *, model: str, system: str, messages: list,
                 tools: list | None, thinking: dict | None,
                 max_tokens: int) -> CompletionResponse:
        ...


def _model_dump(value: Any) -> dict:
    if isinstance(value, dict):
        return dict(value)
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump(exclude_none=True)
    return {}


class AnthropicProvider:
    """Anthropic Messages adapter preserving the existing request behavior."""

    def __init__(self, client: Any | None = None) -> None:
        self._client = client if client is not None else anthropic.Anthropic()

    def complete(self, *, model: str, system: str, messages: list,
                 tools: list | None, thinking: dict | None,
                 max_tokens: int) -> CompletionResponse:
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": [{
                "type": "text", "text": system,
                "cache_control": dict(CACHE_CONTROL),
            }],
            "messages": _cached_transcript(messages),
        }
        if tools:
            kwargs["tools"] = tools
        if thinking is not None:
            kwargs["thinking"] = thinking
        response = self._client.messages.create(**kwargs)
        blocks: list[ContentBlock] = []
        for block in response.content:
            kind = getattr(block, "type", "")
            raw = _model_dump(block) or None
            if kind == "text":
                blocks.append(ContentBlock(
                    type="text", text=getattr(block, "text", ""), raw=raw))
            elif kind == "tool_use":
                blocks.append(ContentBlock(
                    type="tool_use", id=getattr(block, "id", ""),
                    name=getattr(block, "name", ""),
                    input=dict(getattr(block, "input", {}) or {}), raw=raw,
                ))
            else:
                blocks.append(ContentBlock(type=kind or "unknown", raw=raw))
        usage = response.usage
        return CompletionResponse(
            content=blocks,
            stop_reason=getattr(response, "stop_reason", "") or "",
            usage=CompletionUsage(
                input_tokens=getattr(usage, "input_tokens", 0) or 0,
                output_tokens=getattr(usage, "output_tokens", 0) or 0,
                cache_write_tokens=(
                    getattr(usage, "cache_creation_input_tokens", 0) or 0),
                cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            ),
        )


def _tool_schema(tool: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", {"type": "object"}),
        },
    }


def _tool_arguments(raw: str | None, name: str) -> dict:
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise ProviderResponseError(
            f"OpenRouter returned invalid JSON arguments for tool {name!r}: {raw!r}"
        ) from exc
    if not isinstance(value, dict):
        raise ProviderResponseError(
            f"OpenRouter returned non-object arguments for tool {name!r}: {value!r}"
        )
    return value


def _openrouter_reasoning(thinking: dict | None) -> dict | None:
    if thinking is None:
        return None
    kind = thinking.get("type")
    if kind == "disabled":
        return {"effort": "none"}
    budget = thinking.get("budget_tokens")
    if budget:
        return {"max_tokens": int(budget)}
    if kind in {"enabled", "adaptive"}:
        return {"enabled": True}
    return None


def _supports_auto_cache(model: str) -> bool:
    """Whether OpenRouter will honor a top-level ``cache_control`` field.

    Automatic caching is an Anthropic-family feature there; sending the field
    to an unrelated provider risks a rejected request.
    """
    return "claude" in model.lower()


def _openrouter_messages(system: str, messages: list) -> list[dict]:
    converted: list[dict] = [{
        "role": "system",
        "content": [{
            "type": "text", "text": system,
            "cache_control": dict(CACHE_CONTROL),
        }],
    }]
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role == "assistant" and isinstance(content, list):
            texts: list[str] = []
            calls: list[dict] = []
            reasoning: list[dict] = []
            for block in content:
                value = block.as_dict() if isinstance(block, ContentBlock) else block
                kind = value.get("type") if isinstance(value, dict) else ""
                if kind == "text":
                    texts.append(str(value.get("text", "")))
                elif kind == "tool_use":
                    calls.append({
                        "id": value["id"], "type": "function",
                        "function": {
                            "name": value["name"],
                            "arguments": json.dumps(value.get("input", {})),
                        },
                    })
                elif kind == "reasoning":
                    detail = value.get("detail")
                    if isinstance(detail, dict):
                        reasoning.append(detail)
                    else:
                        raw = dict(value)
                        raw.pop("type", None)
                        reasoning.append(raw)
            assistant: dict[str, Any] = {
                "role": "assistant", "content": "\n".join(texts) or None,
            }
            if calls:
                assistant["tool_calls"] = calls
            if reasoning:
                assistant["reasoning_details"] = reasoning
            converted.append(assistant)
            continue
        if role == "user" and isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    converted.append({
                        "role": "tool",
                        "tool_call_id": block["tool_use_id"],
                        "content": str(block.get("content", "")),
                    })
                elif isinstance(block, dict) and block.get("type") == "text":
                    converted.append({"role": "user", "content": block.get("text", "")})
            continue
        converted.append({"role": role, "content": content})
    return converted


class OpenRouterProvider:
    """OpenRouter adapter over its OpenAI-compatible Chat Completions API."""

    def __init__(self, *, client: Any | None = None, api_key: str | None = None,
                 base_url: str = config.OPENROUTER_BASE_URL,
                 site_url: str = config.OPENROUTER_SITE_URL,
                 app_name: str = config.OPENROUTER_APP_NAME) -> None:
        if client is not None:
            self._client = client
            return
        key = (api_key or os.environ.get("OPENROUTER_API_KEY", "")).strip()
        if not key:
            raise config.ConfigurationError(
                "OPENROUTER_API_KEY is required when provider=openrouter"
            )
        headers = {}
        if site_url:
            headers["HTTP-Referer"] = site_url
        if app_name:
            headers["X-OpenRouter-Title"] = app_name
        self._client = OpenAI(
            base_url=base_url.rstrip("/"), api_key=key,
            default_headers=headers or None,
        )

    def complete(self, *, model: str, system: str, messages: list,
                 tools: list | None, thinking: dict | None,
                 max_tokens: int) -> CompletionResponse:
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": _openrouter_messages(system, messages),
        }
        if tools:
            kwargs["tools"] = [_tool_schema(tool) for tool in tools]
        extra_body: dict[str, Any] = {}
        reasoning = _openrouter_reasoning(thinking)
        if reasoning is not None:
            extra_body["reasoning"] = reasoning
        # The chat-completions transcript flattens tool results into string
        # `tool` messages, which carry no per-block breakpoint. OpenRouter's
        # top-level field caches up to the last cacheable block instead and
        # advances that breakpoint as the transcript grows.
        if _supports_auto_cache(model):
            extra_body["cache_control"] = dict(CACHE_CONTROL)
        if extra_body:
            kwargs["extra_body"] = extra_body
        response = self._client.chat.completions.create(**kwargs)
        embedded_error = getattr(response, "error", None)
        if embedded_error:
            raise ProviderResponseError(f"OpenRouter error: {embedded_error}")
        if not getattr(response, "choices", None):
            raise ProviderResponseError("OpenRouter returned no completion choices")
        choice = response.choices[0]
        choice_error = getattr(choice, "error", None)
        if choice_error:
            raise ProviderResponseError(f"OpenRouter completion error: {choice_error}")
        message = choice.message
        blocks: list[ContentBlock] = []
        content = getattr(message, "content", None)
        if isinstance(content, str) and content:
            blocks.append(ContentBlock(type="text", text=content))
        elif isinstance(content, list):
            for item in content:
                value = _model_dump(item) if not isinstance(item, dict) else item
                if value.get("type") in {"text", "output_text"}:
                    blocks.append(ContentBlock(
                        type="text", text=str(value.get("text", ""))))
        for detail in getattr(message, "reasoning_details", None) or []:
            raw = _model_dump(detail) if not isinstance(detail, dict) else dict(detail)
            blocks.append(ContentBlock(
                type="reasoning", raw={"type": "reasoning", "detail": raw}))
        for call in getattr(message, "tool_calls", None) or []:
            function = call.function
            name = getattr(function, "name", "") or ""
            blocks.append(ContentBlock(
                type="tool_use", id=getattr(call, "id", "") or "",
                name=name,
                input=_tool_arguments(getattr(function, "arguments", None), name),
            ))

        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        details = getattr(usage, "prompt_tokens_details", None)
        cache_read = getattr(details, "cached_tokens", 0) or 0
        cache_write = getattr(details, "cache_write_tokens", 0) or 0
        uncached_input = max(0, prompt_tokens - cache_read - cache_write)
        reported_cost = getattr(usage, "cost", None)
        return CompletionResponse(
            content=blocks,
            stop_reason=(
                "tool_use" if getattr(choice, "finish_reason", "") == "tool_calls"
                else (getattr(choice, "finish_reason", "") or "")
            ),
            usage=CompletionUsage(
                input_tokens=uncached_input,
                output_tokens=getattr(usage, "completion_tokens", 0) or 0,
                cache_write_tokens=cache_write,
                cache_read_tokens=cache_read,
                cost_usd=(float(reported_cost) if reported_cost is not None else None),
            ),
        )


class LLMClient:
    """Provider-neutral client that records every normalized completion."""

    def __init__(self, ledger: Ledger, max_tokens: int = 8192, *,
                 provider: str = config.DEFAULT_PROVIDER,
                 openrouter_base_url: str = config.OPENROUTER_BASE_URL,
                 openrouter_site_url: str = config.OPENROUTER_SITE_URL,
                 openrouter_app_name: str = config.OPENROUTER_APP_NAME,
                 adapter: ProviderAdapter | None = None) -> None:
        self.provider = config.normalize_provider(provider)
        if adapter is not None:
            self._adapter = adapter
        elif self.provider == "anthropic":
            self._adapter = AnthropicProvider()
        else:
            self._adapter = OpenRouterProvider(
                base_url=openrouter_base_url,
                site_url=openrouter_site_url,
                app_name=openrouter_app_name,
            )
        self.ledger = ledger
        self.max_tokens = max_tokens

    @classmethod
    def for_run(cls, ledger: Ledger, cfg: config.RunConfig, *,
                provider: str | None = None,
                max_tokens: int | None = None) -> "LLMClient":
        """Build a client from shared run settings for agent or judge calls."""
        return cls(
            ledger,
            max_tokens=cfg.max_tokens if max_tokens is None else max_tokens,
            provider=provider or cfg.provider,
            openrouter_base_url=cfg.openrouter_base_url,
            openrouter_site_url=cfg.openrouter_site_url,
            openrouter_app_name=cfg.openrouter_app_name,
        )

    def complete(self, *, role: str, model: str, system: str, messages: list,
                 tools: list | None = None,
                 thinking: dict | None = None) -> CompletionResponse:
        routed_model = config.api_model(self.provider, model)
        request = {
            "system": system,
            "messages": messages,
            "tools": tools,
            "thinking": thinking,
        }
        # Pinned prices support a true pre-call ceiling. Arbitrary OpenRouter
        # models remain usable via their exact provider-reported post-call cost.
        if config.has_pinned_price(routed_model):
            self.ledger.ensure_capacity(config.request_cost_upper_bound(
                routed_model, request, max_output_tokens=self.max_tokens))
        response = self._adapter.complete(
            model=routed_model,
            system=system,
            messages=messages,
            tools=tools,
            thinking=thinking,
            max_tokens=self.max_tokens,
        )
        self.ledger.record_usage(role, routed_model, response.usage)
        return response

"""Stirrup (Artificial Analysis) wrapped as an AgentRuntime.

Stirrup is embedded as a library: its agent loop and code_exec tool do the
work, while two thin subclasses bolt it onto our accounting and workspace:

* `LedgerLiteLLMClient` records every completion's token usage into the shared
  Ledger (pinned pricing; raises BudgetExceeded mid-run, which Stirrup's loop
  does not swallow).
* `WorkspaceExecToolProvider` roots Stirrup's local code_exec at the task's
  Workspace instead of a fresh temp dir, and leaves cleanup to the Workspace.

Cost caveat: Stirrup's TokenUsage has no cache split, so all input tokens are
billed at the full input rate — a conservative overestimate.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from fusion import config
from fusion.llm import Ledger
from fusion.workspace import Workspace

from .base import RuntimeResult, RuntimeUnavailable, task_prompt

try:
    from stirrup import Agent
    from stirrup.clients.litellm_client import LiteLLMClient
    from stirrup.core.models import AssistantMessage, ChatMessage, Tool, ToolMessage
    from stirrup.tools.code_backends.local import LocalCodeExecToolProvider
    from stirrup.utils.logging import AgentLogger
    _IMPORT_ERROR: Exception | None = None
except ImportError as exc:  # pragma: no cover - exercised only without the dep
    _IMPORT_ERROR = exc

SYSTEM_PROMPT = (
    "You are an expert software engineer fixing a bug in a repository. "
    "The repository is in your working directory. Explore it, make the "
    "smallest correct fix, run the tests to verify, then finish."
)


def _brief(value, limit: int = 140) -> str:
    """One-line, length-capped repr (matches fusion's audit-trail format)."""
    s = " ".join(str(value).split())
    return s if len(s) <= limit else s[: limit - 1] + "…"


class LedgerLiteLLMClient(LiteLLMClient):
    """LiteLLM client that records every completion into a fusion Ledger.

    Step limiting is left to Stirrup's own max_turns (which warns the model
    and ends gracefully); this wrapper only meters and budget-guards.
    """

    def __init__(self, *, model: str, ledger: Ledger, role: str,
                 pricing_model: str, **kwargs) -> None:
        super().__init__(model=model, **kwargs)
        self._ledger = ledger
        self._role = role
        self._pricing_model = pricing_model
        self.calls = 0

    async def generate(self, messages: list["ChatMessage"],
                       tools: dict[str, "Tool"]) -> "AssistantMessage":
        msg = await super().generate(messages, tools)
        self.calls += 1
        u = msg.token_usage
        # No cache split in Stirrup's TokenUsage: bill all input at full rate.
        self._ledger.record_tokens(self._role, self._pricing_model,
                                   input_tokens=u.input, output_tokens=u.output)
        return msg


class WorkspaceExecToolProvider(LocalCodeExecToolProvider):
    """Local code_exec rooted at an existing directory the Workspace owns.

    The parent creates (and on exit deletes) its own temp dir; here the
    directory is the task workspace, so setup just points at it and teardown
    must NOT remove it — Workspace.cleanup() owns that.
    """

    def __init__(self, root: Path, **kwargs) -> None:
        super().__init__(**kwargs)
        self._root = Path(root)

    async def __aenter__(self):
        self._temp_dir = self._root
        return self.get_code_exec_tool(description=self._description)

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self._temp_dir = None


class StirrupRuntime:
    """Adapter over Stirrup's agent loop with code_exec in the task workspace."""

    name = "stirrup"
    # LiteLLM provider prefix. Subclasses (e.g. OpenRouterRuntime) swap this to
    # route the same loop through a different gateway; the bare model id is kept
    # for pricing lookups, which are provider-agnostic.
    provider = "anthropic"

    def preflight(self) -> tuple[bool, str]:
        if _IMPORT_ERROR is not None:
            return False, f"pip install 'stirrup[litellm]' ({_IMPORT_ERROR})"
        return True, "stirrup embedded"

    def run(self, task: dict, ws: Workspace, *, model: str,
            ledger: Ledger, cfg: config.RunConfig) -> RuntimeResult:
        if _IMPORT_ERROR is not None:
            raise RuntimeUnavailable(str(_IMPORT_ERROR))
        # Normalize so a provider-prefixed id (e.g. "anthropic/claude-sonnet-5"
        # routed via OpenRouter) still attributes to the "main" role.
        role = "main" if config._normalize(model) == config.MODEL_MAIN else "sidekick"
        client = LedgerLiteLLMClient(
            model=f"{self.provider}/{model}", ledger=ledger, role=role,
            pricing_model=model,
        )
        return asyncio.run(self._arun(task, ws, client, cfg))

    async def _arun(self, task: dict, ws: Workspace,
                    client: LedgerLiteLLMClient,
                    cfg: config.RunConfig) -> RuntimeResult:
        agent = Agent(
            client=client,
            name="stirrup-solver",
            system_prompt=SYSTEM_PROMPT,
            tools=[WorkspaceExecToolProvider(ws.root)],
            max_turns=cfg.max_steps,
            # Quiet the rich spinner/summary panels; the eval layer owns logging.
            logger=AgentLogger(show_spinner=False, level=logging.WARNING),
        )
        # cache_on_interrupt=False: no SIGINT handler, and no caching of the
        # workspace when BudgetExceeded aborts the session mid-run.
        async with agent.session(cache_on_interrupt=False) as session:
            finish_params, history, run_metadata = await session.run(
                task_prompt(task))
        summary = (finish_params.reason if finish_params is not None
                   else "(max turns reached without finish)")
        return RuntimeResult(
            summary=summary,
            steps=client.calls,
            finished=finish_params is not None,
            trace=_trace_from_history(history),
            extra={"stirrup_run_metadata_keys": sorted(run_metadata)},
        )


class OpenRouterRuntime(StirrupRuntime):
    """Stirrup's LiteLLM loop routed through the OpenRouter gateway.

    LiteLLM addresses OpenRouter models as ``openrouter/<id>`` and reads the
    key from ``OPENROUTER_API_KEY`` in the environment, so this is the exact
    Stirrup path with a different provider prefix — no separate integration.
    """

    name = "openrouter"
    provider = "openrouter"

    def preflight(self) -> tuple[bool, str]:
        ok, reason = super().preflight()
        if not ok:
            return ok, reason
        if not os.environ.get("OPENROUTER_API_KEY"):
            return False, "set OPENROUTER_API_KEY (get one at openrouter.ai/keys)"
        return True, "openrouter via stirrup/litellm"


def _trace_from_history(history: list) -> list[dict]:
    """Flatten Stirrup's per-turn message history into fusion audit records."""
    trace: list[dict] = []
    step = 0
    for turn in history or []:
        results: dict[str, "ToolMessage"] = {}
        for m in turn:
            if isinstance(m, ToolMessage) and m.tool_call_id:
                results[m.tool_call_id] = m
        for m in turn:
            if not isinstance(m, AssistantMessage):
                continue
            step += 1
            for tc in m.tool_calls:
                res = results.get(tc.tool_call_id or "")
                trace.append({
                    "step": step, "role": "main", "tool": tc.name,
                    "input": _brief(tc.arguments, 80),
                    "outcome": _brief(res.content, 140) if res is not None else "",
                    "is_error": bool(res is not None and not getattr(res, "success", True)),
                })
    return trace

"""The existing fusion tool-use loop wrapped as an AgentRuntime.

Kept so all runtimes flow through one orchestration code path; comparing
`baseline-fusion` against the legacy `frontier_only` variant proves the new
plumbing is behavior-identical. Slated for removal once parity passes.
"""

from __future__ import annotations

from fusion import config
from fusion.agent import Agent
from fusion.llm import Ledger, LLMClient
from fusion.orchestrator import SOLVER_SYSTEM
from fusion.tools import FINISH_TOOL, READ_TOOLS, WRITE_TOOLS, WorkspaceTools
from fusion.workspace import Workspace

from .base import RuntimeResult, task_prompt


class FusionRuntime:
    """Adapter over fusion's Agent loop with the full read+write tool surface."""

    name = "fusion"

    def preflight(self) -> tuple[bool, str]:
        return True, "built-in"

    def run(self, task: dict, ws: Workspace, *, model: str,
            ledger: Ledger, cfg: config.RunConfig) -> RuntimeResult:
        client = LLMClient.for_run(ledger, cfg)
        tools = WorkspaceTools(ws)
        role = "main" if model == cfg.main_model else "sidekick"
        thinking = cfg.main_thinking if role == "main" else cfg.sidekick_thinking
        agent = Agent(
            role=role, model=model, system=SOLVER_SYSTEM,
            tools=READ_TOOLS + WRITE_TOOLS + [FINISH_TOOL],
            client=client, thinking=thinking, max_steps=cfg.max_steps,
        )
        result = agent.run(task_prompt(task), tools.execute)
        return RuntimeResult(summary=result.text, steps=result.steps,
                             finished=result.finished, trace=result.trace)

"""The comparison variants.

Each policy solves one task and returns a PolicyResult. They share the harness;
they differ only in *which* agent(s) run and *what tool surface* the main agent
has. Seams are left for the deferred variants (routed / self-verifying).
"""

from __future__ import annotations

# PolicyResult and the diff/score/cleanup packaging now live in the
# orchestrator layer (shared by all runtimes); this module keeps only the
# legacy fusion-loop variants until the parity milestone retires them.
from orchestrator.variants import PolicyResult, finalize, make_ws
from runtimes.base import task_prompt

from . import config, orchestrator
from .agent import Agent
from .llm import BudgetExceeded, Ledger, LLMClient
from .tools import (FINISH_TOOL, READ_TOOLS, SCOUT_TOOL, WRITE_TOOLS,
                    WorkspaceTools)


def _run_single_agent(task: dict, cfg: config.RunConfig, *, variant: str,
                      model: str, thinking: dict | None) -> PolicyResult:
    """frontier_only / sidekick_only: one agent, full read+write toolset."""
    ledger = Ledger(cap_usd=cfg.budget_usd)
    ws = make_ws(task)
    client = LLMClient(ledger, max_tokens=cfg.max_tokens)
    tools = WorkspaceTools(ws)
    agent = Agent(
        role=("main" if model == config.MODEL_MAIN else "sidekick"),
        model=model, system=orchestrator.SOLVER_SYSTEM,
        tools=READ_TOOLS + WRITE_TOOLS + [FINISH_TOOL],
        client=client, thinking=thinking, max_steps=cfg.max_steps,
    )
    return _finalize(variant, task, ws, ledger, agent, tools.execute)


def _run_scout(task: dict, cfg: config.RunConfig) -> PolicyResult:
    """scout (A): Sonnet main whose reads are delegated to a Haiku Scout."""
    ledger = Ledger(cap_usd=cfg.budget_usd)
    ws = make_ws(task)
    client = LLMClient(ledger, max_tokens=cfg.max_tokens)
    tools = WorkspaceTools(ws)
    scout_trace: list = []

    def execute(name: str, inp: dict) -> tuple[str, bool]:
        if name == "scout":
            try:
                m, sub_trace = orchestrator.make_scout_map(
                    inp["question"], ws, client,
                    max_steps=cfg.scout_max_steps,
                    sidekick_model=config.MODEL_SIDEKICK,
                    thinking=cfg.sidekick_thinking,
                )
                scout_trace.extend(sub_trace)
                return m, False
            except BudgetExceeded:
                raise
            except Exception as exc:
                return f"ERROR: scout failed: {exc}", True
        return tools.execute(name, inp)

    # Main gets scout + write tools, but NOT raw read tools: exploration is delegated.
    agent = Agent(
        role="main", model=config.MODEL_MAIN, system=orchestrator.SOLVER_SYSTEM,
        tools=[SCOUT_TOOL] + WRITE_TOOLS + [FINISH_TOOL],
        client=client, thinking=cfg.main_thinking, max_steps=cfg.max_steps,
    )
    return _finalize("scout", task, ws, ledger, agent, execute, scout_trace=scout_trace)


def _finalize(variant, task, ws, ledger, agent, execute,
              scout_trace: list | None = None) -> PolicyResult:
    budget_hit = False
    err = ""
    steps = 0
    finished = False
    trace: list = []
    summary = ""
    try:
        result = agent.run(task_prompt(task), execute)
        summary = result.text
        steps, finished, trace = result.steps, result.finished, result.trace
    except BudgetExceeded as exc:
        budget_hit = True
        summary = f"(budget hit) {exc}"
    except Exception as exc:  # keep the run alive; record the failure
        err = f"{type(exc).__name__}: {exc}"
    return finalize(variant, task, ws, ledger, summary=summary, steps=steps,
                    finished=finished, trace=trace, scout_trace=scout_trace,
                    budget_hit=budget_hit, error=err)


# --- registry ---------------------------------------------------------------
def run_variant(name: str, task: dict, cfg: config.RunConfig) -> PolicyResult:
    if name == "frontier_only":
        return _run_single_agent(task, cfg, variant=name,
                                  model=config.MODEL_MAIN, thinking=cfg.main_thinking)
    if name == "sidekick_only":
        return _run_single_agent(task, cfg, variant=name,
                                  model=config.MODEL_SIDEKICK, thinking=cfg.sidekick_thinking)
    if name == "scout":
        return _run_scout(task, cfg)
    raise ValueError(f"unknown variant: {name}")


ALL_VARIANTS = ["frontier_only", "sidekick_only", "scout"]

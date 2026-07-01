"""The comparison variants.

Each policy solves one task and returns a PolicyResult. They share the harness;
they differ only in *which* agent(s) run and *what tool surface* the main agent
has. Seams are left for the deferred variants (routed / self-verifying).
"""

from __future__ import annotations

from dataclasses import dataclass

from . import config, orchestrator
from .agent import Agent
from .llm import BudgetExceeded, Ledger, LLMClient
from .tools import (FINISH_TOOL, READ_TOOLS, SCOUT_TOOL, WRITE_TOOLS,
                    WorkspaceTools)
from .workspace import Workspace


@dataclass
class PolicyResult:
    variant: str
    resolved: bool
    diff: str
    summary: str
    ledger: dict
    budget_hit: bool = False
    error: str = ""
    resolve_detail: str = ""


def _task_prompt(task: dict) -> str:
    return (
        f"Bug report:\n{task['problem_statement']}\n\n"
        f"The repository is in your working directory. Fix it and make the tests pass."
    )


def _make_ws(task: dict) -> Workspace:
    """Native tasks are copied to a temp dir; swebench tasks are edited in place
    (the venv's editable install points at the clone), reset via git first."""
    if task.get("in_place"):
        if task.get("reset"):
            task["reset"](task)
        return Workspace(task["template_dir"], task["test_cmd"],
                         use_git=True, test_timeout=task.get("test_timeout", 600))
    return Workspace.from_template(task["template_dir"], task["test_cmd"],
                                   test_timeout=task.get("test_timeout", 120))


def _run_single_agent(task: dict, cfg: config.RunConfig, *, variant: str,
                      model: str, thinking: dict | None) -> PolicyResult:
    """frontier_only / sidekick_only: one agent, full read+write toolset."""
    ledger = Ledger(cap_usd=cfg.budget_usd)
    ws = _make_ws(task)
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
    ws = _make_ws(task)
    client = LLMClient(ledger, max_tokens=cfg.max_tokens)
    tools = WorkspaceTools(ws)

    def execute(name: str, inp: dict) -> tuple[str, bool]:
        if name == "scout":
            try:
                m = orchestrator.make_scout_map(
                    inp["question"], ws, client,
                    max_steps=cfg.scout_max_steps,
                    sidekick_model=config.MODEL_SIDEKICK,
                    thinking=cfg.sidekick_thinking,
                )
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
    return _finalize("scout", task, ws, ledger, agent, execute)


def _finalize(variant, task, ws, ledger, agent, execute) -> PolicyResult:
    budget_hit = False
    err = ""
    try:
        result = agent.run(_task_prompt(task), execute)
        summary = result.text
    except BudgetExceeded as exc:
        budget_hit = True
        summary = f"(budget hit) {exc}"
    except Exception as exc:  # keep the run alive; record the failure
        err = f"{type(exc).__name__}: {exc}"
        summary = ""
    # Capture the agent's diff BEFORE scoring: the scorer applies/reverts the
    # gold test patch and must not pollute the recorded change.
    diff = ws.diff()
    # swebench tasks score the SWE-bench way (gold test patch + FAIL/PASS_TO_PASS);
    # native tasks just run their own test command.
    detail = ""
    if task.get("scorer"):
        try:
            scored = task["scorer"](task)
            if isinstance(scored, tuple):
                resolved, detail = bool(scored[0]), str(scored[1])
            else:
                resolved = bool(scored)
        except Exception as exc:  # noqa: BLE001
            resolved, err = False, err or f"scorer: {exc}"
    else:
        resolved, _ = ws.run_tests()
    out = PolicyResult(
        variant=variant, resolved=resolved, diff=diff, summary=summary,
        ledger=ledger.summary(), budget_hit=budget_hit, error=err,
        resolve_detail=detail,
    )
    ws.cleanup()
    return out


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

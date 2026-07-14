"""Prompts and the Scout delegation wiring shared by the policies."""

from __future__ import annotations

from .agent import Agent
from .llm import LLMClient
from .tools import WorkspaceTools
from .workspace import Workspace

SOLVER_SYSTEM = """You are an expert software engineer fixing a bug in a repository.
Work in this loop: locate the defect, make the minimal correct change, then run the
tests to confirm they pass. Keep the change tightly scoped to the reported problem —
do not refactor, rename, or add unrelated code. NEVER modify the repository's
existing test files: grading restores them to their original state and runs the
official tests, so any "fix" made inside a test file is discarded and scores zero.
The fix must live in the library/source code. When the tests pass, call finish()
with a one-line summary. If tests still fail after a few attempts, call finish()
anyway with what you found."""

SCOUT_SYSTEM = """You are a read-only repository Scout. You never edit code. Your job
is to answer a locate/understand question from the main engineer and return a COMPACT,
CITED map: the specific files and line ranges that matter (as `path:line`), plus one
short clause each on why they are relevant and what the likely defect is. Do not paste
large file bodies — cite and summarize. Be precise and brief; the engineer pays for
every token you return to them."""

PLANNER_SYSTEM = """You are a read-only repository Planner-Locator. You never edit code.
A senior engineer will author the fix FROM YOUR BRIEF ALONE, without exploring the repository
themselves, so the brief must be self-contained and correct. Explore with your read-only tools,
then return a COMPACT brief with exactly these sections:

1. ROOT CAUSE — 1-2 sentences naming the actual defect.
2. EDIT PLAN — the specific `path:line` site(s) to change and precisely what the change is.
3. CODE CONTEXT — for each edit site, paste the exact current lines (with a few lines of
   surrounding context) VERBATIM, so the engineer can find the precise text to replace without
   reopening the file.
4. TESTS — the test path or pytest `-k` expression that exercises this fix.

Be precise and brief; cite only what you actually read. Getting the location and the verbatim
code context right matters far more than length — a vague or wrong location makes the whole
brief worthless."""


def make_scout_map(question: str, ws: Workspace, client: LLMClient,
                   max_steps: int, sidekick_model: str,
                   thinking: dict | None) -> tuple[str, list]:
    """Spin a fresh read-only Scout agent (own context). Returns (cited map,
    the scout's own tool-call trace) so the run is fully auditable."""
    from .tools import READ_TOOLS
    tools = WorkspaceTools(ws)
    scout = Agent(
        role="sidekick", model=sidekick_model, system=SCOUT_SYSTEM,
        tools=READ_TOOLS, client=client, thinking=thinking, max_steps=max_steps,
    )
    prompt = (
        f"The main engineer needs help locating/understanding something to fix a bug.\n"
        f"Question: {question}\n\n"
        f"Explore the repo with your read-only tools, then reply with the cited map."
    )
    result = scout.run(prompt, tools.execute)
    return (result.text or "(scout returned no map)"), result.trace


def make_plan_brief(task_prompt_text: str, ws: Workspace, client: LLMClient,
                    max_steps: int, planner_model: str,
                    thinking: dict | None) -> tuple[str, list]:
    """Spin a read-only Planner-Locator (the cheap model). It both localizes the
    defect and drafts the fix plan, returning a self-contained brief the frontier
    author works from without exploring the repo. Returns (brief, planner trace)."""
    from .tools import READ_TOOLS
    tools = WorkspaceTools(ws)
    planner = Agent(
        role="sidekick", model=planner_model, system=PLANNER_SYSTEM,
        tools=READ_TOOLS, client=client, thinking=thinking, max_steps=max_steps,
    )
    prompt = ("Locate the defect and write the self-contained fix brief for this task.\n\n"
              + task_prompt_text)
    result = planner.run(prompt, tools.execute)
    return (result.text or "(planner returned no brief)"), result.trace

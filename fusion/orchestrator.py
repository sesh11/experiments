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


def make_scout_map(question: str, ws: Workspace, client: LLMClient,
                   max_steps: int, sidekick_model: str,
                   thinking: dict | None) -> str:
    """Spin a fresh read-only Scout agent (own context) and return its cited map."""
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
    return result.text or "(scout returned no map)"

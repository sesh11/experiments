"""The runtime contract every agent harness adapter implements.

A runtime is handed a task, a Workspace rooted at the code to fix, a model id,
and a Ledger. It must:
  * edit files only under ``ws.root`` (diff capture relies on this),
  * record every LLM call's usage into ``ledger`` (pinned pricing keeps cost
    numbers comparable across harnesses),
  * let ``BudgetExceeded`` propagate — the orchestrator turns it into a
    budget-capped result row.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from fusion.config import RunConfig
from fusion.llm import Ledger
from fusion.workspace import Workspace


@dataclass
class RuntimeResult:
    """What a runtime hands back after attempting one task."""
    summary: str                                  # final agent text
    steps: int                                    # agent turns / tool-loop steps used
    finished: bool                                # did the agent declare completion?
    trace: list = field(default_factory=list)     # audit records: step/role/tool/input/outcome/is_error
    extra: dict = field(default_factory=dict)     # runtime-specific artifacts (e.g. pi's own cost figure)


class RuntimeUnavailable(RuntimeError):
    """The runtime's external dependency (package, binary) is missing."""


def task_prompt(task: dict) -> str:
    """The shared task framing, identical across runtimes so runs are comparable."""
    return (
        f"Bug report:\n{task['problem_statement']}\n\n"
        f"The repository is in your working directory. Fix it and make the tests pass."
    )


@runtime_checkable
class AgentRuntime(Protocol):
    """One agent harness (fusion loop, Stirrup, pi) behind a uniform interface."""

    name: str

    def preflight(self) -> tuple[bool, str]:
        """Return (ready, reason). A False result must explain how to install."""
        ...

    def run(self, task: dict, ws: Workspace, *, model: str,
            ledger: Ledger, cfg: RunConfig) -> RuntimeResult:
        """Solve `task` by editing files under `ws.root`; record usage in `ledger`."""
        ...

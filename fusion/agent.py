"""A generic tool-using agent loop. One instance = one warm-cache context.

The agent owns its own `messages` list and model id, so the main agent and the
sidekick never share a transcript — delegation passes compact briefs, not the
full history. The actual tool execution is delegated to a callback so the
orchestrator can intercept certain tools (e.g. `scout`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .llm import LLMClient


@dataclass
class AgentResult:
    text: str                        # final assistant text (or scout map)
    steps: int
    finished: bool                   # True if the agent called finish()
    transcript_tail: str = ""        # last assistant text, for debugging
    trace: list = field(default_factory=list)  # per-tool-call audit records


ToolExecutor = Callable[[str, dict], tuple[str, bool]]  # (name, input) -> (result, is_error)


def _brief(value, limit: int = 140) -> str:
    """One-line, length-capped repr of a tool input value or result."""
    s = " ".join(str(value).split())
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _input_brief(inp: dict) -> str:
    """Compact `k=v` view of a tool's arguments for the audit trail."""
    parts = []
    for k, v in inp.items():
        parts.append(f"{k}={_brief(v, 80)}")
    return ", ".join(parts)


class Agent:
    def __init__(self, *, role: str, model: str, system: str, tools: list,
                 client: LLMClient, thinking: dict | None = None,
                 max_steps: int = 14):
        self.role = role
        self.model = model
        self.system = system
        self.tools = tools
        self.client = client
        self.thinking = thinking
        self.max_steps = max_steps
        self.messages: list = []

    def run(self, user_message: str, execute: ToolExecutor) -> AgentResult:
        self.messages.append({"role": "user", "content": user_message})
        last_text = ""
        finished = False
        trace: list = []

        for step in range(1, self.max_steps + 1):
            resp = self.client.complete(
                role=self.role, model=self.model, system=self.system,
                messages=self.messages, tools=self.tools, thinking=self.thinking,
            )
            self.messages.append({"role": "assistant", "content": resp.content})

            text_blocks = [b.text for b in resp.content if b.type == "text"]
            if text_blocks:
                last_text = "\n".join(text_blocks)

            if resp.stop_reason != "tool_use":
                return AgentResult(text=last_text, steps=step, finished=finished,
                                   transcript_tail=last_text, trace=trace)

            tool_results = []
            for block in resp.content:
                if block.type != "tool_use":
                    continue
                if block.name == "finish":
                    finished = True
                    summary = block.input.get("summary", "")
                    last_text = summary or last_text
                    trace.append({"step": step, "role": self.role, "tool": "finish",
                                  "input": _brief(summary, 120), "outcome": "task marked complete",
                                  "is_error": False})
                    tool_results.append({
                        "type": "tool_result", "tool_use_id": block.id,
                        "content": "OK: task marked complete.",
                    })
                    continue
                result, is_error = execute(block.name, block.input)
                trace.append({"step": step, "role": self.role, "tool": block.name,
                              "input": _input_brief(block.input),
                              "outcome": _brief(result, 140), "is_error": is_error})
                tool_results.append({
                    "type": "tool_result", "tool_use_id": block.id,
                    "content": result, "is_error": is_error,
                })

            self.messages.append({"role": "user", "content": tool_results})
            if finished:
                return AgentResult(text=last_text, steps=step, finished=True,
                                   transcript_tail=last_text, trace=trace)

        return AgentResult(text=last_text, steps=self.max_steps, finished=False,
                           transcript_tail=last_text, trace=trace)

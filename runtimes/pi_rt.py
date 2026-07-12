"""pi (Mario Zechner's minimal coding agent) wrapped as an AgentRuntime.

pi is driven as a subprocess in JSON mode with its own default tool surface
(read/bash/edit/write) and system prompt, cwd'd into the task workspace.
Hermetic flags disable sessions, extensions, skills, and CLAUDE.md discovery
so runs are reproducible and never touch the user's pi setup.

Accounting is post-hoc: pi has no turn-limit flag, so the wall-clock timeout
(cfg.runtime_timeout_s) is the in-flight guard; token totals are summed from
the JSON stream at session end and recorded once. Cost is recomputed from
pinned pricing — pi's own figure (kept in extra["pi_reported_cost_usd"]) uses
its bundled registry, which may not know newer model ids.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

from fusion import config
from fusion.llm import Ledger
from fusion.workspace import Workspace

from .base import RuntimeResult, RuntimeUnavailable, task_prompt

_INSTALL_HINT = "npm i -g @mariozechner/pi-coding-agent (or set PI_BIN)"


def _find_pi() -> str | None:
    """Locate the pi binary: PI_BIN override first, then PATH."""
    override = os.environ.get("PI_BIN")
    if override:
        return override if os.path.exists(override) else None
    return shutil.which("pi")


def _brief(value, limit: int = 140) -> str:
    """One-line, length-capped repr (matches fusion's audit-trail format)."""
    s = " ".join(str(value).split())
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _result_text(result: dict) -> str:
    """Flatten a pi tool result ({"content": [{type,text}...]}) to plain text."""
    parts = [b.get("text", "") for b in (result or {}).get("content", [])
             if isinstance(b, dict)]
    return "\n".join(p for p in parts if p)


class PiRuntime:
    """Adapter that shells out to pi in non-interactive JSON mode."""

    name = "pi"

    def preflight(self) -> tuple[bool, str]:
        pi_bin = _find_pi()
        if pi_bin is None:
            return False, _INSTALL_HINT
        return True, pi_bin

    def run(self, task: dict, ws: Workspace, *, model: str,
            ledger: Ledger, cfg: config.RunConfig) -> RuntimeResult:
        pi_bin = _find_pi()
        if pi_bin is None:
            raise RuntimeUnavailable(_INSTALL_HINT)
        cmd = [
            pi_bin, "--provider", "anthropic", "--model", model,
            "--thinking", "off", "--mode", "json", "--print",
            "--no-session", "--no-extensions", "--no-skills",
            "--no-prompt-templates", "--no-themes", "--no-context-files",
            task_prompt(task),
        ]
        timed_out = False
        try:
            proc = subprocess.run(
                cmd, cwd=ws.root, capture_output=True, text=True,
                timeout=cfg.runtime_timeout_s, env=os.environ.copy(),
            )
            stdout, stderr, returncode = proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            returncode = -1
            if isinstance(stdout, bytes):
                stdout = stdout.decode(errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")

        events = _parse_events(stdout)
        totals, pi_cost, steps, last_text, last_stop, api_error = _digest(events)
        role = "main" if model == config.MODEL_MAIN else "sidekick"
        # Post-hoc accounting: one record for the whole session, cost from
        # pinned pricing. May raise BudgetExceeded (caught by the orchestrator).
        ledger.record_tokens(
            role, model,
            input_tokens=totals["input"], output_tokens=totals["output"],
            cache_write_tokens=totals["cacheWrite"],
            cache_read_tokens=totals["cacheRead"],
        )
        recomputed = ledger.by_role[role].cost_usd
        extra = {
            "pi_reported_cost_usd": round(pi_cost, 6),
            "recomputed_cost_usd": round(recomputed, 6),
        }
        if pi_cost and abs(pi_cost - recomputed) / max(recomputed, 1e-9) > 0.10:
            print(f"    ! pi self-reported cost ${pi_cost:.4f} deviates >10% "
                  f"from pinned-pricing recompute ${recomputed:.4f} "
                  f"(pi's registry may not know '{model}')")
        if timed_out:
            raise TimeoutError(
                f"pi hit the {cfg.runtime_timeout_s}s wall-clock cap "
                f"(usage recorded: {totals})")
        if api_error:
            raise RuntimeError(f"pi API error: {_brief(api_error, 200)}")
        if returncode != 0:
            raise RuntimeError(
                f"pi exited {returncode}: {_brief(stderr or stdout, 200)}")
        return RuntimeResult(
            summary=last_text or "(no final text)",
            steps=steps,
            finished=last_stop == "stop",
            trace=_trace_from_events(events),
            extra=extra,
        )


def _parse_events(stdout: str) -> list[dict]:
    events = []
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _digest(events: list[dict]) -> tuple[dict, float, int, str, str, str]:
    """Sum usage over assistant message_end events; pull steps/summary/status."""
    totals = {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}
    pi_cost = 0.0
    steps = 0
    last_text = ""
    last_stop = ""
    api_error = ""
    for e in events:
        if e.get("type") != "message_end":
            continue
        msg = e.get("message", {})
        if msg.get("role") != "assistant":
            continue
        steps += 1
        usage = msg.get("usage", {})
        for key in totals:
            totals[key] += int(usage.get(key, 0) or 0)
        pi_cost += float((usage.get("cost") or {}).get("total", 0) or 0)
        texts = [b.get("text", "") for b in msg.get("content", [])
                 if isinstance(b, dict) and b.get("type") == "text"]
        if any(texts):
            last_text = "\n".join(t for t in texts if t)
        last_stop = msg.get("stopReason", "") or last_stop
        if msg.get("errorMessage"):
            api_error = msg["errorMessage"]
    return totals, pi_cost, steps, last_text, last_stop, api_error


def _trace_from_events(events: list[dict]) -> list[dict]:
    """Build fusion audit records from toolCall blocks + tool_execution_end."""
    results: dict[str, dict] = {}
    for e in events:
        if e.get("type") == "tool_execution_end" and e.get("toolCallId"):
            results[e["toolCallId"]] = e
    trace: list[dict] = []
    step = 0
    for e in events:
        if e.get("type") != "message_end":
            continue
        msg = e.get("message", {})
        if msg.get("role") != "assistant":
            continue
        step += 1
        for block in msg.get("content", []):
            if not isinstance(block, dict) or block.get("type") != "toolCall":
                continue
            res = results.get(block.get("id", ""), {})
            trace.append({
                "step": step, "role": "main", "tool": block.get("name", "?"),
                "input": _brief(block.get("arguments", {}), 80),
                "outcome": _brief(_result_text(res.get("result", {})), 140),
                "is_error": bool(res.get("isError")),
            })
    return trace

"""pi (Mario Zechner's minimal coding agent) wrapped as an AgentRuntime.

pi is driven as a subprocess in JSON mode with its own default tool surface
(read/bash/edit/write) and system prompt, cwd'd into the task workspace.
Hermetic flags disable sessions, extensions, skills, and CLAUDE.md discovery
so runs are reproducible and never touch the user's pi setup.

pi has no turn-limit flag, so the wall-clock timeout (cfg.runtime_timeout_s)
remains an in-flight guard. Usage is monitored from the live JSON stream and pi
is stopped before another full-context request could exceed the cell cap. Token
totals are consolidated into the shared ledger at session end. Cost is
recomputed from pinned pricing — pi's own figure (kept in
extra["pi_reported_cost_usd"]) may use different registry prices.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

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
        # pi only reports usage in its JSON event stream. Read that stream live
        # and stop before another turn when a full-context request could exceed
        # the cell cap. This converts the former post-hoc-only guard into a hard
        # bound with one request's maximum reserved before every next turn.
        next_call_ceiling = config.absolute_request_cost_upper_bound(
            model, max_output_tokens=cfg.max_tokens)
        ledger.ensure_capacity(next_call_ceiling)
        timed_out = [False]
        budget_stopped = False
        lines: list[str] = []
        live_cost = 0.0
        pi_config_dir = Path(ws.root) / ".fusion_pi"
        pi_config_dir.mkdir(exist_ok=True)
        (pi_config_dir / "models.json").write_text(json.dumps({
            "providers": {
                "anthropic": {
                    "modelOverrides": {model: {"maxTokens": cfg.max_tokens}}
                }
            }
        }))
        pi_env = os.environ.copy()
        pi_env["PI_CODING_AGENT_DIR"] = str(pi_config_dir)
        pi_env["PI_CODING_AGENT_SESSION_DIR"] = str(pi_config_dir / "sessions")
        with tempfile.TemporaryFile(mode="w+") as err_file:
            proc = subprocess.Popen(
                cmd, cwd=ws.root, stdout=subprocess.PIPE, stderr=err_file,
                text=True, env=pi_env,
            )

            def kill_on_timeout() -> None:
                if proc.poll() is None:
                    timed_out[0] = True
                    proc.kill()

            timer = threading.Timer(cfg.runtime_timeout_s, kill_on_timeout)
            timer.daemon = True
            timer.start()
            assert proc.stdout is not None
            try:
                for line in proc.stdout:
                    lines.append(line)
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if event.get("type") != "message_end":
                        continue
                    msg = event.get("message", {})
                    if msg.get("role") != "assistant":
                        continue
                    usage = msg.get("usage", {})
                    live_cost += config.cost_for(
                        model,
                        input_tokens=int(usage.get("input", 0) or 0),
                        output_tokens=int(usage.get("output", 0) or 0),
                        cache_read_tokens=int(usage.get("cacheRead", 0) or 0),
                        cache_write_tokens=int(usage.get("cacheWrite", 0) or 0),
                    )
                    if (msg.get("stopReason") != "stop"
                            and live_cost + next_call_ceiling > ledger.cap_usd):
                        budget_stopped = True
                        proc.terminate()
                        break
                returncode = proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                returncode = proc.wait()
            finally:
                timer.cancel()
                proc.stdout.close()
            err_file.seek(0)
            stderr = err_file.read()
        stdout = "".join(lines)

        events = _parse_events(stdout)
        totals, pi_cost, steps, last_text, last_stop, api_error = _digest(events)
        role = "main" if model == config.MODEL_MAIN else "sidekick"
        # Consolidate the live-observed stream into the shared ledger using
        # pinned pricing (the scheduler only consumes this common interface).
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
            "budget_enforcement": "live-event-stream",
        }
        if pi_cost and abs(pi_cost - recomputed) / max(recomputed, 1e-9) > 0.10:
            extra["pricing_warning"] = (
                f"pi self-reported cost ${pi_cost:.4f} deviates >10% from "
                f"pinned-pricing recompute ${recomputed:.4f} "
                f"(pi's registry may not know '{model}')")
        if budget_stopped:
            from fusion.llm import BudgetExceeded
            raise BudgetExceeded(
                f"pi stopped before its next turn could exceed the "
                f"${ledger.cap_usd:.2f} cell cap (spent ${ledger.total_cost:.2f})")
        if timed_out[0]:
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

"""Per-run audit logging: a terminal summary you can read at a glance, plus a
full-detail log file per run so a failure is never a guess.

Every (task, variant) run produces:
  * a compact multi-line block on stdout — verdict, what the agent changed,
    whether it touched test files, the scoring breakdown, and cost split;
  * a `results/runs/<stamp>/<task>__<variant>.log` file with the complete
    tool-call trace, the full diff, and the actual pytest output from scoring.

The whole point: separate "the agent failed" from "the scorer/env failed".
If the diff is empty or full of ERRORs -> agent. If the diff is a real fix but
FAIL_TO_PASS still fails -> read the captured pytest output to see why.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

# Path fragments that mean "this is a test file" — edits here are discarded by
# SWE-bench scoring, so touching them is almost always an agent mistake.
_TEST_RE = re.compile(r"(^|/)(tests?|testing)/|(^|/)(test_|conftest)|_test\.py")


def _diff_files(diff: str) -> list[str]:
    """Paths a unified diff touches (from `+++ b/...` / `diff --git` lines)."""
    files: list[str] = []
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            files.append(line[6:].strip())
        elif line.startswith("diff --git "):
            tok = line.split()[-1]
            files.append(tok[2:] if tok.startswith("b/") else tok)
    seen: dict[str, None] = {}
    for f in files:
        if f and f != "dev/null":
            seen.setdefault(f, None)
    return list(seen)


def _touched_tests(files: list[str]) -> list[str]:
    return [f for f in files if _TEST_RE.search(f)]


def _fmt_trace(trace: list, indent: str = "  ") -> str:
    lines = []
    for t in trace:
        mark = "✗" if t.get("is_error") else " "
        lines.append(f"{indent}{mark} [{t.get('step','?'):>2}] {t.get('tool','?')}"
                     f"({t.get('input','')})")
        outcome = t.get("outcome", "")
        if outcome:
            lines.append(f"{indent}       -> {outcome}")
    return "\n".join(lines) or f"{indent}(no tool calls recorded)"


class Audit:
    def __init__(self, out_dir: str | Path = "results"):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir = Path(out_dir) / "runs" / stamp
        self.dir.mkdir(parents=True, exist_ok=True)
        self.jsonl = self.dir / "runs.jsonl"
        self.count = 0

    def record(self, *, task: dict, variant: str, res, quality, would_merge,
               remaining_before: float) -> str:
        """Write the full per-run log + JSONL row; return a terminal summary."""
        self.count += 1
        files = _diff_files(res.diff or "")
        test_files = _touched_tests(files)
        art = res.score_artifacts or {}
        iid = task["instance_id"]

        # --- full detail log file ------------------------------------------
        log_path = self.dir / f"{iid}__{variant}.log"
        parts: list[str] = []
        parts.append(f"=== {iid} :: {variant} ===")
        parts.append(f"resolved: {res.resolved}   detail: {res.resolve_detail}")
        parts.append(f"cost: ${res.ledger.get('total_cost_usd', 0):.4f}  "
                     f"(main ${res.ledger.get('main_cost_usd', 0)} / "
                     f"sidekick ${res.ledger.get('sidekick_cost_usd', 0)})")
        parts.append(f"steps: {res.steps}   finished(finish() called): {res.finished}   "
                     f"budget_hit: {res.budget_hit}")
        if quality is not None:
            parts.append(f"judge quality: {quality}   would_merge: {would_merge}")
        if res.error:
            parts.append(f"ERROR: {res.error}")
        parts.append("")
        parts.append(f"--- diff touches {len(files)} file(s); "
                     f"test files touched: {test_files or 'none'} ---")
        parts.append(res.diff or "(no diff)")
        parts.append("")
        parts.append("--- main agent tool-call trace ---")
        parts.append(_fmt_trace(res.trace))
        if res.scout_trace:
            parts.append("")
            parts.append("--- scout (sidekick) tool-call trace ---")
            parts.append(_fmt_trace(res.scout_trace))
        parts.append("")
        parts.append("--- scoring evidence ---")
        if not art:
            parts.append("(native task: scored by its own test command, no SWE-bench artifacts)")
        elif not art.get("apply_ok", True):
            parts.append("gold test patch FAILED TO APPLY even after resetting test files.")
            parts.append("=> this is a scoring/patch problem, not necessarily the agent's fix.")
        else:
            parts.append(f"FAIL_TO_PASS (rc={art.get('f2p_rc')}) ids={art.get('f2p_ids')}")
            parts.append(art.get("f2p_output", ""))
            parts.append("")
            parts.append(f"PASS_TO_PASS sample (rc={art.get('p2p_rc')}, "
                         f"n={art.get('p2p_n')})")
            parts.append(art.get("p2p_output", ""))
        log_path.write_text("\n".join(parts))

        # --- structured JSONL row ------------------------------------------
        row = {
            "task": iid, "variant": variant, "resolved": res.resolved,
            "resolve_detail": res.resolve_detail,
            "diff_files": files, "test_files_touched": test_files,
            "steps": res.steps, "finished": res.finished,
            "budget_hit": res.budget_hit, "error": res.error,
            "quality": quality, "would_merge": would_merge,
            "cost_usd": res.ledger.get("total_cost_usd", 0),
            "main_cost_usd": res.ledger.get("main_cost_usd", 0),
            "sidekick_cost_usd": res.ledger.get("sidekick_cost_usd", 0),
            "score_artifacts": art, "log_file": str(log_path),
        }
        with self.jsonl.open("a") as fh:
            fh.write(json.dumps(row) + "\n")

        return self._terminal(iid, variant, res, files, test_files, art,
                              quality, remaining_before, log_path)

    def _terminal(self, iid, variant, res, files, test_files, art,
                  quality, remaining_before, log_path) -> str:
        verdict = "✅ RESOLVED" if res.resolved else "❌ unresolved"
        lines = [
            f"▶ {iid} :: {variant}   (remaining ${remaining_before:.2f})",
            f"    {verdict}  |  {res.resolve_detail}",
            f"    changed {len(files)} file(s){self._flag_tests(test_files)}; "
            f"steps={res.steps} finished={res.finished}"
            + ("  ⚠ BUDGET HIT" if res.budget_hit else ""),
        ]
        # The one-line "why" — the single most useful diagnostic per run.
        lines.append(f"    why: {self._why(res, files, test_files, art)}")
        lines.append(
            f"    cost ${res.ledger.get('total_cost_usd', 0):.4f} "
            f"(main ${res.ledger.get('main_cost_usd', 0)} / "
            f"sidekick ${res.ledger.get('sidekick_cost_usd', 0)})"
            + (f"  quality={quality}" if quality is not None else ""))
        if res.error:
            lines.append(f"    error: {res.error}")
        lines.append(f"    log: {log_path}")
        return "\n".join(lines)

    @staticmethod
    def _flag_tests(test_files: list[str]) -> str:
        return f"  ⚠ TOUCHED TESTS: {test_files}" if test_files else ""

    @staticmethod
    def _why(res, files, test_files, art) -> str:
        """A single clause attributing the outcome — agent vs. scorer/env."""
        if res.resolved:
            return "fix applied and official tests pass"
        if res.budget_hit:
            return "hit the per-task budget cap mid-run (raise --per-task) — NOT a real failure"
        if not files:
            return "agent produced NO diff (nothing was changed) — agent-side"
        if test_files:
            return f"agent edited test file(s) {test_files}; those edits are discarded by scoring — agent-side"
        if art and not art.get("apply_ok", True):
            return "gold test patch failed to apply — scoring/patch-side, inspect the log"
        if art.get("f2p_rc") in (2, 3, 4, 5):
            return f"pytest errored (rc={art.get('f2p_rc')}) collecting FAIL_TO_PASS — env/scoring-side, not a wrong fix"
        if art.get("p2p_rc") not in (0, None) and art.get("f2p_rc") == 0:
            return "fix passed FAIL_TO_PASS but broke PASS_TO_PASS — real regression, agent-side"
        if art.get("f2p_rc") == 1:
            return "real diff present but FAIL_TO_PASS still fails — likely wrong/incomplete fix (see pytest output in log)"
        return "see log for details"

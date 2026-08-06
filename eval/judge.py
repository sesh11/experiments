"""A lightweight "would you merge this?" rubric judge.

FrontierCode's whole point is that a passing test isn't the same as mergeable
code. We echo that quality axis at tiny scale: one judge call per (variant, task)
scores the diff 0-100 across correctness, scope discipline, and style, and says
whether they'd merge it. Cost is returned so it can be charged to the budget.
"""

from __future__ import annotations

import json

from fusion import config
from fusion.llm import Ledger, LLMClient

_JUDGE_SYSTEM = """You are a senior maintainer doing code review. You are given a bug
report and a candidate diff. Decide whether you would merge it as-is. Judge:
correctness (does it actually fix the reported bug), scope (minimal and on-target, no
unrelated churn), and style (fits a clean codebase). Reply with ONLY a JSON object:
{"score": <0-100 int>, "would_merge": <bool>, "rationale": "<one sentence>"}."""


def judge_merge(task: dict, diff: str, tests_pass: bool,
                cfg: config.RunConfig | None = None) -> dict:
    """Returns {score, would_merge, rationale, cost_usd}."""
    cfg = cfg or config.RunConfig()
    ledger = Ledger(cap_usd=float("inf"))
    client = LLMClient.for_run(
        ledger, cfg, provider=cfg.judge_provider, max_tokens=400)
    if len(diff) > 40000:  # cap judge input on huge real-repo diffs
        diff = diff[:40000] + "\n... (diff truncated for review) ..."
    user = (
        f"Bug report:\n{task['problem_statement']}\n\n"
        f"Automated tests currently pass: {tests_pass}\n\n"
        f"Candidate diff:\n{diff if diff.strip() else '(no changes made)'}"
    )
    resp = client.complete(
        role="judge", model=cfg.judge_model,
        thinking={"type": "disabled"}, system=_JUDGE_SYSTEM,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(block.text for block in resp.content
                   if block.type == "text").strip()
    data = _parse(text)
    data["cost_usd"] = round(ledger.total_cost, 4)
    return data


def _parse(text: str) -> dict:
    try:
        start, end = text.index("{"), text.rindex("}") + 1
        obj = json.loads(text[start:end])
        score = max(0, min(100, int(obj.get("score", 0))))
        return {
            "score": score,
            "would_merge": obj.get("would_merge") is True,
            "rationale": str(obj.get("rationale", ""))[:300],
        }
    except Exception:
        return {"score": 0, "would_merge": False, "rationale": f"unparseable: {text[:120]}"}

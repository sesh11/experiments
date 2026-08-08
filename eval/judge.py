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


_MAX_PROBLEM_CHARS = 40000
_MAX_DIFF_CHARS = 40000
_MAX_OUTPUT_TOKENS = 400


def _judge_user(task: dict, diff: str, tests_pass: bool) -> str:
    problem = str(task["problem_statement"])
    if len(problem) > _MAX_PROBLEM_CHARS:
        problem = problem[:_MAX_PROBLEM_CHARS] + "\n... (bug report truncated) ..."
    if len(diff) > _MAX_DIFF_CHARS:
        diff = diff[:_MAX_DIFF_CHARS] + "\n... (diff truncated for review) ..."
    return (
        f"Bug report:\n{problem}\n\n"
        f"Automated tests currently pass: {tests_pass}\n\n"
        f"Candidate diff:\n{diff if diff.strip() else '(no changes made)'}"
    )


def max_cost_upper_bound(task: dict, cfg: config.RunConfig | None = None,
                         diff: str = "", *, worst_case: bool = False) -> float:
    """Conservative reservation for the optional judge stage.

    UTF-8 bytes upper-bound tokenizer output for the capped text request. The
    provider-neutral judge makes one call; unused reserved dollars are released.
    """
    cfg = cfg or config.RunConfig()
    if worst_case:
        diff = "x" * _MAX_DIFF_CHARS
    user = _judge_user(task, diff, False)
    routed_model = config.api_model(cfg.judge_provider, cfg.judge_model)
    one_call = config.request_cost_upper_bound(
        routed_model,
        {"system": _JUDGE_SYSTEM, "messages": [{"role": "user", "content": user}]},
        max_output_tokens=_MAX_OUTPUT_TOKENS,
    )
    return round(one_call + 0.01, 6)


def judge_merge(task: dict, diff: str, tests_pass: bool,
                cfg: config.RunConfig | None = None) -> dict:
    """Returns {score, would_merge, rationale, cost_usd}."""
    cfg = cfg or config.RunConfig()
    ledger = Ledger(cap_usd=cfg.budget_usd)
    client = LLMClient.for_run(
        ledger, cfg, provider=cfg.judge_provider, max_tokens=_MAX_OUTPUT_TOKENS)
    user = _judge_user(task, diff, tests_pass)
    resp = client.complete(
        role="judge", model=cfg.judge_model,
        thinking={"type": "disabled"}, system=_JUDGE_SYSTEM,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(block.text for block in resp.content
                   if block.type == "text").strip()
    data = _parse(text)
    data["cost_usd"] = ledger.total_cost
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

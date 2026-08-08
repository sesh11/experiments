"""A lightweight "would you merge this?" rubric judge.

FrontierCode's whole point is that a passing test isn't the same as mergeable
code. We echo that quality axis at tiny scale: one judge call per (variant, task)
scores the diff 0-100 across correctness, scope discipline, and style, and says
whether they'd merge it. Cost is returned so it can be charged to the budget.
"""

from __future__ import annotations

import json

import anthropic

from fusion import config

_JUDGE_SYSTEM = """You are a senior maintainer doing code review. You are given a bug
report and a candidate diff. Decide whether you would merge it as-is. Judge:
correctness (does it actually fix the reported bug), scope (minimal and on-target, no
unrelated churn), and style (fits a clean codebase). Reply with ONLY a JSON object:
{"score": <0-100 int>, "would_merge": <bool>, "rationale": "<one sentence>"}."""

_SCHEMA = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "score": {"type": "integer"},
            "would_merge": {"type": "boolean"},
            "rationale": {"type": "string"},
        },
        "required": ["score", "would_merge", "rationale"],
        "additionalProperties": False,
    },
}

_MAX_PROBLEM_CHARS = 40000
_MAX_DIFF_CHARS = 40000
_MAX_OUTPUT_TOKENS = 400
_MAX_ATTEMPTS = 2  # structured-output request plus the compatibility fallback


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


def max_cost_upper_bound(task: dict, diff: str = "", *, worst_case: bool = False) -> float:
    """Conservative reservation for the optional judge stage.

    UTF-8 bytes are used as an upper bound on text tokens, and two full calls
    are reserved because older SDK/provider combinations may need the plain-JSON
    fallback. This intentionally over-reserves; unused dollars are released.
    """
    if worst_case:
        diff = "x" * _MAX_DIFF_CHARS
    user = _judge_user(task, diff, False)
    input_bound = len((_JUDGE_SYSTEM + user).encode("utf-8"))
    one_call = config.cost_for(
        config.JUDGE_MODEL, input_tokens=input_bound,
        output_tokens=_MAX_OUTPUT_TOKENS,
    )
    return round(one_call * _MAX_ATTEMPTS + 0.01, 6)


def judge_merge(task: dict, diff: str, tests_pass: bool) -> dict:
    """Returns {score, would_merge, rationale, cost_usd}."""
    client = anthropic.Anthropic()
    user = _judge_user(task, diff, tests_pass)
    try:
        resp = client.messages.create(
            model=config.JUDGE_MODEL,
            max_tokens=_MAX_OUTPUT_TOKENS,
            thinking={"type": "disabled"},
            system=_JUDGE_SYSTEM,
            messages=[{"role": "user", "content": user}],
            output_config={"format": _SCHEMA},
        )
    except Exception:
        # Structured outputs unsupported / call failed → fall back to plain parse.
        resp = client.messages.create(
            model=config.JUDGE_MODEL, max_tokens=_MAX_OUTPUT_TOKENS,
            thinking={"type": "disabled"}, system=_JUDGE_SYSTEM,
            messages=[{"role": "user", "content": user}],
        )

    cost = config.cost_for(
        config.JUDGE_MODEL,
        input_tokens=resp.usage.input_tokens or 0,
        output_tokens=resp.usage.output_tokens or 0,
        cache_write_tokens=getattr(resp.usage, "cache_creation_input_tokens", 0) or 0,
        cache_read_tokens=getattr(resp.usage, "cache_read_input_tokens", 0) or 0,
    )
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    data = _parse(text)
    data["cost_usd"] = cost
    return data


def _parse(text: str) -> dict:
    try:
        start, end = text.index("{"), text.rindex("}") + 1
        obj = json.loads(text[start:end])
        return {
            "score": int(obj.get("score", 0)),
            "would_merge": bool(obj.get("would_merge", False)),
            "rationale": str(obj.get("rationale", ""))[:300],
        }
    except Exception:
        return {"score": 0, "would_merge": False, "rationale": f"unparseable: {text[:120]}"}

from __future__ import annotations

from eval import judge
from fusion import config
from fusion.llm import CompletionResponse, ContentBlock


def test_judge_uses_independent_provider_and_normalized_client(monkeypatch) -> None:
    seen = {}

    class Client:
        def __init__(self, ledger) -> None:
            self.ledger = ledger

        def complete(self, **kwargs):
            seen.update(kwargs)
            self.ledger.record_tokens(
                "judge", "openai/gpt-5",
                input_tokens=10, output_tokens=5, cost_usd=0.02,
            )
            return CompletionResponse(
                content=[ContentBlock(
                    type="text",
                    text='{"score": 91, "would_merge": true, "rationale": "Clean."}',
                )],
                stop_reason="stop",
                usage=None,  # fake client owns accounting in this focused test
            )

    def make_client(cls, ledger, cfg, **kwargs):
        seen["provider"] = kwargs["provider"]
        seen["max_tokens"] = kwargs["max_tokens"]
        return Client(ledger)

    monkeypatch.setattr(judge.LLMClient, "for_run", classmethod(make_client))
    cfg = config.RunConfig(
        provider="anthropic",
        judge_provider="openrouter",
        judge_model="openai/gpt-5",
    )
    result = judge.judge_merge(
        {"problem_statement": "Fix it"}, "diff --git", True, cfg)

    assert seen["provider"] == "openrouter"
    assert seen["model"] == "openai/gpt-5"
    assert seen["max_tokens"] == 400
    assert result == {
        "score": 91, "would_merge": True,
        "rationale": "Clean.", "cost_usd": 0.02,
    }


def test_judge_parser_fails_closed_on_non_json() -> None:
    assert judge._parse("not json")["would_merge"] is False


def test_judge_parser_clamps_score_and_requires_json_boolean() -> None:
    parsed = judge._parse('{"score": 120, "would_merge": "false"}')
    assert parsed["score"] == 100
    assert parsed["would_merge"] is False

"""Offline end-to-end smoke for an OpenRouter-backed native eval variant."""

from __future__ import annotations

from eval import tasks
from fusion import config, llm, policies
from fusion.llm import CompletionResponse, CompletionUsage, ContentBlock


def test_openrouter_variant_edits_workspace_and_passes_tests(monkeypatch) -> None:
    scripted = [
        ContentBlock(
            type="tool_use", id="call-read", name="read_file",
            input={"path": "slugify.py"},
        ),
        ContentBlock(
            type="tool_use", id="call-edit", name="str_replace",
            input={
                "path": "slugify.py", "old_str": "return text",
                "new_str": 'return text.strip("-")',
            },
        ),
        ContentBlock(
            type="tool_use", id="call-test", name="run_tests", input={},
        ),
        ContentBlock(
            type="tool_use", id="call-finish", name="finish",
            input={"summary": "Trimmed leading and trailing separators."},
        ),
    ]

    class Adapter:
        def __init__(self) -> None:
            self.calls = 0
            self.models = []

        def complete(self, **kwargs):
            self.models.append(kwargs["model"])
            block = scripted[self.calls]
            self.calls += 1
            return CompletionResponse(
                content=[block], stop_reason="tool_use",
                usage=CompletionUsage(
                    input_tokens=10, output_tokens=2, cost_usd=0.01),
            )

    adapter = Adapter()
    monkeypatch.setattr(llm, "OpenRouterProvider", lambda **kwargs: adapter)
    cfg = config.RunConfig(
        provider="openrouter",
        main_model="anthropic/claude-sonnet-5",
        sidekick_model="anthropic/claude-haiku-4-5",
        max_steps=6,
    )
    result = policies.run_variant("frontier_only", tasks.load_native(1)[0], cfg)

    assert result.resolved is True
    assert result.finished is True
    assert result.error == ""
    assert 'return text.strip("-")' in result.diff
    assert [item["tool"] for item in result.trace] == [
        "read_file", "str_replace", "run_tests", "finish",
    ]
    assert result.ledger["main_cost_usd"] == 0.04
    assert adapter.models == ["anthropic/claude-sonnet-5"] * 4

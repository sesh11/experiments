from __future__ import annotations

from types import SimpleNamespace

import pytest

from fusion import config
from fusion.llm import (
    AnthropicProvider,
    CompletionResponse,
    CompletionUsage,
    ContentBlock,
    Ledger,
    LLMClient,
    OpenRouterProvider,
    ProviderResponseError,
)


class Recorder:
    def __init__(self, response) -> None:
        self.response = response
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def anthropic_client(response):
    recorder = Recorder(response)
    return SimpleNamespace(messages=recorder), recorder


def openrouter_client(response):
    recorder = Recorder(response)
    return SimpleNamespace(chat=SimpleNamespace(
        completions=recorder,
    )), recorder


def test_anthropic_adapter_preserves_messages_tools_and_cache_usage() -> None:
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="looking"),
            SimpleNamespace(type="tool_use", id="tool-1", name="read_file",
                            input={"path": "a.py"}),
        ],
        stop_reason="tool_use",
        usage=SimpleNamespace(
            input_tokens=10, output_tokens=4,
            cache_creation_input_tokens=20, cache_read_input_tokens=30,
        ),
    )
    client, recorder = anthropic_client(response)
    result = AnthropicProvider(client).complete(
        model="claude-sonnet-5", system="system",
        messages=[{"role": "user", "content": "hello"}],
        tools=[{"name": "read_file", "input_schema": {"type": "object"}}],
        thinking={"type": "disabled"}, max_tokens=100,
    )

    assert recorder.kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert recorder.kwargs["thinking"] == {"type": "disabled"}
    assert result.stop_reason == "tool_use"
    assert result.content[1].input == {"path": "a.py"}
    assert result.usage == CompletionUsage(
        input_tokens=10, output_tokens=4,
        cache_write_tokens=20, cache_read_tokens=30,
    )


def test_openrouter_adapter_converts_tool_transcript_and_usage() -> None:
    response = SimpleNamespace(
        choices=[SimpleNamespace(
            finish_reason="stop",
            message=SimpleNamespace(
                content="done", tool_calls=None, reasoning_details=None),
        )],
        usage=SimpleNamespace(
            prompt_tokens=100, completion_tokens=9, cost=0.0123,
            prompt_tokens_details=SimpleNamespace(
                cached_tokens=60, cache_write_tokens=20),
        ),
    )
    client, recorder = openrouter_client(response)
    adapter = OpenRouterProvider(client=client)
    result = adapter.complete(
        model="anthropic/claude-sonnet-5", system="system",
        messages=[
            {"role": "user", "content": "inspect"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "I'll read it"},
                {"type": "tool_use", "id": "call-1", "name": "read_file",
                 "input": {"path": "a.py"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "call-1", "content": "x = 1"},
            ]},
        ],
        tools=[{
            "name": "read_file", "description": "Read a file",
            "input_schema": {"type": "object", "properties": {}},
        }],
        thinking={"type": "disabled"}, max_tokens=200,
    )

    sent = recorder.kwargs
    assert sent["messages"][0]["role"] == "system"
    assert sent["messages"][0]["content"][0]["cache_control"] == {
        "type": "ephemeral"
    }
    assert sent["messages"][2]["tool_calls"][0]["function"]["arguments"] == (
        '{"path": "a.py"}'
    )
    assert sent["messages"][3] == {
        "role": "tool", "tool_call_id": "call-1", "content": "x = 1",
    }
    assert sent["tools"][0]["function"]["parameters"]["type"] == "object"
    assert sent["extra_body"] == {"reasoning": {"effort": "none"}}
    assert result.content[0].text == "done"
    assert result.usage == CompletionUsage(
        input_tokens=20, output_tokens=9,
        cache_write_tokens=20, cache_read_tokens=60,
        cost_usd=0.0123,
    )


def test_openrouter_adapter_normalizes_tool_calls() -> None:
    response = SimpleNamespace(
        choices=[SimpleNamespace(
            finish_reason="tool_calls",
            message=SimpleNamespace(
                content=None, reasoning_details=None,
                tool_calls=[SimpleNamespace(
                    id="call-7",
                    function=SimpleNamespace(
                        name="str_replace",
                        arguments='{"path":"x.py","old":"a","new":"b"}',
                    ),
                )],
            ),
        )],
        usage=SimpleNamespace(
            prompt_tokens=5, completion_tokens=3, cost=0.001,
            prompt_tokens_details=None,
        ),
    )
    client, _ = openrouter_client(response)
    result = OpenRouterProvider(client=client).complete(
        model="openai/gpt-5", system="s", messages=[], tools=None,
        thinking=None, max_tokens=20,
    )
    assert result.stop_reason == "tool_use"
    assert result.content == [ContentBlock(
        type="tool_use", id="call-7", name="str_replace",
        input={"path": "x.py", "old": "a", "new": "b"},
    )]


def test_openrouter_preserves_reasoning_details_across_tool_turns() -> None:
    response = SimpleNamespace(
        choices=[SimpleNamespace(
            finish_reason="stop",
            message=SimpleNamespace(
                content="done", tool_calls=None, reasoning_details=None),
        )],
        usage=SimpleNamespace(
            prompt_tokens=1, completion_tokens=1, cost=0,
            prompt_tokens_details=None,
        ),
    )
    client, recorder = openrouter_client(response)
    OpenRouterProvider(client=client).complete(
        model="openai/gpt-5", system="s",
        messages=[{
            "role": "assistant",
            "content": [{
                "type": "reasoning",
                "detail": {"type": "reasoning.encrypted", "data": "signed"},
            }],
        }],
        tools=None, thinking=None, max_tokens=20,
    )
    assert recorder.kwargs["messages"][1]["reasoning_details"] == [{
        "type": "reasoning.encrypted", "data": "signed",
    }]


def test_openrouter_rejects_invalid_tool_arguments() -> None:
    response = SimpleNamespace(
        choices=[SimpleNamespace(
            finish_reason="tool_calls",
            message=SimpleNamespace(
                content=None, reasoning_details=None,
                tool_calls=[SimpleNamespace(
                    id="call-1",
                    function=SimpleNamespace(name="read_file", arguments="not json"),
                )],
            ),
        )],
        usage=None,
    )
    client, _ = openrouter_client(response)
    with pytest.raises(ProviderResponseError, match="invalid JSON arguments"):
        OpenRouterProvider(client=client).complete(
            model="openai/gpt-5", system="s", messages=[], tools=None,
            thinking=None, max_tokens=20,
        )


def test_llm_client_routes_model_and_uses_reported_cost() -> None:
    class Adapter:
        seen_model = ""

        def complete(self, **kwargs):
            self.seen_model = kwargs["model"]
            return CompletionResponse(
                content=[ContentBlock(type="text", text="ok")],
                stop_reason="stop",
                usage=CompletionUsage(input_tokens=2, output_tokens=1, cost_usd=0.25),
            )

    ledger = Ledger(cap_usd=1)
    adapter = Adapter()
    client = LLMClient(ledger, provider="openrouter", adapter=adapter)
    client.complete(
        role="main", model="claude-sonnet-5", system="s", messages=[])
    assert adapter.seen_model == "anthropic/claude-sonnet-5"
    assert ledger.total_cost == 0.25
    assert ledger.summary()["main_calls"] == 1


def test_openrouter_requires_a_key_without_an_injected_client(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(config.ConfigurationError, match="OPENROUTER_API_KEY"):
        OpenRouterProvider()

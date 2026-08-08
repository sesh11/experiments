"""End-to-end OpenRouter protocol smoke through the real OpenAI SDK."""

from __future__ import annotations

import json

import httpx
from openai import OpenAI

from fusion.agent import Agent
from fusion.llm import Ledger, LLMClient, OpenRouterProvider


def test_real_openai_sdk_tool_round_trip_and_cost_accounting() -> None:
    requests: list[dict] = []
    responses = [
        {
            "id": "gen-1",
            "object": "chat.completion",
            "created": 1,
            "model": "anthropic/claude-sonnet-5",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "reasoning_details": [{
                        "type": "reasoning.encrypted", "data": "signed",
                    }],
                    "tool_calls": [{
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path":"app.py"}',
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {
                "prompt_tokens": 10, "completion_tokens": 3,
                "total_tokens": 13, "cost": 0.01,
                "prompt_tokens_details": {
                    "cached_tokens": 0, "cache_write_tokens": 0,
                },
            },
        },
        {
            "id": "gen-2",
            "object": "chat.completion",
            "created": 2,
            "model": "anthropic/claude-sonnet-5",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "smoke complete"},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 20, "completion_tokens": 2,
                "total_tokens": 22, "cost": 0.02,
                "prompt_tokens_details": {
                    "cached_tokens": 5, "cache_write_tokens": 0,
                },
            },
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=responses[len(requests) - 1])

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        sdk = OpenAI(
            api_key="smoke-key", base_url="https://openrouter.test/v1",
            http_client=http_client,
        )
        provider = OpenRouterProvider(client=sdk)
        ledger = Ledger(cap_usd=1)
        client = LLMClient(
            ledger, provider="openrouter", adapter=provider, max_tokens=100)
        agent = Agent(
            role="main", model="anthropic/claude-sonnet-5", system="system",
            tools=[{
                "name": "read_file", "description": "Read a file",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            }],
            client=client, thinking={"type": "disabled"}, max_steps=3,
        )
        result = agent.run(
            "inspect", lambda name, args: (f"contents:{args['path']}", False))
    finally:
        http_client.close()

    assert result.text == "smoke complete"
    assert result.steps == 2
    assert requests[0]["model"] == "anthropic/claude-sonnet-5"
    assert requests[0]["messages"][0]["content"][0]["cache_control"] == {
        "type": "ephemeral"
    }
    assert requests[0]["reasoning"] == {"effort": "none"}
    assert requests[0]["tools"][0]["function"]["name"] == "read_file"
    assert requests[1]["messages"][-1] == {
        "role": "tool", "tool_call_id": "call-1", "content": "contents:app.py",
    }
    assert requests[1]["messages"][-2]["reasoning_details"] == [{
        "type": "reasoning.encrypted", "data": "signed",
    }]
    assert ledger.total_cost == 0.03
    assert ledger.summary()["main_cache_read_tokens"] == 5

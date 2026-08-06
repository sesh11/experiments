from __future__ import annotations

from copy import deepcopy

from fusion.agent import Agent
from fusion.llm import CompletionResponse, CompletionUsage, ContentBlock


def test_agent_executes_openrouter_style_normalized_tool_round_trip() -> None:
    class Client:
        def __init__(self) -> None:
            self.messages = []
            self.calls = 0

        def complete(self, **kwargs):
            self.calls += 1
            self.messages.append(deepcopy(kwargs["messages"]))
            if self.calls == 1:
                return CompletionResponse(
                    content=[ContentBlock(
                        type="tool_use", id="call-1", name="read_file",
                        input={"path": "a.py"},
                    )],
                    stop_reason="tool_use", usage=CompletionUsage(),
                )
            return CompletionResponse(
                content=[ContentBlock(type="text", text="fixed")],
                stop_reason="stop", usage=CompletionUsage(),
            )

    client = Client()
    agent = Agent(
        role="main", model="anthropic/claude-sonnet-5", system="s",
        tools=[], client=client, max_steps=3,
    )
    result = agent.run(
        "fix it", lambda name, args: (f"contents of {args['path']}", False))

    assert result.text == "fixed"
    assert result.steps == 2
    assert result.trace[0]["tool"] == "read_file"
    second_messages = client.messages[1]
    assert second_messages[-2]["content"] == [{
        "type": "tool_use", "id": "call-1",
        "name": "read_file", "input": {"path": "a.py"},
    }]
    assert second_messages[-1]["content"][0]["tool_use_id"] == "call-1"

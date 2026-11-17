from __future__ import annotations

import json

import httpx

from labscribe.agent.llm import LLMClient
from labscribe.config import Config


def test_parse_tool_calls():
    payload = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_sop_step",
                                "arguments": json.dumps(
                                    {"experiment_id": "exp-titration", "step_id": "S1"}
                                ),
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {"total_tokens": 42},
    }
    response = LLMClient._parse(payload, latency_ms=12.0)
    assert response.finish_reason == "tool_calls"
    assert len(response.tool_calls) == 1
    call = response.tool_calls[0]
    assert call.id == "call_1"
    assert call.name == "get_sop_step"
    assert call.arguments["step_id"] == "S1"
    assert call.error is None
    assert response.latency_ms == 12.0


def test_invalid_tool_arguments_are_flagged():
    payload = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "call_x",
                            "type": "function",
                            "function": {"name": "write_experiment_record", "arguments": "{not json"},
                        }
                    ]
                }
            }
        ]
    }
    response = LLMClient._parse(payload, latency_ms=1.0)
    assert response.tool_calls[0].error is not None
    assert response.tool_calls[0].arguments == {}


def test_chat_with_mock_transport(monkeypatch, tmp_path):
    monkeypatch.setenv("LABSCRIBE_LLM_API_KEY", "test-key")
    cfg = Config.load(
        overrides={
            "agent": {"base_url": "https://example.invalid", "model": "deepseek-flash"},
            "store": {"db_path": str(tmp_path / "x.db")},
        }
    )
    client = LLMClient(cfg)
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("Authorization")
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "已记录。"},
                    }
                ],
                "usage": {"total_tokens": 10},
            },
        )

    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    response = client.chat(
        [{"role": "user", "content": "ping"}],
        tools=[{"type": "function", "function": {"name": "x", "parameters": {}}}],
    )
    assert response.content == "已记录。"
    assert captured["auth"] == "Bearer test-key"
    assert captured["body"]["model"] == "deepseek-flash"
    assert captured["body"]["tool_choice"] == "auto"
    assert captured["body"]["tools"][0]["function"]["name"] == "x"
    client.close()

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from ..config import Config
from ..schema import new_id


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str = "{}"
    error: str | None = None


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str = ""
    latency_ms: float = 0.0
    usage: dict[str, Any] = field(default_factory=dict)


class LLMError(RuntimeError):
    pass


class LLMClient:
    """Minimal OpenAI-compatible chat-completions client with tool calling."""

    def __init__(self, cfg: Config):
        base_url = str(cfg.get("agent.base_url", "https://api.deepseek.com")).rstrip("/")
        self.endpoint = f"{base_url}/chat/completions"
        self.model = str(cfg.get("agent.model", "deepseek-flash"))
        self.temperature = float(cfg.get("agent.temperature", 0.1))
        self.timeout = float(cfg.get("agent.timeout_s", 60))
        self.max_tokens = int(cfg.get("agent.max_tokens", 900))
        self.api_key = cfg.api_key()
        self.max_retries = 2
        self._client = httpx.Client(timeout=self.timeout)

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
        max_tokens: int | None = None,
    ) -> LLMResponse:
        if not self.api_key:
            raise LLMError(
                "no LLM API key: set LABSCRIBE_LLM_API_KEY or DEEPSEEK_API_KEY in the environment/.env"
            )
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            started = time.perf_counter()
            try:
                response = self._client.post(self.endpoint, json=payload, headers=headers)
                latency_ms = (time.perf_counter() - started) * 1000.0
                if response.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    time.sleep(1.0 * (2**attempt))
                    continue
                if response.status_code >= 400:
                    body = response.text[:300].replace("\n", " ")
                    raise LLMError(f"LLM HTTP {response.status_code}: {body}")
                data = response.json()
                return self._parse(data, latency_ms)
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(1.0 * (2**attempt))
                    continue
                raise LLMError(f"LLM network error: {exc}") from exc
            except json.JSONDecodeError as exc:
                raise LLMError(f"LLM returned invalid JSON: {exc}") from exc
        raise LLMError(f"LLM request failed: {last_error}")

    @staticmethod
    def _parse(data: dict[str, Any], latency_ms: float) -> LLMResponse:
        choices = data.get("choices") or []
        if not choices:
            raise LLMError(f"LLM response has no choices: {str(data)[:200]}")
        choice = choices[0]
        message = choice.get("message") or {}
        raw_tool_calls = message.get("tool_calls") or []
        tool_calls: list[ToolCall] = []
        for raw in raw_tool_calls:
            function = raw.get("function") or {}
            name = function.get("name") or ""
            raw_arguments = function.get("arguments") or "{}"
            error = None
            arguments: Any = {}
            try:
                arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
            except json.JSONDecodeError as exc:
                error = f"invalid tool arguments JSON: {exc}"
            if not isinstance(arguments, dict):
                arguments = {"value": arguments}
            tool_calls.append(
                ToolCall(
                    id=raw.get("id") or new_id("call"),
                    name=name,
                    arguments=arguments,
                    raw_arguments=raw_arguments if isinstance(raw_arguments, str) else "{}",
                    error=error,
                )
            )
        return LLMResponse(
            content=(message.get("content") or "").strip(),
            tool_calls=tool_calls,
            raw_tool_calls=raw_tool_calls,
            finish_reason=str(choice.get("finish_reason", "")),
            latency_ms=latency_ms,
            usage=data.get("usage") or {},
        )

    def close(self) -> None:
        self._client.close()

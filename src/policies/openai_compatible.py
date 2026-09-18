"""Policy adapter for any OpenAI-compatible chat-completions endpoint."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any
from urllib import error, request

from loguru import logger

from src.policies.code_execution import DEFAULT_MAX_TOKENS, SYSTEM_PROMPT, clean_action

Transport = Callable[[str, dict[str, Any], dict[str, str], float], dict[str, Any]]


def _chat_completions_url(endpoint: str) -> str:
    endpoint = endpoint.rstrip("/")
    if endpoint.endswith("/chat/completions"):
        return endpoint
    return f"{endpoint}/chat/completions"


def _post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
) -> dict[str, Any]:
    body = json.dumps(payload).encode()
    req = request.Request(url, data=body, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read())
    except error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"model endpoint returned HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"could not reach model endpoint: {exc.reason}") from exc


def _text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") in {"text", "output_text"}
        ]
        if parts:
            return "".join(parts)
    raise ValueError("chat-completions response did not contain text content")


class OpenAICompatiblePolicy:
    """Use vLLM, Ollama, LM Studio, or a hosted OpenAI-compatible model.

    Configure it with constructor arguments or these environment variables:

    - ENVOY_MODEL_ENDPOINT (for example http://localhost:8000/v1)
    - ENVOY_MODEL_ID
    - ENVOY_MODEL_API_KEY (optional)
    - ENVOY_MODEL_EXTRA_JSON (optional request fields)
    """

    def __init__(
        self,
        endpoint: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
        timeout: float = 120.0,
        extra_body: dict[str, Any] | None = None,
        transport: Transport = _post_json,
    ) -> None:
        self.endpoint = endpoint or os.environ.get("ENVOY_MODEL_ENDPOINT", "")
        self.model = model or os.environ.get("ENVOY_MODEL_ID", "")
        self.api_key = api_key or os.environ.get("ENVOY_MODEL_API_KEY")
        if not self.endpoint or not self.model:
            raise ValueError("Set ENVOY_MODEL_ENDPOINT and ENVOY_MODEL_ID for openai_compatible")

        configured_extra = os.environ.get("ENVOY_MODEL_EXTRA_JSON")
        if extra_body is None and configured_extra:
            parsed = json.loads(configured_extra)
            if not isinstance(parsed, dict):
                raise ValueError("ENVOY_MODEL_EXTRA_JSON must contain a JSON object")
            extra_body = parsed

        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.extra_body = extra_body or {}
        self._transport = transport
        self.history: list[dict[str, str]] = []
        self.config = {
            "backend": "openai_compatible",
            "endpoint": self.endpoint,
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

    def act(self, observation: str) -> str:
        self.history.append({"role": "user", "content": observation})
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *self.history]
        payload = {
            **self.extra_body,
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = self._transport(
            _chat_completions_url(self.endpoint), payload, headers, self.timeout
        )
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("invalid chat-completions response shape") from exc

        raw_action = _text_content(content)
        self.history.append({"role": "assistant", "content": raw_action})
        action = clean_action(raw_action)
        logger.debug(
            f"OpenAICompatiblePolicy ({self.model}) action ({len(action)} chars): {action[:100]}..."
        )
        return action

    def reset(self) -> None:
        self.history = []

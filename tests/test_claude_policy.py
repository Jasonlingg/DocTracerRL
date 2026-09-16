"""Offline checks for ClaudePolicy's model/temperature handling."""

import anthropic
import pytest

from src.policies.claude_policy import ClaudePolicy


class _Response:
    def __init__(self, text: str, thinking: str | None = None):
        blocks = []
        if thinking is not None:
            blocks.append(type("ThinkingBlock", (), {"thinking": thinking})())
        blocks.append(type("TextBlock", (), {"text": text})())
        self.content = blocks


class _FakeMessages:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        if hasattr(result, "content"):
            return result
        return _Response(result)


def _bad_request(message: str) -> anthropic.BadRequestError:
    request = anthropic._base_client.httpx.Request("POST", "https://api.anthropic.com")
    response = anthropic._base_client.httpx.Response(400, request=request, json={
        "type": "error", "error": {"type": "invalid_request_error", "message": message},
    })
    return anthropic.BadRequestError(message, response=response, body=None)


def test_model_override_env_var(monkeypatch):
    monkeypatch.setenv("CLAUDE_MODEL_PATH", "claude-sonnet-5")
    monkeypatch.setattr(anthropic, "Anthropic", lambda api_key=None: object())
    policy = ClaudePolicy()
    assert policy.model == "claude-sonnet-5"


def test_explicit_model_wins_over_env_var(monkeypatch):
    monkeypatch.setenv("CLAUDE_MODEL_PATH", "claude-sonnet-5")
    monkeypatch.setattr(anthropic, "Anthropic", lambda api_key=None: object())
    policy = ClaudePolicy(model="claude-haiku-4-5-20251001")
    assert policy.model == "claude-haiku-4-5-20251001"


def test_falls_back_when_model_rejects_temperature(monkeypatch):
    fake = _FakeMessages([
        _bad_request("`temperature` is deprecated for this model."),
        "print(1)",
    ])
    monkeypatch.setattr(
        anthropic, "Anthropic",
        lambda api_key=None: type("Client", (), {"messages": fake})(),
    )
    policy = ClaudePolicy(model="claude-sonnet-5")
    action = policy.act("go")
    assert action.strip() == "print(1)"
    assert "temperature" in fake.calls[0]
    assert "temperature" not in fake.calls[1]


def test_second_turn_does_not_resend_temperature_after_fallback(monkeypatch):
    fake = _FakeMessages([
        _bad_request("`temperature` is deprecated for this model."),
        "step_one()",
        "step_two()",
    ])
    monkeypatch.setattr(
        anthropic, "Anthropic",
        lambda api_key=None: type("Client", (), {"messages": fake})(),
    )
    policy = ClaudePolicy(model="claude-sonnet-5")
    policy.act("first")
    policy.act("second")
    assert all("temperature" not in call for call in fake.calls[1:])


def test_skips_thinking_block_and_uses_the_final_text_block(monkeypatch):
    """A model with native thinking on by default (e.g. claude-sonnet-5) returns
    a ThinkingBlock before the actual text response; content[0].text would
    crash on it since ThinkingBlock has no .text attribute."""
    fake = _FakeMessages([_Response("search('x')", thinking="reasoning about x...")])
    monkeypatch.setattr(
        anthropic, "Anthropic",
        lambda api_key=None: type("Client", (), {"messages": fake})(),
    )
    policy = ClaudePolicy(model="claude-sonnet-5")
    action = policy.act("go")
    assert action.strip() == "search('x')"


def test_raises_a_clear_error_when_response_has_no_text_block(monkeypatch):
    fake = _FakeMessages([type("Empty", (), {"content": []})()])
    monkeypatch.setattr(
        anthropic, "Anthropic",
        lambda api_key=None: type("Client", (), {"messages": fake})(),
    )
    policy = ClaudePolicy(model="claude-sonnet-5")
    with pytest.raises(ValueError, match="no text content"):
        policy.act("go")


def test_unrelated_bad_request_is_not_swallowed(monkeypatch):
    fake = _FakeMessages([_bad_request("invalid api key")])
    monkeypatch.setattr(
        anthropic, "Anthropic",
        lambda api_key=None: type("Client", (), {"messages": fake})(),
    )
    policy = ClaudePolicy(model="claude-sonnet-5")
    with pytest.raises(anthropic.BadRequestError):
        policy.act("go")

import pytest

from src.policies.openai_compatible import OpenAICompatiblePolicy
from src.policies.protocol import Policy
from src.policies.registry import POLICY_SPECS, policy_catalog, requires_anthropic_key


def test_endpoint_policy_uses_the_shared_action_contract():
    calls = []

    def transport(url, payload, headers, timeout):
        calls.append((url, payload, headers, timeout))
        return {"choices": [{"message": {"content": "```python\nprint(search('masking'))\n```"}}]}

    policy = OpenAICompatiblePolicy(
        endpoint="http://localhost:8000/v1/",
        model="any-model",
        api_key="secret",
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        transport=transport,
    )

    assert isinstance(policy, Policy)
    assert policy.act("Question: why mask tokens?") == "print(search('masking'))"
    url, payload, headers, timeout = calls[0]
    assert url == "http://localhost:8000/v1/chat/completions"
    assert payload["model"] == "any-model"
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][-1] == {"role": "user", "content": "Question: why mask tokens?"}
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert headers["Authorization"] == "Bearer secret"
    assert timeout == 120.0

    policy.reset()
    assert policy.history == []


def test_endpoint_policy_accepts_structured_text_content():
    policy = OpenAICompatiblePolicy(
        endpoint="http://localhost:11434/v1/chat/completions",
        model="local",
        transport=lambda *args: {
            "choices": [
                {"message": {"content": [{"type": "text", "text": "SUBMIT: answer CITATIONS: []"}]}}
            ]
        },
    )
    assert policy.act("Question") == "SUBMIT: answer CITATIONS: []"


def test_endpoint_policy_requires_endpoint_and_model(monkeypatch):
    for name in ("ENVOY_MODEL_ENDPOINT", "ENVOY_MODEL_ID"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="ENVOY_MODEL_ENDPOINT"):
        OpenAICompatiblePolicy()


def test_policy_catalog_reports_endpoint_configuration(monkeypatch):
    monkeypatch.delenv("ENVOY_MODEL_ENDPOINT", raising=False)
    monkeypatch.delenv("ENVOY_MODEL_ID", raising=False)
    unavailable = {item["name"]: item for item in policy_catalog()}
    assert not unavailable["openai_compatible"]["available"]

    monkeypatch.setenv("ENVOY_MODEL_ENDPOINT", "http://localhost:8000/v1")
    monkeypatch.setenv("ENVOY_MODEL_ID", "qwen")
    available = {item["name"]: item for item in policy_catalog()}
    assert available["openai_compatible"]["available"]


def test_endpoint_policy_does_not_require_anthropic_credentials():
    assert not requires_anthropic_key(["openai_compatible"])
    assert requires_anthropic_key(["claude_policy"])
    assert not POLICY_SPECS["openai_compatible"].select_by_default

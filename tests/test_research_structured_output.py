"""Regression checks for the malformed final actions found in the GPU pilot."""

import io
import json

import pytest

from src.eval.artifacts import configuration_hash
from src.research.action_schema import ACTION_SCHEMA, validate_structured_action
from src.research.agent import EndpointPolicy, parse_action

ABSTENTION = {"action": "submit", "answer": {
    "claims": [], "recommendation": "Insufficient evidence; measure this directly.",
    "limitations": ["The requested experiment is absent."],
}}


@pytest.mark.parametrize("action", [
    {"action": "papers", "arguments": {}},
    {"action": "search_papers", "arguments": {"query": "support", "top_k": 3}},
    {"action": "paper", "arguments": {"doc_id": "paper"}},
    {"action": "passage", "arguments": {"doc_id": "paper", "start": 0, "length": 1600}},
    ABSTENTION,
    {"action": "submit", "answer": {**ABSTENTION["answer"], "claims": [
        {"text": "A finding", "evidence": [{"doc_id": "paper", "start": 0, "end": 20}]},
    ]}},
])
def test_all_actions_and_abstention_agree_with_dispatcher(action):
    raw = json.dumps(action)
    assert validate_structured_action(raw) == parse_action(raw) == action


@pytest.mark.parametrize("raw", [
    json.dumps(ABSTENTION) + "\n\n```",
    json.dumps({**ABSTENTION, "evidence": []}),
    '{"action":"shell","arguments":{"command":"ls"}}',
    '{"action":"passage","arguments":{"doc_id":"p","start":true}}',
    '{"action":"submit","answer":{"claims":"not a list"}}',
])
def test_endpoint_fails_closed_when_server_ignores_schema(monkeypatch, raw):
    def respond(request, timeout):
        return io.BytesIO(json.dumps({"choices": [
            {"finish_reason": "stop", "message": {"content": raw}},
        ]}).encode())

    monkeypatch.setattr("src.research.agent.urlopen", respond)
    policy = EndpointPolicy("http://localhost/v1", "test", "revision",
                            structured_output="json_schema")
    with pytest.raises(RuntimeError, match="invalid structured JSON"):
        policy.act("Question")


def test_schema_is_sent_and_recorded_without_changing_old_policy(monkeypatch):
    requests = []

    def respond(request, timeout):
        requests.append(json.loads(request.data))
        return io.BytesIO(json.dumps({"choices": [
            {"finish_reason": "stop", "message": {"content": json.dumps(ABSTENTION)}},
        ]}).encode())

    monkeypatch.setattr("src.research.agent.urlopen", respond)
    old = EndpointPolicy("http://localhost/v1", "test", "revision")
    new = EndpointPolicy("http://localhost/v1", "test", "revision",
                         structured_output="json_schema")
    old.act("Question")
    new.act("Question")
    assert "response_format" not in requests[0]
    assert "structured_output" not in old.config
    assert requests[1]["response_format"]["json_schema"]["schema"] == ACTION_SCHEMA
    assert new.config["action_schema_hash"] == configuration_hash(ACTION_SCHEMA)
    assert configuration_hash(old.config) != configuration_hash(new.config)

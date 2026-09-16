"""Tests for query_papers: the tool wrapper a main agent calls to consult the papers."""

import json
from pathlib import Path

import pytest

from src.research.agent import load_snapshot
from src.research.knowledge_tool import QUERY_PAPERS_TOOL, query_papers
from src.research.papers import build_snapshot

META = '''<meta property="og:url" content="https://arxiv.org/abs/2503.09516v1">
<meta name="citation_title" content="A retrieval study">
<meta name="citation_abstract" content="We study search. Evidence has limitations.">
<meta name="citation_date" content="2025/03/12">
<meta name="citation_online_date" content="2025/03/12">'''
HTML = '<article><h2>Methods</h2>' + ''.join(
    f'<p>Search experiment {i} uses a fixed corpus.</p>' for i in range(6)
) + '</article>'


@pytest.fixture
def snapshot(tmp_path):
    path = tmp_path / "snapshot"
    build_snapshot(["2503.09516v1"], path,
                   downloader=lambda u: (HTML if "/html/" in u else META).encode(), delay=0)
    return path


class ScriptedPolicy:
    config = {"backend": "scripted_test", "model": "none", "seed": 0}

    def __init__(self, doc_id):
        self.step = 0
        self.doc_id = doc_id

    def act(self, observation):
        self.step += 1
        if self.step == 1:
            return json.dumps({"action": "search_papers",
                               "arguments": {"query": "search", "top_k": 2}})
        if self.step == 2:
            return json.dumps({"action": "passage",
                               "arguments": {"doc_id": self.doc_id, "start": 0, "length": 16}})
        return json.dumps({"action": "submit", "answer": {
            "claims": [{"text": "The authors study search.", "evidence": [
                {"doc_id": self.doc_id, "start": 0, "end": 16},
            ]}],
            "recommendation": "Inference: test retrieval on our own questions.",
            "limitations": ["This fixture establishes no real model performance."],
        }})


def test_query_papers_returns_a_narrow_tool_result_and_writes_a_durable_artifact(
    snapshot, tmp_path
):
    _, docs = load_snapshot(snapshot)
    doc_id = next(iter(docs))
    run_dir = tmp_path / "runs"
    run_dir.mkdir()

    result = query_papers(snapshot, "What was studied?", ScriptedPolicy(doc_id), run_dir)

    assert result["status"] == "submitted"
    assert result["submission"]["claims"][0]["evidence"][0]["doc_id"] == doc_id
    assert result["checks"]["claims_with_valid_source_spans"] == 1
    assert {"status", "submission", "checks", "protocol", "corpus_hash",
           "run_id", "artifact"} == set(result)
    assert Path(result["artifact"]).exists()


def test_query_papers_works_identically_for_any_policy_implementing_act(snapshot, tmp_path):
    """The harness must not special-case a specific model; two different Policy
    classes calling the same tool should exercise identical code and produce the
    same protocol identity."""
    _, docs = load_snapshot(snapshot)
    doc_id = next(iter(docs))
    run_dir = tmp_path / "runs"
    run_dir.mkdir()

    class DifferentScriptedPolicy(ScriptedPolicy):
        config = {"backend": "a_totally_different_model", "model": "not-qwen", "seed": 1}

    first = query_papers(snapshot, "What was studied?", ScriptedPolicy(doc_id), run_dir)
    second = query_papers(snapshot, "What was studied?",
                          DifferentScriptedPolicy(doc_id), run_dir)
    assert first["status"] == second["status"] == "submitted"
    assert first["protocol"] == second["protocol"]


def test_query_papers_tool_schema_is_a_minimal_stable_contract():
    assert QUERY_PAPERS_TOOL["name"] == "query_papers"
    assert QUERY_PAPERS_TOOL["input_schema"]["required"] == ["question"]
    assert QUERY_PAPERS_TOOL["input_schema"]["additionalProperties"] is False

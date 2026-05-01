"""Tests for BM25 sparse RAG baseline."""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from src.env.corpus import Corpus
from src.policies.sparse_rag import SparseRAGPolicy


@pytest.fixture(scope="module")
def corpus() -> Corpus:
    subprocess.run(["python", "scripts/setup_corpus.py"], check=True, capture_output=True)
    c = Corpus(corpus_path="data/corpus")
    c.load()
    return c


def _mock_claude(text: str) -> MagicMock:
    client = MagicMock()
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    client.messages.create.return_value = msg
    return client


def test_bm25_retrieves_relevant_docs(corpus: Corpus) -> None:
    """BM25 should rank apex_corp_2024_financial high for an apex revenue query."""
    policy = SparseRAGPolicy(corpus=corpus, top_k=3)
    policy._ensure_index()
    scores = policy._bm25.get_scores(
        "apex corp revenue".split()
    )
    top_idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:3]
    top_doc_ids = [policy._docs[i]["doc_id"] for i in top_idx]
    assert any("apex_corp" in d for d in top_doc_ids)


def test_act_returns_submit_format(corpus: Corpus) -> None:
    """Policy.act passes BM25 top-k context to Claude; expects SUBMIT/CITATIONS back."""
    canned = 'SUBMIT: $142.5M CITATIONS: ["apex_corp_2024_financial"]'
    with patch("src.policies.sparse_rag.Anthropic", return_value=_mock_claude(canned)):
        policy = SparseRAGPolicy(corpus=corpus, top_k=2)
        action = policy.act("Question: What was Apex Corp's 2024 revenue?")
    assert action.startswith("SUBMIT:")
    assert "CITATIONS:" in action
    assert "apex_corp_2024_financial" in action


def test_act_only_runs_once(corpus: Corpus) -> None:
    """Subsequent calls after answering should short-circuit to a 'no answer' submission."""
    canned = 'SUBMIT: foo CITATIONS: []'
    with patch("src.policies.sparse_rag.Anthropic", return_value=_mock_claude(canned)):
        policy = SparseRAGPolicy(corpus=corpus, top_k=2)
        policy.act("Question: anything")
        second = policy.act("Question: anything")
    assert second.startswith("SUBMIT:")
    assert "No answer available" in second

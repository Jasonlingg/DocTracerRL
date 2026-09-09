"""Tests for corpus loading and search."""

import subprocess

import pytest

from src.env.corpus import Corpus


@pytest.fixture(scope="module")
def corpus() -> Corpus:
    """Load the synthetic corpus (generate if needed)."""
    subprocess.run(["python", "scripts/setup_corpus.py"], check=True, capture_output=True)
    c = Corpus(corpus_path="data/corpus")
    c.load()
    return c


def test_document_count(corpus: Corpus) -> None:
    docs = corpus.list_documents()
    assert len(docs) == 43


def test_search_returns_results(corpus: Corpus) -> None:
    results = corpus.search("revenue")
    assert len(results) > 0
    assert results[0].score > 0


def test_search_relevance(corpus: Corpus) -> None:
    results = corpus.search("Apex Corp revenue financial")
    doc_ids = [r.doc_id for r in results]
    assert any("apex_corp" in d for d in doc_ids)


def test_read_existing_doc(corpus: Corpus) -> None:
    text = corpus.read("apex_corp_2024_financial")
    assert text is not None
    assert "Apex Corp" in text
    assert "$142.5M" in text


def test_read_missing_doc(corpus: Corpus) -> None:
    assert corpus.read("nonexistent_doc") is None


def test_list_documents_has_info(corpus: Corpus) -> None:
    docs = corpus.list_documents()
    for doc in docs:
        assert doc.doc_id
        assert doc.title
        assert doc.chars > 0

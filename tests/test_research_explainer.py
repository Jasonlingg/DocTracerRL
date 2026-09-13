"""Tests for the bounded handoff from retrieval to explanation."""

import json

import pytest

from src.research.explainer import (
    build_evidence_packet,
    check_explanation,
    run_explanation,
)


def retriever_run():
    evidence = {
        "doc_id": "paper_a",
        "start": 10,
        "end": 20,
        "quote": "exact text",
        "quote_origin": "snapshot_materialized",
    }
    checked = {
        "doc_id": "paper_a",
        "usable_source_span": True,
        "source_url": "https://example.test/paper_a",
        "coverage": "html_paragraphs",
    }
    return {
        "run_id": "retriever-1",
        "protocol": "research-tools-v2",
        "status": "submitted",
        "question": {"id": "q1", "question": "What does the paper show?"},
        "corpus_hash": "a" * 64,
        "submission": {
            "claims": [{"text": "Candidate claim", "evidence": [evidence]}],
            "recommendation": "Inference: test it.",
            "limitations": ["One paper only."],
        },
        "checks": {"claims": [{"evidence": [checked]}]},
    }


def test_packet_contains_exact_evidence_but_marks_candidate_claims_untrusted():
    packet = build_evidence_packet(retriever_run())
    assert packet["evidence"][0]["evidence_id"] == "E1"
    assert packet["evidence"][0]["quote"] == "exact text"
    assert packet["candidate_claims"][0]["evidence_ids"] == ["E1"]
    assert "untrusted" in packet["warning"]


def test_explainer_may_only_reference_evidence_in_packet():
    packet = build_evidence_packet(retriever_run())
    explanation = {
        "answer": "The supplied passage supports a limited statement [E1].",
        "claims": [{"text": "A limited statement", "evidence_ids": ["E1"]}],
        "limitations": ["No broader conclusion is supported."],
    }
    checks = check_explanation(explanation, packet)
    assert checks["claims_with_valid_evidence"] == 1
    assert checks["semantic_support"] == "not_reviewed"
    explanation["claims"][0]["evidence_ids"] = ["invented"]
    with pytest.raises(ValueError, match="unknown evidence"):
        check_explanation(explanation, packet)


def test_scripted_explainer_saves_a_reviewable_artifact(tmp_path):
    retriever_path = tmp_path / "retriever.json"
    retriever_path.write_text(json.dumps(retriever_run()))

    class ScriptedExplainer:
        config = {"backend": "scripted_test", "model": "none"}

        def explain(self, packet):
            assert packet["question_id"] == "q1"
            return json.dumps({
                "answer": "A plain explanation [E1].",
                "claims": [{"text": "A limited statement", "evidence_ids": ["E1"]}],
                "limitations": ["This is a test fixture."],
            })

    output = tmp_path / "explanation.json"
    result = run_explanation(retriever_path, ScriptedExplainer(), output)
    assert result["status"] == "submitted"
    assert result["checks"]["claims_with_valid_evidence"] == 1
    assert "exact text" in output.with_suffix(".md").read_text()

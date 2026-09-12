"""Offline tests for the research benchmark boundary and scorer."""

import json
from pathlib import Path

import pytest

from src.eval.artifacts import configuration_hash
from src.research.benchmark import (
    REVIEW_SCHEMA_VERSION,
    SCHEMA_VERSION,
    make_review_template,
    score_benchmark,
    validate_benchmark,
)


def benchmark():
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark_id": "test-pilot",
        "corpus_hash": "a" * 64,
        "reserved_doc_ids": ["paper_a"],
        "questions": [{
            "id": "pilot_01",
            "split": "pilot_evaluation",
            "question": "What does the paper establish?",
            "expected_answerability": "sufficient",
            "required_doc_ids": ["paper_a"],
            "minimum_distinct_sources": 1,
            "grader_notes": ["Do not overclaim."],
        }],
    }


def run_artifact():
    return {
        "question": {"id": "pilot_01"},
        "corpus_hash": "a" * 64,
        "protocol": "research-tools-v2",
        "policy": {"model": "test", "seed": 42},
        "prompt_hash": "prompt-a",
        "max_steps": 10,
        "status": "submitted",
        "trajectory": [
            {"action": {
                "action": "search_papers", "arguments": {"query": "test", "top_k": 3}
            }, "output": "[]"},
            {"action": {"action": "submit", "answer": {}}},
        ],
        "submission": {
            "claims": [{
                "text": "A supported claim",
                "evidence": [{
                    "doc_id": "paper_a", "start": 0, "end": 5, "quote": "Study"
                }],
            }],
            "recommendation": "Inference: test it.",
            "limitations": [],
        },
        "checks": {"claims_with_valid_source_spans": 1},
    }


def test_benchmark_reserves_entire_snapshot_and_rejects_leakage():
    data = benchmark()
    manifest = {"corpus_hash": "a" * 64, "papers": [{"doc_id": "paper_a"}]}
    validate_benchmark(data, manifest)
    with pytest.raises(ValueError, match="all snapshot documents"):
        validate_benchmark(data, {
            "corpus_hash": "a" * 64,
            "papers": [{"doc_id": "paper_a"}, {"doc_id": "paper_b"}],
        })
    data["questions"].append(dict(data["questions"][0]))
    with pytest.raises(ValueError, match="unique"):
        validate_benchmark(data)


def test_checked_in_pilot_is_valid_and_keeps_all_source_papers_out_of_training():
    data = json.loads(Path("data/research/benchmark_pilot_v1.json").read_text())
    validate_benchmark(data)
    assert len(data["questions"]) == 8
    assert len(data["reserved_doc_ids"]) == 6
    assert {question["expected_answerability"] for question in data["questions"]} == {
        "sufficient", "insufficient"
    }


def test_score_separates_provenance_from_human_support(tmp_path):
    data = benchmark()
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "pilot_01.json").write_text(json.dumps(run_artifact()))
    automatic = score_benchmark(data, runs)
    assert automatic["automatic"]["source_valid_claim_rate"] == 1.0
    assert automatic["automatic"]["required_document_recall"] == 1.0
    assert automatic["questions"][0]["tool_action_steps"] == 1
    assert automatic["experiment_identity"]["protocols"] == ["research-tools-v2"]
    assert automatic["human"] is None

    reviews = make_review_template(data, runs)
    review = reviews["reviews"][0]
    review["claim_reviews"][0]["support"] = "unsupported"
    review.update({
        "answerability_handled": False,
        "recommendation_faithful": False,
        "relevant_evidence_missed": True,
        "usefulness": 0,
        "failure_category": "unsupported_synthesis",
    })
    scored = score_benchmark(data, runs, reviews)
    assert scored["human"]["fully_supported_claim_rate"] == 0.0
    assert scored["human"]["unsupported_claim_rate"] == 1.0


def test_incomplete_review_cannot_produce_human_metrics(tmp_path):
    data = benchmark()
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "pilot_01.json").write_text(json.dumps(run_artifact()))
    reviews = make_review_template(data, runs)
    assert reviews["schema_version"] == REVIEW_SCHEMA_VERSION
    assert reviews["benchmark_hash"] == configuration_hash(data)
    with pytest.raises(ValueError, match="answerability_handled"):
        score_benchmark(data, runs, reviews)

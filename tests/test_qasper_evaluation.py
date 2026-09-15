"""Protect the held-out QASPER split from leakage and ambiguous labels."""

import copy

import pytest

from src.research.qasper_evaluation import validate_qasper_evaluation_plan


def fixtures():
    question = {
        "id": "q1",
        "source_paper_id": "eval-paper",
        "usage": "reserved_evaluation",
        "expected_answerability": "sufficient",
        "conversion_issues": [],
        "ambiguous_text_evidence_count": 0,
        "gold_evidence": [{"doc_id": "eval", "start": 1, "end": 2}],
    }
    question_set = {
        "source_dataset": "allenai/qasper",
        "source_revision": "revision",
        "source_split": "validation",
        "corpus_hash": "eval-hash",
        "questions": [question],
    }
    evaluation_manifest = {
        "source_split": "validation",
        "corpus_hash": "eval-hash",
        "papers": [{"qasper_paper_id": "eval-paper"}],
    }
    training_manifest = {
        "source_split": "train",
        "corpus_hash": "train-hash",
        "papers": [{"qasper_paper_id": "train-paper"}],
    }
    plan = {
        "schema_version": "qasper-held-out-eval-v1",
        "status": "locked_before_research_sft",
        "source_dataset": "allenai/qasper",
        "source_revision": "revision",
        "source_split": "validation",
        "snapshot_corpus_hash": "eval-hash",
        "training_snapshot_corpus_hash": "train-hash",
        "paper_overlap_with_training_snapshot": 0,
        "question_ids": ["q1"],
        "coverage": {"question_count": 1, "unique_paper_count": 1},
    }
    return plan, question_set, evaluation_manifest, training_manifest


def test_valid_paper_disjoint_plan_passes():
    result = validate_qasper_evaluation_plan(*fixtures())
    assert result["status"] == "passed"
    assert result["paper_overlap_with_training_snapshot"] == 0


def test_training_paper_leakage_is_rejected():
    plan, questions, evaluation, training = copy.deepcopy(fixtures())
    training["papers"].append({"qasper_paper_id": "eval-paper"})
    with pytest.raises(ValueError, match="leaked"):
        validate_qasper_evaluation_plan(plan, questions, evaluation, training)


def test_ambiguous_question_is_rejected():
    plan, questions, evaluation, training = copy.deepcopy(fixtures())
    questions["questions"][0]["expected_answerability"] = "annotator_disagreement"
    with pytest.raises(ValueError, match="disagreement"):
        validate_qasper_evaluation_plan(plan, questions, evaluation, training)

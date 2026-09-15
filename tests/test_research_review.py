"""Tests for the small blind code-execution research comparison."""

from __future__ import annotations

import json

import pytest

from src.eval.research_review import prepare_review, score_review


def _write_fixture(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "paper.json").write_text(
        json.dumps(
            {
                "doc_id": "paper",
                "title": "Paper",
                "text": "alpha supported result omega",
            }
        )
    )
    questions = tmp_path / "questions.json"
    questions.write_text(
        json.dumps(
            {
                "questions": [
                    {
                        "id": "q1",
                        "question": "What result?",
                        "answer": "supported result",
                        "expected_answerability": "sufficient",
                        "required_doc_ids": ["paper"],
                        "grader_notes": ["Check support."],
                    }
                ]
            }
        )
    )

    def transcript(label, answer, evidence):
        return [
            {
                "question_id": "q1",
                "question": "What result?",
                "run_label": label,
                "policy": label,
                "run_id": label,
                "status": "completed",
                "predicted_answer": answer,
                "predicted_citations": ["paper"],
                "predicted_evidence": evidence,
                "duration_seconds": 1.0,
                "trajectory": [{"action": 'search("result")', "observation": "found"}],
            }
        ]

    base = tmp_path / "base.json"
    trained = tmp_path / "trained.json"
    base.write_text(json.dumps(transcript("base", "guess", [])))
    trained.write_text(
        json.dumps(
            transcript("trained", "supported result", [{"doc_id": "paper", "start": 6, "end": 22}])
        )
    )
    return questions, corpus, base, trained


def test_prepare_review_blinds_systems_and_materializes_exact_spans(tmp_path):
    questions, corpus, base, trained = _write_fixture(tmp_path)
    review, key, automatic = prepare_review(questions, corpus, [base, trained], seed=42)
    assert len(review["rows"]) == 2
    assert all("system" not in row for row in review["rows"])
    assert {item["system"] for item in key["assignments"]} == {"base", "trained"}
    trained_blind_id = next(
        item["blind_id"] for item in key["assignments"] if item["system"] == "trained"
    )
    trained_row = next(row for row in review["rows"] if row["blind_id"] == trained_blind_id)
    assert trained_row["evidence"][0]["quote"] == "supported result"
    assert automatic["systems"]["trained"]["questions_with_valid_evidence_rate"] == 1.0
    assert automatic["systems"]["base"]["questions_with_valid_evidence_rate"] == 0.0


def test_score_review_reports_supported_answers_and_paired_winner(tmp_path):
    questions, corpus, base, trained = _write_fixture(tmp_path)
    review, key, _ = prepare_review(questions, corpus, [base, trained], seed=42)
    for row in review["rows"]:
        system = next(
            item["system"] for item in key["assignments"] if item["blind_id"] == row["blind_id"]
        )
        row["verdict"] = "pass" if system == "trained" else "fail"
    score = score_review(review, key)
    assert score["systems"]["trained"]["supported_answer_rate"] == 1.0
    assert score["paired_wins"]["trained"] == 1

    review["rows"][0]["verdict"] = None
    with pytest.raises(ValueError, match="needs pass"):
        score_review(review, key)

"""Check that QASPER smoke scoring keeps automatic and semantic judgments separate."""

import json

from src.research.qasper_smoke import score_qasper_smoke


def test_score_tracks_gold_reach_and_abstention_without_claiming_correctness(tmp_path):
    questions = {
        "corpus_hash": "a" * 64,
        "questions": [
            {
                "id": "answerable",
                "answer_types": ["extractive"],
                "expected_answerability": "sufficient",
                "target_doc_ids": ["paper"],
                "gold_evidence": [{"doc_id": "paper", "start": 10, "end": 20}],
            },
            {
                "id": "unanswerable",
                "answer_types": ["unanswerable"],
                "expected_answerability": "insufficient",
                "target_doc_ids": ["paper"],
                "gold_evidence": [],
            },
        ],
    }
    plan = {
        "experiment_id": "fixture",
        "snapshot_corpus_hash": "a" * 64,
        "question_ids": ["answerable", "unanswerable"],
    }
    runs = tmp_path / "runs"
    runs.mkdir()
    (tmp_path / "questions.json").write_text(json.dumps(questions))
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    answerable = {
        "status": "submitted",
        "submission": {
            "claims": [
                {
                    "text": "A claim",
                    "evidence": [{"doc_id": "paper", "start": 12, "end": 18}],
                }
            ]
        },
        "checks": {"claims": [{"has_valid_source_span": True}]},
        "trajectory": [
            {"action": {"action": "paper", "arguments": {"doc_id": "paper"}}},
            {"action": {"action": "paper", "arguments": {"doc_id": "paper"}}},
            {"raw_action": "not JSON", "output": "Action rejected"},
        ],
    }
    abstention = {
        "status": "submitted",
        "submission": {"claims": []},
        "checks": {"claims": []},
        "trajectory": [],
    }
    (runs / "answerable.json").write_text(json.dumps(answerable))
    (runs / "unanswerable.json").write_text(json.dumps(abstention))

    score = score_qasper_smoke(tmp_path / "plan.json", tmp_path / "questions.json", runs)
    assert score["automatic"]["submission_rate"] == 1.0
    assert score["automatic"]["answerable_gold_evidence_reach_rate"] == 1.0
    assert score["automatic"]["unanswerable_abstention_rate"] == 1.0
    assert score["automatic"]["questions_with_duplicate_tool_actions"] == 1
    assert score["questions"][0]["answer_correct"] is None
    assert score["decision"]["semantic_review_required"] is True

"""Checks for the QASPER within-paper retrieval diagnostic."""

from src.research.qasper_retrieval import evaluate_qasper_paper_search


class FakeTools:
    paper_reranker = None

    def search_paper(self, doc_id, query, top_k):
        assert (doc_id, query, top_k) == ("p", "Where is the answer?", 5)
        return [
            {"doc_id": "p", "start": 0, "end": 10, "section": "Intro", "score": 2.0},
            {"doc_id": "p", "start": 20, "end": 40, "section": "Method", "score": 1.0},
        ]


def test_qasper_paper_search_scores_overlap_and_flags_missing_gold():
    questions = {
        "question_set_id": "qasper-train", "corpus_hash": "abc", "questions": [
            {
                "id": "hit", "source_question": "Where is the answer?",
                "target_doc_ids": ["p"], "expected_answerability": "sufficient",
                "gold_evidence": [{"doc_id": "p", "start": 25, "end": 30}],
            },
            {
                "id": "missing", "source_question": "No label", "target_doc_ids": ["p"],
                "expected_answerability": "sufficient", "gold_evidence": [],
            },
            {
                "id": "abstain", "source_question": "Unknown", "target_doc_ids": ["p"],
                "expected_answerability": "insufficient", "gold_evidence": [],
            },
        ],
    }
    result = evaluate_qasper_paper_search(questions, FakeTools())
    assert result["gold_evidence_recall_at"] == {"1": 0.0, "3": 1.0, "5": 1.0}
    assert result["decision"]["pass"] is True
    assert result["answerable_without_gold_ids"] == ["missing"]
    assert result["questions"][0]["results"][1]["overlaps_gold"] is True

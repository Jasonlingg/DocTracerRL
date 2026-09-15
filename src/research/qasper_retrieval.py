"""Evaluate within-paper retrieval against QASPER human evidence paragraphs."""

from __future__ import annotations


def _overlaps(hit: dict, evidence: dict) -> bool:
    return (
        hit.get("doc_id") == evidence.get("doc_id")
        and hit["start"] < evidence["end"]
        and hit["end"] > evidence["start"]
    )


def evaluate_qasper_paper_search(question_set: dict, tools, top_ks=(1, 3, 5)) -> dict:
    """Measure retrieval only; this does not score answers or semantic support."""
    if question_set.get("corpus_hash") is None:
        raise ValueError("QASPER question set must identify its corpus")
    maximum = max(top_ks)
    rows = []
    answerable_without_gold = []
    for question in question_set.get("questions", []):
        if question.get("expected_answerability") != "sufficient":
            continue
        evidence = question.get("gold_evidence", [])
        if not evidence:
            answerable_without_gold.append(question["id"])
            continue
        hits = tools.search_paper(
            question["target_doc_ids"][0], question["source_question"], top_k=maximum
        )
        hit_at = {
            str(k): any(_overlaps(hit, item) for hit in hits[:k] for item in evidence)
            for k in top_ks
        }
        rows.append({
            "question_id": question["id"],
            "doc_id": question["target_doc_ids"][0],
            "query": question["source_question"],
            "hit_at": hit_at,
            "results": [
                {
                    "start": hit["start"], "end": hit["end"],
                    "section": hit["section"], "score": hit["score"],
                    "overlaps_gold": any(_overlaps(hit, item) for item in evidence),
                }
                for hit in hits
            ],
        })
    denominator = len(rows)
    recall = {
        str(k): round(sum(row["hit_at"][str(k)] for row in rows) / denominator, 4)
        if denominator else None
        for k in top_ks
    }
    return {
        "schema_version": "qasper-known-paper-retrieval-v1",
        "question_set_id": question_set.get("question_set_id"),
        "corpus_hash": question_set["corpus_hash"],
        "retriever": tools.paper_reranker.config if tools.paper_reranker else {
            "kind": "lexical_paragraph", "version": "v1"
        },
        "answerable_with_gold_count": denominator,
        "answerable_without_gold_count": len(answerable_without_gold),
        "answerable_without_gold_ids": answerable_without_gold,
        "gold_evidence_recall_at": recall,
        "decision": {
            "metric": "gold_evidence_recall_at_3",
            "threshold": 0.8,
            "pass": recall.get("3") is not None and recall["3"] >= 0.8,
        },
        "questions": rows,
        "note": (
            "Uses the original question as the query and human evidence only for scoring. "
            "Evidence overlap does not measure answer correctness or semantic support."
        ),
    }

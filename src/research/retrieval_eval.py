"""CPU-only retrieval checks for development question sets."""

from __future__ import annotations

import json
from pathlib import Path

from src.eval.artifacts import configuration_hash
from src.research.agent import load_snapshot
from src.research.tools_runtime import ResearchTools


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def evaluate_retrieval(snapshot: Path, questions_path: Path, top_k: int = 5) -> dict:
    """Measure whether a fixed lexical query surfaces its intended papers.

    This checks document discovery only. It does not judge passage relevance,
    claim support, answer quality, or a model's ability to formulate searches.
    """
    if not 1 <= top_k <= 10:
        raise ValueError("top_k must be 1..10")
    manifest, _ = load_snapshot(snapshot)
    question_set = json.loads(questions_path.read_text())
    if question_set.get("schema_version") != "research-questions-v1":
        raise ValueError("question set must use schema_version research-questions-v1")
    if question_set.get("corpus_hash") != manifest["corpus_hash"]:
        raise ValueError("question set and snapshot corpus hashes differ")
    questions = question_set.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError("question set needs at least one question")

    tools = ResearchTools(snapshot / "corpus")
    rows = []
    total_targets = 0
    total_hits = 0
    for question in questions:
        query = question.get("retrieval_query")
        targets = question.get("target_doc_ids")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"{question.get('id')} needs a retrieval_query")
        if not isinstance(targets, list) or any(not isinstance(item, str) for item in targets):
            raise ValueError(f"{question.get('id')} needs target_doc_ids")
        results = tools.search_papers(query, top_k=top_k)
        retrieved = list(dict.fromkeys(result["doc_id"] for result in results))
        hits = [doc_id for doc_id in targets if doc_id in retrieved]
        missed = [doc_id for doc_id in targets if doc_id not in retrieved]
        total_targets += len(targets)
        total_hits += len(hits)
        rows.append({
            "question_id": question["id"],
            "retrieval_query": query,
            "target_doc_ids": targets,
            "retrieved_doc_ids": retrieved,
            "target_hits": hits,
            "target_misses": missed,
            "all_targets_found": len(hits) == len(targets) if targets else None,
            "top_passages": [
                {
                    "doc_id": result["doc_id"],
                    "start": result["start"],
                    "end": result["end"],
                    "section": result["section"],
                    "score": result["score"],
                }
                for result in results
            ],
        })

    evaluable = [row for row in rows if row["target_doc_ids"]]
    return {
        "schema_version": "research-retrieval-check-v1",
        "question_set_id": question_set.get("question_set_id"),
        "question_set_hash": configuration_hash(question_set),
        "corpus_hash": manifest["corpus_hash"],
        "retriever": "lexical-best-passage-per-document-v2",
        "top_k_passages": top_k,
        "question_count": len(rows),
        "evaluable_question_count": len(evaluable),
        "metrics": {
            "target_document_recall": _ratio(total_hits, total_targets),
            "all_target_documents_found_rate": _ratio(
                sum(row["all_targets_found"] is True for row in evaluable), len(evaluable)
            ),
            "any_target_document_found_rate": _ratio(
                sum(bool(row["target_hits"]) for row in evaluable), len(evaluable)
            ),
        },
        "questions": rows,
        "note": (
            "Routing diagnostic only. A retrieved passage may still be irrelevant "
            "or fail to support a claim."
        ),
    }

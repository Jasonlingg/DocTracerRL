"""Validate a locked, paper-disjoint QASPER evaluation plan."""

from __future__ import annotations


def validate_qasper_evaluation_plan(
    plan: dict,
    question_set: dict,
    evaluation_manifest: dict,
    training_manifest: dict,
) -> dict:
    if plan.get("schema_version") != "qasper-held-out-eval-v1":
        raise ValueError("Unsupported QASPER evaluation plan schema")
    if plan.get("status") != "locked_before_research_sft":
        raise ValueError("QASPER evaluation plan must be locked before research SFT")
    for field in ("source_dataset", "source_revision", "source_split"):
        if plan.get(field) != question_set.get(field):
            raise ValueError(f"Evaluation plan and question set disagree on {field}")
    corpus_hash = evaluation_manifest.get("corpus_hash")
    if (
        plan.get("snapshot_corpus_hash") != corpus_hash
        or question_set.get("corpus_hash") != corpus_hash
    ):
        raise ValueError("Evaluation plan, questions, and snapshot corpus hashes differ")
    if plan.get("training_snapshot_corpus_hash") != training_manifest.get("corpus_hash"):
        raise ValueError("Training snapshot hash differs from the locked plan")
    if plan["source_split"] == training_manifest.get("source_split"):
        raise ValueError("Evaluation and training use the same QASPER source split")

    training_papers = {paper["qasper_paper_id"] for paper in training_manifest.get("papers", [])}
    evaluation_papers = {
        paper["qasper_paper_id"] for paper in evaluation_manifest.get("papers", [])
    }
    overlap = training_papers & evaluation_papers
    if overlap or plan.get("paper_overlap_with_training_snapshot") != 0:
        raise ValueError(f"Training paper families leaked into evaluation: {sorted(overlap)}")

    question_lookup = {question["id"]: question for question in question_set.get("questions", [])}
    question_ids = plan.get("question_ids", [])
    if not question_ids or len(question_ids) != len(set(question_ids)):
        raise ValueError("Evaluation question IDs must be non-empty and unique")
    missing = set(question_ids) - set(question_lookup)
    if missing:
        raise ValueError(f"Unknown evaluation question IDs: {sorted(missing)}")
    selected = [question_lookup[question_id] for question_id in question_ids]
    for question in selected:
        if question.get("usage") != "reserved_evaluation":
            raise ValueError("Evaluation questions must be reserved_evaluation rows")
        if question.get("expected_answerability") not in {"sufficient", "insufficient"}:
            raise ValueError("Evaluation questions cannot contain answerability disagreement")
        if question.get("conversion_issues") or question.get("ambiguous_text_evidence_count"):
            raise ValueError("Evaluation questions must have unambiguous text evidence")
        if question["expected_answerability"] == "sufficient" and not question.get("gold_evidence"):
            raise ValueError("Answerable evaluation questions need gold text evidence")

    coverage = plan.get("coverage", {})
    if coverage.get("question_count") != len(selected):
        raise ValueError("Locked question count does not match question IDs")
    if coverage.get("unique_paper_count") != len({q["source_paper_id"] for q in selected}):
        raise ValueError("Locked paper count does not match selected questions")
    return {
        "status": "passed",
        "question_count": len(selected),
        "unique_paper_count": coverage["unique_paper_count"],
        "paper_overlap_with_training_snapshot": 0,
        "source_split": plan["source_split"],
        "snapshot_corpus_hash": corpus_hash,
    }

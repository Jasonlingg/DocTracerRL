"""Automatic, provenance-only scoring for the fixed QASPER smoke experiment."""

from __future__ import annotations

import json
from pathlib import Path


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _cited_spans(run: dict) -> list[dict]:
    submission = run.get("submission") or {}
    return [
        evidence
        for claim in submission.get("claims", [])
        for evidence in claim.get("evidence", [])
        if isinstance(evidence, dict)
    ]


def _overlaps_gold(citation: dict, question: dict) -> bool:
    start, end = citation.get("start"), citation.get("end")
    if type(start) is not int or type(end) is not int:
        return False
    return any(
        citation.get("doc_id") == gold.get("doc_id")
        and start < gold.get("end", -1)
        and end > gold.get("start", -1)
        for gold in question.get("gold_evidence", [])
    )


def score_qasper_smoke(plan_path: Path, questions_path: Path, runs_dir: Path) -> dict:
    plan = json.loads(plan_path.read_text())
    question_set = json.loads(questions_path.read_text())
    if question_set.get("corpus_hash") != plan.get("snapshot_corpus_hash"):
        raise ValueError("Smoke plan and question-set corpus hashes differ")
    questions = {item["id"]: item for item in question_set.get("questions", [])}
    if missing := set(plan.get("question_ids", [])) - set(questions):
        raise ValueError(f"Smoke questions missing from question set: {sorted(missing)}")

    rows = []
    total_claims = 0
    valid_source_claims = 0
    for question_id in plan["question_ids"]:
        question = questions[question_id]
        path = runs_dir / f"{question_id}.json"
        run = json.loads(path.read_text()) if path.exists() else {"status": "missing"}
        citations = _cited_spans(run)
        claims = (run.get("submission") or {}).get("claims", [])
        checks = run.get("checks") or {}
        valid_claim_count = sum(
            item.get("has_valid_source_span") is True for item in checks.get("claims", [])
        )
        total_claims += len(claims)
        valid_source_claims += valid_claim_count
        tool_actions = [
            record["action"]
            for record in run.get("trajectory", [])
            if isinstance(record.get("action"), dict)
            and record["action"].get("action") != "submit"
        ]
        action_keys = [json.dumps(action, sort_keys=True) for action in tool_actions]
        duplicate_actions = len(action_keys) - len(set(action_keys))
        expected = question["expected_answerability"]
        submitted = run.get("status") == "submitted"
        rows.append(
            {
                "question_id": question_id,
                "answer_types": question["answer_types"],
                "expected_answerability": expected,
                "status": run.get("status"),
                "error": run.get("error"),
                "steps": len(run.get("trajectory", [])),
                "claim_count": len(claims),
                "valid_source_claim_count": valid_claim_count,
                "target_document_cited": any(
                    item.get("doc_id") in question["target_doc_ids"] for item in citations
                ),
                "gold_evidence_reached": (
                    any(_overlaps_gold(item, question) for item in citations)
                    if expected == "sufficient"
                    else None
                ),
                "abstained": submitted and not claims if expected == "insufficient" else None,
                "duplicate_tool_actions": duplicate_actions,
                "answer_correct": None,
                "semantic_support": "not_reviewed",
            }
        )

    submitted_count = sum(row["status"] == "submitted" for row in rows)
    answerable = [row for row in rows if row["expected_answerability"] == "sufficient"]
    unanswerable = [row for row in rows if row["expected_answerability"] == "insufficient"]
    gold_hits = sum(row["gold_evidence_reached"] is True for row in answerable)
    abstentions = sum(row["abstained"] is True for row in unanswerable)
    protocol_pass = submitted_count >= 4
    evidence_access_pass = gold_hits >= 3
    expected_signal_pass = (
        protocol_pass and evidence_access_pass and abstentions == len(unanswerable)
    )
    return {
        "schema_version": "qasper-known-paper-smoke-score-v1",
        "experiment_id": plan["experiment_id"],
        "snapshot_corpus_hash": plan["snapshot_corpus_hash"],
        "question_count": len(rows),
        "automatic": {
            "submission_rate": _ratio(submitted_count, len(rows)),
            "valid_source_claim_rate": _ratio(valid_source_claims, total_claims),
            "answerable_gold_evidence_reach_rate": _ratio(gold_hits, len(answerable)),
            "unanswerable_abstention_rate": _ratio(abstentions, len(unanswerable)),
            "questions_with_duplicate_tool_actions": sum(
                row["duplicate_tool_actions"] > 0 for row in rows
            ),
        },
        "decision": {
            "protocol_threshold_pass": protocol_pass,
            "evidence_access_threshold_pass": evidence_access_pass,
            "numeric_decision_rule_pass": protocol_pass and evidence_access_pass,
            "full_expected_signal_pass": expected_signal_pass,
            "semantic_review_required": True,
        },
        "questions": rows,
        "note": (
            "Gold-evidence overlap measures whether a citation intersects a human evidence "
            "paragraph. It does not establish answer correctness or semantic support."
        ),
    }

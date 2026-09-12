"""Reproducible scoring for evidence-grounded research-agent runs."""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.eval.artifacts import configuration_hash

SCHEMA_VERSION = "research-benchmark-v1"
REVIEW_SCHEMA_VERSION = "research-human-review-v1"
SCORE_SCHEMA_VERSION = "research-benchmark-score-v1"
SUPPORT_LABELS = {"supported", "partial", "unsupported"}
FAILURE_CATEGORIES = {
    "none",
    "no_submission",
    "tool_protocol",
    "retrieval",
    "evidence_selection",
    "unsupported_synthesis",
    "answerability",
    "other",
}


def load_benchmark(path: Path, snapshot_manifest: dict | None = None) -> dict:
    benchmark = json.loads(path.read_text())
    validate_benchmark(benchmark, snapshot_manifest)
    return benchmark


def validate_benchmark(benchmark: dict, snapshot_manifest: dict | None = None) -> None:
    """Reject ambiguous splits, duplicate IDs, and corpus leakage hazards."""
    if benchmark.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"benchmark schema_version must be {SCHEMA_VERSION}")
    if not isinstance(benchmark.get("benchmark_id"), str) or not benchmark["benchmark_id"]:
        raise ValueError("benchmark_id must be a non-empty string")
    corpus_hash = benchmark.get("corpus_hash")
    if not isinstance(corpus_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", corpus_hash):
        raise ValueError("corpus_hash must be a SHA-256 hex digest")
    reserved = benchmark.get("reserved_doc_ids")
    if not isinstance(reserved, list) or not reserved or len(reserved) != len(set(reserved)):
        raise ValueError("reserved_doc_ids must be a non-empty unique list")
    questions = benchmark.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError("benchmark needs at least one question")
    ids = [question.get("id") for question in questions if isinstance(question, dict)]
    if len(ids) != len(questions) or any(not isinstance(item, str) or not item for item in ids):
        raise ValueError("every question needs a non-empty id")
    if len(ids) != len(set(ids)):
        raise ValueError("benchmark question IDs must be unique")
    for question in questions:
        if question.get("split") != "pilot_evaluation":
            raise ValueError("pilot benchmark questions must use split=pilot_evaluation")
        if not isinstance(question.get("question"), str) or not question["question"].strip():
            raise ValueError(f"{question['id']} needs question text")
        if question.get("expected_answerability") not in {"sufficient", "insufficient"}:
            raise ValueError(f"{question['id']} needs expected_answerability")
        required = question.get("required_doc_ids")
        if not isinstance(required, list) or any(doc_id not in reserved for doc_id in required):
            raise ValueError(f"{question['id']} required_doc_ids must be reserved documents")
        minimum = question.get("minimum_distinct_sources")
        if type(minimum) is not int or minimum < 0:
            raise ValueError(f"{question['id']} needs a non-negative minimum_distinct_sources")
        if not isinstance(question.get("grader_notes"), list):
            raise ValueError(f"{question['id']} needs grader_notes")
    if snapshot_manifest is not None:
        if snapshot_manifest.get("corpus_hash") != corpus_hash:
            raise ValueError("benchmark and snapshot corpus hashes differ")
        snapshot_ids = {paper["doc_id"] for paper in snapshot_manifest.get("papers", [])}
        if set(reserved) != snapshot_ids:
            raise ValueError("all snapshot documents must be reserved from training")


def _load_runs(benchmark: dict, runs_dir: Path) -> tuple[list[dict], list[str]]:
    runs = []
    missing = []
    for question in benchmark["questions"]:
        path = runs_dir / f"{question['id']}.json"
        if not path.exists():
            missing.append(question["id"])
            continue
        run = json.loads(path.read_text())
        if run.get("question", {}).get("id") != question["id"]:
            raise ValueError(f"run question mismatch in {path}")
        if run.get("corpus_hash") != benchmark["corpus_hash"]:
            raise ValueError(f"run corpus mismatch in {path}")
        runs.append(run)
    return runs, missing


def make_review_template(benchmark: dict, runs_dir: Path) -> dict:
    """Create a claim-level review form containing the evidence a reviewer must inspect."""
    runs, missing = _load_runs(benchmark, runs_dir)
    entries = []
    questions = {question["id"]: question for question in benchmark["questions"]}
    for run in runs:
        question_id = run["question"]["id"]
        submission = run.get("submission") or {}
        claims = submission.get("claims", [])
        entries.append({
            "question_id": question_id,
            "expected_answerability": questions[question_id]["expected_answerability"],
            "status": run.get("status"),
            "claim_reviews": [
                {
                    "claim_index": index,
                    "claim": claim.get("text"),
                    "evidence": [
                        {
                            "doc_id": item.get("doc_id"),
                            "start": item.get("start"),
                            "end": item.get("end"),
                            "quote": item.get("quote"),
                        }
                        for item in claim.get("evidence", [])
                    ],
                    "support": None,
                    "notes": "",
                }
                for index, claim in enumerate(claims)
            ],
            "answerability_handled": None,
            "recommendation_faithful": None,
            "relevant_evidence_missed": None,
            "usefulness": None,
            "failure_category": None,
            "notes": "",
        })
    return {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "benchmark_id": benchmark["benchmark_id"],
        "benchmark_hash": configuration_hash(benchmark),
        "review_status": "incomplete",
        "instructions": {
            "support": "Choose supported, partial, or unsupported for every claim.",
            "answerability_handled": "True only if the answer answered or abstained as expected.",
            "recommendation_faithful": "True only if advice adds no unsupported factual premise.",
            "relevant_evidence_missed": "True when an important available passage was omitted.",
            "usefulness": "0 unusable, 1 partly useful, 2 useful for a project decision.",
        },
        "missing_run_ids": missing,
        "reviews": entries,
    }


def _validate_reviews(benchmark: dict, reviews: dict, runs: dict[str, dict]) -> list[dict]:
    if reviews.get("schema_version") != REVIEW_SCHEMA_VERSION:
        raise ValueError(f"review schema_version must be {REVIEW_SCHEMA_VERSION}")
    if reviews.get("benchmark_hash") != configuration_hash(benchmark):
        raise ValueError("review template was created for a different benchmark")
    entries = reviews.get("reviews")
    if not isinstance(entries, list):
        raise ValueError("reviews must be a list")
    seen = set()
    for entry in entries:
        question_id = entry.get("question_id")
        if question_id in seen or question_id not in runs:
            raise ValueError("reviews contain duplicate or unknown question IDs")
        seen.add(question_id)
        if entry.get("answerability_handled") not in {True, False}:
            raise ValueError(f"{question_id} answerability_handled is incomplete")
        if entry.get("recommendation_faithful") not in {True, False}:
            raise ValueError(f"{question_id} recommendation_faithful is incomplete")
        if entry.get("relevant_evidence_missed") not in {True, False}:
            raise ValueError(f"{question_id} relevant_evidence_missed is incomplete")
        if entry.get("usefulness") not in {0, 1, 2}:
            raise ValueError(f"{question_id} usefulness must be 0, 1, or 2")
        if entry.get("failure_category") not in FAILURE_CATEGORIES:
            raise ValueError(f"{question_id} has an invalid failure_category")
        claim_reviews = entry.get("claim_reviews")
        if not isinstance(claim_reviews, list):
            raise ValueError(f"{question_id} claim_reviews must be a list")
        expected_indices = set(range(len((runs[question_id].get("submission") or {}).get(
            "claims", []
        ))))
        actual_indices = {claim.get("claim_index") for claim in claim_reviews}
        if actual_indices != expected_indices or len(claim_reviews) != len(expected_indices):
            raise ValueError(f"{question_id} must review every submitted claim exactly once")
        for claim in claim_reviews:
            if claim.get("support") not in SUPPORT_LABELS:
                raise ValueError(f"{question_id} has an incomplete claim review")
    if seen != set(runs):
        raise ValueError("human metrics require a review for every available run")
    return entries


def score_benchmark(benchmark: dict, runs_dir: Path, reviews: dict | None = None) -> dict:
    """Report mechanical provenance and human judgments as separate metric groups."""
    runs, missing = _load_runs(benchmark, runs_dir)
    questions = {question["id"]: question for question in benchmark["questions"]}
    rows = []
    for run in runs:
        question_id = run["question"]["id"]
        expected = questions[question_id]
        submission = run.get("submission") or {}
        claims = submission.get("claims", [])
        checks = run.get("checks") or {}
        cited_docs = {
            item.get("doc_id")
            for claim in claims
            for item in claim.get("evidence", [])
            if isinstance(item.get("doc_id"), str)
        }
        required = set(expected["required_doc_ids"])
        actions = [str(step.get("action", "")) for step in run.get("trajectory", [])]
        outputs = [str(step.get("output", "")) for step in run.get("trajectory", [])]
        rows.append({
            "question_id": question_id,
            "status": run.get("status"),
            "submitted": run.get("status") == "submitted",
            "claim_count": len(claims),
            "claims_with_valid_source_spans": checks.get("claims_with_valid_source_spans", 0),
            "distinct_sources": len(cited_docs),
            "minimum_distinct_sources": expected["minimum_distinct_sources"],
            "required_doc_hits": len(required & cited_docs),
            "required_doc_count": len(required),
            "trajectory_steps": len(run.get("trajectory", [])),
            "tool_action_steps": sum(
                bool(re.search(r"\b(?:papers|search_papers|paper|passage)\s*\(", action))
                for action in actions
            ),
            "execution_error_steps": sum(
                "Traceback" in output or "SyntaxError" in output for output in outputs
            ),
        })
    claim_count = sum(row["claim_count"] for row in rows)
    source_valid = sum(row["claims_with_valid_source_spans"] for row in rows)
    required_count = sum(row["required_doc_count"] for row in rows)
    result = {
        "schema_version": SCORE_SCHEMA_VERSION,
        "benchmark_id": benchmark["benchmark_id"],
        "benchmark_hash": configuration_hash(benchmark),
        "run_count": len(rows),
        "missing_run_ids": missing,
        "automatic": {
            "coverage": _ratio(len(rows), len(benchmark["questions"])),
            "submission_rate": _ratio(sum(row["submitted"] for row in rows), len(rows)),
            "source_valid_claim_rate": _ratio(source_valid, claim_count),
            "required_document_recall": _ratio(
                sum(row["required_doc_hits"] for row in rows), required_count
            ),
            "source_count_requirement_rate": _ratio(
                sum(
                    row["distinct_sources"] >= row["minimum_distinct_sources"] for row in rows
                ),
                len(rows),
            ),
            "average_trajectory_steps": _mean([row["trajectory_steps"] for row in rows]),
            "execution_error_step_rate": _ratio(
                sum(row["execution_error_steps"] for row in rows),
                sum(row["trajectory_steps"] for row in rows),
            ),
            "note": "These metrics verify execution and provenance, not semantic support.",
        },
        "human": None,
        "questions": rows,
    }
    if reviews is not None:
        entries = _validate_reviews(
            benchmark, reviews, {run["question"]["id"]: run for run in runs}
        )
        labels = [
            claim["support"] for entry in entries for claim in entry.get("claim_reviews", [])
        ]
        result["human"] = {
            "review_coverage": _ratio(len(entries), len(rows)),
            "claim_support_counts": {
                label: labels.count(label) for label in sorted(SUPPORT_LABELS)
            },
            "fully_supported_claim_rate": _ratio(labels.count("supported"), len(labels)),
            "unsupported_claim_rate": _ratio(labels.count("unsupported"), len(labels)),
            "answerability_accuracy": _ratio(
                sum(entry["answerability_handled"] for entry in entries), len(entries)
            ),
            "faithful_recommendation_rate": _ratio(
                sum(entry["recommendation_faithful"] for entry in entries), len(entries)
            ),
            "missed_relevant_evidence_rate": _ratio(
                sum(entry["relevant_evidence_missed"] for entry in entries), len(entries)
            ),
            "average_usefulness_0_to_2": _mean([entry["usefulness"] for entry in entries]),
            "failure_category_counts": {
                category: sum(entry["failure_category"] == category for entry in entries)
                for category in sorted(FAILURE_CATEGORIES)
            },
        }
    return result


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _mean(values: list[int]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def score_markdown(score: dict) -> str:
    automatic = score["automatic"]
    lines = [
        f"# Benchmark score: {score['benchmark_id']}",
        "",
        f"Runs: {score['run_count']} · Missing: {len(score['missing_run_ids'])}",
        "",
        "## Automatic checks",
        "",
        f"- Submission rate: {_percent(automatic['submission_rate'])}",
        f"- Claims with valid source spans: {_percent(automatic['source_valid_claim_rate'])}",
        f"- Required-document recall: {_percent(automatic['required_document_recall'])}",
        f"- Source-count requirement: {_percent(automatic['source_count_requirement_rate'])}",
        f"- Execution-error step rate: {_percent(automatic['execution_error_step_rate'])}",
        "",
        automatic["note"],
        "",
        "## Human review",
        "",
    ]
    human = score.get("human")
    if human is None:
        lines.append("Not supplied. Do not interpret source validity as answer correctness.")
    else:
        lines += [
            f"- Review coverage: {_percent(human['review_coverage'])}",
            f"- Fully supported claims: {_percent(human['fully_supported_claim_rate'])}",
            f"- Unsupported claims: {_percent(human['unsupported_claim_rate'])}",
            f"- Answerability accuracy: {_percent(human['answerability_accuracy'])}",
            f"- Faithful recommendations: {_percent(human['faithful_recommendation_rate'])}",
            f"- Average usefulness (0–2): {human['average_usefulness_0_to_2']}",
        ]
    if score["missing_run_ids"]:
        lines += ["", "Missing run IDs: " + ", ".join(score["missing_run_ids"])]
    return "\n".join(lines) + "\n"


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"

"""Blind human review for code-execution research evaluations."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

VERDICTS = {"pass", "partial", "fail"}
VERDICT_SCORE = {"pass": 2, "partial": 1, "fail": 0}


def load_question_set(path: Path) -> dict:
    data = json.loads(path.read_text())
    questions = data.get("questions") if isinstance(data, dict) else data
    if not isinstance(questions, list) or not questions:
        raise ValueError("Question file must be a list or contain a non-empty questions list")
    ids = [question.get("id") for question in questions]
    if any(not isinstance(item, str) or not item for item in ids) or len(ids) != len(set(ids)):
        raise ValueError("Questions need unique non-empty IDs")
    return {"metadata": data if isinstance(data, dict) else {}, "questions": questions}


def load_corpus(path: Path) -> dict[str, dict]:
    documents = {}
    for item in sorted(path.glob("*.json")):
        document = json.loads(item.read_text())
        doc_id = document.get("doc_id")
        if not isinstance(doc_id, str) or not isinstance(document.get("text"), str):
            raise ValueError(f"Invalid corpus document: {item}")
        documents[doc_id] = document
    if not documents:
        raise ValueError("Corpus contains no JSON documents")
    return documents


def _load_runs(paths: list[Path], expected_ids: set[str]) -> list[dict]:
    runs = []
    systems = set()
    for path in paths:
        rows = json.loads(path.read_text())
        if not isinstance(rows, list):
            raise ValueError(f"Transcript must contain a list: {path}")
        row_ids = {row.get("question_id") for row in rows}
        if row_ids != expected_ids or len(rows) != len(expected_ids):
            raise ValueError(f"Transcript question IDs do not match the frozen set: {path}")
        labels = {str(row.get("run_label") or row.get("policy") or "unknown") for row in rows}
        if len(labels) != 1:
            raise ValueError(f"Transcript mixes system labels: {path}")
        system = labels.pop()
        if system in systems:
            raise ValueError(f"Duplicate system label: {system}")
        systems.add(system)
        runs.append({"path": str(path), "system": system, "rows": rows})
    if len(runs) != 2:
        raise ValueError("The minimal blind comparison requires exactly two transcript files")
    return runs


def materialize_evidence(items: object, documents: dict[str, dict]) -> list[dict]:
    """Attach exact corpus text and mark malformed or out-of-range spans."""
    if not isinstance(items, list):
        return []
    materialized = []
    for item in items:
        if not isinstance(item, dict):
            materialized.append({"valid": False, "error": "evidence item is not an object"})
            continue
        doc_id, start, end = item.get("doc_id"), item.get("start"), item.get("end")
        document = documents.get(doc_id) if isinstance(doc_id, str) else None
        valid = bool(
            document is not None
            and type(start) is int
            and type(end) is int
            and 0 <= start < end <= len(document["text"])
        )
        resolved = {"doc_id": doc_id, "start": start, "end": end, "valid": valid}
        if valid:
            resolved["quote"] = document["text"][start:end]
        else:
            resolved["error"] = "document or character range is invalid"
        materialized.append(resolved)
    return materialized


def _automatic_metrics(rows: list[dict], questions: dict[str, dict], documents: dict) -> dict:
    submitted = 0
    spans = []
    questions_with_valid_evidence = 0
    required_hits = 0
    required_total = 0
    error_steps = 0
    duplicate_steps = 0
    total_steps = 0
    durations = []
    for row in rows:
        submitted += bool(row.get("predicted_answer"))
        evidence = materialize_evidence(row.get("predicted_evidence"), documents)
        spans.extend(item["valid"] for item in evidence)
        questions_with_valid_evidence += any(item["valid"] for item in evidence)
        required = set(questions[row["question_id"]].get("required_doc_ids", []))
        cited = set(row.get("predicted_citations") or [])
        required_hits += len(required & cited)
        required_total += len(required)
        actions = []
        for step in row.get("trajectory", []):
            action = str(step.get("action", "")).strip()
            observation = str(step.get("observation", ""))
            if action and not action.upper().startswith("SUBMIT:"):
                duplicate_steps += action in actions
                actions.append(action)
            error_steps += any(
                marker in observation
                for marker in ("Traceback", "SyntaxError", "timed out", "ERROR:")
            )
            total_steps += 1
        durations.append(float(row.get("duration_seconds", 0.0)))
    count = len(rows)
    return {
        "question_count": count,
        "submission_rate": round(submitted / count, 4),
        "valid_evidence_span_rate": round(sum(spans) / len(spans), 4) if spans else 0.0,
        "questions_with_valid_evidence_rate": round(questions_with_valid_evidence / count, 4),
        "required_document_recall": (
            round(required_hits / required_total, 4) if required_total else 1.0
        ),
        "execution_error_step_rate": round(error_steps / total_steps, 4) if total_steps else 0.0,
        "repeated_action_step_rate": (
            round(duplicate_steps / total_steps, 4) if total_steps else 0.0
        ),
        "average_steps": round(total_steps / count, 2),
        "average_duration_seconds": round(sum(durations) / count, 2),
        "note": "These checks measure execution and provenance, not semantic support.",
    }


def prepare_review(
    question_path: Path,
    corpus_path: Path,
    run_paths: list[Path],
    seed: int = 42,
) -> tuple[dict, dict, dict]:
    question_set = load_question_set(question_path)
    question_list = question_set["questions"]
    questions = {question["id"]: question for question in question_list}
    documents = load_corpus(corpus_path)
    runs = _load_runs(run_paths, set(questions))
    rng = random.Random(seed)
    review_rows = []
    key_rows = []
    counter = 1
    rows_by_system = {
        run["system"]: {row["question_id"]: row for row in run["rows"]} for run in runs
    }
    systems = [run["system"] for run in runs]
    for question in question_list:
        order = systems[:]
        rng.shuffle(order)
        for system in order:
            row = rows_by_system[system][question["id"]]
            blind_id = f"R{counter:03d}"
            counter += 1
            review_rows.append(
                {
                    "blind_id": blind_id,
                    "question_id": question["id"],
                    "question": question["question"],
                    "expected_answerability": question.get("expected_answerability"),
                    "reference_answer": question.get("answer"),
                    "grader_notes": question.get("grader_notes", []),
                    "answer": row.get("predicted_answer", ""),
                    "citations": row.get("predicted_citations", []),
                    "evidence": materialize_evidence(row.get("predicted_evidence"), documents),
                    "verdict": None,
                    "notes": "",
                }
            )
            key_rows.append(
                {
                    "blind_id": blind_id,
                    "question_id": question["id"],
                    "system": system,
                    "policy": row.get("policy"),
                    "run_id": row.get("run_id"),
                }
            )
    review = {
        "schema_version": "code-exec-blind-review-v1",
        "status": "incomplete",
        "instructions": {
            "pass": "Answers the question, major claims are supported, and limitations are honest.",
            "partial": "Useful, but misses an important point, span, or qualification.",
            "fail": "Wrong, unsupported, misleading, empty, or fails to answer.",
        },
        "rows": review_rows,
    }
    key = {
        "schema_version": "code-exec-blind-key-v1",
        "seed": seed,
        "assignments": key_rows,
    }
    automatic = {
        "schema_version": "code-exec-automatic-score-v1",
        "systems": {
            run["system"]: _automatic_metrics(run["rows"], questions, documents) for run in runs
        },
    }
    return review, key, automatic


def score_review(review: dict, key: dict) -> dict:
    assignments = {item["blind_id"]: item for item in key.get("assignments", [])}
    rows = review.get("rows")
    if not isinstance(rows, list) or set(assignments) != {row.get("blind_id") for row in rows}:
        raise ValueError("Review and blind key do not contain the same rows")
    by_system: dict[str, Counter] = defaultdict(Counter)
    paired: dict[str, dict[str, str]] = defaultdict(dict)
    for row in rows:
        verdict = row.get("verdict")
        if verdict not in VERDICTS:
            raise ValueError(f"{row.get('blind_id')} needs pass, partial, or fail")
        assignment = assignments[row["blind_id"]]
        system = assignment["system"]
        by_system[system][verdict] += 1
        by_system[system]["total"] += 1
        paired[assignment["question_id"]][system] = verdict
    systems = sorted(by_system)
    if len(systems) != 2:
        raise ValueError("Blind key must contain exactly two systems")
    summary = {}
    for system in systems:
        counts = by_system[system]
        total = counts["total"]
        summary[system] = {
            "pass": counts["pass"],
            "partial": counts["partial"],
            "fail": counts["fail"],
            "supported_answer_rate": round(counts["pass"] / total, 4),
            "mean_review_score_0_to_2": round(
                sum(VERDICT_SCORE[label] * counts[label] for label in VERDICTS) / total, 4
            ),
        }
    wins = {system: 0 for system in systems}
    ties = 0
    for outcomes in paired.values():
        if set(outcomes) != set(systems):
            raise ValueError("Every question must contain both systems")
        left, right = systems
        difference = VERDICT_SCORE[outcomes[left]] - VERDICT_SCORE[outcomes[right]]
        if difference > 0:
            wins[left] += 1
        elif difference < 0:
            wins[right] += 1
        else:
            ties += 1
    return {
        "schema_version": "code-exec-human-score-v1",
        "systems": summary,
        "paired_wins": wins,
        "paired_ties": ties,
        "decision_note": (
            "Treat the trained model as improved only if its supported-answer rate is higher and "
            "automatic execution metrics show no unacceptable regression."
        ),
    }


def review_markdown(review: dict) -> str:
    lines = [
        "# Blind AI-paper evaluation",
        "",
        "For each row, enter `pass`, `partial`, or `fail` in `review.json`. System identities "
        "are in",
        "`blind-key.json`; leave that file closed until review is complete.",
        "",
    ]
    for row in review["rows"]:
        lines += [
            f"## {row['blind_id']} · {row['question_id']}",
            "",
            f"**Question:** {row['question']}",
            "",
            f"**Reference:** {row['reference_answer']}",
            "",
            f"**Answer:** {row['answer'] or '*No submitted answer*'}",
            "",
            f"**Citations:** {', '.join(row['citations']) or '*None*'}",
            "",
            "**Evidence:**",
            "",
        ]
        if row["evidence"]:
            for evidence in row["evidence"]:
                if evidence["valid"]:
                    lines += [
                        f"- `{evidence['doc_id']}:{evidence['start']}-{evidence['end']}`",
                        "",
                        f"> {evidence['quote'].replace(chr(10), ' ')}",
                        "",
                    ]
                else:
                    lines += [f"- Invalid span: `{evidence}`", ""]
        else:
            lines += ["- *No exact evidence submitted*", ""]
        lines += ["**Verdict:** pending", "", "**Notes:**", ""]
    return "\n".join(lines)


def score_markdown(score: dict) -> str:
    lines = ["# Blind AI-paper evaluation score", ""]
    for system, values in score["systems"].items():
        lines += [
            f"## {system}",
            "",
            f"- Supported-answer rate: {values['supported_answer_rate']:.1%}",
            f"- Pass / partial / fail: {values['pass']} / {values['partial']} / {values['fail']}",
            f"- Mean review score: {values['mean_review_score_0_to_2']:.2f} / 2",
            f"- Paired wins: {score['paired_wins'][system]}",
            "",
        ]
    lines += [f"Paired ties: {score['paired_ties']}", "", score["decision_note"], ""]
    return "\n".join(lines)

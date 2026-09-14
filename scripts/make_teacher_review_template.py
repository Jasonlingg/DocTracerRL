"""Build a fill-in-the-blanks human review template for teacher trajectory runs.

Adapts src/research/benchmark.py's make_review_template for open-ended dev
questions (data/research/questions.json), which have no expected_answerability
or required_doc_ids the way the locked benchmark_pilot_v1.json does.
"""

import argparse
import json
from pathlib import Path

from src.eval.artifacts import configuration_hash
from src.research.benchmark import REVIEW_SCHEMA_VERSION


def build_template(question_set: dict, runs_dir: Path, run_prefix: str) -> dict:
    entries = []
    missing = []
    for question in question_set["questions"]:
        question_id = question["id"]
        run_path = runs_dir / f"{run_prefix}{question_id}.json"
        if not run_path.exists():
            missing.append(question_id)
            continue
        run = json.loads(run_path.read_text())
        submission = run.get("submission") or {}
        claims = submission.get("claims", [])
        entries.append({
            "question_id": question_id,
            "question": question["question"],
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
        "source": "teacher_trajectories",
        "question_set_hash": configuration_hash(question_set),
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=Path("data/research/questions.json"))
    parser.add_argument("--runs-dir", type=Path, default=Path("out/research/teacher_trajectories"))
    parser.add_argument("--run-prefix", default="teacher_",
                        help="Filename prefix before the question id, e.g. teacher_research_01.json")
    parser.add_argument("--output", type=Path,
                        default=Path("out/research/teacher_trajectories/human-review-template.json"))
    args = parser.parse_args()

    question_set = json.loads(args.questions.read_text())
    template = build_template(question_set, args.runs_dir, args.run_prefix)
    args.output.write_text(json.dumps(template, indent=2, ensure_ascii=False) + "\n")

    reviewable = sum(1 for entry in template["reviews"] if entry["claim_reviews"])
    print(f"Wrote {len(template['reviews'])} run reviews "
          f"({reviewable} with claims to review, {len(template['missing_run_ids'])} missing) "
          f"to {args.output}")


if __name__ == "__main__":
    raise SystemExit(main())

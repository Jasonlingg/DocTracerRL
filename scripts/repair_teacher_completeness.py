"""Regenerate QASPER-train teacher trajectories flagged incomplete on manual review.

Unlike repair_teacher_abstentions.py, this does not give the teacher any ground-truth
label — the question is genuinely answerable and the model already found real evidence,
it just stopped searching before covering every part of a multi-part question. The fix
is a generic verification instruction (not privileged information), so no stripping
concern applies: any deployed model should ideally do this too.

Usage:
  python scripts/repair_teacher_completeness.py <question_id> [<question_id> ...]
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.env.corpus import Corpus
from src.env.document_env import DocumentExplorationEnv
from src.eval.harness import run_single
from src.policies.claude_policy import ClaudePolicy
from src.policies.teacher_hint_policy import TeacherHintPolicy

BENCHMARK = Path("out/research/qasper-train-teacher-v1/benchmark.json")
CORPUS_PATH = Path("out/research/qasper-train-teacher-v1/corpus")
RAW_TEACHER_FILE = Path("out/research/qasper-teacher-trajectories/teacher.json")
OUTPUT = Path("out/research/qasper-teacher-trajectories/completeness_repaired.json")

VERIFICATION_HINT = (
    "Before submitting, list every distinct fact or sub-part the question asks for "
    "(a question can have more than one part, e.g. two different numbers). Confirm "
    "you have found explicit textual support for each part individually before you "
    "submit. Do not submit a partial answer as if it were complete, and do not state "
    "any detail you did not find explicit support for. If the paper describes more "
    "than one quantity that could reasonably answer the question (e.g. two differently "
    "sized datasets used at different stages), report all of them rather than picking "
    "the one that seems most central — do not narrow the scope of the question yourself."
)


def main() -> None:
    load_dotenv(override=True)
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set")

    target_ids = set(sys.argv[1:])
    if not target_ids:
        raise SystemExit("Pass one or more question IDs to regenerate")

    benchmark = json.loads(BENCHMARK.read_text())
    questions = [q for q in benchmark["questions"] if q["id"] in target_ids]
    raw_results = {r["question_id"]: r for r in json.loads(RAW_TEACHER_FILE.read_text())}

    logger.info(f"Regenerating {len(questions)} question(s) with a completeness check: "
                f"{[q['id'] for q in questions]}")

    corpus = Corpus(corpus_path=str(CORPUS_PATH))
    corpus.load(build_index=False)
    env = DocumentExplorationEnv(
        corpus=corpus,
        questions=questions,
        max_steps=10,
        use_docker=False,
        corpus_path=str(CORPUS_PATH),
        require_evidence=False,
    )

    repaired = []
    for idx, q in enumerate(questions):
        logger.info(f"[{q['id']}] gold: {q['answer']!r}")
        logger.info(f"[{q['id']}] old (incomplete) answer was: "
                    f"{raw_results[q['id']]['predicted_answer'][:150]}...")
        inner = ClaudePolicy()
        policy = TeacherHintPolicy(inner, hint=VERIFICATION_HINT)
        result = run_single(env, policy, idx)
        result_dict = json.loads(result.model_dump_json())
        result_dict["question_id"] = q["id"]
        result_dict["run_label"] = "qasper_teacher_sonnet5_completeness_repaired"
        repaired.append(result_dict)
        logger.info(f"[{q['id']}] new answer: {result_dict['predicted_answer']}")
        logger.info(f"[{q['id']}] reward={result_dict['reward']:.3f}")

    env.close()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(repaired, indent=2))
    logger.info(f"Wrote {len(repaired)} repaired trajectories to {OUTPUT}")


if __name__ == "__main__":
    main()

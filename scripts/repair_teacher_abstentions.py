"""Regenerate the QASPER-train teacher trajectories that failed to abstain.

Two of the 20 trajectories in out/research/qasper-teacher-trajectories/teacher.json
are on questions the benchmark marks expected_answerability="insufficient", and the
blind teacher (Claude Sonnet, no hint) answered them anyway instead of abstaining.

This script regenerates only those two, wrapping the teacher policy in
TeacherHintPolicy so it is told the ground-truth answerability up front. That hint
never reaches the environment or the harness's logged question/trajectory text, so a
model later trained on the repaired trajectory still has to infer answerability from
the real search/read steps, not from a label it won't have at inference time. The
original (raw, failed) run stays untouched in teacher.json for comparison; the
regenerated ones are written separately and then merged into the final SFT-ready set.

Usage:
  python scripts/repair_teacher_abstentions.py
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
OUTPUT = Path("out/research/qasper-teacher-trajectories/repaired.json")

TEACHER_ONLY_HINT = (
    "TEACHER-ONLY NOTE (ground truth, for your calibration only — do not reference "
    "this note in your search steps or final answer): this question is confirmed "
    "UNANSWERABLE from this specific paper. You must still demonstrate real due "
    "diligence: run at least 2-3 distinct searches (different queries/sections) "
    "before concluding. Once those searches turn up no explicit textual support, "
    "stop searching and submit an honest abstention within 5 steps total — do not "
    "keep searching indefinitely trying to rule out every possible section, and do "
    "not guess or hedge toward an inferred answer."
)


def main() -> None:
    load_dotenv(override=True)
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set")

    benchmark = json.loads(BENCHMARK.read_text())
    questions = [
        q for q in benchmark["questions"] if q["expected_answerability"] == "insufficient"
    ]
    raw_results = {r["question_id"]: r for r in json.loads(RAW_TEACHER_FILE.read_text())}

    logger.info(f"Repairing {len(questions)} unanswerable question(s): "
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
        logger.info(f"[{q['id']}] raw (blind) answer was: "
                    f"{raw_results[q['id']]['predicted_answer'][:120]}...")
        inner = ClaudePolicy()
        policy = TeacherHintPolicy(inner, hint=TEACHER_ONLY_HINT)
        result = run_single(env, policy, idx)
        result_dict = json.loads(result.model_dump_json())
        result_dict["question_id"] = q["id"]
        result_dict["run_label"] = "qasper_teacher_sonnet5_repaired"
        repaired.append(result_dict)
        logger.info(f"[{q['id']}] repaired answer: {result_dict['predicted_answer'][:200]}")
        logger.info(f"[{q['id']}] reward={result_dict['reward']:.3f}")

    env.close()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(repaired, indent=2))
    logger.info(f"Wrote {len(repaired)} repaired trajectories to {OUTPUT}")


if __name__ == "__main__":
    main()

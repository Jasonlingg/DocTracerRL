"""Generate a large batch of QASPER-train teacher trajectories for Qwen3-8B SFT.

Unlike the earlier one-off repair scripts, this runs the full stratified benchmark
(out/research/qasper-train-teacher-v2, 300 questions: 100 "insufficient" + 200
"sufficient") through Claude Sonnet 5, writing incrementally so a crash, rate
limit, or Ctrl-C doesn't lose completed work.

"Insufficient" questions get the ground-truth answerability as teacher-only
supervision (TeacherHintPolicy) — never exported to the student, see
src/policies/teacher_hint_policy.py. "Sufficient" questions rely on the
verification instruction now baked permanently into claude_prompts.SYSTEM_PROMPT
(no per-question hint needed).

Supports --workers > 1 for concurrent generation: each worker gets its own fresh
DocumentExplorationEnv + ClaudePolicy instance (no shared mutable state), the
same pattern src/eval/harness.py already uses for parallel eval.

Usage:
  python scripts/generate_qasper_teacher_batch.py \
      --benchmark out/research/qasper-train-teacher-v2/benchmark.json \
      --corpus out/research/qasper-train-teacher-v2/corpus \
      --output out/research/qasper-teacher-batch-v2/trajectories.jsonl \
      --workers 8
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import typer
from dotenv import load_dotenv
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.env.corpus import Corpus
from src.env.document_env import DocumentExplorationEnv
from src.eval.harness import run_single
from src.policies.claude_policy import ClaudePolicy
from src.policies.teacher_hint_policy import TeacherHintPolicy

INSUFFICIENT_HINT = (
    "TEACHER-ONLY NOTE (ground truth, for your calibration only — do not reference "
    "this note in your search steps or final answer): this question is confirmed "
    "UNANSWERABLE from this specific paper. You must still demonstrate real due "
    "diligence: run at least 2-3 distinct searches (different queries/sections) "
    "before concluding. Once those searches turn up no explicit textual support, "
    "stop searching and submit an honest abstention within 5 steps total — do not "
    "keep searching indefinitely trying to rule out every possible section, and do "
    "not guess or hedge toward an inferred answer."
)

MAX_CONSECUTIVE_FAILURES = 8

_write_lock = threading.Lock()
_consecutive_failures = 0
_stop = threading.Event()


def _run_one(q: dict, corpus: Corpus, corpus_path: str, max_steps: int) -> dict:
    env = DocumentExplorationEnv(
        corpus=corpus,
        questions=[q],
        max_steps=max_steps,
        use_docker=False,
        corpus_path=corpus_path,
        require_evidence=False,
    )
    try:
        inner = ClaudePolicy()
        policy = TeacherHintPolicy(inner, hint=INSUFFICIENT_HINT) \
            if q["expected_answerability"] == "insufficient" else inner
        result = run_single(env, policy, 0)
        result_dict = json.loads(result.model_dump_json())
        result_dict["question_id"] = q["id"]
        result_dict["expected_answerability"] = q["expected_answerability"]
        return result_dict
    finally:
        env.close()


def main(
    benchmark: Path = typer.Option(...),
    corpus_path: str = typer.Option(..., "--corpus"),
    output: Path = typer.Option(...),
    max_steps: int = typer.Option(10),
    workers: int = typer.Option(8, help="Concurrent episodes. Each gets its own env+policy."),
    limit: int = typer.Option(None, help="Only run the first N questions (for smoke testing)"),
) -> None:
    load_dotenv(override=True)
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set")

    questions = json.loads(benchmark.read_text())["questions"]
    if limit:
        questions = questions[:limit]

    output.parent.mkdir(parents=True, exist_ok=True)
    done_ids = set()
    if output.exists():
        for line in output.read_text().splitlines():
            if line.strip():
                done_ids.add(json.loads(line)["question_id"])
        logger.info(f"Resuming: {len(done_ids)} already done, skipping those")

    corpus = Corpus(corpus_path=corpus_path)
    corpus.load(build_index=False)

    remaining = [q for q in questions if q["id"] not in done_ids]
    logger.info(f"Generating {len(remaining)}/{len(questions)} trajectories "
                f"-> {output} (workers={workers})")

    global _consecutive_failures
    ok, failed = 0, 0

    with ThreadPoolExecutor(max_workers=workers) as pool, output.open("a") as f:
        futures = {
            pool.submit(_run_one, q, corpus, corpus_path, max_steps): q
            for q in remaining
        }
        for i, fut in enumerate(as_completed(futures)):
            if _stop.is_set():
                for pending in futures:
                    pending.cancel()
                break
            q = futures[fut]
            try:
                result_dict = fut.result()
                with _write_lock:
                    f.write(json.dumps(result_dict) + "\n")
                    f.flush()
                ok += 1
                _consecutive_failures = 0
                logger.info(f"[{i+1}/{len(remaining)}] {q['id']} "
                            f"({q['expected_answerability']}) reward={result_dict['reward']:.3f} "
                            f"steps={result_dict['steps']}")
            except Exception as exc:
                failed += 1
                _consecutive_failures += 1
                logger.error(f"[{i+1}/{len(remaining)}] {q['id']} FAILED: {exc}")
                if _consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    logger.error(f"{MAX_CONSECUTIVE_FAILURES} consecutive failures — "
                                 "stopping (likely out of credits or rate-limited).")
                    _stop.set()

    logger.info(f"Done. {ok} generated, {failed} failed. Output: {output}")


if __name__ == "__main__":
    typer.run(main)

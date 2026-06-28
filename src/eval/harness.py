"""Evaluation harness: run policies through the environment and collect results."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from loguru import logger
from pydantic import BaseModel, Field

from src.env.corpus import Corpus
from src.env.document_env import DocumentExplorationEnv, StepRecord


class EvalResult(BaseModel):
    question_id: str
    question: str
    policy_name: str
    reward: float
    answer_score: float
    citation_precision: float
    citation_recall: float
    efficiency_bonus: float
    steps: int
    trajectory: list[StepRecord] = Field(default_factory=list)
    predicted_answer: str = ""
    predicted_citations: list[str] = Field(default_factory=list)
    duration_seconds: float = 0.0


def run_single(
    env: DocumentExplorationEnv,
    policy: object,
    question_idx: int,
) -> EvalResult:
    """Run a single policy on a single question through the environment."""
    # Reset
    if hasattr(policy, "reset"):
        policy.reset()
    obs = env.reset(question_idx=question_idx)
    q = env.questions[question_idx % len(env.questions)]

    start = time.time()
    steps = 0
    reward = 0.0
    predicted_answer = ""
    predicted_citations: list[str] = []
    answer_score = 0.0
    cit_p = 0.0
    cit_r = 0.0
    eff = 0.0

    done = False
    while not done:
        action = policy.act(obs)
        obs, reward, done, info = env.step(action)
        steps += 1

        if done and "reward_breakdown" in info:
            rb = info["reward_breakdown"]
            answer_score = rb.answer_score
            cit_p = rb.citation_precision
            cit_r = rb.citation_recall
            eff = rb.efficiency_bonus
            predicted_answer = info.get("predicted_answer", "")
            predicted_citations = info.get("predicted_citations", [])

    duration = time.time() - start
    trajectory = env.get_trajectory()

    return EvalResult(
        question_id=q["id"],
        question=q["question"],
        policy_name=type(policy).__name__,
        reward=reward,
        answer_score=answer_score,
        citation_precision=cit_p,
        citation_recall=cit_r,
        efficiency_bonus=eff,
        steps=steps,
        trajectory=trajectory,
        predicted_answer=predicted_answer,
        predicted_citations=predicted_citations,
        duration_seconds=duration,
    )


def _run_one_question(
    corpus: Corpus,
    questions: list[dict],
    q_idx: int,
    policy_name: str,
    policy_factory: Callable,
    max_steps: int,
    use_docker: bool | None,
    corpus_path: str,
) -> EvalResult:
    """Run one question with a fresh env + policy instance (safe for parallel use)."""
    q = questions[q_idx]
    env = DocumentExplorationEnv(
        corpus=corpus,
        questions=questions,
        max_steps=max_steps,
        use_docker=use_docker,
        corpus_path=corpus_path,
    )
    policy = policy_factory()
    try:
        result = run_single(env, policy, q_idx)
        result.policy_name = policy_name
        return result
    except Exception as e:
        logger.error(f"  → FAILED {q['id']}: {e}")
        return EvalResult(
            question_id=q["id"],
            question=q["question"],
            policy_name=policy_name,
            reward=0.0,
            answer_score=0.0,
            citation_precision=0.0,
            citation_recall=0.0,
            efficiency_bonus=0.0,
            steps=0,
            duration_seconds=0.0,
        )
    finally:
        try:
            env.close()
        except Exception:
            pass


def run_eval(
    corpus: Corpus,
    questions: list[dict],
    policies: dict[str, object],
    max_steps: int = 10,
    use_docker: bool | None = None,
    corpus_path: str = "data/corpus",
    question_ids: list[str] | None = None,
    workers: int = 1,
) -> list[EvalResult]:
    """Run all policies on all (or selected) questions.

    Set workers > 1 to parallelize across questions — each worker gets its own
    env + policy instance so there is no shared mutable state.
    policy values may be either instances (workers=1) or zero-arg callables
    (workers >= 1). Instances are wrapped in a lambda automatically.
    """
    results: list[EvalResult] = []

    if question_ids:
        q_indices = [i for i, q in enumerate(questions) if q["id"] in question_ids]
    else:
        q_indices = list(range(len(questions)))

    # Normalise: wrap plain instances in a factory so parallel path always has a callable.
    factories: dict[str, Callable] = {}
    for name, p in policies.items():
        factories[name] = p if callable(p) and not hasattr(p, "act") else (lambda _p=p: _p)

    try:
        for policy_name, factory in factories.items():
            logger.info(f"Running policy: {policy_name} (workers={workers})")

            if workers <= 1:
                for q_idx in q_indices:
                    q = questions[q_idx]
                    logger.info(f"  Question {q['id']}: {q['question'][:60]}...")
                    result = _run_one_question(
                        corpus, questions, q_idx, policy_name, factory,
                        max_steps, use_docker, corpus_path,
                    )
                    results.append(result)
                    logger.info(
                        f"  → reward={result.reward:.3f}, steps={result.steps}, "
                        f"time={result.duration_seconds:.1f}s"
                    )
            else:
                futures = {}
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    for q_idx in q_indices:
                        fut = pool.submit(
                            _run_one_question,
                            corpus, questions, q_idx, policy_name, factory,
                            max_steps, use_docker, corpus_path,
                        )
                        futures[fut] = q_idx

                    for fut in as_completed(futures):
                        result = fut.result()
                        results.append(result)
                        logger.info(
                            f"  {result.question_id} → reward={result.reward:.3f}, "
                            f"steps={result.steps}, time={result.duration_seconds:.1f}s"
                        )

    except KeyboardInterrupt:
        logger.warning("Eval interrupted — returning partial results")

    return results

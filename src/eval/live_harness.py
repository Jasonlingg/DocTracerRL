"""Parallel eval harness that streams step-by-step events via callbacks.

Runs multiple policies on the same question concurrently, calling
on_step() after each action so the viewer can stream results in real time.
"""

from __future__ import annotations

import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import Callable

from loguru import logger

from src.env.corpus import Corpus
from src.env.document_env import DocumentExplorationEnv


@dataclass
class StepEvent:
    policy: str
    question_id: str
    question: str
    step: int
    action: str
    observation: str
    reward: float
    done: bool
    total_reward: float
    answer_score: float
    citation_precision: float
    citation_recall: float
    predicted_answer: str
    predicted_citations: list[str]
    elapsed: float

    def to_dict(self) -> dict:
        return asdict(self)


def run_policy_live(
    corpus: Corpus,
    question: dict,
    question_idx: int,
    policy: object,
    policy_name: str,
    max_steps: int,
    corpus_path: str,
    on_step: Callable[[StepEvent], None],
) -> None:
    """Run a single policy on a single question, calling on_step after each action."""
    env = DocumentExplorationEnv(
        corpus=corpus,
        questions=[question],
        max_steps=max_steps,
        use_docker=None,
        corpus_path=corpus_path,
    )
    try:
        policy.reset()
        obs = env.reset(question_idx=0)
        start = time.time()

        for step_num in range(1, max_steps + 1):
            action = policy.act(obs)
            obs, reward, done, info = env.step(action)

            breakdown = info.get("reward_breakdown")
            event = StepEvent(
                policy=policy_name,
                question_id=question["id"],
                question=question["question"],
                step=step_num,
                action=action,
                observation=obs,
                reward=reward,
                done=done,
                total_reward=breakdown.total if breakdown else 0.0,
                answer_score=breakdown.answer_score if breakdown else 0.0,
                citation_precision=breakdown.citation_precision if breakdown else 0.0,
                citation_recall=breakdown.citation_recall if breakdown else 0.0,
                predicted_answer=info.get("predicted_answer", ""),
                predicted_citations=info.get("predicted_citations", []),
                elapsed=time.time() - start,
            )
            on_step(event)
            if done:
                break
    except Exception as e:
        logger.error(f"Error in {policy_name}: {e}\n{traceback.format_exc()}")
        on_step(StepEvent(
            policy=policy_name, question_id=question["id"],
            question=question["question"], step=0, action="",
            observation=f"ERROR: {e}", reward=0.0, done=True,
            total_reward=0.0, answer_score=0.0, citation_precision=0.0,
            citation_recall=0.0, predicted_answer="", predicted_citations=[],
            elapsed=0.0,
        ))
    finally:
        env.close()


def run_parallel(
    corpus: Corpus,
    question: dict,
    policies: dict[str, object],
    max_steps: int,
    corpus_path: str,
    on_step: Callable[[StepEvent], None],
) -> None:
    """Run all policies on a question in parallel, streaming steps via on_step."""
    with ThreadPoolExecutor(max_workers=len(policies)) as pool:
        futures = {
            pool.submit(
                run_policy_live, corpus, question, 0, policy, name,
                max_steps, corpus_path, on_step,
            ): name
            for name, policy in policies.items()
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
            except Exception as e:
                logger.error(f"Policy {name} crashed: {e}")

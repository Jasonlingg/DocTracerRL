"""Verifiable reward functions for the document exploration environment.

Computes answer accuracy (token overlap F1) and citation precision/recall.
This is what a GRPO training loop optimizes. Max reward = 1.0.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


def parse_submission(action: str) -> tuple[str, list[str]] | None:
    """Parse a SUBMIT action into (answer, citations).

    Returns None if the action is not a submission. Used by both the
    Gym-style env (`document_env.py`) and the verifiers wrapper
    (`verifiers_env.py`).
    """
    if not action.strip().upper().startswith("SUBMIT:"):
        return None

    # Extract answer (between SUBMIT: and CITATIONS:)
    match = re.search(
        r"SUBMIT:\s*(.*?)\s*CITATIONS:\s*(\[.*?\])",
        action,
        re.DOTALL | re.IGNORECASE,
    )
    if match:
        answer = match.group(1).strip()
        try:
            citations = json.loads(match.group(2))
        except json.JSONDecodeError:
            citations = []
        return answer, citations

    # Fallback: just SUBMIT with no citations
    match = re.search(r"SUBMIT:\s*(.*)", action, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip(), []

    return None


@dataclass
class RewardBreakdown:
    answer_score: float
    citation_precision: float
    citation_recall: float
    citation_f1: float
    efficiency_bonus: float
    total: float


def _tokenize(text: str) -> list[str]:
    """Lowercase and split into alpha-numeric tokens."""
    return re.findall(r"\w+", text.lower())


def score_answer(predicted: str, gold: str) -> float:
    """Token overlap F1 between predicted and gold answer."""
    pred_tokens = _tokenize(predicted)
    gold_tokens = _tokenize(gold)

    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0

    pred_set = set(pred_tokens)
    gold_set = set(gold_tokens)
    common = pred_set & gold_set

    if not common:
        return 0.0

    precision = len(common) / len(pred_set)
    recall = len(common) / len(gold_set)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def score_citations(
    predicted: list[str], gold: list[str]
) -> dict[str, float]:
    """Compute citation precision, recall, and F1."""
    if not gold:
        return {
            "precision": 1.0 if not predicted else 0.0,
            "recall": 1.0,
            "f1": 1.0 if not predicted else 0.0,
        }
    if not predicted:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    pred_set = set(predicted)
    gold_set = set(gold)
    correct = pred_set & gold_set

    precision = len(correct) / len(pred_set) if pred_set else 0.0
    recall = len(correct) / len(gold_set) if gold_set else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


STEP_BONUS = 0.015  # per exploration step before SUBMIT, max 0.12 for 8 steps
STEP_HIT_REWARD = 0.02  # per step where retrieved content overlaps gold answer (max 3 hits = 0.06)


def compute_reward(
    predicted_answer: str,
    predicted_citations: list[str],
    gold_answer: str,
    gold_citations: list[str],
    steps_taken: int,
    max_steps: int,
) -> RewardBreakdown:
    """Compute the full verifiable reward signal.

    Formula: step_bonus + 0.5 * answer_F1 + 0.25 * cit_P + 0.25 * cit_R
    step_bonus = 0.015 * min(steps_taken - 1, 8) — rewards exploration depth.

    Replacing the flat format bonus (0.1 for any SUBMIT) with a per-step bonus
    creates variance within GRPO groups even when all answers are wrong: rollouts
    that explore 5 steps get 0.06 while 1-step submissions get 0.0. The flat bonus
    collapsed all wrong-answer rollouts to identical rewards, causing all groups to
    be skipped and zero gradient updates.
    """
    ans = score_answer(predicted_answer, gold_answer)
    cit = score_citations(predicted_citations, gold_citations)

    outcome = 0.5 * ans + 0.25 * cit["precision"] + 0.25 * cit["recall"]
    exploration_steps = max(0, steps_taken - 1)
    step_bonus = STEP_BONUS * min(exploration_steps, 8)
    total = outcome + step_bonus

    return RewardBreakdown(
        answer_score=ans,
        citation_precision=cit["precision"],
        citation_recall=cit["recall"],
        citation_f1=cit["f1"],
        efficiency_bonus=step_bonus,
        total=total,
    )

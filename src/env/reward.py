"""Verifiable reward functions for the document exploration environment.

Computes answer accuracy (token overlap F1) and citation precision/recall.
This is what a GRPO training loop optimizes. Max reward = 1.0.
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel

REWARD_VERSION = "outcome-v1"
REWARD_WEIGHTS = {"answer": 0.8, "citation_precision": 0.1, "citation_recall": 0.1}


def parse_submission_details(
    action: str,
) -> tuple[str, list[str], list[dict]] | None:
    """Parse a submission, including optional exact source spans.

    The evidence suffix is deliberately optional so old MuSiQue trajectories and
    adapters keep their original submission protocol. Research evaluation can
    require it without changing the training reward.
    """
    if not action.strip().upper().startswith("SUBMIT:"):
        return None

    body = re.sub(r"^\s*SUBMIT:\s*", "", action, count=1, flags=re.IGNORECASE)
    citations_match = re.search(r"\bCITATIONS:\s*", body, re.IGNORECASE)
    if citations_match is None:
        return body.strip(), [], []

    answer = body[:citations_match.start()].strip()
    tail = body[citations_match.end():].lstrip()
    try:
        citations_value, consumed = json.JSONDecoder().raw_decode(tail)
        citations = (
            citations_value
            if isinstance(citations_value, list)
            and all(isinstance(item, str) for item in citations_value)
            else []
        )
    except json.JSONDecodeError:
        return answer, [], []

    evidence: list[dict] = []
    remainder = tail[consumed:].strip()
    evidence_match = re.match(r"^EVIDENCE:\s*", remainder, re.IGNORECASE)
    if evidence_match is not None:
        try:
            value, _ = json.JSONDecoder().raw_decode(remainder[evidence_match.end():].lstrip())
            if isinstance(value, list) and all(isinstance(item, dict) for item in value):
                evidence = value
        except json.JSONDecodeError:
            pass
    return answer, citations, evidence


def parse_submission(action: str) -> tuple[str, list[str]] | None:
    """Parse a SUBMIT action into (answer, citations).

    Returns None if the action is not a submission. Used by both the
    Gym-style env (`document_env.py`) and the verifiers wrapper
    (`verifiers_env.py`).
    """
    parsed = parse_submission_details(action)
    if parsed is None:
        return None
    answer, citations, _ = parsed
    return answer, citations


class RewardBreakdown(BaseModel):
    answer_score: float
    citation_precision: float
    citation_recall: float
    citation_f1: float
    efficiency_bonus: float
    total: float
    reward_version: str = REWARD_VERSION


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


def compute_reward(
    predicted_answer: str,
    predicted_citations: list[str],
    gold_answer: str,
    gold_citations: list[str],
    steps_taken: int,
    max_steps: int,
) -> RewardBreakdown:
    """Outcome-only baseline: 0.8 * answer F1 + 0.1 * citation P + 0.1 * citation R.

    Step arguments and efficiency_bonus remain for caller/artifact compatibility.
    Extra actions never earn reward; process metrics are recorded separately.
    """
    ans = score_answer(predicted_answer, gold_answer)
    cit = score_citations(predicted_citations, gold_citations)

    outcome = (
        REWARD_WEIGHTS["answer"] * ans
        + REWARD_WEIGHTS["citation_precision"] * cit["precision"]
        + REWARD_WEIGHTS["citation_recall"] * cit["recall"]
    )

    return RewardBreakdown(
        answer_score=ans,
        citation_precision=cit["precision"],
        citation_recall=cit["recall"],
        citation_f1=cit["f1"],
        efficiency_bonus=0.0,
        total=outcome,
    )

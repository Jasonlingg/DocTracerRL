"""Scoring functions for evaluation. Thin wrapper around reward.py."""

from __future__ import annotations

from src.env.reward import (
    RewardBreakdown,
    compute_reward,
    score_answer,
    score_citations,
)

__all__ = [
    "compute_reward",
    "score_answer",
    "score_citations",
    "RewardBreakdown",
]

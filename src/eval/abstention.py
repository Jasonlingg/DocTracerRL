"""Behaviour-level abstention metrics.

The headline outcome reward is misleading for abstention: it scores answer
token-overlap F1 against gold, and gold for an unanswerable question is the terse
string "Unanswerable". A *good* verbose abstention ("Unanswerable — the paper
never states X, though it does discuss Y") therefore scores low against it. A
model whose abstention behaviour genuinely improved can show a *falling* reward.

So abstention is measured as behaviour instead: did the model decline, yes or no.
Two rates matter, and they trade off against each other:

  abstention_recall      — of questions that are genuinely unanswerable, how many
                           did it decline? Higher is better.
  false_abstention_rate  — of questions that ARE answerable, how many did it
                           wrongly decline? Lower is better. This is the
                           conservative-shift check: training on an abstention-
                           heavy mix can produce a model that refuses everything,
                           which reward alone would not distinguish from success.

Detection lives here rather than in the export or scoring scripts so the
definition used to filter training data and the one used to score results cannot
drift apart.
"""

from __future__ import annotations

import re

ABSTAIN_WORDS = re.compile(
    r"unanswerable|does not (state|specify|explicitly|mention|provide|contain)|"
    r"cannot be (determined|established|answered)|no (explicit )?(information|mention)|"
    r"not (explicitly )?(stated|specified|provided)",
    re.I,
)

# An abstention phrase immediately undercut by a confident claim — "the paper does
# not state X. However, based on the full text, X is Y." That is an answer wearing
# an abstention's clothes, and counts as NOT abstaining.
HEDGE_THEN_GUESS = re.compile(
    r"(unanswerable|does not|cannot be|no explicit|not stated|not specified)"
    r"[^.]*\.\s*(however|but|although|that said|based on)",
    re.I,
)


def is_abstention(answer: str) -> bool:
    """True when the answer genuinely declines rather than merely hedging."""
    if not answer or not answer.strip():
        return False
    if not ABSTAIN_WORDS.search(answer):
        return False
    return not HEDGE_THEN_GUESS.search(answer)


def abstention_metrics(
    results: list[dict], questions: list[dict]
) -> dict[str, float | int]:
    """Score a run against a benchmark's expected_answerability labels.

    `results` are eval records carrying question_id and predicted_answer;
    `questions` are benchmark entries carrying id and expected_answerability.
    """
    answerability = {q["id"]: q["expected_answerability"] for q in questions}

    unanswerable_total = unanswerable_abstained = 0
    answerable_total = answerable_abstained = 0

    for record in results:
        label = answerability.get(record["question_id"])
        abstained = is_abstention(record.get("predicted_answer", ""))
        if label == "insufficient":
            unanswerable_total += 1
            unanswerable_abstained += abstained
        elif label == "sufficient":
            answerable_total += 1
            answerable_abstained += abstained

    return {
        "unanswerable_total": unanswerable_total,
        "correctly_abstained": unanswerable_abstained,
        "abstention_recall": (
            unanswerable_abstained / unanswerable_total if unanswerable_total else 0.0
        ),
        "answerable_total": answerable_total,
        "wrongly_abstained": answerable_abstained,
        "false_abstention_rate": (
            answerable_abstained / answerable_total if answerable_total else 0.0
        ),
    }

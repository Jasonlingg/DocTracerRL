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

import math
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


def fisher_exact_two_sided(table: tuple[tuple[int, int], tuple[int, int]]) -> float:
    """Return the two-sided Fisher exact p-value for a 2x2 contingency table.

    This small stdlib implementation keeps behavioral-eval scoring reproducible
    without adding SciPy as a project dependency. It uses the same probability-
    ordering definition as ``scipy.stats.fisher_exact``: sum every table with
    the observed margins whose probability is no greater than the observed one.
    """
    (a, b), (c, d) = table
    if min(a, b, c, d) < 0:
        raise ValueError("Fisher table counts must be non-negative")

    row_1 = a + b
    row_2 = c + d
    col_1 = a + c
    total = row_1 + row_2
    if total == 0:
        return 1.0

    denominator = math.comb(total, row_1)

    def probability(cell: int) -> float:
        return (
            math.comb(col_1, cell)
            * math.comb(total - col_1, row_1 - cell)
            / denominator
        )

    lower = max(0, row_1 - (total - col_1))
    upper = min(row_1, col_1)
    observed = probability(a)
    tolerance = observed * 1e-12
    return min(
        1.0,
        sum(
            probability(cell)
            for cell in range(lower, upper + 1)
            if probability(cell) <= observed + tolerance
        ),
    )


def compare_abstention_runs(
    base_results: list[dict],
    candidate_results: list[dict],
    questions: list[dict],
    *,
    alpha: float = 0.05,
    false_abstention_delta_limit: float = 0.15,
) -> dict:
    """Apply the pre-registered success rule to base and candidate runs."""
    base = abstention_metrics(base_results, questions)
    candidate = abstention_metrics(candidate_results, questions)

    expected_ids = {question["id"] for question in questions}
    for name, results in (("base", base_results), ("candidate", candidate_results)):
        result_ids = [record["question_id"] for record in results]
        duplicates = sorted({item for item in result_ids if result_ids.count(item) > 1})
        missing = sorted(expected_ids - set(result_ids))
        extra = sorted(set(result_ids) - expected_ids)
        if duplicates or missing or extra:
            raise ValueError(
                f"{name} run does not match benchmark IDs: "
                f"duplicates={duplicates}, missing={missing}, extra={extra}"
            )

    p_value = fisher_exact_two_sided(
        (
            (
                int(base["correctly_abstained"]),
                int(base["unanswerable_total"] - base["correctly_abstained"]),
            ),
            (
                int(candidate["correctly_abstained"]),
                int(candidate["unanswerable_total"] - candidate["correctly_abstained"]),
            ),
        )
    )
    recall_improved = candidate["abstention_recall"] > base["abstention_recall"]
    false_abstention_delta = (
        candidate["false_abstention_rate"] - base["false_abstention_rate"]
    )
    guardrail_passed = false_abstention_delta <= false_abstention_delta_limit

    if recall_improved and p_value < alpha and guardrail_passed:
        decision = "success"
    elif recall_improved:
        decision = "partial"
    else:
        decision = "failure"

    return {
        "base": base,
        "candidate": candidate,
        "abstention_recall_delta": (
            candidate["abstention_recall"] - base["abstention_recall"]
        ),
        "false_abstention_rate_delta": false_abstention_delta,
        "fisher_exact_two_sided_p": p_value,
        "alpha": alpha,
        "false_abstention_delta_limit": false_abstention_delta_limit,
        "guardrail_passed": guardrail_passed,
        "decision": decision,
    }

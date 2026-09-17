"""Tests for the abstention scorer.

Written before any SFT results exist, deliberately: defining what counts as a
successful abstention after seeing the numbers would let the definition drift
toward whatever the model happened to do.
"""

import pytest

from src.eval.abstention import (
    abstention_metrics,
    compare_abstention_runs,
    fisher_exact_two_sided,
    is_abstention,
)


def test_plain_refusal_counts_as_abstention():
    assert is_abstention("Unanswerable — the paper does not state this.")
    assert is_abstention("The paper does not explicitly specify the cluster size.")


def test_confident_answer_is_not_abstention():
    assert not is_abstention("The dataset contains 645 hand-labeled articles.")


def test_hedge_then_guess_is_not_abstention():
    """The failure mode we found in real teacher output: an abstention phrase
    used as preamble to a confident answer."""
    assert not is_abstention(
        "The paper does not explicitly state what languages are used. However, "
        "based on the full text, it is evaluated on Yahoo, Amazon, and Yelp."
    )


def test_empty_answer_is_not_abstention():
    """A timeout produces no answer; that is a failure, not a principled refusal."""
    assert not is_abstention("")
    assert not is_abstention("   ")


def _bench():
    return [
        {"id": "u1", "expected_answerability": "insufficient"},
        {"id": "u2", "expected_answerability": "insufficient"},
        {"id": "a1", "expected_answerability": "sufficient"},
        {"id": "a2", "expected_answerability": "sufficient"},
    ]


def test_metrics_separate_correct_from_false_abstention():
    results = [
        {"question_id": "u1", "predicted_answer": "Unanswerable — not stated."},
        {"question_id": "u2", "predicted_answer": "The answer is 42."},
        {"question_id": "a1", "predicted_answer": "The model uses BERT."},
        {"question_id": "a2", "predicted_answer": "The paper does not state this."},
    ]
    m = abstention_metrics(results, _bench())

    assert m["correctly_abstained"] == 1 and m["unanswerable_total"] == 2
    assert m["abstention_recall"] == 0.5
    # a2 was answerable but refused — conservative shift
    assert m["wrongly_abstained"] == 1 and m["answerable_total"] == 2
    assert m["false_abstention_rate"] == 0.5


def test_model_that_refuses_everything_is_caught():
    """Reward alone could look mediocre-but-fine here; false_abstention_rate
    must expose it as broken."""
    results = [
        {"question_id": q["id"], "predicted_answer": "Unanswerable — not stated."}
        for q in _bench()
    ]
    m = abstention_metrics(results, _bench())
    assert m["abstention_recall"] == 1.0
    assert m["false_abstention_rate"] == 1.0


def test_missing_questions_are_ignored_not_counted_as_failures():
    m = abstention_metrics(
        [{"question_id": "unknown", "predicted_answer": "whatever"}], _bench()
    )
    assert m["unanswerable_total"] == 0 and m["answerable_total"] == 0


def test_fisher_exact_matches_preregistered_example():
    # The preregistration states that 4/20 -> 12/20 gives p approximately 0.02.
    assert fisher_exact_two_sided(((4, 16), (12, 8))) == pytest.approx(0.0225, abs=0.0001)


def test_comparison_applies_significance_and_guardrail():
    questions = [
        *[
            {"id": f"u{i}", "expected_answerability": "insufficient"}
            for i in range(20)
        ],
        *[
            {"id": f"a{i}", "expected_answerability": "sufficient"}
            for i in range(20)
        ],
    ]

    def make_results(correct: int, false: int) -> list[dict]:
        return [
            {
                "question_id": question["id"],
                "predicted_answer": (
                    "Unanswerable — not stated."
                    if (
                        question["id"].startswith("u")
                        and int(question["id"][1:]) < correct
                    )
                    or (
                        question["id"].startswith("a")
                        and int(question["id"][1:]) < false
                    )
                    else "The paper provides an answer."
                ),
            }
            for question in questions
        ]

    result = compare_abstention_runs(
        make_results(correct=4, false=1),
        make_results(correct=12, false=3),
        questions,
    )
    assert result["decision"] == "success"
    assert result["guardrail_passed"] is True


def test_comparison_rejects_incomplete_run():
    with pytest.raises(ValueError, match="missing"):
        compare_abstention_runs(
            [{"question_id": "u1", "predicted_answer": "answer"}],
            [],
            [{"id": "u1", "expected_answerability": "insufficient"}],
        )

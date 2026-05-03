"""Tests for reward scoring functions."""

from src.env.reward import compute_reward, score_answer, score_citations


class TestScoreAnswer:
    def test_exact_match(self) -> None:
        assert score_answer("hello world", "hello world") == 1.0

    def test_partial_overlap(self) -> None:
        score = score_answer("the revenue was 42M", "revenue was 42M dollars")
        assert 0.5 < score < 1.0

    def test_no_overlap(self) -> None:
        assert score_answer("foo bar", "baz qux") == 0.0

    def test_empty_gold(self) -> None:
        assert score_answer("", "") == 1.0
        assert score_answer("something", "") == 0.0

    def test_empty_predicted(self) -> None:
        assert score_answer("", "expected answer") == 0.0

    def test_case_insensitive(self) -> None:
        assert score_answer("HELLO World", "hello WORLD") == 1.0


class TestScoreCitations:
    def test_perfect_match(self) -> None:
        result = score_citations(["doc1", "doc2"], ["doc1", "doc2"])
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0

    def test_partial_match(self) -> None:
        result = score_citations(["doc1", "doc3"], ["doc1", "doc2"])
        assert result["precision"] == 0.5
        assert result["recall"] == 0.5

    def test_superset_predicted(self) -> None:
        result = score_citations(["doc1", "doc2", "doc3"], ["doc1", "doc2"])
        assert abs(result["precision"] - 2 / 3) < 0.01
        assert result["recall"] == 1.0

    def test_empty_gold(self) -> None:
        result = score_citations([], [])
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0

    def test_empty_predicted(self) -> None:
        result = score_citations([], ["doc1"])
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0


class TestComputeReward:
    def test_perfect_submission(self) -> None:
        result = compute_reward(
            predicted_answer="answer here",
            predicted_citations=["doc1"],
            gold_answer="answer here",
            gold_citations=["doc1"],
            steps_taken=1,
            max_steps=10,
        )
        assert result.answer_score == 1.0
        assert result.citation_precision == 1.0
        assert result.citation_recall == 1.0
        assert result.total == 1.0  # max reward with no efficiency bonus

    def test_wrong_answer(self) -> None:
        result = compute_reward(
            predicted_answer="wrong",
            predicted_citations=["doc1"],
            gold_answer="correct answer",
            gold_citations=["doc1"],
            steps_taken=5,
            max_steps=10,
        )
        assert result.answer_score < 0.5
        assert result.citation_precision == 1.0

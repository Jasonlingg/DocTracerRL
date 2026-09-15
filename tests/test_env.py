"""Tests for the Gym-compatible document exploration environment."""

import subprocess

import pytest

from src.env.corpus import Corpus
from src.env.document_env import DocumentExplorationEnv, parse_submission


@pytest.fixture(scope="module")
def corpus() -> Corpus:
    subprocess.run(["python", "scripts/setup_corpus.py"], check=True, capture_output=True)
    c = Corpus(corpus_path="data/corpus")
    c.load(build_index=False)
    return c


@pytest.fixture(scope="module")
def questions() -> list[dict]:
    return [{
        "id": "test_q",
        "question": "What was Apex Corp's net income in 2024?",
        "answer": "Apex Corp's net income was $28.2M in 2024.",
        "expected_citations": ["apex_corp_2024_financial"],
    }]


@pytest.fixture
def env(corpus: Corpus, questions: list[dict]) -> DocumentExplorationEnv:
    e = DocumentExplorationEnv(
        corpus=corpus, questions=questions, max_steps=5, use_docker=False,
    )
    yield e
    e.close()


def test_reset_returns_observation(env: DocumentExplorationEnv) -> None:
    obs = env.reset(question_idx=0)
    assert "Question:" in obs
    assert "Apex Corp" in obs


def test_step_with_code(env: DocumentExplorationEnv) -> None:
    env.reset(question_idx=0)
    obs, reward, done, info = env.step('print("hello")')
    assert "hello" in obs
    assert reward == 0.0
    assert done is False


def test_step_with_submit(env: DocumentExplorationEnv) -> None:
    env.reset(question_idx=0)
    obs, reward, done, info = env.step(
        'SUBMIT: Net income was $28.2M CITATIONS: ["apex_corp_2024_financial"] '
        'EVIDENCE: [{"doc_id":"apex_corp_2024_financial","start":0,"end":20}]'
    )
    assert done is True
    assert reward > 0.0
    assert "reward_breakdown" in info
    assert info["predicted_evidence"] == [{
        "doc_id": "apex_corp_2024_financial", "start": 0, "end": 20,
    }]


def test_max_steps_triggers_done(env: DocumentExplorationEnv) -> None:
    env.reset(question_idx=0)
    for _ in range(5):
        obs, reward, done, info = env.step('print("step")')
        if done:
            break
    assert done is True


def test_trajectory_recorded(env: DocumentExplorationEnv) -> None:
    env.reset(question_idx=0)
    env.step('print("a")')
    env.step('SUBMIT: answer CITATIONS: []')
    traj = env.get_trajectory()
    assert len(traj) == 2


class TestParseSubmission:
    def test_valid_submission(self) -> None:
        result = parse_submission('SUBMIT: answer here CITATIONS: ["doc1", "doc2"]')
        assert result is not None
        answer, citations = result
        assert answer == "answer here"
        assert citations == ["doc1", "doc2"]

    def test_not_a_submission(self) -> None:
        assert parse_submission('print("hello")') is None

    def test_submit_no_citations(self) -> None:
        result = parse_submission("SUBMIT: just the answer")
        assert result is not None
        assert result[0] == "just the answer"
        assert result[1] == []

    def test_case_insensitive(self) -> None:
        result = parse_submission('submit: answer CITATIONS: ["doc1"]')
        assert result is not None

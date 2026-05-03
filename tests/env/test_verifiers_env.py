"""Tests for the native verifiers env.

Scope is minimal because the full verifiers training stack (vllm, bitsandbytes,
trl) only runs on CUDA. These tests verify the env's logic in isolation:

1. Imports + class hierarchy
2. Dataset construction (column names, info payload)
3. setup_state spawns a REPL session
4. env_response detects SUBMIT and short-circuits
5. env_response runs non-SUBMIT code in the REPL
6. reward_func scores SUBMIT correctly and returns 0 when absent

Tests skip if `verifiers` or `datasets` aren't importable.
"""

from __future__ import annotations

import asyncio
import subprocess

import pytest

verifiers = pytest.importorskip("verifiers")
pytest.importorskip("datasets")

from src.env.corpus import Corpus  # noqa: E402
from src.env.verifiers_env import (  # noqa: E402
    SYSTEM_PROMPT,
    DocumentExplorationVerifiersEnv,
    reward_func,
)


@pytest.fixture(scope="module")
def corpus() -> Corpus:
    subprocess.run(
        ["python", "scripts/setup_corpus.py"], check=True, capture_output=True
    )
    c = Corpus(corpus_path="data/corpus")
    c.load()
    return c


@pytest.fixture(scope="module")
def questions() -> list[dict]:
    return [
        {
            "id": "test_q",
            "question": "What was Apex Corp's net income in 2024?",
            "answer": "Apex Corp's net income was $28.2M in 2024.",
            "expected_citations": ["apex_corp_2024_financial"],
            "answer_aliases": [],
        }
    ]


@pytest.fixture
def env(
    corpus: Corpus, questions: list[dict]
) -> DocumentExplorationVerifiersEnv:
    return DocumentExplorationVerifiersEnv(
        corpus=corpus,
        questions=questions,
        max_steps=5,
        use_docker=False,
    )


def _make_state(answer: str, expected_citations: list[str]) -> "verifiers.State":
    state = verifiers.State(
        input={
            "info": {
                "q_id": "test_q",
                "expected_citations": expected_citations,
                "answer_aliases": [],
            },
            "answer": answer,
            "task": "default",
            "example_id": 0,
            "prompt": [],
        }
    )
    state["trajectory"] = []
    return state


def test_imports_resolve() -> None:
    assert callable(reward_func)
    assert isinstance(SYSTEM_PROMPT, str) and "SUBMIT:" in SYSTEM_PROMPT
    assert issubclass(
        DocumentExplorationVerifiersEnv, verifiers.MultiTurnEnv
    )


def test_dataset_shape(env: DocumentExplorationVerifiersEnv) -> None:
    """Dataset is built from questions; verifiers prepends system_prompt."""
    assert env.max_turns == 5
    cols = env.dataset.column_names
    for required in ("question", "answer", "info", "prompt", "example_id"):
        assert required in cols, f"missing column {required!r}"
    row = env.dataset[0]
    assert row["info"]["q_id"] == "test_q"
    assert row["info"]["expected_citations"] == ["apex_corp_2024_financial"]
    prompt = row["prompt"]
    assert prompt[0]["role"] == "system" and "SUBMIT:" in prompt[0]["content"]
    assert prompt[1]["role"] == "user"


def test_setup_state_spawns_repl(env: DocumentExplorationVerifiersEnv) -> None:
    """setup_state must allocate a fresh REPL session per rollout."""
    state = _make_state("x", [])
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(env.setup_state(state))
        assert state.get("_repl") is not None
        assert state.get("_submitted") is False
    finally:
        loop.run_until_complete(env.cleanup_repl(state))
        loop.close()


def test_env_response_handles_submit(
    env: DocumentExplorationVerifiersEnv,
) -> None:
    """A SUBMIT message marks the rollout submitted; @vf.stop fires."""
    state = _make_state(
        "Apex Corp's net income was $28.2M in 2024.",
        ["apex_corp_2024_financial"],
    )
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(env.setup_state(state))
        assistant_msg = verifiers.AssistantMessage(
            content='SUBMIT: $28.2M CITATIONS: ["apex_corp_2024_financial"]'
        )
        response = loop.run_until_complete(
            env.env_response([assistant_msg], state)
        )
        assert state["_submitted"] is True
        assert isinstance(response, list) and len(response) == 1
        assert loop.run_until_complete(env.submitted(state)) is True
    finally:
        loop.run_until_complete(env.cleanup_repl(state))
        loop.close()


def test_env_response_runs_repl_code(
    env: DocumentExplorationVerifiersEnv,
) -> None:
    """A non-SUBMIT message is executed in the REPL; obs is returned."""
    state = _make_state("x", [])
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(env.setup_state(state))
        msg = verifiers.AssistantMessage(content='print(2 + 2)')
        response = loop.run_until_complete(env.env_response([msg], state))
        assert state["_submitted"] is False
        text = response[0].content if hasattr(response[0], "content") else response[0]["content"]
        assert "4" in text
    finally:
        loop.run_until_complete(env.cleanup_repl(state))
        loop.close()


def test_reward_func_scores_submit() -> None:
    completion = [
        verifiers.AssistantMessage(
            content='SUBMIT: Net income was $28.2M CITATIONS: ["apex_corp_2024_financial"]'
        )
    ]
    info = {"expected_citations": ["apex_corp_2024_financial"]}
    reward = reward_func(
        completion=completion,
        answer="Apex Corp's net income was $28.2M in 2024.",
        info=info,
    )
    assert reward > 0.5


def test_reward_func_no_submit_returns_zero() -> None:
    completion = [
        verifiers.AssistantMessage(content='print("still exploring")')
    ]
    reward = reward_func(
        completion=completion,
        answer="anything",
        info={"expected_citations": []},
    )
    assert reward == 0.0

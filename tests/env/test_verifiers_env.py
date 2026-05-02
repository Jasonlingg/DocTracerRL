"""Tests for the verifiers-compatible wrapper around DocumentExplorationEnv.

Scope is minimal because the verifiers training stack (vllm, bitsandbytes,
verifiers-rl) only runs on CUDA. These tests verify:

1. The wrapper module imports cleanly when `verifiers` and `datasets` are
   available.
2. The wrapper class instantiates and produces a well-formed dataset.
3. `setup_state` and `env_response` advance state correctly when driven
   directly (no model in the loop).
4. `reward_func` returns the expected reward for a synthetic SUBMIT trajectory.

The tests skip if `verifiers` or `datasets` aren't importable.
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


def test_imports_resolve() -> None:
    """Smoke check: the public API symbols are importable."""
    assert callable(reward_func)
    assert isinstance(SYSTEM_PROMPT, str) and "SUBMIT:" in SYSTEM_PROMPT
    assert issubclass(
        DocumentExplorationVerifiersEnv, verifiers.MultiTurnEnv
    )


def test_wrapper_instantiates(env: DocumentExplorationVerifiersEnv) -> None:
    """Constructing the wrapper builds the dataset and registers stop/cleanup."""
    assert env.max_turns == 5
    assert env.dataset is not None
    cols = env.dataset.column_names
    # Verifiers auto-injects example_id, prompt, task; we contribute the rest.
    for required in ("question", "answer", "info", "prompt", "example_id"):
        assert required in cols, f"missing column {required!r}"
    row = env.dataset[0]
    assert row["info"]["q_idx"] == 0
    assert row["info"]["q_id"] == "test_q"
    assert row["info"]["expected_citations"] == ["apex_corp_2024_financial"]
    # prompt is system + user, with our SYSTEM_PROMPT prepended.
    prompt = row["prompt"]
    assert prompt[0]["role"] == "system" and "SUBMIT:" in prompt[0]["content"]
    assert prompt[1]["role"] == "user"


def test_setup_state_creates_inner_env(
    env: DocumentExplorationVerifiersEnv,
) -> None:
    """setup_state must build a fresh inner env keyed off info.q_idx."""
    state = verifiers.State(
        input={
            "info": {"q_idx": 0, "q_id": "test_q", "expected_citations": []},
            "answer": "x",
            "task": "default",
            "example_id": 0,
            "prompt": [],
        }
    )
    state["trajectory"] = []
    asyncio.get_event_loop().run_until_complete(env.setup_state(state))
    try:
        assert state.get("_inner_env") is not None
        assert state.get("_inner_done") is False
        assert state.get("_submitted") is False
    finally:
        # Release the REPL session spawned by reset().
        asyncio.get_event_loop().run_until_complete(
            env.cleanup_inner.__wrapped__(env, state)  # type: ignore[attr-defined]
            if hasattr(env.cleanup_inner, "__wrapped__")
            else env.cleanup_inner(state)
        )


def test_env_response_handles_submit(
    env: DocumentExplorationVerifiersEnv,
) -> None:
    """A SUBMIT message should mark the rollout submitted+done in state."""
    state = verifiers.State(
        input={
            "info": {
                "q_idx": 0,
                "q_id": "test_q",
                "expected_citations": ["apex_corp_2024_financial"],
            },
            "answer": "Apex Corp's net income was $28.2M in 2024.",
            "task": "default",
            "example_id": 0,
            "prompt": [],
        }
    )
    state["trajectory"] = []

    loop = asyncio.get_event_loop()
    loop.run_until_complete(env.setup_state(state))

    assistant_msg = verifiers.AssistantMessage(
        content='SUBMIT: $28.2M CITATIONS: ["apex_corp_2024_financial"]'
    )
    response = loop.run_until_complete(
        env.env_response([assistant_msg], state)
    )

    assert state["_submitted"] is True
    assert state["_inner_done"] is True
    assert isinstance(response, list) and len(response) == 1

    # Verify the @vf.stop predicate fires.
    assert loop.run_until_complete(env.submitted(state)) is True

    loop.run_until_complete(env.cleanup_inner(state))


def test_reward_func_scores_submit() -> None:
    """reward_func parses SUBMIT from completion and returns compute_reward.total."""
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
        state=None,
    )
    # answer F1 > 0, citations precise+complete, plus efficiency bonus; safely > 0.5.
    assert reward > 0.5


def test_reward_func_no_submit_returns_zero() -> None:
    """Trajectories without a SUBMIT line score 0."""
    completion = [
        verifiers.AssistantMessage(content='print("still exploring")')
    ]
    reward = reward_func(
        completion=completion,
        answer="anything",
        info={"expected_citations": []},
        state=None,
    )
    assert reward == 0.0

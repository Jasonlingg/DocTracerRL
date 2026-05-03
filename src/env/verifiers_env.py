"""Native verifiers env for the document exploration task.

Subclasses `vf.MultiTurnEnv` directly — no wrapping of the gym-style
`DocumentExplorationEnv`. This env owns the REPL lifecycle, runs blocking
REPL calls off the asyncio event loop, and exposes a clean reward function
that is a pure function of (completion, answer, info).

Reuses primitives from elsewhere in the codebase:
- `src.env.repl.PersistentREPL`     — REPL session management (auto-selects Docker / local)
- `src.env.tools.TOOL_PREAMBLE`     — injected into every REPL session
- `src.env.reward.parse_submission` — parses 'SUBMIT: <ans> CITATIONS: [...]'
- `src.env.reward.compute_reward`   — terminal scoring (F1 + cit P/R + efficiency)

Verifiers public API used:
- `vf.MultiTurnEnv.env_response(messages, state, **kwargs) -> Messages` (async)
- `vf.MultiTurnEnv.setup_state(state, **kwargs) -> State` (async, per-rollout init)
- `@vf.stop` coroutines for stop conditions
- `@vf.cleanup` coroutine for end-of-rollout teardown
"""

from __future__ import annotations

import asyncio
from typing import Any

import verifiers as vf

from src.env.corpus import Corpus
from src.env.repl import PersistentREPL
from src.env.reward import compute_reward, parse_submission

SYSTEM_PROMPT = """You are an agent that explores a document corpus by writing Python code.

Available tools (already imported into a persistent Python REPL):
  search(query, top_k=5)              → doc-level keyword search
  search(query, method="chunk")       → chunk-level search (finds buried facts)
  read(doc_id)                        → full document text
  extract(doc_id, regex)              → regex matches from a doc
  search_within(doc_id, query)        → search inside a specific doc
  verify(doc_id, claim)               → check if a claim is supported by a doc
  list_docs()                         → list all docs

Each assistant turn must be EITHER a single block of executable Python code OR
a SUBMIT line — never both, never English prose, never markdown fences.

Variables persist across turns. Use print() to see output.

When you have the answer, respond with ONLY:
  SUBMIT: <answer> CITATIONS: ["doc_id_1", "doc_id_2"]
"""


def _last_message_content(message: Any) -> str:
    """Extract a plain-text content string from a vf.Message or raw dict."""
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    pieces: list[str] = []
    for part in content:
        text = getattr(part, "text", None)
        if text is None and isinstance(part, dict):
            text = part.get("text")
        if isinstance(text, str):
            pieces.append(text)
    return "\n".join(pieces)


class DocumentExplorationVerifiersEnv(vf.MultiTurnEnv):
    """Native verifiers env: each rollout owns its own REPL subprocess."""

    def __init__(
        self,
        corpus: Corpus,
        questions: list[dict],
        max_steps: int = 10,
        use_docker: bool | None = False,
        corpus_path: str = "data/corpus",
        repl_timeout: int = 30,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> None:
        self._corpus = corpus
        self._use_docker = use_docker
        self._corpus_path = corpus_path
        self._repl_timeout = repl_timeout
        self._max_steps = max_steps

        dataset = self._build_dataset(questions)

        super().__init__(
            dataset=dataset,
            system_prompt=system_prompt or SYSTEM_PROMPT,
            max_turns=max_steps,
            **kwargs,
        )

    @staticmethod
    def _build_dataset(questions: list[dict]) -> Any:
        """Build a HuggingFace Dataset from the questions list.

        `datasets` is imported lazily — it lives under the `[training]` extra.
        """
        from datasets import Dataset

        rows = [
            {
                "question": q["question"],
                "answer": q["answer"],
                "info": {
                    "q_id": q["id"],
                    "expected_citations": q.get("expected_citations", []),
                    "answer_aliases": q.get("answer_aliases", []),
                },
            }
            for q in questions
        ]
        return Dataset.from_list(rows)

    async def setup_state(self, state: vf.State, **kwargs: Any) -> vf.State:
        """Per-rollout init: spawn a fresh REPL subprocess off the event loop."""
        state = await super().setup_state(state, **kwargs)
        repl = PersistentREPL(
            use_docker=self._use_docker, corpus_path=self._corpus_path,
        )
        await asyncio.to_thread(repl.start_session)
        state["_repl"] = repl
        state["_submitted"] = False
        return state

    async def env_response(
        self, messages: vf.Messages, state: vf.State, **kwargs: Any
    ) -> vf.Messages:
        """Execute the assistant's action in the REPL (or detect SUBMIT)."""
        last_msg = messages[-1] if messages else None
        action = _last_message_content(last_msg).strip()

        # SUBMIT short-circuits — no REPL execution, no further turns.
        if parse_submission(action) is not None:
            state["_submitted"] = True
            return [vf.UserMessage(content="[Answer submitted]")]

        # Otherwise run the action in the REPL. Blocking subprocess call goes
        # OFF the asyncio loop so concurrent rollouts in a GRPO group aren't
        # serialized.
        repl: PersistentREPL = state["_repl"]
        obs = await asyncio.to_thread(repl.execute, action, self._repl_timeout)
        return [vf.UserMessage(content=obs)]

    @vf.stop
    async def submitted(self, state: vf.State) -> bool:
        return bool(state.get("_submitted", False))

    @vf.cleanup
    async def cleanup_repl(self, state: vf.State) -> None:
        repl = state.get("_repl")
        if repl is not None:
            try:
                await asyncio.to_thread(repl.kill_session)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Reward function — pure function of (completion, answer, info).
# Verifiers passes these by signature inspection; no state coupling.
# ---------------------------------------------------------------------------


def reward_func(
    completion: vf.Messages | str,
    answer: str,
    info: dict | None = None,
    **kwargs: Any,
) -> float:
    """Score a finished rollout against the gold answer + citations.

    Pulls the last assistant message from `completion`, parses its SUBMIT line,
    and delegates to `compute_reward`. Returns 0.0 if no SUBMIT was emitted
    (malformed or premature termination).
    """
    info = info or {}

    # Find the SUBMIT line + count assistant turns (= steps taken).
    submit_text = ""
    steps_taken = 0
    if isinstance(completion, str):
        submit_text = completion
        steps_taken = 1
    elif isinstance(completion, list):
        for msg in completion:
            role = getattr(msg, "role", None) or (
                msg.get("role") if isinstance(msg, dict) else None
            )
            if role == "assistant":
                steps_taken += 1
                submit_text = _last_message_content(msg)  # keep last assistant msg

    parsed = parse_submission(submit_text)
    if parsed is None:
        return 0.0
    pred_answer, pred_citations = parsed

    rb = compute_reward(
        predicted_answer=pred_answer,
        predicted_citations=pred_citations,
        gold_answer=answer,
        gold_citations=info.get("expected_citations", []),
        steps_taken=max(steps_taken, 1),
        max_steps=10,
    )
    return rb.total

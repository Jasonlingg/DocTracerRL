"""verifiers-compatible wrapper around DocumentExplorationEnv for GRPO training.

Wraps the existing Gym-style `DocumentExplorationEnv` so it can be driven by
Will Brown's `verifiers` library (https://github.com/willccbb/verifiers, v0.1.12).
Each "assistant turn" is one chunk of Python code (or a `SUBMIT:` line). The
env's response is the REPL stdout. The rollout terminates when the model emits
`SUBMIT:` or `max_turns` is reached. Reward is computed at completion via the
module-level `reward_func`.

The verifiers public API used here:
- `vf.MultiTurnEnv.env_response(messages, state, **kwargs) -> Messages` (async)
- `@vf.stop` decorated coroutines on the subclass for stop conditions
- `setup_state(state) -> state` for per-rollout initialization
- Reward functions are plain callables registered on a `vf.Rubric`; verifiers
  passes them `prompt`, `completion`, `answer`, `state`, `task`, `info` by
  inspecting their signature.
"""

from __future__ import annotations

from typing import Any

import verifiers as vf

from src.env.corpus import Corpus
from src.env.document_env import DocumentExplorationEnv
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

Variables persist across turns. Use print() to see output. Track discoveries
in a dict (e.g. known_facts = {}).

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
    # Content may be str or a list of typed content parts; we only need str here.
    if isinstance(content, str):
        return content
    # Best-effort flatten of structured content (text parts only).
    pieces: list[str] = []
    for part in content:
        text = getattr(part, "text", None)
        if text is None and isinstance(part, dict):
            text = part.get("text")
        if isinstance(text, str):
            pieces.append(text)
    return "\n".join(pieces)


class DocumentExplorationVerifiersEnv(vf.MultiTurnEnv):
    """verifiers-compatible MultiTurnEnv for the document exploration task.

    Each rollout owns its own `DocumentExplorationEnv` instance (created in
    `setup_state`) so that the inner env's mutable state (REPL session,
    step counter, episode info) is isolated from concurrent rollouts.
    """

    def __init__(
        self,
        corpus: Corpus,
        questions: list[dict],
        max_steps: int = 10,
        use_docker: bool | None = False,
        corpus_path: str = "data/corpus",
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> None:
        # Stash config for per-rollout inner-env construction.
        self._corpus = corpus
        self._questions = questions
        self._max_steps = max_steps
        self._use_docker = use_docker
        self._corpus_path = corpus_path

        # Map question id → list-index so we can recover the row from a dataset
        # row's `info` dict regardless of dataset shuffling.
        self._qid_to_idx: dict[str, int] = {
            q["id"]: i for i, q in enumerate(questions)
        }

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

        Each row needs at minimum: `question` (drives `prompt` formatting),
        `answer` (gold for scoring), and `info` (carrying expected_citations,
        answer_aliases, and the question id for inner-env lookup).

        We import `datasets` lazily because it's an optional dependency declared
        under `[training]` — callers building only the inner env shouldn't need
        it.
        """
        from datasets import Dataset

        rows: list[dict] = []
        for i, q in enumerate(questions):
            rows.append(
                {
                    "question": q["question"],
                    "answer": q["answer"],
                    "info": {
                        "q_id": q["id"],
                        "q_idx": i,
                        "expected_citations": q.get("expected_citations", []),
                        "answer_aliases": q.get("answer_aliases", []),
                    },
                }
            )
        return Dataset.from_list(rows)

    def _resolve_q_idx(self, state: vf.State) -> int:
        """Recover the inner env's question index from the dataset row.

        Prefer `info["q_idx"]` (set at dataset construction). Fall back to
        looking up `info["q_id"]` in our id→idx map. Last resort: `example_id`
        (auto-added by verifiers if absent — though we set it via info).
        """
        info = state.get("info") or {}
        if isinstance(info, dict):
            if "q_idx" in info:
                return int(info["q_idx"])
            qid = info.get("q_id")
            if qid is not None and qid in self._qid_to_idx:
                return self._qid_to_idx[qid]
        example_id = state.get("example_id")
        if isinstance(example_id, int):
            return example_id
        raise RuntimeError(
            "Cannot resolve question index from state.info or state.example_id"
        )

    async def setup_state(self, state: vf.State, **kwargs: Any) -> vf.State:
        """Per-rollout init: build a fresh inner env and reset it."""
        state = await super().setup_state(state, **kwargs)
        inner = DocumentExplorationEnv(
            corpus=self._corpus,
            questions=self._questions,
            max_steps=self._max_steps,
            use_docker=self._use_docker,
            corpus_path=self._corpus_path,
        )
        q_idx = self._resolve_q_idx(state)
        # `reset` returns the system+question observation; verifiers already
        # serves the same content via `system_prompt` + `prompt`, so we discard.
        inner.reset(question_idx=q_idx)
        state["_inner_env"] = inner
        state["_inner_done"] = False
        state["_submitted"] = False
        return state

    async def env_response(
        self, messages: vf.Messages, state: vf.State, **kwargs: Any
    ) -> vf.Messages:
        """Run the latest assistant action in the inner env and return obs."""
        inner: DocumentExplorationEnv = state["_inner_env"]

        # Last message is the assistant's just-emitted code or SUBMIT line.
        last_msg = messages[-1] if messages else None
        action = _last_message_content(last_msg).strip()

        # Inner env computes terminal reward on SUBMIT; we ignore it here
        # because the verifiers Rubric scores the rollout end-to-end via
        # `reward_func` below. We DO need `done` to fire @vf.stop.
        obs, _reward, done, info = inner.step(action)

        state["_inner_done"] = bool(done)
        state["_last_step_info"] = info
        if parse_submission(action) is not None:
            state["_submitted"] = True

        # If SUBMIT, inner returns empty obs — short-circuit termination via
        # @vf.stop predicate; we still need to return *something* the verifiers
        # framework can append. An empty user message is fine.
        return [vf.UserMessage(content=obs or "")]

    @vf.stop
    async def submitted(self, state: vf.State) -> bool:
        """Stop as soon as the model emitted a SUBMIT line."""
        return bool(state.get("_submitted", False))

    @vf.stop
    async def inner_env_done(self, state: vf.State) -> bool:
        """Stop if the inner env's max_steps was reached."""
        return bool(state.get("_inner_done", False))

    @vf.cleanup
    async def cleanup_inner(self, state: vf.State) -> None:
        """Release the inner env's REPL session at end of rollout."""
        inner = state.get("_inner_env")
        if inner is not None:
            try:
                inner.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Reward function (module-level so verifiers can introspect its signature).
# ---------------------------------------------------------------------------


def reward_func(
    completion: vf.Messages | str,
    answer: str,
    info: dict | None = None,
    state: vf.State | None = None,
    **kwargs: Any,
) -> float:
    """Compute the verifiable reward for a finished rollout.

    Pulls the last assistant message from `completion`, parses the SUBMIT line,
    and delegates to `compute_reward`. Returns 0.0 if no SUBMIT was emitted.

    `info` carries `expected_citations` (gold cites) and `answer_aliases`
    (we don't use aliases here yet — the answer F1 is token-overlap on the
    canonical gold answer; aliases would require a small change to
    `score_answer`).
    """
    info = info or {}
    text: str
    if isinstance(completion, str):
        text = completion
    elif isinstance(completion, list) and completion:
        # Find the last assistant message; SUBMIT should be in it.
        text = ""
        for msg in reversed(completion):
            role = getattr(msg, "role", None) or (
                msg.get("role") if isinstance(msg, dict) else None
            )
            if role == "assistant":
                text = _last_message_content(msg)
                break
        if not text:
            text = _last_message_content(completion[-1])
    else:
        return 0.0

    parsed = parse_submission(text)
    if parsed is None:
        return 0.0
    pred_answer, pred_citations = parsed

    # `state["_last_step_info"]["step"]` is the inner env's true step count if
    # available; otherwise estimate from completion length.
    steps_taken = 1
    max_steps = 10
    if state is not None:
        last_info = state.get("_last_step_info") or {}
        if isinstance(last_info, dict) and "step" in last_info:
            steps_taken = int(last_info["step"])
        inner = state.get("_inner_env")
        if inner is not None:
            max_steps = getattr(inner, "max_steps", max_steps)

    rb = compute_reward(
        predicted_answer=pred_answer,
        predicted_citations=pred_citations,
        gold_answer=answer,
        gold_citations=info.get("expected_citations", []),
        steps_taken=steps_taken,
        max_steps=max_steps,
    )
    return rb.total

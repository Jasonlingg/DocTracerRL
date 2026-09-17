"""Wraps a policy to give it privileged, teacher-only supervision.

Used only for generating training trajectories: the wrapped policy (the
"teacher", e.g. Claude Sonnet) sees a ground-truth hint appended to the first
observation of an episode, so it reliably makes the right call on hard
questions (e.g. whether a paper actually answers the question at all).

The hint is never returned to the caller and never touches anything the
harness logs or exports (EvalResult.question, StepRecord.observation) — those
are built from the environment's own text, which this wrapper never sees or
modifies. A model trained on the resulting trajectory only ever sees the
plain question and the teacher's real search/read/submit steps, so it still
has to learn to infer answerability from evidence, not from a label it will
never have at inference time.
"""

from __future__ import annotations


class TeacherHintPolicy:
    def __init__(self, inner: object, hint: str) -> None:
        self._inner = inner
        self._hint = hint
        self._used_hint = False

    def act(self, observation: str) -> str:
        if not self._used_hint:
            observation = f"{self._hint}\n\n{observation}"
            self._used_hint = True
        return self._inner.act(observation)

    def reset(self) -> None:
        self._used_hint = False
        if hasattr(self._inner, "reset"):
            self._inner.reset()

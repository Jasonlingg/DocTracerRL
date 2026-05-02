"""Gym-compatible RL environment for document exploration.

Observation space: string (question + execution output from last step)
Action space: string (Python code to execute, or "SUBMIT: <answer> CITATIONS: [...]")
Reward: 0 during exploration, verifiable score on submission
"""

from __future__ import annotations

from dataclasses import dataclass, field

from loguru import logger

from src.env.corpus import Corpus
from src.env.repl import PersistentREPL
from src.env.reward import (
    RewardBreakdown,
    compute_reward,
    parse_submission,
)

# Re-exported so `from src.env.document_env import parse_submission` keeps working.
__all__ = [
    "DocumentExplorationEnv",
    "EpisodeInfo",
    "StepRecord",
    "SYSTEM_PREAMBLE",
    "parse_submission",
]


@dataclass
class StepRecord:
    step: int
    action: str
    observation: str
    reward: float
    done: bool


@dataclass
class EpisodeInfo:
    question_id: str
    question: str
    gold_answer: str
    gold_citations: list[str]
    trajectory: list[StepRecord] = field(default_factory=list)
    final_reward: RewardBreakdown | None = None


SYSTEM_PREAMBLE = """You have a Python REPL with these functions loaded:
  search(query, top_k=5)              → [{"doc_id", "title", "chunk", "score"}]
  search(query, method="chunk")       → chunk-level search (finds buried facts)
  read(doc_id)                        → full document text
  extract(doc_id, pattern)            → regex matches
  search_within(doc_id, query)        → search inside a specific document
  verify(doc_id, claim)               → check if a claim is supported by a doc
  list_docs()                         → [{"doc_id", "title", "chars"}]

TIP: Track discoveries as you go: known_facts = {}
     known_facts["company_x"] = "Apex Corp"
     Then use known_facts values in subsequent searches.

Respond with ONLY Python code. Use print() to see output.
When done, respond: SUBMIT: <answer> CITATIONS: ["id1", "id2"]
"""


class DocumentExplorationEnv:
    """Gym-compatible RL environment for document exploration via code execution."""

    def __init__(
        self,
        corpus: Corpus,
        questions: list[dict],
        max_steps: int = 10,
        use_docker: bool | None = None,
        corpus_path: str = "data/corpus",
    ) -> None:
        self.corpus = corpus
        self.questions = questions
        self.max_steps = max_steps
        self._use_docker = use_docker
        self._corpus_path = corpus_path
        self.repl = PersistentREPL(
            use_docker=use_docker, corpus_path=corpus_path,
        )
        self._episode: EpisodeInfo | None = None
        self._step_count: int = 0
        self._done: bool = False
        self._question_idx: int = 0

    def reset(self, question_idx: int | None = None) -> str:
        """Start a new episode. Returns initial observation (question + tools)."""
        # Clean up previous session
        try:
            self.repl.kill_session()
        except Exception:
            pass

        # Pick question
        if question_idx is not None:
            self._question_idx = question_idx
        q = self.questions[self._question_idx % len(self.questions)]

        self._episode = EpisodeInfo(
            question_id=q["id"],
            question=q["question"],
            gold_answer=q["answer"],
            gold_citations=q.get("expected_citations", []),
        )
        self._step_count = 0
        self._done = False

        # Start fresh REPL
        self.repl = PersistentREPL(
            use_docker=self._use_docker, corpus_path=self._corpus_path,
        )
        self.repl.start_session()

        # Build initial observation
        observation = f"{SYSTEM_PREAMBLE}\n\nQuestion: {q['question']}\n"
        logger.info(f"Episode started: {q['id']} — {q['question'][:80]}")
        return observation

    def step(self, action: str) -> tuple[str, float, bool, dict]:
        """Take an action. Returns (observation, reward, done, info)."""
        if self._done:
            return "", 0.0, True, {"error": "Episode already done"}
        if self._episode is None:
            raise RuntimeError("Call reset() before step()")

        self._step_count += 1

        # Check if this is a submission
        submission = parse_submission(action)

        if submission is not None:
            answer, citations = submission
            reward_breakdown = compute_reward(
                predicted_answer=answer,
                predicted_citations=citations,
                gold_answer=self._episode.gold_answer,
                gold_citations=self._episode.gold_citations,
                steps_taken=self._step_count,
                max_steps=self.max_steps,
            )
            self._episode.final_reward = reward_breakdown
            self._done = True

            record = StepRecord(
                step=self._step_count,
                action=action,
                observation=f"Submitted. Reward: {reward_breakdown.total:.3f}",
                reward=reward_breakdown.total,
                done=True,
            )
            self._episode.trajectory.append(record)

            logger.info(
                f"Episode ended: reward={reward_breakdown.total:.3f} "
                f"(answer={reward_breakdown.answer_score:.2f}, "
                f"cit_p={reward_breakdown.citation_precision:.2f}, "
                f"cit_r={reward_breakdown.citation_recall:.2f}, "
                f"eff={reward_breakdown.efficiency_bonus:.2f})"
            )

            info = {
                "step": self._step_count,
                "reward_breakdown": reward_breakdown,
                "predicted_answer": answer,
                "predicted_citations": citations,
            }
            return "", reward_breakdown.total, True, info

        # Execute code in REPL
        observation = self.repl.execute(action, timeout=30)

        # If SyntaxError, add a hint to help the model recover
        if "SyntaxError" in observation:
            observation += (
                "\n\nHINT: Your response had a syntax error. "
                "Respond with ONLY Python code, no English text. Example:\n"
                "results = search(\"your query\")\n"
                "for r in results:\n"
                "    print(r[\"doc_id\"], r[\"title\"])"
            )

        # Add step counter so agent knows urgency
        remaining = self.max_steps - self._step_count
        if remaining <= 3:
            observation += f"\n\n[Step {self._step_count}/{self.max_steps} — {remaining} steps remaining. Submit soon!]"
        else:
            observation += f"\n\n[Step {self._step_count}/{self.max_steps}]"

        # Check if max steps reached
        done = self._step_count >= self.max_steps
        if done:
            self._done = True
            logger.warning(f"Max steps ({self.max_steps}) reached — episode timeout")

        record = StepRecord(
            step=self._step_count,
            action=action,
            observation=observation,
            reward=0.0,
            done=done,
        )
        self._episode.trajectory.append(record)

        info = {"step": self._step_count, "code": action, "output": observation}
        return observation, 0.0, done, info

    def get_trajectory(self) -> list[StepRecord]:
        """Return the full trajectory for this episode."""
        if self._episode is None:
            return []
        return self._episode.trajectory

    def get_episode_info(self) -> EpisodeInfo | None:
        """Return full episode info including question and reward."""
        return self._episode

    def close(self) -> None:
        """Clean up REPL session."""
        try:
            self.repl.kill_session()
        except Exception:
            pass

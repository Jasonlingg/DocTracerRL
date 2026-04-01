"""Single-shot baseline: minimal retrieval (top_k=3), answer in 1 step.

Even less context than naive RAG. Tests how much retrieval matters
when you can only look at a few chunks.
"""

from __future__ import annotations

import anthropic
from loguru import logger

from src.env.corpus import Corpus

SINGLE_SHOT_PROMPT = """Answer the following question based on the limited context below.
If the context is insufficient, give your best answer based on what's available.
Include citations as document IDs.

Format your response exactly as:
SUBMIT: <your answer> CITATIONS: ["doc_id_1", "doc_id_2"]

Context:
{context}

Question: {question}"""


class SingleShotPolicy:
    """Baseline: minimal retrieval (top_k=3), answer in one step."""

    def __init__(
        self,
        corpus: Corpus,
        model: str = "claude-haiku-4-5-20251001",
        top_k: int = 3,
        api_key: str | None = None,
    ) -> None:
        self.corpus = corpus
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.top_k = top_k
        self._question: str | None = None
        self._answered: bool = False

    def act(self, observation: str) -> str:
        """Extract question, retrieve top-3, submit."""
        if self._answered:
            return 'SUBMIT: No answer available CITATIONS: []'

        if self._question is None:
            self._question = self._extract_question(observation)

        results = self.corpus.search(self._question, top_k=self.top_k)

        context_parts = []
        for r in results:
            context_parts.append(f"[{r.doc_id}] {r.chunk}")
        context = "\n\n".join(context_parts)

        prompt = SINGLE_SHOT_PROMPT.format(context=context, question=self._question)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )

        self._answered = True
        action = response.content[0].text.strip()
        logger.debug(f"SingleShot answer: {action[:100]}...")
        return action

    def reset(self) -> None:
        self._question = None
        self._answered = False

    @staticmethod
    def _extract_question(observation: str) -> str:
        for line in observation.split("\n"):
            if line.strip().startswith("Question:"):
                return line.strip()[len("Question:"):].strip()
        return observation.split("\n")[-1].strip()

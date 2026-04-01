"""Naive RAG baseline: retrieve top-k chunks, pass to Claude, answer in 1 step.

This is the standard RAG approach — no iteration, no exploration.
Should perform well on single-document questions but poorly on multi-hop.
"""

from __future__ import annotations

import anthropic
from loguru import logger

from src.env.corpus import Corpus

RAG_PROMPT = """Answer the following question using ONLY the provided context.
Include citations as document IDs.

Format your response exactly as:
SUBMIT: <your answer> CITATIONS: ["doc_id_1", "doc_id_2"]

Context:
{context}

Question: {question}"""


class NaiveRAGPolicy:
    """Baseline: retrieve top-k chunks and answer in one step."""

    def __init__(
        self,
        corpus: Corpus,
        model: str = "claude-haiku-4-5-20251001",
        top_k: int = 10,
        max_context_chars: int = 30_000,
        api_key: str | None = None,
    ) -> None:
        self.corpus = corpus
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.top_k = top_k
        self.max_context_chars = max_context_chars
        self._question: str | None = None
        self._answered: bool = False

    def act(self, observation: str) -> str:
        """Extract question from first observation, retrieve, and submit."""
        if self._answered:
            return 'SUBMIT: No answer available CITATIONS: []'

        # Extract question from observation
        if self._question is None:
            self._question = self._extract_question(observation)

        # Retrieve top-k chunks, then fetch full docs for better context
        results = self.corpus.search(self._question, top_k=self.top_k)

        seen_docs: set[str] = set()
        context_parts: list[str] = []
        total_chars = 0
        for r in results:
            if r.doc_id in seen_docs:
                continue
            seen_docs.add(r.doc_id)
            text = self.corpus.read(r.doc_id)
            if not text:
                continue
            # Cap individual docs at 2000 chars to leave room for more docs
            if len(text) > 2000:
                text = text[:2000] + "\n[truncated]"
            if total_chars + len(text) > self.max_context_chars:
                break
            context_parts.append(f"=== [{r.doc_id}] ===\n{text}")
            total_chars += len(text)
        context = "\n\n".join(context_parts)

        # Ask Claude with retrieved context
        prompt = RAG_PROMPT.format(context=context, question=self._question)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )

        self._answered = True
        action = response.content[0].text.strip()
        logger.debug(f"NaiveRAG answer: {action[:100]}...")
        return action

    def reset(self) -> None:
        self._question = None
        self._answered = False

    @staticmethod
    def _extract_question(observation: str) -> str:
        """Pull the question text from the initial observation."""
        for line in observation.split("\n"):
            if line.strip().startswith("Question:"):
                return line.strip()[len("Question:"):].strip()
        return observation.split("\n")[-1].strip()

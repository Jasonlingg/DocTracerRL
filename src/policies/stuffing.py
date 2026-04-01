"""Context stuffing baseline: concatenate all documents, answer in 1 step.

Represents the "just give it everything" approach. Tests whether the model
can find needles in the haystack without targeted retrieval.
"""

from __future__ import annotations

import anthropic
from loguru import logger

from src.env.corpus import Corpus

STUFFING_PROMPT = """Answer the following question using ONLY the provided documents.
Include citations as document IDs.

Format your response exactly as:
SUBMIT: <your answer> CITATIONS: ["doc_id_1", "doc_id_2"]

Documents:
{documents}

Question: {question}"""


class ContextStuffingPolicy:
    """Baseline: concatenate docs into context, answer in one step.

    For small corpora: stuffs everything. For large corpora: retrieves
    top-k docs by relevance first, then stuffs those.
    """

    def __init__(
        self,
        corpus: Corpus,
        model: str = "claude-haiku-4-5-20251001",
        max_context_chars: int = 100_000,
        api_key: str | None = None,
    ) -> None:
        self.corpus = corpus
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_context_chars = max_context_chars
        self._question: str | None = None
        self._answered: bool = False

    def _corpus_fits(self) -> bool:
        """Check if the entire corpus fits within max_context_chars."""
        return sum(d.chars for d in self.corpus.list_documents()) < self.max_context_chars

    def _stuff_all(self) -> str:
        """Concatenate all documents (small corpus mode)."""
        doc_parts = []
        total_chars = 0
        for doc_info in self.corpus.list_documents():
            text = self.corpus.read(doc_info.doc_id)
            if text and total_chars + len(text) < self.max_context_chars:
                doc_parts.append(f"=== [{doc_info.doc_id}] {doc_info.title} ===\n{text}")
                total_chars += len(text)
        return "\n\n".join(doc_parts)

    def _stuff_topk(self, question: str, top_k: int = 20) -> str:
        """Retrieve top-k docs by embedding search, stuff those (large corpus mode)."""
        results = self.corpus.search(question, top_k=top_k)
        seen_docs: set[str] = set()
        doc_parts: list[str] = []
        total_chars = 0
        for r in results:
            if r.doc_id in seen_docs:
                continue
            seen_docs.add(r.doc_id)
            text = self.corpus.read(r.doc_id)
            if text and total_chars + len(text) < self.max_context_chars:
                doc_parts.append(f"=== [{r.doc_id}] ===\n{text}")
                total_chars += len(text)
        return "\n\n".join(doc_parts)

    def act(self, observation: str) -> str:
        """Extract question, stuff docs into context, submit."""
        if self._answered:
            return 'SUBMIT: No answer available CITATIONS: []'

        if self._question is None:
            self._question = self._extract_question(observation)

        if self._corpus_fits():
            documents = self._stuff_all()
            logger.info(f"ContextStuffing: stuffing all docs ({len(documents)} chars)")
        else:
            documents = self._stuff_topk(self._question)
            logger.info(f"ContextStuffing: large corpus — top-k retrieval ({len(documents)} chars)")

        prompt = STUFFING_PROMPT.format(documents=documents, question=self._question)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )

        self._answered = True
        action = response.content[0].text.strip()
        logger.debug(f"ContextStuffing answer: {action[:100]}...")
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

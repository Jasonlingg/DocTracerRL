"""Sparse RAG baseline: retrieve top-k docs by BM25, pass to Claude, answer in 1 step.

Mirrors NaiveRAGPolicy but uses BM25 (rank_bm25) instead of dense FAISS retrieval,
so the only difference between the two is the retrieval method — apples-to-apples
sparse vs dense comparison.
"""

from __future__ import annotations

import re

from anthropic import Anthropic
from loguru import logger
from rank_bm25 import BM25Okapi

from src.env.corpus import Corpus

RAG_PROMPT = """Answer the following question using ONLY the provided context.
Include citations as document IDs.

Format your response exactly as:
SUBMIT: <your answer> CITATIONS: ["doc_id_1", "doc_id_2"]

Context:
{context}

Question: {question}"""


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class SparseRAGPolicy:
    """Baseline: BM25 retrieval over full documents, answer in one step."""

    def __init__(
        self,
        corpus: Corpus,
        model: str = "claude-haiku-4-5-20251001",
        top_k: int = 10,
        max_context_chars: int = 30_000,
        api_key: str | None = None,
    ) -> None:
        self.corpus = corpus
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.top_k = top_k
        self.max_context_chars = max_context_chars
        self._question: str | None = None
        self._answered: bool = False
        self._bm25: BM25Okapi | None = None
        self._docs: list[dict] = []

    def _ensure_index(self) -> None:
        if self._bm25 is not None:
            return
        # Pull all documents from the corpus via the public list_documents +
        # get_document API; BM25 indexes whole-doc text.
        self._docs = [
            self.corpus.get_document(d.doc_id)
            for d in self.corpus.list_documents()
        ]
        self._docs = [d for d in self._docs if d is not None]
        self._bm25 = BM25Okapi([_tokenize(d["text"]) for d in self._docs])

    def act(self, observation: str) -> str:
        if self._answered:
            return 'SUBMIT: No answer available CITATIONS: []'

        if self._question is None:
            self._question = self._extract_question(observation)

        self._ensure_index()
        scores = self._bm25.get_scores(_tokenize(self._question))
        top_idx = sorted(range(len(scores)), key=lambda i: -scores[i])[: self.top_k]

        context_parts: list[str] = []
        total_chars = 0
        for i in top_idx:
            doc = self._docs[i]
            text = doc["text"]
            if len(text) > 2000:
                text = text[:2000] + "\n[truncated]"
            if total_chars + len(text) > self.max_context_chars:
                break
            context_parts.append(f"=== [{doc['doc_id']}] ===\n{text}")
            total_chars += len(text)
        context = "\n\n".join(context_parts)

        prompt = RAG_PROMPT.format(context=context, question=self._question)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )

        self._answered = True
        action = response.content[0].text.strip()
        logger.debug(f"SparseRAG answer: {action[:100]}...")
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

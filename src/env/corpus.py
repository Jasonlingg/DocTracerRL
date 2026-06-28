"""Load, chunk, embed, and search a document corpus with FAISS."""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from loguru import logger
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer


class Chunk(BaseModel):
    doc_id: str
    chunk_id: int
    text: str
    title: str


class SearchResult(BaseModel):
    doc_id: str
    chunk_id: int
    chunk: str
    score: float


class DocumentInfo(BaseModel):
    doc_id: str
    title: str
    chars: int


class Corpus:
    """Document corpus with vector search over chunks."""

    def __init__(
        self,
        corpus_path: str = "data/corpus",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        top_k: int = 5,
    ) -> None:
        self.corpus_path = Path(corpus_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self._documents: dict[str, dict] = {}
        self._chunks: list[Chunk] = []
        self._index: faiss.IndexFlatIP | None = None
        self._embedder: SentenceTransformer | None = None
        self._embedding_model = embedding_model

    @property
    def embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            logger.info(f"Loading embedding model: {self._embedding_model}")
            self._embedder = SentenceTransformer(self._embedding_model)
        return self._embedder

    def load(self) -> None:
        """Load documents from corpus_path, chunk, embed, and build FAISS index."""
        self._load_documents()
        self._chunk_documents()
        self._build_index()
        logger.info(
            f"Corpus loaded: {len(self._documents)} docs, "
            f"{len(self._chunks)} chunks"
        )

    def _load_documents(self) -> None:
        """Load .json and .txt files from corpus directory."""
        if not self.corpus_path.exists():
            raise FileNotFoundError(f"Corpus path not found: {self.corpus_path}")

        for path in sorted(self.corpus_path.iterdir()):
            if path.suffix == ".json":
                with open(path) as f:
                    doc = json.load(f)
                self._documents[doc["doc_id"]] = doc
            elif path.suffix == ".txt":
                text = path.read_text()
                doc_id = path.stem
                self._documents[doc_id] = {
                    "doc_id": doc_id,
                    "title": doc_id,
                    "text": text,
                    "metadata": {"type": "text"},
                }

        logger.info(f"Loaded {len(self._documents)} documents")

    def _chunk_documents(self) -> None:
        """Split documents into overlapping chunks."""
        self._chunks = []
        for doc in self._documents.values():
            text = doc["text"]
            title = doc.get("title", doc["doc_id"])
            start = 0
            chunk_id = 0
            while start < len(text):
                end = start + self.chunk_size
                chunk_text = text[start:end]
                self._chunks.append(Chunk(
                    doc_id=doc["doc_id"],
                    chunk_id=chunk_id,
                    text=chunk_text,
                    title=title,
                ))
                chunk_id += 1
                start += self.chunk_size - self.chunk_overlap

    def _build_index(self) -> None:
        """Embed chunks and build FAISS inner-product index."""
        if not self._chunks:
            logger.warning("No chunks to index")
            return

        texts = [c.text for c in self._chunks]
        embeddings = self.embedder.encode(texts, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype=np.float32)

        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)

    def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """Embed query and search FAISS index, returning top-k results."""
        if self._index is None or self._index.ntotal == 0:
            return []

        k = top_k or self.top_k
        q_emb = self.embedder.encode([query], normalize_embeddings=True)
        q_emb = np.array(q_emb, dtype=np.float32)

        scores, indices = self._index.search(q_emb, min(k, self._index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self._chunks[idx]
            results.append(SearchResult(
                doc_id=chunk.doc_id,
                chunk_id=chunk.chunk_id,
                chunk=chunk.text,
                score=float(score),
            ))
        return results

    def read(self, doc_id: str) -> str | None:
        """Return full document text by doc_id."""
        doc = self._documents.get(doc_id)
        if doc is None:
            return None
        return doc["text"]

    def list_documents(self) -> list[DocumentInfo]:
        """Return list of all documents with basic info."""
        return [
            DocumentInfo(
                doc_id=doc["doc_id"],
                title=doc.get("title", doc["doc_id"]),
                chars=len(doc["text"]),
            )
            for doc in self._documents.values()
        ]

    def get_document(self, doc_id: str) -> dict | None:
        """Return full document dict by doc_id."""
        return self._documents.get(doc_id)

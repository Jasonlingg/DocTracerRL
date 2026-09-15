"""Read-only research tools used by the validated agent action dispatcher."""

import json
import math
import re
from pathlib import Path


class ResearchTools:
    """A bounded, read-only API over one frozen corpus directory."""

    def __init__(self, corpus_dir: Path, paper_reranker=None):
        self.documents = {
            doc["doc_id"]: doc
            for path in sorted(corpus_dir.glob("*.json"))
            for doc in [json.loads(path.read_text())]
        }
        if not self.documents:
            raise ValueError("research corpus contains no documents")
        self.paper_reranker = paper_reranker

    def papers(self):
        return [
            {
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "submitted": doc["metadata"]["submitted"],
                "coverage": doc["metadata"]["coverage"],
                "source_kind": doc["metadata"].get("source_kind", "arxiv_paper"),
            }
            for doc in self.documents.values()
        ]

    def paper(self, doc_id):
        """Metadata includes version, source URL, dates, and available text coverage."""
        if not isinstance(doc_id, str) or doc_id not in self.documents:
            raise ValueError("doc_id must identify a paper in this snapshot")
        doc = self.documents[doc_id]
        return {
            "doc_id": doc_id,
            "title": doc["title"],
            **doc["metadata"],
            "sections": doc["sections"],
        }

    def passage(self, doc_id, start=0, length=1600):
        """Exact, bounded text with character offsets into this frozen snapshot."""
        if not isinstance(doc_id, str) or doc_id not in self.documents:
            raise ValueError("doc_id must identify a paper in this snapshot")
        doc = self.documents[doc_id]
        if type(start) is not int or type(length) is not int:
            raise ValueError("start and length must be integers")
        if not 0 <= start < len(doc["text"]) or not 1 <= length <= 3000:
            raise ValueError("start must be in the document; length must be 1..3000")
        end = min(start + length, len(doc["text"]))
        return {
            "doc_id": doc_id,
            "start": start,
            "end": end,
            "quote": doc["text"][start:end],
            "source_url": doc["metadata"]["source_url"],
            "coverage": doc["metadata"]["coverage"],
            "source_kind": doc["metadata"].get("source_kind", "arxiv_paper"),
            "source_path": doc["metadata"].get("source_path"),
            "pages": [section["page"] for section in doc["sections"]
                      if "page" in section and section["start"] < end and section["end"] > start],
        }

    def search_papers(self, query, top_k=5):
        """Rank paragraph windows lexically; inspect passages before drawing conclusions."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if type(top_k) is not int or not 1 <= top_k <= 10:
            raise ValueError("top_k must be 1..10")
        terms = set(re.findall(r"\w+", query.lower()))
        docs = self.documents
        df = {
            term: sum(
                term in set(re.findall(r"\w+", document["text"].lower()))
                for document in docs.values()
            )
            for term in terms
        }
        results = []
        for doc in docs.values():
            for section in doc["sections"]:
                for start in range(section["start"], section["end"], 1200):
                    end = min(start + 1600, section["end"])
                    words = re.findall(r"\w+", doc["text"][start:end].lower())
                    score = sum(
                        math.log(1 + len(docs) / (1 + df[term])) * words.count(term)
                        / (1 + len(words) / 200)
                        for term in terms
                    )
                    if score:
                        results.append({
                            **self.passage(doc["doc_id"], start, end - start),
                            "title": doc["title"],
                            "section": section["section"],
                            "score": round(score, 4),
                        })
        results.sort(key=lambda result: (-result["score"], result["doc_id"], result["start"]))
        # Search is the discovery step; return each paper's strongest matching
        # window so repeated passages from one document cannot hide other papers.
        best_by_document = {}
        for result in results:
            best_by_document.setdefault(result["doc_id"], result)
        return list(best_by_document.values())[:top_k]

    def search_paper(self, doc_id, query, top_k=5):
        """Rank bounded text windows inside one known paper."""
        if not isinstance(doc_id, str) or doc_id not in self.documents:
            raise ValueError("doc_id must identify a paper in this snapshot")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if type(top_k) is not int or not 1 <= top_k <= 10:
            raise ValueError("top_k must be 1..10")

        doc = self.documents[doc_id]
        candidates = []
        cursor = 0
        for paragraph in doc["text"].split("\n\n"):
            start = doc["text"].find(paragraph, cursor)
            end = start + len(paragraph)
            cursor = end + 2
            if not paragraph.strip():
                continue
            section = next(
                item for item in doc["sections"] if item["start"] <= start < item["end"]
            )
            candidates.append({
                "paragraph_start": start,
                "paragraph_end": end,
                "section": section,
                "ranking_text": f'{section["section"]}. {paragraph}',
            })

        if self.paper_reranker is not None:
            ranked = self.paper_reranker.rank(
                query, [candidate["ranking_text"] for candidate in candidates]
            )
            retriever = self.paper_reranker.config
        else:
            terms = set(re.findall(r"\w+", query.lower()))
            ranked = []
            for index, candidate in enumerate(candidates):
                words = re.findall(r"\w+", candidate["ranking_text"].lower())
                score = sum(words.count(term) / (1 + len(words) / 200) for term in terms)
                if score:
                    ranked.append((index, score))
            ranked.sort(key=lambda item: (-item[1], candidates[item[0]]["paragraph_start"]))
            retriever = {"kind": "lexical_paragraph", "version": "v1"}

        results = []
        for index, score in ranked[:top_k]:
            candidate = candidates[index]
            section = candidate["section"]
            center = (candidate["paragraph_start"] + candidate["paragraph_end"]) // 2
            start = max(section["start"], center - 1200)
            end = min(section["end"], start + 2400)
            start = max(section["start"], end - 2400)
            results.append({
                **self.passage(doc_id, start, end - start),
                "title": doc["title"],
                "section": section["section"],
                "score": round(score, 4),
                "retriever": retriever,
            })
        return results

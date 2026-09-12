"""Self-contained tools appended to the existing Python REPL preamble.

Only standard-library imports: this source also runs inside the sandbox image.
"""

import json
import math
import os
import re
from pathlib import Path


def _papers():
    return {
        doc["doc_id"]: doc
        for path in sorted(Path(os.environ["CORPUS_DIR"]).glob("*.json"))
        for doc in [json.loads(path.read_text())]
    }


def paper(doc_id):
    """Metadata includes version, source URL, dates, and available text coverage."""
    doc = _papers()[doc_id]
    return {"doc_id": doc_id, "title": doc["title"], **doc["metadata"],
            "sections": doc["sections"]}


def passage(doc_id, start=0, length=1600):
    """Exact, bounded text with character offsets into this frozen snapshot."""
    doc = _papers()[doc_id]
    if type(start) is not int or type(length) is not int:
        raise ValueError("start and length must be integers")
    if not 0 <= start < len(doc["text"]) or not 1 <= length <= 3000:
        raise ValueError("start must be in the document; length must be 1..3000")
    end = min(start + length, len(doc["text"]))
    return {"doc_id": doc_id, "start": start, "end": end,
            "quote": doc["text"][start:end], "source_url": doc["metadata"]["source_url"],
            "coverage": doc["metadata"]["coverage"]}


def search_papers(query, top_k=5):
    """Rank paragraph windows lexically; inspect passages before drawing conclusions."""
    if type(top_k) is not int or not 1 <= top_k <= 10:
        raise ValueError("top_k must be 1..10")
    terms = set(re.findall(r"\w+", query.lower()))
    docs = _papers()
    df = {term: sum(term in set(re.findall(r"\w+", d["text"].lower()))
                    for d in docs.values()) for term in terms}
    results = []
    for doc in docs.values():
        for section in doc["sections"]:
            for start in range(section["start"], section["end"], 1200):
                end = min(start + 1600, section["end"])
                words = re.findall(r"\w+", doc["text"][start:end].lower())
                score = sum(math.log(1 + len(docs) / (1 + df[t])) * words.count(t)
                            / (1 + len(words) / 200) for t in terms)
                if score:
                    results.append({
                        **passage(doc["doc_id"], start, end - start),
                        "title": doc["title"], "section": section["section"],
                        "score": round(score, 4),
                    })
    results.sort(key=lambda r: (-r["score"], r["doc_id"], r["start"]))
    return results[:top_k]


def papers():
    return [{"doc_id": d["doc_id"], "title": d["title"],
             "submitted": d["metadata"]["submitted"],
             "coverage": d["metadata"]["coverage"]} for d in _papers().values()]

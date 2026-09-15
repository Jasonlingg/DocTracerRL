"""Tool preamble injected into the REPL at session start.

Defines search(), read(), extract(), aggregate(), search_within(), verify()
functions that operate on the mounted corpus directory. Includes memoization
so re-running the cumulative script doesn't repeat expensive file reads.
"""

TOOL_PREAMBLE = '''
import json
import math
import os
import re
from pathlib import Path
from collections import Counter

# Memoization cache — persists across cumulative script re-runs
if "_memo" not in dir():
    _memo = {}

_corpus_env = os.environ.get("CORPUS_DIR", "")
if _corpus_env and Path(_corpus_env).exists():
    CORPUS_DIR = Path(_corpus_env)
elif Path("/workspace/data/corpus").exists():
    CORPUS_DIR = Path("/workspace/data/corpus")
elif Path("/workspace/corpus").exists():
    CORPUS_DIR = Path("/workspace/corpus")
else:
    CORPUS_DIR = Path("data/corpus")

def _load_doc(doc_id: str) -> dict | None:
    """Load a document by ID, with memoization."""
    if doc_id in _memo:
        return _memo[doc_id]
    path = CORPUS_DIR / f"{doc_id}.json"
    if not path.exists():
        return None
    with open(path) as f:
        doc = json.load(f)
    _memo[doc_id] = doc
    return doc

def _load_all_docs() -> list[dict]:
    """Load all documents, with memoization."""
    if "_all_docs" in _memo:
        return _memo["_all_docs"]
    docs = []
    for path in sorted(CORPUS_DIR.glob("*.json")):
        with open(path) as f:
            docs.append(json.load(f))
    _memo["_all_docs"] = docs
    return docs

def _get_chunks(chunk_size=500, chunk_overlap=100) -> list[dict]:
    """Build overlapping text chunks from all documents, with memoization."""
    cache_key = f"_chunks_{chunk_size}_{chunk_overlap}"
    if cache_key in _memo:
        return _memo[cache_key]
    chunks = []
    for doc in _load_all_docs():
        text = doc["text"]
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append({
                "doc_id": doc["doc_id"],
                "title": doc.get("title", doc["doc_id"]),
                "text": text[start:end],
                "start": start,
            })
            start += chunk_size - chunk_overlap
    _memo[cache_key] = chunks
    return chunks

def _idf_scores() -> dict[str, float]:
    """Compute IDF (inverse document frequency) for all terms."""
    if "_idf" in _memo:
        return _memo["_idf"]
    docs = _load_all_docs()
    n = len(docs)
    df = Counter()
    for doc in docs:
        terms = set(doc["text"].lower().split())
        for t in terms:
            df[t] += 1
    # Smoothed IDF stays positive even for a one-document corpus. The old
    # log(n / (1 + freq)) formula made every term negative when n == 1, so
    # search() discarded every matching result via its score > 0 check.
    idf = {t: math.log((n + 1) / (freq + 1)) + 1.0 for t, freq in df.items()}
    _memo["_idf"] = idf
    return idf

def search(query: str, top_k: int = 5, method: str = "keyword") -> list[dict]:
    """Search over corpus. method: 'keyword' (BM25-like) or 'chunk' (chunk-level TF-IDF).

    'keyword' — scores whole documents by TF-IDF, returns top matches with preview.
    'chunk'   — scores individual 500-char chunks, finds buried facts in long docs.
    """
    query_terms = query.lower().split()
    idf = _idf_scores()

    if method == "chunk":
        chunks = _get_chunks()
        scored = []
        for chunk in chunks:
            text_lower = chunk["text"].lower()
            score = sum(text_lower.count(t) * idf.get(t, 1.0) for t in query_terms)
            if score > 0:
                scored.append({
                    "doc_id": chunk["doc_id"],
                    "title": chunk["title"],
                    "chunk": chunk["text"],
                    "score": round(score, 2),
                    "offset": chunk["start"],
                })
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]
    else:
        # Default: document-level TF-IDF search
        results = []
        for doc in _load_all_docs():
            text_lower = doc["text"].lower()
            score = sum(text_lower.count(t) * idf.get(t, 1.0) for t in query_terms)
            if score > 0:
                results.append({
                    "doc_id": doc["doc_id"],
                    "title": doc.get("title", doc["doc_id"]),
                    "chunk": doc["text"][:500],
                    "score": round(score, 2),
                })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

def read(doc_id: str) -> str:
    """Read full document text by ID."""
    doc = _load_doc(doc_id)
    if doc is None:
        return f"ERROR: Document '{doc_id}' not found"
    return doc["text"]

def extract(doc_id: str, pattern: str) -> list[str]:
    """Extract text matching a regex pattern from a document."""
    doc = _load_doc(doc_id)
    if doc is None:
        return [f"ERROR: Document '{doc_id}' not found"]
    return re.findall(pattern, doc["text"])

def aggregate(doc_ids: list[str], field: str) -> list[dict]:
    """Extract a JSON metadata field across multiple documents."""
    results = []
    for doc_id in doc_ids:
        doc = _load_doc(doc_id)
        if doc is None:
            results.append({"doc_id": doc_id, "error": "not found"})
            continue
        value = doc.get("metadata", {}).get(field)
        results.append({"doc_id": doc_id, field: value})
    return results

def search_within(doc_id: str, query: str, top_k: int = 3) -> list[dict]:
    """Search within a specific document. Returns the most relevant 500-char windows."""
    doc = _load_doc(doc_id)
    if doc is None:
        return [{"error": f"Document '{doc_id}' not found"}]
    text = doc["text"]
    query_terms = query.lower().split()
    windows = []
    step = 200
    for start in range(0, len(text), step):
        end = start + 500
        window = text[start:end]
        score = sum(window.lower().count(t) for t in query_terms)
        if score > 0:
            windows.append({"text": window, "offset": start, "score": score})
    windows.sort(key=lambda x: x["score"], reverse=True)
    return windows[:top_k]

def verify(doc_id: str, claim: str) -> dict:
    """Check if a claim's keywords appear in a document. Returns bool + matching excerpt."""
    doc = _load_doc(doc_id)
    if doc is None:
        return {"found": False, "error": f"Document '{doc_id}' not found"}
    text_lower = doc["text"].lower()
    claim_terms = claim.lower().split()
    matches = sum(1 for t in claim_terms if t in text_lower)
    ratio = matches / len(claim_terms) if claim_terms else 0
    if ratio < 0.4:
        return {"found": False, "match_ratio": round(ratio, 2)}
    # Find best matching window
    best_score, best_start = 0, 0
    for start in range(0, len(doc["text"]) - 200, 100):
        window = doc["text"][start:start+200].lower()
        score = sum(window.count(t) for t in claim_terms)
        if score > best_score:
            best_score = score
            best_start = start
    excerpt = doc["text"][best_start:best_start+200]
    return {"found": True, "match_ratio": round(ratio, 2), "excerpt": excerpt}

def list_docs() -> list[dict]:
    """List all available documents."""
    return [
        {"doc_id": d["doc_id"], "title": d.get("title", d["doc_id"]), "chars": len(d["text"])}
        for d in _load_all_docs()
    ]

print(
    "Tools loaded: search(), read(), extract(), aggregate(), "
    "search_within(), verify(), list_docs()"
)
print(f"Corpus: {len(list_docs())} documents available")
print("TIP: search(q) for doc-level, search(q, method='chunk') for chunk-level search")
'''

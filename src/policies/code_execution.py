"""Provider-independent prompt and action parsing for code-execution agents."""

from __future__ import annotations

import re

SYSTEM_PROMPT = """You are an agent exploring a document corpus via Python code.

Tools (already imported):
  search(query, top_k=5)         → [{"doc_id", "title", "chunk", "score"}]
  search(query, method="chunk")  → chunk-level search for buried facts
  read(doc_id)                   → full document text
  extract(doc_id, regex)         → regex matches from a doc
  search_within(doc_id, query)   → relevant windows inside a specific doc
  verify(doc_id, claim)          → {"found", "match_ratio", "excerpt"}
  list_docs()                    → [{"doc_id", "title", "chars"}]

Each turn: write Python code OR a SUBMIT line. Never both. Never prose. Never markdown.
Variables persist across turns. Use print() to see output.

Before you SUBMIT: does the question have more than one distinct part (e.g. two
different numbers or facts)? If so, confirm you have explicit support for EACH
part, not just the one that seems most central, before submitting.

When you have the answer:
SUBMIT: <your answer> CITATIONS: ["doc_id_1", "doc_id_2"]
"""

DEFAULT_MAX_TOKENS = 1024


def clean_action(text: str) -> str:
    """Extract a SUBMIT line or bare code block from raw model output."""
    stripped = text.strip()

    submit_match = re.search(r"(SUBMIT:\s*.+)", stripped, re.DOTALL | re.IGNORECASE)
    if submit_match:
        return submit_match.group(1).strip()

    if stripped.startswith("```python") and stripped.endswith("```"):
        stripped = stripped[len("```python") :][:-3].strip()
    elif stripped.startswith("```") and stripped.endswith("```"):
        stripped = "\n".join(stripped.split("\n")[1:-1]).strip()

    code_blocks = re.findall(r"```(?:python)?\n(.*?)```", stripped, re.DOTALL)
    if code_blocks:
        return "\n".join(code_blocks).strip()

    return stripped

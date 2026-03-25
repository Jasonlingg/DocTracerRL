"""Reference policy: Claude decides what code to write at each step.

This validates that the environment rewards good exploration strategies.
Future work replaces this with an open-weight model trained via GRPO.
"""

from __future__ import annotations

import re

import anthropic
from loguru import logger

SYSTEM_PROMPT = """You are a Python programmer exploring a document corpus. You can ONLY respond with Python code or a SUBMIT line. Never respond with English prose, explanations, or XML.

AVAILABLE FUNCTIONS (already loaded):
  search(query, top_k=5)           → doc-level keyword search: [{"doc_id", "title", "chunk", "score"}]
  search(query, method="chunk")    → chunk-level search (finds buried facts in long documents)
  read(doc_id)                     → full document text (use ONLY for short docs)
  extract(doc_id, pattern)         → regex matches from a document
  search_within(doc_id, query)     → search inside a specific document for relevant 500-char windows
  verify(doc_id, claim)            → check if a claim's keywords appear in a doc (fast relevance check)
  list_docs()                      → [{"doc_id", "title", "chars"}]

RULES:
1. Each response must be EITHER executable Python code OR a SUBMIT line. Never both.
2. Do NOT write English sentences. Do NOT explain your thinking. Just write code.
3. Do NOT use XML tags, markdown fences, or any non-Python syntax.
4. Use print() to see results. Variables persist between steps.
5. When you have the answer, respond with ONLY: SUBMIT: <answer> CITATIONS: ["id1", "id2"]
6. Keep each response to ONE sub-question per step. Variables persist, so you will see results next step.

STRATEGY — follow these steps for multi-hop questions:
- Initialize known_facts = {} on step 1. Store EVERY discovery.
- Break the question into sub-questions. Solve one per step.
- When a document references another entity, person, or place, SEARCH for it next.
- PREFER search_within(doc_id, query) over read(doc_id). It returns only the relevant 500-char windows.
- Use verify(doc_id, claim) BEFORE reading — it's a fast check if a doc is relevant.
- Use extract(doc_id, pattern) to pull specific values (dates, numbers, names) with regex.
- NEVER call read() on the same document twice. NEVER call read() on docs longer than 1000 chars.
- If search returns nothing useful, try different keywords or method="chunk".
- Submit as soon as you have enough facts. Do not over-explore.

EXAMPLE STEP 1 — search, then verify before reading:
known_facts = {}
results = search("person X employer")
for r in results:
    print(r["doc_id"], r["title"], r["score"])
# Check which doc is actually relevant
for r in results[:3]:
    v = verify(r["doc_id"], "person X employer")
    print(r["doc_id"], v["found"], v.get("excerpt", "")[:100])

EXAMPLE STEP 2 — search_within a long document (PREFERRED over read):
windows = search_within("some_doc_id", "headquarters location")
for w in windows:
    print(w["text"])
known_facts["employer"] = "Company Y"
known_facts["hq_city"] = "City Z"

EXAMPLE STEP 3 — extract specific values with regex:
matches = extract("another_doc_id", r"City Z.*?(?:station|terminal|depot)[^.]*")
for m in matches:
    print(m)

EXAMPLE STEP 4 — follow a chain: use discovery from step 2 in a new search:
results = search(f"bus station {known_facts['hq_city']}")
for r in results:
    print(r["doc_id"], r["title"], r["score"])

EXAMPLE STEP 5 — submit with all citations:
SUBMIT: The answer based on discovered facts. CITATIONS: ["doc_id_1", "doc_id_2"]"""


def _truncate_to_first_step(code: str) -> str:
    """If the model generated multiple steps, keep only the first one.

    Haiku often outputs '# Step 1 ... # Step 2 ... # Step 3 ...' as one block.
    Code for later steps references variables that don't exist yet (because
    the agent hasn't seen earlier steps' output). This is not a policy choice
    we're constraining — it's broken code that will SyntaxError/NameError.
    """
    lines = code.split("\n")
    step_markers = []
    for i, line in enumerate(lines):
        if re.match(r"^#\s*Step\s+\d", line.strip(), re.IGNORECASE):
            step_markers.append(i)

    # Only truncate if there are 2+ step markers (multi-step dump) — cut at the second marker
    if len(step_markers) >= 2:
        code = "\n".join(lines[: step_markers[1]]).strip()

    return code


class ClaudePolicy:
    """Reference policy using Claude API as the agent brain."""

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> None:
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.history: list[dict] = []

    def act(self, observation: str) -> str:
        """Given an observation, return an action (code or SUBMIT)."""
        self.history.append({"role": "user", "content": observation})

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=SYSTEM_PROMPT,
            messages=self.history,
        )

        action = response.content[0].text
        self.history.append({"role": "assistant", "content": action})

        # Clean raw model output into executable Python
        action = self._clean_action(action)

        logger.debug(f"Claude action ({len(action)} chars): {action[:100]}...")
        return action

    def reset(self) -> None:
        """Clear history for a new episode."""
        self.history = []

    @staticmethod
    def _clean_action(text: str) -> str:
        """Extract executable Python from model output, stripping fences, XML, and prose."""
        stripped = text.strip()

        # If there's a SUBMIT line anywhere, extract and return it
        # (model is ready to answer — don't try to run code too)
        submit_match = re.search(
            r"(SUBMIT:\s*.*?CITATIONS:\s*\[.*?\])",
            stripped,
            re.DOTALL | re.IGNORECASE,
        )
        if not submit_match:
            submit_match = re.search(
                r"(SUBMIT:\s*.+)",
                stripped,
                re.IGNORECASE,
            )
        if submit_match:
            return submit_match.group(1).strip()

        # Strip markdown code fences
        if stripped.startswith("```python") and stripped.endswith("```"):
            stripped = stripped[len("```python"):][:-3].strip()
        elif stripped.startswith("```") and stripped.endswith("```"):
            lines = stripped.split("\n")
            stripped = "\n".join(lines[1:-1]).strip()

        # Strip XML function_calls (Haiku sometimes generates these)
        if "<function_calls>" in stripped or "<invoke" in stripped:
            stripped = re.sub(r"</?function_calls>", "", stripped)
            stripped = re.sub(r"</?invoke[^>]*>", "", stripped)
            stripped = re.sub(r"</?parameter[^>]*>", "", stripped)

        # Extract code from markdown fences embedded in prose
        code_blocks = re.findall(r"```(?:python)?\n(.*?)```", stripped, re.DOTALL)
        if code_blocks:
            stripped = "\n".join(code_blocks).strip()

        # Filter: keep only lines that look like Python code, drop prose
        lines = stripped.split("\n")
        code_lines = []
        for line in lines:
            s = line.strip()
            # Keep blank lines (they're valid Python)
            if not s:
                code_lines.append(line)
                continue
            # Keep lines that look like code
            if re.match(
                r"^("
                r"#|"                           # comments
                r"[a-zA-Z_]\w*\s*[=(.\[]|"      # assignment, call, attribute, index
                r"from |import |"               # imports
                r"print\(|"                     # print
                r"for |if |elif |else:|while |" # control flow
                r"def |class |"                 # definitions
                r"return |yield |"              # returns
                r"try:|except |finally:|"       # exception handling
                r"with |"                       # context managers
                r"raise |assert |"              # raise/assert
                r"pass|break|continue|"         # simple statements
                r"\)|"                          # closing paren (continuation)
                r"\]|"                          # closing bracket
                r"\}|"                          # closing brace
                r"\"\"\"|'''|"                  # docstrings
                r"@"                            # decorators
                r")",
                s,
            ):
                code_lines.append(line)
            # else: drop the line (it's prose)

        result = "\n".join(code_lines).strip()
        result = result if result else stripped

        # Haiku often generates ALL steps at once (# Step 1 ... # Step 2 ...).
        # Truncate to just the first step to avoid running code that depends
        # on output the agent hasn't seen yet.
        result = _truncate_to_first_step(result)

        return result

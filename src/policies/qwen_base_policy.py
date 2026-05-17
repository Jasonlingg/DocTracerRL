"""Qwen2.5-1.5B-Instruct baseline (untrained) — isolates training contribution.

Uses Together.ai's OpenAI-compatible endpoint so no local GPU is needed for eval.
Set TOGETHER_API_KEY in .env (or environment) before running.

This policy is structurally identical to qwen_sft_policy / grpo_policy — the only
difference is the model string (base vs. fine-tuned checkpoint). Having all three
in the final comparison table is what makes the base → sft → grpo delta credible.
"""

from __future__ import annotations

import os
import re

from loguru import logger

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

When you have the answer:
SUBMIT: <your answer> CITATIONS: ["doc_id_1", "doc_id_2"]
"""

# Together.ai model ID for Qwen2.5-1.5B-Instruct
TOGETHER_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


class QwenBasePolicy:
    """Untrained Qwen2.5-1.5B-Instruct via Together.ai inference API."""

    def __init__(
        self,
        model: str = TOGETHER_MODEL,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        api_key: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("pip install openai") from e

        key = api_key or os.environ.get("TOGETHER_API_KEY")
        if not key:
            raise ValueError("Set TOGETHER_API_KEY in .env or environment")

        from openai import OpenAI
        self.client = OpenAI(
            api_key=key,
            base_url="https://api.together.xyz/v1",
        )
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.history: list[dict] = []

    def act(self, observation: str) -> str:
        self.history.append({"role": "user", "content": observation})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + self.history,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        action = response.choices[0].message.content or ""
        self.history.append({"role": "assistant", "content": action})
        action = self._clean_action(action)

        logger.debug(f"QwenBase action ({len(action)} chars): {action[:100]}...")
        return action

    def reset(self) -> None:
        self.history = []

    @staticmethod
    def _clean_action(text: str) -> str:
        stripped = text.strip()

        submit_match = re.search(
            r"(SUBMIT:\s*.*?CITATIONS:\s*\[.*?\])", stripped, re.DOTALL | re.IGNORECASE
        ) or re.search(r"(SUBMIT:\s*.+)", stripped, re.IGNORECASE)
        if submit_match:
            return submit_match.group(1).strip()

        if stripped.startswith("```python") and stripped.endswith("```"):
            stripped = stripped[len("```python"):][:-3].strip()
        elif stripped.startswith("```") and stripped.endswith("```"):
            stripped = "\n".join(stripped.split("\n")[1:-1]).strip()

        code_blocks = re.findall(r"```(?:python)?\n(.*?)```", stripped, re.DOTALL)
        if code_blocks:
            return "\n".join(code_blocks).strip()

        return stripped

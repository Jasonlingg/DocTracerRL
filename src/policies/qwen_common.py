"""Shared system prompt, action cleaning, and model loading for local Qwen policies.

qwen_base_policy.py, qwen_sft_policy.py, and grpo_policy.py previously each
carried an independent ~140-line copy of this logic, differing only in
whether a LoRA adapter is applied on top of the base model.
"""

from __future__ import annotations

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


def clean_action(text: str) -> str:
    """Extract a SUBMIT line or bare code block from raw model output."""
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


def load_qwen_tokenizer(base_model: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_qwen_base_model(base_model: str):
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        base_model, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    model.eval()
    return model


def load_qwen_lora_model(base_model: str, checkpoint_path: str):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(
        base_model, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, checkpoint_path)
    model.eval()
    return model


class BaseQwenPolicy:
    """Shared act()/reset() for local Qwen inference. Subclasses set
    self._tokenizer and self._model in __init__ before calling act()."""

    def __init__(self, max_tokens: int = 1024, temperature: float = 0.0) -> None:
        self._max_tokens = max_tokens
        self._temperature = temperature
        self.history: list[dict] = []

    def act(self, observation: str) -> str:
        import torch

        self.history.append({"role": "user", "content": observation})

        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.history
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self._max_tokens,
                do_sample=self._temperature > 0,
                temperature=self._temperature if self._temperature > 0 else None,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        action = self._tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        self.history.append({"role": "assistant", "content": action})
        action = clean_action(action)

        logger.debug(f"{type(self).__name__} action ({len(action)} chars): {action[:100]}...")
        return action

    def reset(self) -> None:
        self.history = []

"""Qwen2.5-1.5B-Instruct + SFT LoRA adapter — mandatory ablation baseline.

Loads a LoRA checkpoint trained by scripts/train_sft.py on top of the Qwen2.5-1.5B
base. Without this in the comparison table, the GRPO contribution is unverifiable.

The three-way comparison is:
  qwen_base_policy  (untrained)  →  qwen_sft_policy  (SFT warmup)  →  grpo_policy  (GRPO)

Run on the same GPU box used for training:
  CHECKPOINT_PATH=checkpoints/sft_qwen_1.5b/final python scripts/run_eval.py \
      --musique --split dev --policy qwen_sft_policy -t 50
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

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


class QwenSFTPolicy:
    """SFT-trained Qwen2.5-7B with LoRA adapter, loaded for local inference."""

    def __init__(
        self,
        checkpoint_path: str | None = None,
        base_model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> None:
        path = checkpoint_path or os.environ.get("CHECKPOINT_PATH")
        if not path:
            raise ValueError(
                "Provide checkpoint_path or set CHECKPOINT_PATH env var "
                "(e.g. checkpoints/sft_qwen_7b/final)"
            )

        base = base_model or os.environ.get("BASE_MODEL_PATH") or DEFAULT_BASE_MODEL

        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError("pip install -e '.[training]'") from e

        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        base_mdl = AutoModelForCausalLM.from_pretrained(
            base,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self._model = PeftModel.from_pretrained(base_mdl, path)
        self._model.eval()

        self._max_tokens = max_tokens
        self._temperature = temperature
        self.history: list[dict] = []

        logger.info(f"QwenSFTPolicy loaded from {path}")

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
        action = self._clean_action(action)

        logger.debug(f"QwenSFT action ({len(action)} chars): {action[:100]}...")
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

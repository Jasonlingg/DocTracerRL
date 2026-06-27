"""Qwen2.5-7B-Instruct baseline (untrained) — isolates training contribution.

Loads the base model locally (no API key needed). Set BASE_MODEL_PATH to a local
directory or leave unset to download from HuggingFace.

This policy is structurally identical to qwen_sft_policy / grpo_policy — the only
difference is no LoRA adapter is applied. Having all three in the comparison table
is what makes the base → sft → grpo delta credible.
"""

from __future__ import annotations

import os

from loguru import logger

from src.policies.qwen_common import BaseQwenPolicy, load_qwen_base_model, load_qwen_tokenizer

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


class QwenBasePolicy(BaseQwenPolicy):
    """Untrained Qwen2.5-7B-Instruct loaded locally for inference."""

    def __init__(
        self,
        base_model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> None:
        super().__init__(max_tokens=max_tokens, temperature=temperature)
        model_path = base_model or os.environ.get("BASE_MODEL_PATH") or DEFAULT_BASE_MODEL

        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
        except ImportError as e:
            raise ImportError("pip install -e '.[training]'") from e

        self._tokenizer = load_qwen_tokenizer(model_path)
        self._model = load_qwen_base_model(model_path)

        logger.info(f"QwenBasePolicy loaded from {model_path}")

"""Qwen2.5-7B-Instruct + SFT LoRA adapter — mandatory ablation baseline.

Loads a LoRA checkpoint trained by scripts/train_sft.py on top of the Qwen2.5-7B
base. Without this in the comparison table, the GRPO contribution is unverifiable.

The three-way comparison is:
  qwen_base_policy  (untrained)  →  qwen_sft_policy  (SFT warmup)  →  grpo_policy  (GRPO)

Run on the same GPU box used for training:
  CHECKPOINT_PATH=checkpoints/sft_qwen_7b/final python scripts/run_eval.py \
      --musique --split dev --policy qwen_sft_policy -t 50
"""

from __future__ import annotations

import os

from loguru import logger

from src.policies.qwen_common import BaseQwenPolicy, load_qwen_lora_model, load_qwen_tokenizer

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


class QwenSFTPolicy(BaseQwenPolicy):
    """SFT-trained Qwen2.5-7B with LoRA adapter, loaded for local inference."""

    def __init__(
        self,
        checkpoint_path: str | None = None,
        base_model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> None:
        super().__init__(max_tokens=max_tokens, temperature=temperature)
        path = checkpoint_path or os.environ.get("CHECKPOINT_PATH")
        if not path:
            raise ValueError(
                "Provide checkpoint_path or set CHECKPOINT_PATH env var "
                "(e.g. checkpoints/sft_qwen_7b/final)"
            )

        base = base_model or os.environ.get("BASE_MODEL_PATH") or DEFAULT_BASE_MODEL

        try:
            import torch  # noqa: F401
            from peft import PeftModel  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
        except ImportError as e:
            raise ImportError("pip install -e '.[training]'") from e

        self._tokenizer = load_qwen_tokenizer(base)
        self._model = load_qwen_lora_model(base, path)

        logger.info(f"QwenSFTPolicy loaded from {path}")

"""GRPO-trained Qwen2.5-7B policy — the headline result.

Structurally identical to qwen_sft_policy but loads the GRPO LoRA adapter instead.
The base → sft → grpo delta is the primary claim of the project:
  - base → sft: format + tool-use learned via imitation
  - sft → grpo: multi-hop exploration strategy learned via RL

Run on the same GPU box used for training:
  CHECKPOINT_PATH=checkpoints/grpo_qwen_7b/final python scripts/run_eval.py \
      --musique --split test --policy grpo_policy
"""

from __future__ import annotations

import os

from loguru import logger

from src.policies.qwen_common import BaseQwenPolicy, load_qwen_lora_model, load_qwen_tokenizer

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


class GRPOPolicy(BaseQwenPolicy):
    """GRPO-trained Qwen2.5-7B with LoRA adapter, loaded for local inference."""

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
                "(e.g. checkpoints/grpo_qwen_7b/final)"
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

        logger.info(f"GRPOPolicy loaded from {path}")

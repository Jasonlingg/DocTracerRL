"""Shared system prompt, action cleaning, and model loading for local Qwen policies.

qwen_base_policy.py, qwen_sft_policy.py, and grpo_policy.py previously each
carried an independent ~140-line copy of this logic, differing only in
whether a LoRA adapter is applied on top of the base model.
"""

from __future__ import annotations

from loguru import logger

from src.policies.code_execution import DEFAULT_MAX_TOKENS, SYSTEM_PROMPT, clean_action


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

    def __init__(
        self,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
    ) -> None:
        self._max_tokens = max_tokens
        self._temperature = temperature
        self.history: list[dict] = []

    def act(self, observation: str) -> str:
        import torch

        self.history.append({"role": "user", "content": observation})

        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.history
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
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

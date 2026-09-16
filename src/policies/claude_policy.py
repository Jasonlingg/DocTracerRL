"""Reference policy: Claude decides what code to write at each step.

This validates that the environment rewards good exploration strategies.
Future work replaces this with an open-weight model trained via GRPO.
"""

from __future__ import annotations

import os

import anthropic
from loguru import logger

from src.policies.claude_action_cleaning import clean_action
from src.policies.claude_prompts import SYSTEM_PROMPT

DEFAULT_MODEL = "claude-haiku-4-5-20251001"


class ClaudePolicy:
    """Reference policy using Claude API as the agent brain.

    Set CLAUDE_MODEL_PATH to use a different model (e.g. as a stronger teacher
    for trajectory distillation) without touching call sites, matching the
    BASE_MODEL_PATH override pattern used by the local Qwen policies.
    """

    def __init__(
        self,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float | None = 0.0,
        api_key: str | None = None,
    ) -> None:
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model or os.environ.get("CLAUDE_MODEL_PATH") or DEFAULT_MODEL
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._temperature_supported = temperature is not None
        self.history: list[dict] = []

    def act(self, observation: str) -> str:
        """Given an observation, return an action (code or SUBMIT)."""
        self.history.append({"role": "user", "content": observation})

        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": SYSTEM_PROMPT,
            "messages": self.history,
        }
        if self._temperature_supported:
            kwargs["temperature"] = self.temperature
        try:
            response = self.client.messages.create(**kwargs)
        except anthropic.BadRequestError as exc:
            if self._temperature_supported and "temperature" in str(exc):
                # Newer model generations (e.g. claude-sonnet-5) reject an
                # explicit temperature entirely rather than just clamping it.
                logger.warning(f"{self.model} rejected temperature; retrying without it")
                self._temperature_supported = False
                kwargs.pop("temperature", None)
                response = self.client.messages.create(**kwargs)
            else:
                raise

        # Skip non-text blocks (e.g. ThinkingBlock, present when a model's
        # native reasoning mode is on by default) and take the final text
        # block, which is the model's actual response after any reasoning.
        text_blocks = [block.text for block in response.content if hasattr(block, "text")]
        if not text_blocks:
            raise ValueError(f"{self.model} response had no text content block")
        action = text_blocks[-1]
        self.history.append({"role": "assistant", "content": action})

        # Clean raw model output into executable Python
        action = clean_action(action)

        logger.debug(f"Claude action ({len(action)} chars): {action[:100]}...")
        return action

    def reset(self) -> None:
        """Clear history for a new episode."""
        self.history = []

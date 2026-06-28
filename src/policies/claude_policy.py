"""Reference policy: Claude decides what code to write at each step.

This validates that the environment rewards good exploration strategies.
Future work replaces this with an open-weight model trained via GRPO.
"""

from __future__ import annotations

import anthropic
from loguru import logger

from src.policies.claude_action_cleaning import clean_action
from src.policies.claude_prompts import SYSTEM_PROMPT


class ClaudePolicy:
    """Reference policy using Claude API as the agent brain."""

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        max_tokens: int = 4096,
        temperature: float = 0.0,
        api_key: str | None = None,
    ) -> None:
        self.client = anthropic.Anthropic(api_key=api_key)
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
        action = clean_action(action)

        logger.debug(f"Claude action ({len(action)} chars): {action[:100]}...")
        return action

    def reset(self) -> None:
        """Clear history for a new episode."""
        self.history = []

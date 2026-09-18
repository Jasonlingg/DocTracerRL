"""The model-independent contract used by Envoy's execution harnesses."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Policy(Protocol):
    """Anything that can choose the next action from an environment observation."""

    def act(self, observation: str) -> str:
        """Return Python code or a terminal SUBMIT action."""

    def reset(self) -> None:
        """Clear episode-local state."""

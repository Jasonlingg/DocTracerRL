"""One policy registry shared by the CLI and trajectory viewer."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any

from src.env.corpus import Corpus
from src.policies.claude_policy import ClaudePolicy
from src.policies.grpo_policy import GRPOPolicy
from src.policies.naive_rag import NaiveRAGPolicy
from src.policies.openai_compatible import OpenAICompatiblePolicy
from src.policies.qwen_base_policy import QwenBasePolicy
from src.policies.qwen_sft_policy import QwenSFTPolicy
from src.policies.single_shot import SingleShotPolicy
from src.policies.sparse_rag import SparseRAGPolicy
from src.policies.stuffing import ContextStuffingPolicy


@dataclass(frozen=True)
class PolicySpec:
    name: str
    label: str
    description: str
    requires_anthropic_key: bool = False
    requires_checkpoint: bool = False
    requires_endpoint: bool = False
    select_by_default: bool = True


POLICY_SPECS = {
    spec.name: spec
    for spec in (
        PolicySpec(
            "claude_policy",
            "Claude explorer",
            "Reference code-execution policy",
            requires_anthropic_key=True,
        ),
        PolicySpec(
            "naive_rag",
            "Claude + dense RAG",
            "Single-pass dense-retrieval baseline",
            requires_anthropic_key=True,
        ),
        PolicySpec(
            "sparse_rag",
            "Claude + sparse RAG",
            "Single-pass BM25 baseline",
            requires_anthropic_key=True,
        ),
        PolicySpec(
            "context_stuffing",
            "Claude + context stuffing",
            "Single-pass full-context baseline",
            requires_anthropic_key=True,
        ),
        PolicySpec(
            "single_shot",
            "Claude single shot",
            "Single-pass minimal baseline",
            requires_anthropic_key=True,
        ),
        PolicySpec(
            "qwen_base_policy",
            "Local Qwen base",
            "Untrained local Hugging Face checkpoint",
        ),
        PolicySpec(
            "qwen_sft_policy",
            "Local Qwen SFT",
            "Local Hugging Face base plus SFT adapter",
            requires_checkpoint=True,
        ),
        PolicySpec(
            "grpo_policy",
            "Local Qwen GRPO",
            "Local Hugging Face base plus GRPO adapter",
            requires_checkpoint=True,
        ),
        PolicySpec(
            "openai_compatible",
            "OpenAI-compatible endpoint",
            "vLLM, Ollama, LM Studio, or a compatible hosted model",
            requires_endpoint=True,
            select_by_default=False,
        ),
    )
}


def policy_catalog(*, allow_anthropic_byok: bool = False) -> list[dict[str, Any]]:
    """Return UI-safe policy metadata and current server-side availability."""
    catalog = []
    for spec in POLICY_SPECS.values():
        available = True
        reason = ""
        if spec.requires_anthropic_key and not (
            allow_anthropic_byok or os.environ.get("ANTHROPIC_API_KEY")
        ):
            available = False
            reason = "requires an Anthropic API key"
        elif spec.requires_checkpoint and not os.environ.get("CHECKPOINT_PATH"):
            available = False
            reason = "set CHECKPOINT_PATH"
        elif spec.requires_endpoint and not (
            os.environ.get("ENVOY_MODEL_ENDPOINT") and os.environ.get("ENVOY_MODEL_ID")
        ):
            available = False
            reason = "set ENVOY_MODEL_ENDPOINT and ENVOY_MODEL_ID"
        catalog.append({**asdict(spec), "available": available, "reason": reason})
    return catalog


def requires_anthropic_key(names: list[str]) -> bool:
    return any(
        POLICY_SPECS.get(name, PolicySpec(name, name, "")).requires_anthropic_key for name in names
    )


def build_policies(
    corpus: Corpus,
    names: list[str] | None = None,
    *,
    api_key: str | None = None,
    as_factories: bool = False,
) -> dict[str, object]:
    """Build selected policies without coupling the harness to model classes."""
    factories = {
        "claude_policy": lambda: ClaudePolicy(api_key=api_key),
        "naive_rag": lambda: NaiveRAGPolicy(corpus=corpus, api_key=api_key),
        "sparse_rag": lambda: SparseRAGPolicy(corpus=corpus, api_key=api_key),
        "context_stuffing": lambda: ContextStuffingPolicy(corpus=corpus, api_key=api_key),
        "single_shot": lambda: SingleShotPolicy(corpus=corpus, api_key=api_key),
        "qwen_base_policy": lambda: QwenBasePolicy(),
        "qwen_sft_policy": lambda: QwenSFTPolicy(),
        "grpo_policy": lambda: GRPOPolicy(),
        "openai_compatible": lambda: OpenAICompatiblePolicy(),
    }
    selected = {
        name: factory
        for name, factory in factories.items()
        if (names is None and POLICY_SPECS[name].select_by_default)
        or (names is not None and name in names)
    }
    if as_factories:
        return selected
    return {name: factory() for name, factory in selected.items()}

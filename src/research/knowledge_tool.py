"""Callable knowledge-query tool: a main agent's interface to the paper-search loop.

Wraps run_question unchanged. This module adds no new agent behavior, only a
narrower result shape and a stable tool schema, so the underlying Qwen/Claude/etc.
policy and the research-tools action protocol stay exactly what they already are.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

from src.research.agent import run_question

QUERY_PAPERS_TOOL = {
    "name": "query_papers",
    "description": (
        "Search the paper library for evidence relevant to a specific question and "
        "return cited findings with exact source spans. Call this only when you need "
        "facts from the paper collection; the tool cannot do anything else."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "minLength": 1,
                "description": "The specific question to research.",
            },
        },
        "required": ["question"],
        "additionalProperties": False,
    },
}


def query_papers(snapshot: Path, question: str, policy: Any, run_dir: Path,
                 max_steps: int = 10, server_hardware: str = "unrecorded") -> dict:
    """Run one knowledge-query call and return a result shaped for a calling agent.

    `policy` is any object implementing `.act(observation) -> str`; this function is
    identical for a base Qwen EndpointPolicy, a trained Qwen checkpoint, or Claude via
    AnthropicPolicy. `run_dir` receives the full run artifact, matching this project's
    convention that every call stays inspectable and reproducible.
    """
    run_id = uuid4().hex
    output = run_dir / f"{run_id}.json"
    result = run_question(
        snapshot=snapshot,
        question={"id": run_id, "question": question},
        policy=policy,
        output=output,
        max_steps=max_steps,
        server_hardware=server_hardware,
    )
    return {
        "status": result["status"],
        "submission": result["submission"],
        "checks": result["checks"],
        "protocol": result["protocol"],
        "corpus_hash": result["corpus_hash"],
        "run_id": result["run_id"],
        "artifact": str(output),
    }

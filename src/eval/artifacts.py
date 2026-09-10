"""Stable experiment identities and version-aware transcript metrics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def outcome_reward(row: dict) -> float:
    """Use recorded outcome; preserve old artifacts without inventing new weights."""
    if "outcome_reward" in row:
        return row["outcome_reward"]
    return row["reward"] - row.get("efficiency_bonus", 0.0)


def comparison_key(row: dict, source: str = "legacy") -> str:
    # Even unlabeled historical GRPO files stay separate. Labels are descriptive;
    # run IDs, not labels alone, establish identity.
    label = row.get("run_label") or row["policy"]
    return f"{label} [{row.get('run_id') or source}]"


def content_hash(path: Path) -> str:
    """Hash file contents and relative names, including all corpus JSON files."""
    digest = hashlib.sha256()
    files = sorted(path.rglob("*.json")) if path.is_dir() else [path]
    for item in files:
        name = item.relative_to(path).as_posix() if path.is_dir() else item.name
        digest.update(name.encode() + b"\0")
        with item.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def configuration_hash(configuration: dict) -> str:
    return hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()

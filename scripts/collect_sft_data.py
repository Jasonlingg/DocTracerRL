"""Collect SFT training data from high-reward Claude trajectories.

Reads one or more run_*.json transcripts, filters to claude_policy episodes
with corrected reward >= threshold, and formats each as a Qwen chat-template
conversation (system + alternating user/assistant turns).

Output: JSONL where each line is {"messages": [...]} ready for trl.SFTTrainer.

Usage:
  python scripts/collect_sft_data.py out/run_train_*.json --out data/sft/qwen_traj_smoke.jsonl
  python scripts/collect_sft_data.py out/run_train_*.json --out data/sft/qwen_traj_full.jsonl --min-reward 0.5
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

console = Console()

SYSTEM_PROMPT = """You are an agent exploring a document corpus via Python code.

Tools (already imported):
  search(query, top_k=5)         → [{"doc_id", "title", "chunk", "score"}]
  search(query, method="chunk")  → chunk-level search for buried facts
  read(doc_id)                   → full document text
  extract(doc_id, regex)         → regex matches from a doc
  search_within(doc_id, query)   → relevant windows inside a specific doc
  verify(doc_id, claim)          → {"found", "match_ratio", "excerpt"}
  list_docs()                    → [{"doc_id", "title", "chars"}]

Each turn: write Python code OR a SUBMIT line. Never both. Never prose. Never markdown.
Variables persist across turns. Use print() to see output.

When you have the answer:
SUBMIT: <your answer> CITATIONS: ["doc_id_1", "doc_id_2"]
"""


def _corrected_reward(r: dict) -> float:
    return 0.5 * r["answer_score"] + 0.25 * r["citation_precision"] + 0.25 * r["citation_recall"]


def _format_conversation(episode: dict) -> dict | None:
    """Convert one episode record into a chat-template messages list.

    Returns None if the trajectory has no SUBMIT action (incomplete episode).
    """
    trajectory = episode.get("trajectory", [])
    if not trajectory:
        return None

    # Verify the last action is a SUBMIT — skip episodes that timed out
    last_action = trajectory[-1]["action"].strip()
    if not last_action.upper().startswith("SUBMIT:"):
        return None

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {episode['question']}"},
    ]

    for step in trajectory:
        action = step["action"].strip()
        observation = step["observation"].strip()

        messages.append({"role": "assistant", "content": action})

        # Don't append a user turn after the SUBMIT — the episode ends there.
        if action.upper().startswith("SUBMIT:"):
            break

        messages.append({"role": "user", "content": observation})

    return {"messages": messages}


def main(
    run_files: list[Path] = typer.Argument(..., help="One or more out/run_*.json files"),
    out: Path = typer.Option(..., "--out", "-o", help="Output JSONL path"),
    min_reward: float = typer.Option(0.5, "--min-reward", "-r",
        help="Minimum corrected reward to include (default 0.5)"),
    policy: str = typer.Option("claude_policy", "--policy",
        help="Policy name to filter (default claude_policy)"),
) -> None:
    all_episodes: list[dict] = []
    for path in run_files:
        results = json.loads(path.read_text())
        filtered = [r for r in results
                    if r["policy"] == policy and _corrected_reward(r) >= min_reward]
        console.print(f"  {path.name}: {len(filtered)}/{sum(1 for r in results if r['policy'] == policy)} "
                      f"{policy} episodes with reward ≥ {min_reward}")
        all_episodes.extend(filtered)

    conversations = []
    skipped = 0
    for ep in all_episodes:
        conv = _format_conversation(ep)
        if conv is None:
            skipped += 1
        else:
            conversations.append(conv)

    if skipped:
        console.print(f"[yellow]Skipped {skipped} episodes with no SUBMIT action[/yellow]")

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for conv in conversations:
            f.write(json.dumps(conv) + "\n")

    console.print(f"\n[green]Wrote {len(conversations)} conversations to {out}[/green]")

    # Report turn-count distribution
    lengths = [len([m for m in c["messages"] if m["role"] == "assistant"]) for c in conversations]
    if lengths:
        console.print(f"  Avg assistant turns per conversation: {sum(lengths)/len(lengths):.1f}")
        console.print(f"  Total (question, action) pairs: {sum(lengths)}")


if __name__ == "__main__":
    typer.run(main)

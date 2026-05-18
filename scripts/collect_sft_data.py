"""Collect SFT training data from high-reward Claude trajectories.

Reads one or more run_*.json transcripts, filters to claude_policy episodes
with corrected reward >= threshold, and formats each as a Qwen chat-template
conversation (system + alternating user/assistant turns).

Output: JSONL where each line is {"messages": [...]} ready for trl.SFTTrainer.

Usage:
  python scripts/collect_sft_data.py out/run_train_*.json --out data/sft/qwen_traj_full.jsonl
  python scripts/collect_sft_data.py out/run_train_*.json --out data/sft/qwen_traj_full.jsonl --min-reward 0.4
"""

from __future__ import annotations

import json
import re
from pathlib import Path

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

MAX_TOKENS = 8000  # approx token budget; conversations over this are dropped


def _corrected_reward(r: dict) -> float:
    return 0.5 * r["answer_score"] + 0.25 * r["citation_precision"] + 0.25 * r["citation_recall"]


def _has_prose(action: str) -> bool:
    """Return True if the action's first non-empty line looks like English prose."""
    first = action.strip().split("\n")[0].strip()
    if not first:
        return False
    return bool(re.match(
        r"^(I('ll| will| can| need)|Let me|To |Here|Sure|First|Now|Next|Step|The |This )",
        first,
        re.IGNORECASE,
    ))


def _format_conversation(episode: dict) -> dict | None:
    """Convert one episode into a chat-template messages list.

    Returns None if the trajectory has no SUBMIT, contains prose-contaminated
    actions, or exceeds the token budget.
    """
    trajectory = episode.get("trajectory", [])
    if not trajectory:
        return None

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

        # Drop conversations where any non-SUBMIT action contains prose
        if not action.upper().startswith("SUBMIT:") and _has_prose(action):
            return None

        messages.append({"role": "assistant", "content": action})

        if action.upper().startswith("SUBMIT:"):
            break

        messages.append({"role": "user", "content": observation})

    # Drop conversations that exceed the token budget (would be silently truncated)
    approx_tokens = sum(len(m["content"]) for m in messages) // 4
    if approx_tokens > MAX_TOKENS:
        return None

    return {"messages": messages}


def main(
    run_files: list[Path] = typer.Argument(..., help="One or more out/run_*.json files"),
    out: Path = typer.Option(..., "--out", "-o", help="Output JSONL path"),
    min_reward: float = typer.Option(0.5, "--min-reward", "-r",
        help="Minimum corrected reward to include (default 0.5)"),
    policy: str = typer.Option("claude_policy", "--policy",
        help="Policy name to filter (default claude_policy)"),
    train_only: bool = typer.Option(True, "--train-only/--all-splits",
        help="Only include train_ question IDs (exclude dev/test leakage)"),
) -> None:
    # Collect all qualifying episodes, deduplicating by question_id
    # (keep highest-reward episode when the same question appears in multiple runs)
    best_by_qid: dict[str, dict] = {}

    for path in run_files:
        results = json.loads(path.read_text())
        filtered = [
            r for r in results
            if r["policy"] == policy and _corrected_reward(r) >= min_reward
        ]
        if train_only:
            filtered = [r for r in filtered if r["question_id"].startswith("train_")]

        for r in filtered:
            qid = r["question_id"]
            if qid not in best_by_qid or _corrected_reward(r) > _corrected_reward(best_by_qid[qid]):
                best_by_qid[qid] = r

        console.print(
            f"  {path.name}: {len(filtered)}/{sum(1 for r in results if r['policy'] == policy)} "
            f"{policy} episodes with reward ≥ {min_reward}"
            + (f" (train only)" if train_only else "")
        )

    episodes = list(best_by_qid.values())
    console.print(f"\nAfter dedup: {len(episodes)} unique questions")

    # Format and filter
    conversations = []
    dropped = {"no_submit": 0, "prose": 0, "too_long": 0}

    for ep in episodes:
        conv = _format_conversation(ep)
        if conv is None:
            traj = ep.get("trajectory", [])
            last = traj[-1]["action"].strip() if traj else ""
            if not last.upper().startswith("SUBMIT:"):
                dropped["no_submit"] += 1
            elif sum(len(m["content"]) for m in []) // 4 > MAX_TOKENS:
                dropped["too_long"] += 1
            else:
                dropped["prose"] += 1
        else:
            conversations.append((ep["question_id"], conv))

    total_dropped = sum(dropped.values())
    if total_dropped:
        console.print(f"[yellow]Dropped: prose={dropped['prose']} too_long={dropped['too_long']} no_submit={dropped['no_submit']}[/yellow]")

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for _, conv in conversations:
            f.write(json.dumps(conv) + "\n")

    console.print(f"\n[green]Wrote {len(conversations)} conversations to {out}[/green]")

    lengths = [sum(1 for m in c["messages"] if m["role"] == "assistant") for _, c in conversations]
    if lengths:
        console.print(f"  Avg assistant turns: {sum(lengths)/len(lengths):.1f}")
        console.print(f"  Total training pairs: {sum(lengths)}")


if __name__ == "__main__":
    typer.run(main)

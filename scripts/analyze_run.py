"""Deep analysis of an eval run transcript.

Produces four analyses that together tell the full story:
  1. Hop-stratified results  — the core proof: does exploration advantage grow with hop count?
  2. Trajectory length dist  — shows multi-step vs one-shot behaviour per policy
  3. Tool usage breakdown    — which tools each policy calls (exploration quality signal)
  4. Side-by-side examples   — 3 representative questions × all policies

Usage:
  python scripts/analyze_run.py out/run_<timestamp>.json
  python scripts/analyze_run.py out/run_<timestamp>.json --questions data/musique/questions/dev_set.json
  python scripts/analyze_run.py out/run_<timestamp>.json --out notes/analysis.md
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

console = Console()

TOOLS = ["search_within", "search", "read", "extract", "verify", "list_docs"]

POLICY_ORDER = [
    "single_shot", "sparse_rag", "naive_rag", "context_stuffing",
    "claude_policy", "qwen_base_policy", "qwen_sft_policy", "grpo_policy",
]


def _corrected_reward(r: dict) -> float:
    return 0.5 * r["answer_score"] + 0.25 * r["citation_precision"] + 0.25 * r["citation_recall"]


def _count_tools(trajectory: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {t: 0 for t in TOOLS}
    for step in trajectory:
        action = step.get("action", "")
        for tool in TOOLS:
            counts[tool] += len(re.findall(rf"\b{tool}\s*\(", action))
    return counts


def _sort_policies(policies: list[str]) -> list[str]:
    ordered = [p for p in POLICY_ORDER if p in policies]
    remainder = sorted(p for p in policies if p not in POLICY_ORDER)
    return ordered + remainder


# ── 1. Hop-stratified results ──────────────────────────────────────────────

def hop_stratified(results: list[dict], hop_map: dict[str, int]) -> Table:
    by_policy_hop: dict[tuple[str, int], list[float]] = defaultdict(list)
    for r in results:
        h = hop_map.get(r["question_id"])
        if h is None:
            continue
        by_policy_hop[(r["policy"], h)].append(_corrected_reward(r))

    hops = sorted({h for _, h in by_policy_hop})
    policies = _sort_policies(sorted({p for p, _ in by_policy_hop}))

    table = Table(title="Hop-Stratified Results (corrected reward, no efficiency bonus)", show_lines=True)
    table.add_column("Policy", style="bold cyan")
    for h in hops:
        table.add_column(f"{h}-hop", justify="right")
    table.add_column("All", justify="right")

    for policy in policies:
        all_rewards = []
        row = [policy]
        for h in hops:
            vals = by_policy_hop.get((policy, h), [])
            all_rewards.extend(vals)
            row.append(f"{sum(vals)/len(vals):.3f}" if vals else "—")
        row.append(f"{sum(all_rewards)/len(all_rewards):.3f}" if all_rewards else "—")
        table.add_row(*row)

    return table


# ── 2. Trajectory length distribution ─────────────────────────────────────

def traj_length_table(results: list[dict]) -> Table:
    by_policy: dict[str, list[int]] = defaultdict(list)
    for r in results:
        by_policy[r["policy"]].append(r["steps"])

    table = Table(title="Trajectory Length Distribution (steps)", show_lines=True)
    table.add_column("Policy", style="bold cyan")
    table.add_column("Mean", justify="right")
    table.add_column("1", justify="right")
    table.add_column("2-3", justify="right")
    table.add_column("4-6", justify="right")
    table.add_column("7-9", justify="right")
    table.add_column("10 (timeout)", justify="right")

    for policy in _sort_policies(sorted(by_policy)):
        steps = by_policy[policy]
        n = len(steps)
        table.add_row(
            policy,
            f"{sum(steps)/n:.1f}",
            f"{sum(1 for s in steps if s == 1)}",
            f"{sum(1 for s in steps if 2 <= s <= 3)}",
            f"{sum(1 for s in steps if 4 <= s <= 6)}",
            f"{sum(1 for s in steps if 7 <= s <= 9)}",
            f"{sum(1 for s in steps if s == 10)}",
        )

    return table


# ── 3. Tool usage breakdown ────────────────────────────────────────────────

def tool_usage_table(results: list[dict]) -> Table:
    by_policy: dict[str, dict[str, int]] = defaultdict(lambda: {t: 0 for t in TOOLS})
    counts_n: dict[str, int] = defaultdict(int)

    for r in results:
        policy = r["policy"]
        counts_n[policy] += 1
        tool_counts = _count_tools(r.get("trajectory", []))
        for tool, count in tool_counts.items():
            by_policy[policy][tool] += count

    table = Table(title="Tool Usage (total calls across all episodes)", show_lines=True)
    table.add_column("Policy", style="bold cyan")
    for tool in TOOLS:
        table.add_column(tool, justify="right")
    table.add_column("calls/ep", justify="right")

    for policy in _sort_policies(sorted(by_policy)):
        n = counts_n[policy]
        total = sum(by_policy[policy].values())
        row = [policy] + [str(by_policy[policy][t]) for t in TOOLS] + [f"{total/n:.1f}"]
        table.add_row(*row)

    return table


# ── 4. Side-by-side trajectory examples ───────────────────────────────────

def side_by_side(results: list[dict], hop_map: dict[str, int], n: int = 3) -> str:
    by_qid: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_qid[r["question_id"]].append(r)

    # Pick n questions where claude_policy submitted (reward > 0) across all hops
    candidates = []
    for qid, runs in by_qid.items():
        claude = next((r for r in runs if r["policy"] == "claude_policy"), None)
        if claude and _corrected_reward(claude) > 0.3:
            candidates.append((hop_map.get(qid, 0), qid))
    candidates.sort()

    chosen = []
    seen_hops: set[int] = set()
    for h, qid in candidates:
        if h not in seen_hops:
            chosen.append(qid)
            seen_hops.add(h)
        if len(chosen) >= n:
            break
    chosen = chosen or [candidates[0][1]] if candidates else []

    lines = ["# Side-by-Side Trajectory Examples\n"]
    for qid in chosen:
        runs = by_qid[qid]
        q_text = runs[0]["question"]
        h = hop_map.get(qid, "?")
        lines.append(f"## {qid} ({h}-hop)\n**Q:** {q_text}\n")
        for r in sorted(runs, key=lambda x: _sort_policies([x["policy"]]).index(x["policy"])
                        if x["policy"] in _sort_policies([x["policy"]]) else 99):
            cr = _corrected_reward(r)
            lines.append(f"### {r['policy']} — reward={cr:.3f}, steps={r['steps']}")
            lines.append(f"**Answer:** {r['predicted_answer'] or '(no submit)'}\n")
            for step in r.get("trajectory", []):
                lines.append(f"**Step {step['step']} action:**")
                lines.append(f"```python\n{step['action'][:400]}\n```")
                lines.append(f"**Obs:** {step['observation'][:300]}\n")
            lines.append("---\n")
    return "\n".join(lines)


# ── CLI ────────────────────────────────────────────────────────────────────

def main(
    run_file: Path = typer.Argument(..., help="Path to out/run_*.json transcript"),
    questions: Optional[Path] = typer.Option(None, "--questions", "-q",
        help="Questions JSON with hops field (auto-detected if omitted)"),
    out: Optional[Path] = typer.Option(None, "--out", "-o",
        help="Write side-by-side examples to this markdown file"),
) -> None:
    results = json.loads(run_file.read_text())

    # Auto-detect questions file from split name in results
    if questions is None:
        first_id = results[0]["question_id"]
        split = first_id.split("_")[0]
        questions = Path(f"data/musique/questions/{split}_set.json")

    hop_map: dict[str, int] = {}
    if questions.exists():
        for q in json.loads(questions.read_text()):
            hop_map[q["id"]] = q["hops"]
    else:
        console.print(f"[yellow]Warning: {questions} not found — hop analysis skipped[/yellow]")

    console.print(hop_stratified(results, hop_map))
    console.print()
    console.print(traj_length_table(results))
    console.print()
    console.print(tool_usage_table(results))

    examples = side_by_side(results, hop_map)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(examples)
        console.print(f"\n[green]Side-by-side examples written to {out}[/green]")
    else:
        console.print("\n[dim]Pass --out <file.md> to save side-by-side trajectory examples[/dim]")

    console.print(f"\n[bold]Headline numbers (corrected reward, no efficiency bonus):[/bold]")
    by_policy: dict[str, list[float]] = defaultdict(list)
    for r in results:
        by_policy[r["policy"]].append(_corrected_reward(r))
    for policy in _sort_policies(sorted(by_policy)):
        vals = by_policy[policy]
        console.print(f"  {policy}: {sum(vals)/len(vals):.3f}")


if __name__ == "__main__":
    typer.run(main)

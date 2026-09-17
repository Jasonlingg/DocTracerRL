"""Replay every recorded teacher trajectory against a fresh REPL to confirm it is
reproducible before it's trusted as an SFT training target.

For each non-SUBMIT step, re-executes the recorded `action` code in a fresh,
persistent REPL session (one per trajectory, matching how it was originally
generated) and compares the freshly produced stdout to the recorded
`observation`. A mismatch means the trajectory is not a faithful demonstration
of real tool use — the corpus, tool, or code behavior has drifted since
generation — and it must not go into the SFT set unreviewed.

Usage:
  python scripts/replay_teacher_trajectories.py \
      out/research/qasper-teacher-trajectories/final_for_sft.json \
      --corpus out/research/qasper-train-teacher-v1/corpus
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from loguru import logger
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.env.corpus import Corpus
from src.env.document_env import DocumentExplorationEnv

console = Console()


def replay_one(entry: dict, benchmark_question: dict, corpus: Corpus, corpus_path: str) -> dict:
    env = DocumentExplorationEnv(
        corpus=corpus,
        questions=[benchmark_question],
        max_steps=10,  # matches the max_steps used for the original teacher generation run
        use_docker=False,
        corpus_path=corpus_path,
        require_evidence=False,
    )
    env.reset(question_idx=0)
    mismatches = []
    try:
        for step in entry["trajectory"]:
            actual, reward, done, _info = env.step(step["action"])
            if done:
                # Submission: env.step() returns "" as the observation (the
                # descriptive "Submitted. Reward: X" text only lives in the
                # trajectory record) — compare the reward instead.
                if abs(reward - entry["reward"]) > 1e-6:
                    mismatches.append({
                        "step": step["step"],
                        "action": step["action"],
                        "expected": f"reward={entry['reward']}",
                        "actual": f"reward={reward}",
                    })
                continue
            expected = step["observation"].strip()
            if actual.strip() != expected:
                mismatches.append({
                    "step": step["step"],
                    "action": step["action"],
                    "expected": expected[:300],
                    "actual": actual.strip()[:300],
                })
    finally:
        env.close()
    return {
        "question_id": entry["question_id"],
        "steps_replayed": len(entry["trajectory"]),
        "mismatches": mismatches,
        "reproducible": not mismatches,
    }


def main(
    trajectories: Path = typer.Argument(..., help="Trajectory file (list of run records)"),
    corpus_path: str = typer.Option(..., "--corpus", help="Corpus directory used to generate them"),
    benchmark: Path = typer.Option(..., "--benchmark", help="Benchmark file with gold answers/citations"),
    output: Path = typer.Option(None, "--output", help="Where to write the replay report"),
) -> None:
    entries = json.loads(trajectories.read_text())
    questions_by_id = {q["id"]: q for q in json.loads(benchmark.read_text())["questions"]}
    console.print(f"Replaying {len(entries)} trajectories against {corpus_path}...")

    corpus = Corpus(corpus_path=corpus_path)
    corpus.load(build_index=False)

    reports = []
    for entry in entries:
        report = replay_one(entry, questions_by_id[entry["question_id"]], corpus, corpus_path)
        reports.append(report)
        status = "[green]OK[/green]" if report["reproducible"] else "[red]MISMATCH[/red]"
        console.print(f"  {entry['question_id']}: {status} ({report['steps_replayed']} steps)")

    bad = [r for r in reports if not r["reproducible"]]
    console.print(f"\n{len(entries) - len(bad)}/{len(entries)} trajectories reproduced exactly.")
    if bad:
        console.print(f"[red]{len(bad)} trajectories had at least one mismatch — do not use "
                       "for SFT until reviewed.[/red]")
        for r in bad:
            console.print(f"  {r['question_id']}: {len(r['mismatches'])} mismatched step(s)")

    if output:
        output.write_text(json.dumps(reports, indent=2))
        console.print(f"Report written to {output}")


if __name__ == "__main__":
    typer.run(main)

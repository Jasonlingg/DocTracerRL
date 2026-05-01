"""CLI for running evaluation: policies through the document exploration environment."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import typer
from loguru import logger
from rich.console import Console

from src.env.corpus import Corpus
from src.eval.harness import run_eval
from src.eval.report import print_results
from src.policies.claude_policy import ClaudePolicy
from src.policies.naive_rag import NaiveRAGPolicy
from src.policies.single_shot import SingleShotPolicy
from src.policies.sparse_rag import SparseRAGPolicy
from src.policies.stuffing import ContextStuffingPolicy

app = typer.Typer(help="RLM Explorer Evaluation CLI")
console = Console()


def load_questions(path: str = "data/questions/eval_set.json") -> list[dict]:
    with open(path) as f:
        return json.load(f)


def build_policies(
    corpus: Corpus,
    policy_names: list[str] | None = None,
) -> dict[str, object]:
    """Build policy instances. If policy_names is None, build all."""
    all_policies = {
        "claude_policy": lambda: ClaudePolicy(),
        "naive_rag": lambda: NaiveRAGPolicy(corpus=corpus),
        "sparse_rag": lambda: SparseRAGPolicy(corpus=corpus),
        "context_stuffing": lambda: ContextStuffingPolicy(corpus=corpus),
        "single_shot": lambda: SingleShotPolicy(corpus=corpus),
    }

    if policy_names:
        return {
            name: factory()
            for name, factory in all_policies.items()
            if name in policy_names
        }
    return {name: factory() for name, factory in all_policies.items()}


@app.command()
def main(
    policy: str | None = typer.Option(
        None, "--policy", "-p", help="Run only this policy (e.g. claude_policy)"
    ),
    question: str | None = typer.Option(
        None, "--question", "-q", help="Run only this question ID (e.g. q01)"
    ),
    hard: bool = typer.Option(
        False, "--hard", help="Use hard multi-hop question set"
    ),
    musique: bool = typer.Option(
        False, "--musique", help="Use MuSiQue corpus + questions (run setup_musique.py first)"
    ),
    split: str = typer.Option(
        "eval", "--split",
        help="Question split (MuSiQue only): eval | train | dev | test",
    ),
    test: int = typer.Option(
        0, "--test", "-t", help="Test mode: only run first N questions (e.g. -t 5)"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Print full trajectories"
    ),
    max_steps: int = typer.Option(
        10, "--max-steps", help="Max steps per episode"
    ),
    questions_path: str = typer.Option(
        "data/questions/eval_set.json", "--questions", help="Path to questions JSON"
    ),
    corpus_path: str = typer.Option(
        "data/corpus", "--corpus", help="Path to corpus directory"
    ),
) -> None:
    """Run evaluation: policies through the document exploration environment."""
    console.print("[bold]RLM Explorer — Evaluation[/bold]\n")

    # MuSiQue overrides corpus + questions paths
    if musique:
        corpus_path = "data/musique/corpus"
        questions_path = f"data/musique/questions/{split}_set.json"
        console.print(
            f"[bold cyan]Using MuSiQue corpus + {split} questions[/bold cyan]"
        )

    # Load corpus
    console.print("Loading corpus...")
    corpus = Corpus(corpus_path=corpus_path)
    corpus.load()

    # Load questions
    if hard and not musique:
        questions_path = "data/questions/hard_eval_set.json"
        console.print("[bold magenta]Using HARD multi-hop question set[/bold magenta]")
    questions = load_questions(questions_path)
    if test > 0:
        questions = questions[:test]
        console.print(f"[yellow]Test mode: using first {test} questions[/yellow]")
    console.print(f"Loaded {len(questions)} questions\n")

    # Build policies
    policy_names = [policy] if policy else None
    policies = build_policies(corpus, policy_names)
    console.print(f"Policies: {', '.join(policies.keys())}\n")

    # Filter questions
    question_ids = [question] if question else None

    # Run evaluation
    results = run_eval(
        corpus=corpus,
        questions=questions,
        policies=policies,
        max_steps=max_steps,
        use_docker=None,  # auto-detect: use Docker if rlm-sandbox image exists
        corpus_path=corpus_path,
        question_ids=question_ids,
    )

    # Always print and save, even on partial results
    console.print()
    if results:
        print_results(results, verbose=verbose)
    save_transcripts(results)


def save_transcripts(results: list) -> None:
    """Save eval results and trajectories as JSON transcripts to out/."""
    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"run_{timestamp}.json"

    transcripts = []
    for r in results:
        transcripts.append({
            "question_id": r.question_id,
            "question": r.question,
            "policy": r.policy_name,
            "reward": r.reward,
            "answer_score": r.answer_score,
            "citation_precision": r.citation_precision,
            "citation_recall": r.citation_recall,
            "efficiency_bonus": r.efficiency_bonus,
            "steps": r.steps,
            "duration_seconds": r.duration_seconds,
            "predicted_answer": r.predicted_answer,
            "predicted_citations": r.predicted_citations,
            "trajectory": [
                {
                    "step": s.step,
                    "action": s.action,
                    "observation": s.observation,
                    "reward": s.reward,
                    "done": s.done,
                }
                for s in r.trajectory
            ],
        })

    with open(out_path, "w") as f:
        json.dump(transcripts, f, indent=2)

    console.print(f"\n[bold green]Transcript saved to {out_path}[/bold green]")


if __name__ == "__main__":
    app()

"""CLI for running evaluation: policies through the document exploration environment."""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import typer
from dotenv import load_dotenv
from rich.console import Console

from src.env.corpus import Corpus
from src.env.reward import REWARD_VERSION, REWARD_WEIGHTS
from src.eval.artifacts import configuration_hash, content_hash
from src.eval.harness import run_eval
from src.eval.report import print_results
from src.policies.claude_policy import ClaudePolicy
from src.policies.grpo_policy import GRPOPolicy
from src.policies.naive_rag import NaiveRAGPolicy
from src.policies.qwen_base_policy import QwenBasePolicy
from src.policies.qwen_sft_policy import QwenSFTPolicy
from src.policies.single_shot import SingleShotPolicy
from src.policies.sparse_rag import SparseRAGPolicy
from src.policies.stuffing import ContextStuffingPolicy

# An exported-but-EMPTY key shadows .env: load_dotenv() defaults to
# override=False and treats "" as already-set, so the blank value wins and
# every downstream key check fails with a confusing "missing API key".
for _k in ("ANTHROPIC_API_KEY", "DEMO_API_KEY"):
    if os.environ.get(_k, None) == "":
        del os.environ[_k]
load_dotenv()

app = typer.Typer(help="RLM Explorer Evaluation CLI")
console = Console()


def load_questions(path: str = "data/questions/eval_set.json") -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and isinstance(data.get("questions"), list):
        return data["questions"]
    if not isinstance(data, list):
        raise ValueError("Question file must be a list or contain a questions list")
    return data


def build_policies(
    corpus: Corpus,
    policy_names: list[str] | None = None,
    as_factories: bool = False,
) -> dict[str, object]:
    """Build policy instances (or factories when as_factories=True).

    Pass as_factories=True when using workers > 1 so each worker thread
    creates its own fresh instance with no shared mutable state.
    """
    all_policies = {
        "claude_policy": lambda: ClaudePolicy(),
        "naive_rag": lambda: NaiveRAGPolicy(corpus=corpus),
        "sparse_rag": lambda: SparseRAGPolicy(corpus=corpus),
        "context_stuffing": lambda: ContextStuffingPolicy(corpus=corpus),
        "single_shot": lambda: SingleShotPolicy(corpus=corpus),
        "qwen_base_policy": lambda: QwenBasePolicy(),
        "qwen_sft_policy": lambda: QwenSFTPolicy(),
        "grpo_policy": lambda: GRPOPolicy(),
    }

    selected = {
        name: factory
        for name, factory in all_policies.items()
        if policy_names is None or name in policy_names
    }
    if as_factories:
        return selected
    return {name: factory() for name, factory in selected.items()}


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
    workers: int = typer.Option(
        1, "--workers", "-w", help="Parallel workers (default 1). Set 4-8 for fast data collection."
    ),
    output: Path | None = typer.Option(None, "--output", help="Exact transcript output path"),
    run_label: str | None = typer.Option(None, "--run-label", help="Checkpoint comparison label"),
    seed: int = typer.Option(42, "--seed"),
    require_evidence: bool = typer.Option(
        False, "--require-evidence", help="Ask for exact source spans in submissions"
    ),
    no_vector_index: bool = typer.Option(
        False, "--no-vector-index", help="Skip unused FAISS index for code-execution policies"
    ),
) -> None:
    """Run evaluation: policies through the document exploration environment."""
    console.print("[bold]RLM Explorer — Evaluation[/bold]\n")
    if output is not None and (output.exists() or output.with_suffix(".manifest.json").exists()):
        raise typer.BadParameter(f"Output already exists: {output}")
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    # Seed optional local torch inference as well as Python/NumPy.
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None:
        torch.manual_seed(seed)

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
    if no_vector_index:
        corpus.load(build_index=False)
    else:
        corpus.load()

    # Load questions
    if hard and not musique:
        questions_path = "data/questions/hard_eval_set.json"
        console.print("[bold magenta]Using HARD multi-hop question set[/bold magenta]")
    questions = load_questions(questions_path)
    if test > 0:
        questions = questions[:test]
        console.print(f"[yellow]Test mode: using first {test} questions[/yellow]")
    if question:
        questions = [q for q in questions if q["id"] == question]
    if not questions:
        raise typer.BadParameter("No questions selected")
    console.print(f"Loaded {len(questions)} questions\n")

    # Build policies (factories when parallel so each worker gets a fresh instance)
    policy_names = [policy] if policy else None
    policies = build_policies(corpus, policy_names, as_factories=workers > 1)
    if not policies:
        raise typer.BadParameter(f"Unknown policy: {policy}")
    console.print(f"Policies: {', '.join(policies.keys())}\n")

    # Filter questions
    question_ids = [question] if question else None

    # Run evaluation
    results = run_eval(
        corpus=corpus,
        questions=questions,
        policies=policies,
        max_steps=max_steps,
        use_docker=None,
        corpus_path=corpus_path,
        question_ids=question_ids,
        workers=workers,
        require_evidence=require_evidence,
    )

    # Always print and save, even on partial results
    console.print()
    if results:
        print_results(results, verbose=verbose)
    protocol = {
        "question_ids": [q["id"] for q in questions],
        "questions_sha256": content_hash(Path(questions_path)),
        "corpus_sha256": content_hash(Path(corpus_path)),
        "max_steps": max_steps, "seed": seed, "reward_version": REWARD_VERSION,
        "workers": workers, "require_evidence": require_evidence,
        "vector_index": not no_vector_index,
        "decoding": sorted({
            json.dumps({"max_tokens": getattr(p, "_max_tokens", None),
                        "temperature": getattr(p, "_temperature", None)}, sort_keys=True)
            for p in policies.values()
        }),
    }
    manifest = {
        **protocol,
        "comparison_id": configuration_hash(protocol),
        "split": split if musique else questions_path,
        "checkpoint_id": os.environ.get("CHECKPOINT_PATH"),
        "base_model": os.environ.get("BASE_MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct")
        if policy in {"qwen_base_policy", "qwen_sft_policy", "grpo_policy"} else None,
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
        ).stdout.strip(),
        "git_status": subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True,
        ).stdout.splitlines(),
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__ if torch is not None else None,
        "gpu": (
            torch.cuda.get_device_name()
            if torch is not None and torch.cuda.is_available()
            else None
        ),
        "workers": workers,
        "policy_settings": {
            name: {"max_tokens": getattr(p, "_max_tokens", None),
                   "temperature": getattr(p, "_temperature", None),
                   "base_revision": getattr(getattr(getattr(p, "_model", None),
                                                    "config", None), "_commit_hash", None)}
            for name, p in policies.items()
        },
    }
    save_transcripts(results, output=output, run_label=run_label, manifest=manifest)
    if len(results) != len(questions) * len(policies) or any(r.status == "error" for r in results):
        raise typer.Exit(1)


def save_transcripts(
    results: list, output: Path | None = None, run_label: str | None = None,
    manifest: dict | None = None,
) -> Path:
    """Save eval results and trajectories as JSON transcripts to out/."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{uuid4().hex[:12]}"
    out_path = output or Path("out") / f"run_{run_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {**(manifest or {}), "run_id": run_id, "run_label": run_label,
                "reward_version": REWARD_VERSION, "reward_weights": REWARD_WEIGHTS}

    transcripts = []
    for r in results:
        transcripts.append({
            "question_id": r.question_id,
            "question": r.question,
            "policy": r.policy_name,
            "run_id": run_id,
            "run_label": run_label or r.policy_name,
            "checkpoint_id": manifest.get("checkpoint_id"),
            "comparison_id": manifest.get("comparison_id"),
            "reward": r.reward,
            "outcome_reward": r.outcome_reward,
            "shaping_reward": r.shaping_reward,
            "episode_return": r.episode_return,
            "reward_version": r.reward_version,
            "reward_weights": REWARD_WEIGHTS,
            "status": r.status,
            "error": r.error,
            "answer_score": r.answer_score,
            "citation_precision": r.citation_precision,
            "citation_recall": r.citation_recall,
            "efficiency_bonus": r.efficiency_bonus,
            "steps": r.steps,
            "duration_seconds": r.duration_seconds,
            "predicted_answer": r.predicted_answer,
            "predicted_citations": r.predicted_citations,
            "predicted_evidence": r.predicted_evidence,
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

    with open(out_path, "x") as f:
        json.dump(transcripts, f, indent=2)
    with out_path.with_suffix(".manifest.json").open("x") as f:
        json.dump(manifest, f, indent=2)

    console.print(f"\n[bold green]Transcript saved to {out_path}[/bold green]")
    return out_path


if __name__ == "__main__":
    app()

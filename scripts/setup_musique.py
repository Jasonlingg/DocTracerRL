"""Download MuSiQue dataset and convert to RLM Explorer corpus + question format.

MuSiQue (Multi-hop Questions via Single-hop Question Composition) provides
multi-hop QA over Wikipedia passages. This creates a corpus too large for
context stuffing, forcing the iterative agent to actually explore.

Usage:
    python scripts/setup_musique.py                    # default: 200 questions from validation
    python scripts/setup_musique.py --split train --num-questions 500
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path

import typer
from rich.console import Console

MUSIQUE_DIR = Path(__file__).parent.parent / "data" / "musique"
CORPUS_DIR = MUSIQUE_DIR / "corpus"
QUESTIONS_DIR = MUSIQUE_DIR / "questions"

console = Console()
app = typer.Typer(help="Download and prepare MuSiQue corpus for RLM Explorer")


def slugify(text: str) -> str:
    """Convert text to a filesystem-safe slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s-]+", "_", text)
    return text[:80]


def download_musique(
    split: str = "validation",
    num_questions: int | None = None,
    skip: int = 0,
) -> list[dict]:
    """Download MuSiQue dataset from HuggingFace.

    Slicing is `examples[skip : skip + num_questions]`. Use `skip` to carve
    a held-out test set out of the validation split (HF doesn't expose MuSiQue's
    real test split — it's hidden behind a leaderboard).
    """
    try:
        from datasets import load_dataset
    except ImportError:
        console.print("[red]Install datasets: pip install datasets[/red]")
        raise SystemExit(1)

    console.print(f"Downloading MuSiQue ({split} split)...")
    ds = load_dataset("bdsaglam/musique", split=split)
    examples = list(ds)

    # Filter to answerable questions only
    examples = [ex for ex in examples if ex.get("answerable", True)]
    console.print(f"  {len(examples)} answerable examples found")

    if skip:
        examples = examples[skip:]
        console.print(f"  Skipping first {skip} examples")

    if num_questions and num_questions < len(examples):
        examples = examples[:num_questions]
        console.print(f"  Using next {num_questions} questions")

    return examples


def build_corpus(examples: list[dict]) -> tuple[dict[str, dict], dict[str, str]]:
    """Extract paragraphs from MuSiQue examples, group by article title.

    Returns:
        documents: {doc_id: doc_dict}
        para_key_to_doc_id: {f"{title}||{text}" -> doc_id}
    """
    articles: dict[str, list[str]] = defaultdict(list)
    seen_paras: set[str] = set()
    para_key_to_doc_id: dict[str, str] = {}

    for ex in examples:
        for para in ex["paragraphs"]:
            title = para["title"]
            text = para["paragraph_text"]
            key = f"{title}||{text}"

            if key not in seen_paras:
                seen_paras.add(key)
                articles[title].append(text)

            doc_id = f"musique_{slugify(title)}"
            para_key_to_doc_id[key] = doc_id

    # Build document objects
    documents: dict[str, dict] = {}
    for title, paragraphs in articles.items():
        doc_id = f"musique_{slugify(title)}"
        full_text = f"{title}\n\n" + "\n\n".join(paragraphs)
        documents[doc_id] = {
            "doc_id": doc_id,
            "title": title,
            "text": full_text,
            "metadata": {
                "type": "wikipedia",
                "source": "musique",
                "paragraphs": len(paragraphs),
            },
        }

    return documents, para_key_to_doc_id


def build_questions(
    examples: list[dict],
    para_key_to_doc_id: dict[str, str],
    id_prefix: str = "m",
) -> list[dict]:
    """Convert MuSiQue examples to RLM Explorer eval format."""
    questions: list[dict] = []

    for i, ex in enumerate(examples):
        if not ex.get("answerable", True):
            continue

        # Map supporting paragraphs to doc_ids
        citations: list[str] = []
        for para in ex["paragraphs"]:
            if para["is_supporting"]:
                key = f"{para['title']}||{para['paragraph_text']}"
                doc_id = para_key_to_doc_id.get(key)
                if doc_id and doc_id not in citations:
                    citations.append(doc_id)

        decomposition = ex.get("question_decomposition", [])
        hops = len(decomposition) if decomposition else 2
        answer_aliases = ex.get("answer_aliases", [])

        questions.append({
            "id": f"{id_prefix}{i:04d}",
            "question": ex["question"],
            "answer": ex["answer"],
            "answer_aliases": answer_aliases,
            "expected_citations": citations,
            "type": f"{hops}hop",
            "difficulty": "hard" if hops >= 3 else "medium",
            "hops": hops,
        })

    return questions


@app.command()
def main(
    split: str = typer.Option("validation", help="HuggingFace dataset split"),
    num_questions: int = typer.Option(200, "--num-questions", "-n", help="Number of questions"),
    skip: int = typer.Option(0, "--skip", help="Skip first N examples (for non-overlapping slices)"),
    output: str = typer.Option(
        "data/musique/questions/eval_set.json",
        "--output", "-o",
        help="Output path for the questions JSON",
    ),
    id_prefix: str = typer.Option(
        "m", "--id-prefix",
        help="Prefix for question IDs (e.g. 'train_', 'dev_', 'test_')",
    ),
) -> None:
    """Download MuSiQue and build corpus + questions for RLM Explorer."""
    examples = download_musique(split=split, num_questions=num_questions, skip=skip)

    console.print("Building corpus from paragraphs...")
    documents, para_map = build_corpus(examples)

    console.print("Building question set...")
    questions = build_questions(examples, para_map, id_prefix=id_prefix)

    # Write corpus (additive — preserves articles from prior splits)
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    for doc in documents.values():
        path = CORPUS_DIR / f"{doc['doc_id']}.json"
        with open(path, "w") as f:
            json.dump(doc, f, indent=2)

    # Write questions to caller-specified path
    q_path = Path(output)
    q_path.parent.mkdir(parents=True, exist_ok=True)
    with open(q_path, "w") as f:
        json.dump(questions, f, indent=2)

    # Stats
    total_chars = sum(len(d["text"]) for d in documents.values())
    console.print(f"\n[bold green]MuSiQue corpus ready:[/bold green]")
    console.print(f"  Documents: {len(documents)} (in {CORPUS_DIR})")
    console.print(f"  Total chars: {total_chars:,} (~{total_chars // 4:,} tokens)")
    console.print(f"  Questions: {len(questions)} (in {q_path})")
    console.print(f"\n  Context stuffing limit: ~200K tokens")
    console.print(f"  Corpus size: ~{total_chars // 4:,} tokens")
    if total_chars // 4 > 200_000:
        console.print("  [bold green]✓ Corpus exceeds context window — stuffing will fail[/bold green]")
    else:
        console.print("  [yellow]⚠ Corpus may still fit in context. Try --num-questions 500+[/yellow]")


if __name__ == "__main__":
    app()

"""Validate, prepare human review, and score research-agent benchmark runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research.agent import load_snapshot  # noqa: E402
from src.research.benchmark import (  # noqa: E402
    load_benchmark,
    make_review_template,
    score_benchmark,
    score_markdown,
)


def _write_new(path: Path, value: dict) -> None:
    if path.suffix != ".json":
        raise ValueError("output must end in .json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="Check benchmark and snapshot identity")
    validate.add_argument("--benchmark", type=Path, required=True)
    validate.add_argument("--snapshot", type=Path, required=True)
    review = commands.add_parser("review-template", help="Create a human-review JSON form")
    review.add_argument("--benchmark", type=Path, required=True)
    review.add_argument("--runs", type=Path, required=True)
    review.add_argument("--output", type=Path, required=True)
    score = commands.add_parser("score", help="Score a directory of run artifacts")
    score.add_argument("--benchmark", type=Path, required=True)
    score.add_argument("--runs", type=Path, required=True)
    score.add_argument("--reviews", type=Path)
    score.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "validate":
            manifest, _ = load_snapshot(args.snapshot)
            benchmark = load_benchmark(args.benchmark, manifest)
            print(
                f"Valid: {benchmark['benchmark_id']} reserves "
                f"{len(benchmark['reserved_doc_ids'])} papers and has "
                f"{len(benchmark['questions'])} questions"
            )
            return 0
        benchmark = load_benchmark(args.benchmark)
        if args.command == "review-template":
            template = make_review_template(benchmark, args.runs)
            _write_new(args.output, template)
            print(f"Review template: {args.output}")
            return 0
        reviews = json.loads(args.reviews.read_text()) if args.reviews else None
        result = score_benchmark(benchmark, args.runs, reviews)
        _write_new(args.output, result)
        args.output.with_suffix(".md").write_text(score_markdown(result))
        print(f"Score: {args.output}; report: {args.output.with_suffix('.md')}")
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

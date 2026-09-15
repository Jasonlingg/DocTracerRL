"""Prepare and score a blind base-versus-trained AI-paper evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.research_review import (  # noqa: E402
    prepare_review,
    review_markdown,
    score_markdown,
    score_review,
)


def write_json(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--questions", type=Path, required=True)
    prepare.add_argument("--corpus", type=Path, required=True)
    prepare.add_argument("--run", type=Path, action="append", required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--seed", type=int, default=42)
    score = commands.add_parser("score")
    score.add_argument("--review", type=Path, required=True)
    score.add_argument("--key", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            args.output.mkdir(parents=True, exist_ok=False)
            review, key, automatic = prepare_review(
                args.questions, args.corpus, args.run, args.seed
            )
            write_json(args.output / "review.json", review)
            write_json(args.output / "blind-key.json", key)
            write_json(args.output / "automatic.json", automatic)
            (args.output / "review.md").write_text(review_markdown(review))
            print(f"Review bundle: {args.output}")
            return 0
        review = json.loads(args.review.read_text())
        key = json.loads(args.key.read_text())
        result = score_review(review, key)
        write_json(args.output, result)
        args.output.with_suffix(".md").write_text(score_markdown(result))
        print(f"Human score: {args.output}")
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

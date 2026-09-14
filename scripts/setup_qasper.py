"""Download a pinned QASPER split and build a research-agent snapshot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research.qasper import (  # noqa: E402
    QASPER_CONFIG,
    QASPER_DATASET,
    QASPER_REVISION,
    build_qasper_snapshot,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="train")
    parser.add_argument("--num-questions", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--revision", default=QASPER_REVISION)
    parser.add_argument(
        "--output", type=Path, required=True, help="New immutable snapshot directory"
    )
    args = parser.parse_args()
    try:
        from datasets import load_dataset
    except ImportError as exc:
        parser.error("Install the QASPER dependency with: pip install -e '.[qasper]'")
        raise AssertionError from exc

    rows = load_dataset(
        QASPER_DATASET,
        QASPER_CONFIG,
        revision=args.revision,
        split=args.split,
    )
    manifest, questions = build_qasper_snapshot(
        rows,
        args.output,
        source_split=args.split,
        revision=args.revision,
        num_questions=args.num_questions,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "snapshot": str(args.output),
                "paper_count": manifest["paper_count"],
                "source_question_count": manifest["source_question_count"],
                "selected_question_count": len(questions["questions"]),
                "corpus_hash": manifest["corpus_hash"],
                "questions": str(args.output / "questions.json"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

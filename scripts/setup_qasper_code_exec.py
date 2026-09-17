"""Download a pinned QASPER split and build a code-execution-protocol benchmark.

Unlike setup_qasper.py (which targets the paused JSON-action protocol in
src/research/), this produces a benchmark.json + corpus that
scripts/run_eval.py and scripts/research_benchmark.py can run and validate
directly, using the same search()/read()/passage() tools as the MuSiQue and
AI-paper pilots.
"""

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
    build_qasper_code_exec_benchmark,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--num-questions", type=int, default=20)
    parser.add_argument("--min-insufficient", type=int, default=None,
                         help="Oversample this many 'insufficient' (unanswerable) questions "
                              "above QASPER's natural ~16%% rate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--revision", default=QASPER_REVISION)
    parser.add_argument(
        "--output", type=Path, required=True, help="New immutable benchmark directory"
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
    manifest, benchmark = build_qasper_code_exec_benchmark(
        rows,
        args.output,
        source_split=args.split,
        revision=args.revision,
        num_questions=args.num_questions,
        seed=args.seed,
        min_insufficient=args.min_insufficient,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "paper_count": manifest["paper_count"],
                "selected_question_count": len(benchmark["questions"]),
                "corpus_hash": manifest["corpus_hash"],
                "benchmark": str(args.output / "benchmark.json"),
                "corpus": str(args.output / "corpus"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

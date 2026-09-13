"""Run a reproducible CPU-only retrieval check over development questions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research.retrieval_eval import evaluate_retrieval  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.suffix != ".json":
            raise ValueError("--output must end in .json")
        if args.output.exists():
            raise FileExistsError(args.output)
        result = evaluate_retrieval(args.snapshot, args.questions, top_k=args.top_k)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        metrics = result["metrics"]
        print(
            f"Checked {result['question_count']} questions: "
            f"target recall={metrics['target_document_recall']}, "
            f"all targets found={metrics['all_target_documents_found_rate']}"
        )
        print(f"Artifact: {args.output}")
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


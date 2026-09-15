"""Export replay-verified QASPER train trajectories for the SFT pilot."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research.qasper_demonstrations import build_qasper_demonstrations  # noqa: E402
from src.research.reranker import (  # noqa: E402
    MS_MARCO_MINILM_MODEL,
    MS_MARCO_MINILM_REVISION,
    CrossEncoderReranker,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--model", default=MS_MARCO_MINILM_MODEL)
    parser.add_argument("--revision", default=MS_MARCO_MINILM_REVISION)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    reranker = CrossEncoderReranker(
        model_id=args.model,
        revision=args.revision,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    report = build_qasper_demonstrations(
        args.questions,
        args.snapshot,
        args.output,
        paper_reranker=reranker,
        top_k=args.top_k,
        max_steps=args.max_steps,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

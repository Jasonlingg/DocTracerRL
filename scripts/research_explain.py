"""Ask a larger model to explain a Qwen evidence packet."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research.explainer import EndpointExplainer, run_explanation  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retriever-run", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--server-hardware", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=2400)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    explainer = EndpointExplainer(
        args.endpoint, args.model, args.revision, args.seed, args.max_tokens
    )
    result = run_explanation(
        args.retriever_run, explainer, args.output, args.server_hardware
    )
    print(f"Status: {result['status']}; review: {args.output.with_suffix('.md')}")
    return 0 if result["status"] == "submitted" else 1


if __name__ == "__main__":
    raise SystemExit(main())

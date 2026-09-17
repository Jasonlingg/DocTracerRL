"""Score a base-versus-SFT abstention evaluation using the pre-registered rule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.eval.abstention import compare_abstention_runs


def _load_json(path: Path) -> dict | list:
    with path.open() as stream:
        return json.load(stream)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    benchmark = _load_json(args.benchmark)
    questions = benchmark["questions"] if isinstance(benchmark, dict) else benchmark
    result = compare_abstention_runs(
        _load_json(args.base),
        _load_json(args.candidate),
        questions,
    )
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Score the fixed known-paper QASPER smoke run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research.qasper_smoke import score_qasper_smoke  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("data/research/qasper_smoke_v1.json"))
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    score = score_qasper_smoke(args.plan, args.questions, args.runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(score, indent=2) + "\n")
    print(json.dumps({"automatic": score["automatic"], "decision": score["decision"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

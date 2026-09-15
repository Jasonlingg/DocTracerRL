"""Validate the locked QASPER evaluation against its snapshots and questions."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research.qasper_evaluation import validate_qasper_evaluation_plan  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    args = parser.parse_args()
    result = validate_qasper_evaluation_plan(
        json.loads(args.plan.read_text()),
        json.loads(args.questions.read_text()),
        json.loads(args.evaluation_manifest.read_text()),
        json.loads(args.training_manifest.read_text()),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

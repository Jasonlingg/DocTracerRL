"""Build the small authored research batch using real, frozen tool observations."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research.demonstrations import build_demonstrations  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, default=Path("data/research/demonstrations_v1.json"))
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, action="append", required=True,
                        help="Repeat for every reserved evaluation manifest")
    parser.add_argument("--output", type=Path, required=True, help="New artifact directory")
    args = parser.parse_args()
    report = build_demonstrations(args.batch, args.snapshot, args.benchmark, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

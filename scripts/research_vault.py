"""Import an Obsidian folder and export cited explanation notes without a plugin."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research.vault import export_explanation, import_vault  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("import")
    ingest.add_argument("--vault", type=Path, required=True)
    ingest.add_argument("--collection", required=True, help="Relative folder inside the vault")
    ingest.add_argument("--output", type=Path, required=True, help="New snapshot outside the vault")
    export = commands.add_parser("export")
    export.add_argument("--vault", type=Path, required=True)
    export.add_argument("--snapshot", type=Path, required=True)
    export.add_argument("--explanation", type=Path, required=True)
    export.add_argument("--note", required=True, help="New relative Markdown path in the vault")
    args = parser.parse_args()
    try:
        if args.command == "import":
            result = import_vault(args.vault, args.collection, args.output)
            print(json.dumps({key: result[key] for key in [
                "status", "corpus_hash", "failures", "skipped"
            ]}, indent=2))
            print(f"Imported {len(result['papers'])} documents into {args.output}")
            return 0 if result["status"] == "complete" else 1
        print(export_explanation(args.explanation, args.snapshot, args.vault, args.note))
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

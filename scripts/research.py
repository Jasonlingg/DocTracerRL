"""Research prototype: discover, snapshot, explore, and inspect evidence."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Support both `python scripts/research.py` and `python -m scripts.research`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research.agent import EndpointPolicy, load_snapshot, run_question  # noqa: E402
from src.research.papers import build_snapshot, discover  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("snapshot", help="Download selected versioned arXiv papers")
    build.add_argument("--sources", type=Path, default=Path("data/research/sources.json"))
    build.add_argument("--output", type=Path, required=True, help="New snapshot directory")
    search = commands.add_parser("discover", help="Discover candidates via a dated arXiv search")
    search.add_argument("--query", default='all:"retrieval augmented" OR all:"search agent"')
    search.add_argument("--since", required=True, help="YYYY-MM-DD")
    search.add_argument("--until", default=datetime.now(timezone.utc).date().isoformat())
    search.add_argument("--limit", type=int, default=20)
    search.add_argument("--output", type=Path, required=True)
    inspect = commands.add_parser("inspect", help="Search saved paper passages without a model")
    inspect.add_argument("--snapshot", type=Path, required=True)
    inspect.add_argument("--query", required=True)
    inspect.add_argument("--top-k", type=int, default=3)
    ask = commands.add_parser("ask", help="Run a multi-turn agent against a frozen snapshot")
    ask.add_argument("--snapshot", type=Path, required=True)
    selection = ask.add_mutually_exclusive_group(required=True)
    selection.add_argument("--question", help="Your own project question")
    selection.add_argument("--question-id", help="ID from the draft development set")
    ask.add_argument("--questions", type=Path, default=Path("data/research/questions.json"))
    ask.add_argument("--endpoint", default="http://localhost:8000/v1")
    ask.add_argument("--model", required=True, help="Model identifier served by the endpoint")
    ask.add_argument("--revision", required=True, help="Model commit/hash used by the server")
    ask.add_argument("--server-hardware", required=True, help="GPU model/count and serving dtype")
    ask.add_argument("--seed", type=int, default=42)
    ask.add_argument("--max-steps", type=int, default=10)
    ask.add_argument("--max-tokens", type=int, default=1800)
    ask.add_argument("--structured-output", choices=["none", "json_schema"], default="none",
                     help="Request constrained JSON generation; recorded as a new policy setting")
    ask.add_argument("--output", type=Path, required=True, help="New .json run artifact")
    args = parser.parse_args()
    try:
        if args.command == "inspect":
            from src.research.tools_runtime import ResearchTools

            load_snapshot(args.snapshot)
            tools = ResearchTools(args.snapshot / "corpus")
            print(json.dumps(
                tools.search_papers(args.query, args.top_k), indent=2, ensure_ascii=False
            ))
            return 0
        if args.command == "snapshot":
            sources = json.loads(args.sources.read_text())
            ids = [p["arxiv_id"] for p in sources["papers"]]
            result = build_snapshot(ids, args.output)
            print(json.dumps(result, indent=2))
            return 0 if result["status"] == "complete" else 1
        if args.command == "discover":
            if args.output.exists():
                raise FileExistsError(args.output)
            result = discover(args.query, args.since, args.until, args.limit)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x") as stream:
                json.dump(result, stream, indent=2)
            print(f"Saved {len(result['papers'])} candidate papers to {args.output}")
            return 0
        if args.output.suffix != ".json":
            raise ValueError("--output must end in .json")
        if args.question_id:
            questions = json.loads(args.questions.read_text())["questions"]
            matches = [q for q in questions if q["id"] == args.question_id]
            if not matches:
                raise ValueError(f"Unknown question ID: {args.question_id}")
            question = matches[0]
        else:
            question = {"id": "custom", "question": args.question, "split": "development",
                        "review_status": "unreviewed"}
        policy = EndpointPolicy(
            args.endpoint, args.model, args.revision, args.seed, args.max_tokens,
            structured_output=args.structured_output,
        )
        result = run_question(
            args.snapshot,
            question,
            policy,
            args.output,
            max_steps=args.max_steps,
            server_hardware=args.server_hardware,
        )
        print(f"Status: {result['status']}; review: {args.output.with_suffix('.md')}")
        return 0 if result["status"] == "submitted" else 1
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

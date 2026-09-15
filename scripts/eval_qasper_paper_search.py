"""Evaluate a version-pinned cross encoder on QASPER within-paper evidence retrieval."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research.agent import load_snapshot  # noqa: E402
from src.research.qasper_retrieval import evaluate_qasper_paper_search  # noqa: E402
from src.research.reranker import (  # noqa: E402
    MS_MARCO_MINILM_MODEL,
    MS_MARCO_MINILM_REVISION,
    CrossEncoderReranker,
)
from src.research.tools_runtime import ResearchTools  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--model", default=MS_MARCO_MINILM_MODEL)
    parser.add_argument("--revision", default=MS_MARCO_MINILM_REVISION)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest, _ = load_snapshot(args.snapshot)
    questions = json.loads(args.questions.read_text())
    if questions.get("corpus_hash") != manifest["corpus_hash"]:
        raise ValueError("QASPER question set and snapshot corpus hashes differ")
    reranker = CrossEncoderReranker(
        model_id=args.model,
        revision=args.revision,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    result = evaluate_qasper_paper_search(
        questions, ResearchTools(args.snapshot / "corpus", paper_reranker=reranker)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({
        "gold_evidence_recall_at": result["gold_evidence_recall_at"],
        "decision": result["decision"],
        "output": str(args.output),
    }, indent=2))
    return 0 if result["decision"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

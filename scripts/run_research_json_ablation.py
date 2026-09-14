"""Compare unchanged Qwen with and without schema-constrained decoding on one server."""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research.agent import EndpointPolicy, load_snapshot, run_question  # noqa: E402
from src.research.benchmark import (  # noqa: E402
    load_benchmark,
    make_review_template,
    score_benchmark,
    score_markdown,
)


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--server-hardware", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest, _ = load_snapshot(args.snapshot)
    benchmark = load_benchmark(args.benchmark, manifest)
    args.output.mkdir(parents=True, exist_ok=False)
    timings = []
    modes = ["none", "json_schema"]
    write_json(args.output / "experiment.json", {
        "kind": "paired harness ablation; no training",
        "hypothesis": "Constrained decoding reduces malformed actions without reducing answer quality",
        "decision_rule": "Adopt only with no schema violations and no reviewed quality regression",
        "order": "Alternate which mode runs first for successive questions",
        "seed": 42, "max_steps": 10, "max_tokens": 1800,
        "model": args.model, "revision": args.revision,
        "server_hardware": args.server_hardware,
        "corpus_hash": manifest["corpus_hash"],
        "question_ids": [q["id"] for q in benchmark["questions"]],
    })
    try:
        probe = EndpointPolicy(args.endpoint, args.model, args.revision,
                               structured_output="json_schema")
        action = probe.act("Connection test: list available papers using the papers action.")
        write_json(args.output / "schema-probe.json", {"action": action, "policy": probe.config})
        for index, question in enumerate(benchmark["questions"]):
            for mode in modes if index % 2 == 0 else list(reversed(modes)):
                policy = EndpointPolicy(args.endpoint, args.model, args.revision,
                                        structured_output=mode)
                started = time.monotonic()
                result = run_question(
                    args.snapshot, question, policy,
                    args.output / mode / "runs" / f"{question['id']}.json",
                    max_steps=10, server_hardware=args.server_hardware,
                )
                timings.append({"question_id": question["id"], "mode": mode,
                                "seconds": round(time.monotonic() - started, 3),
                                "status": result["status"]})
                print(json.dumps(timings[-1]), flush=True)
                # Truncation is a budget failure of this question; retain it and continue.
                # Transport, server, or schema failures stop the comparison for diagnosis.
                if result["status"] == "error" and "Model output truncated" not in result["error"]:
                    raise RuntimeError(result["error"])
    finally:
        write_json(args.output / "timings.json", timings)
        for mode in modes:
            directory = args.output / mode
            directory.mkdir(exist_ok=True)
            score = score_benchmark(benchmark, directory / "runs")
            write_json(directory / "automatic-score.json", score)
            (directory / "automatic-score.md").write_text(score_markdown(score))
            write_json(directory / "review-template.json",
                       make_review_template(benchmark, directory / "runs"))


if __name__ == "__main__":
    main()

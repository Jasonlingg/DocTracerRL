"""Replay authored demonstrations against a real snapshot and export reviewable conversations."""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.eval.artifacts import configuration_hash, content_hash
from src.research.action_schema import validate_structured_action
from src.research.agent import SYSTEM_PROMPT, load_snapshot, run_question


def paper_family(doc_id: str) -> str:
    """Keep versions of one arXiv paper in the same split."""
    return re.sub(r"v\d+$", "", doc_id)


def _ngrams(text: str) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", text.casefold())
    return {tuple(words[i:i + 4]) for i in range(max(0, len(words) - 3))}


def validate_batch(batch: dict, manifest: dict, benchmarks: list[dict]) -> None:
    if batch.get("schema_version") != "research-demonstrations-v1":
        raise ValueError("Unsupported demonstration schema")
    if batch.get("corpus_hash") != manifest["corpus_hash"]:
        raise ValueError("Demonstration corpus hash mismatch")
    if not benchmarks:
        raise ValueError("At least one benchmark exclusion manifest is required")
    reserved = {paper_family(doc) for b in benchmarks for doc in b["reserved_doc_ids"]}
    present = {paper_family(p["doc_id"]) for p in manifest["papers"]}
    if reserved & present:
        raise ValueError(f"Reserved paper families in demonstration corpus: {reserved & present}")
    held_out = [q for b in benchmarks for q in b["questions"]]
    examples = batch.get("examples", [])
    ids = [e["id"] for e in examples]
    if not examples or len(ids) != len(set(ids)):
        raise ValueError("Examples must have unique IDs and the batch must not be empty")
    for example in examples:
        if example.get("split") != "development":
            raise ValueError("Demonstrations must be development-only")
        if example.get("exclude_from_training"):
            raise ValueError("An explicitly excluded question cannot become a demonstration")
        for question in held_out:
            left, right = _ngrams(example["question"]), _ngrams(question["question"])
            overlap = len(left & right) / max(1, min(len(left), len(right)))
            if (example["id"] == question["id"]
                    or example["question"].strip().casefold()
                    == question["question"].strip().casefold() or overlap >= 0.35):
                raise ValueError(f"Possible benchmark question reuse: {example['id']}")
        actions = example["actions"]
        if not 3 <= len(actions) <= batch["max_steps"]:
            raise ValueError("Demonstrations need a bounded multi-step trajectory")
        for action in actions:
            validate_structured_action(json.dumps(action))
        if (actions[-1]["action"] != "submit"
                or any(a["action"] == "submit" for a in actions[:-1])):
            raise ValueError("Exactly one final submission is required")
        review = example["review"]
        if (review.get("status") != "complete" or not review.get("reviewer_type")
                or not review.get("answer_complete")
                or not review.get("recommendation_faithful") or not review.get("notes")):
            raise ValueError("An explicit, complete review is required")
        claims = actions[-1]["answer"]["claims"]
        labels = review["claim_reviews"]
        if (len(labels) != len(claims)
                or {r["claim_index"] for r in labels} != set(range(len(claims)))
                or any(r.get("support") != "supported" or not r.get("notes") for r in labels)):
            raise ValueError("Each claim needs a supported label and review rationale")
        if example.get("expected_answerability") not in {"sufficient", "insufficient"}:
            raise ValueError("Expected answerability is required")
        if not claims and example["expected_answerability"] != "insufficient":
            raise ValueError("An answerable example must contain supported claims")


class _ReplayPolicy:
    def __init__(self, example: dict, batch_hash: str):
        self.actions = iter(example["actions"])
        self.config = {"backend": "assistant_authored_replay", "model": "none",
                       "seed": 0, "batch_hash": batch_hash,
                       "note": "No teacher endpoint or Qwen inference was run."}

    def act(self, observation: str) -> str:
        return json.dumps(next(self.actions), ensure_ascii=False)


def validate_replay(run: dict) -> None:
    """Check evidence was actually exposed by a tool before the authored final answer."""
    if run["status"] != "submitted":
        raise ValueError(f"Replay did not submit: {run['question']['id']}")
    known_docs = set()
    spans = set()
    for step in run["trajectory"]:
        action = step["action"]
        if action["action"] in {"paper", "passage"}:
            if action["arguments"]["doc_id"] not in known_docs:
                raise ValueError("A paper ID was used before discovery")
        if action["action"] == "submit":
            for claim in action["answer"]["claims"]:
                for e in claim["evidence"]:
                    if (e["doc_id"], e["start"], e["end"]) not in spans:
                        raise ValueError("Cited evidence was not observed in the trajectory")
            continue
        output = json.loads(step["output"])
        for item in output if isinstance(output, list) else [output]:
            if isinstance(item, dict) and "doc_id" in item:
                known_docs.add(item["doc_id"])
                if {"start", "end", "quote"} <= item.keys():
                    spans.add((item["doc_id"], item["start"], item["end"]))
    checks = run["checks"]
    if (checks["claims_with_valid_source_spans"] != checks["claim_count"]
            or checks["invalid_quote_count"]):
        raise ValueError("Replay contains invalid source evidence")


def build_demonstrations(batch_path: Path, snapshot: Path, benchmark_paths: list[Path],
                         output: Path) -> dict:
    batch = json.loads(batch_path.read_text())
    manifest, _ = load_snapshot(snapshot)
    benchmarks = [json.loads(p.read_text()) for p in benchmark_paths]
    validate_batch(batch, manifest, benchmarks)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.mkdir(parents=True)
    batch_hash = configuration_hash(batch)
    conversations, runs, reviews = [], [], []
    for example in batch["examples"]:
        question = {k: example[k] for k in ("id", "split", "question", "focus")}
        run = run_question(snapshot, question, _ReplayPolicy(example, batch_hash),
                           output / "runs" / f"{example['id']}.json",
                           max_steps=batch["max_steps"], server_hardware="none; CPU replay")
        validate_replay(run)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for step in run["trajectory"]:
            messages.extend([
                {"role": "user", "content": step["observation"]},
                {"role": "assistant", "content": step["raw_action"]},
            ])
        conversations.append({"id": example["id"], "messages": messages,
                              "metadata": {"split": "development", "batch_hash": batch_hash,
                                           "review": example["review"],
                                           "origin": "assistant_authored_replay"}})
        reviews.append({"id": example["id"], "question": example["question"],
                        "submission": run["submission"], "review": example["review"]})
        runs.append(run)
    (output / "conversations.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in conversations)
    )
    (output / "review.json").write_text(json.dumps(reviews, indent=2, ensure_ascii=False) + "\n")
    report = {
        "batch_id": batch["batch_id"], "batch_hash": batch_hash,
        "corpus_hash": manifest["corpus_hash"], "provenance": batch["provenance"],
        "exclusion_hashes": [configuration_hash(b) for b in benchmarks],
        "run_artifact_hash": content_hash(output / "runs"),
        "example_count": len(runs), "assistant_turn_count": sum(
            len(run["trajectory"]) for run in runs),
        "claim_count": sum(run["checks"]["claim_count"] for run in runs),
        "abstention_count": sum(not run["submission"]["claims"] for run in runs),
        "mechanical_validation": "passed",
        "semantic_review": "assistant self-review; independent review pending",
        "contamination_check": "paper families and question lexical overlap passed; "
                               "semantic paraphrase review still required",
        "training_ready": False,
        "remaining_gates": ["independent example review", "lock larger final benchmark",
                            "base-model structured-output experiment"],
        "training_contract": "Use the Qwen chat template; train on assistant action tokens only. "
                             "System, user, and tool-observation tokens are conditioning context.",
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report

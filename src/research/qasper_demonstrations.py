"""Build replay-verified tool trajectories from QASPER's human train labels."""

from __future__ import annotations

import json
from pathlib import Path

from src.eval.artifacts import configuration_hash, content_hash
from src.research.agent import SYSTEM_PROMPT, load_snapshot, run_question
from src.research.demonstrations import validate_replay
from src.research.tools_runtime import ResearchTools

GENERATOR_VERSION = "qasper-demonstrations-v1"


def _overlaps(left: dict, right: dict) -> bool:
    return (
        left.get("doc_id") == right.get("doc_id")
        and left["start"] < right["end"]
        and left["end"] > right["start"]
    )


def _covered_annotation(question: dict, hits: list[dict]):
    """Choose a human answer only when every one of its evidence spans was retrieved."""
    for annotation in question.get("answer_annotations", []):
        evidence = annotation.get("evidence", [])
        if annotation.get("unanswerable") or not evidence:
            continue
        selected = []
        for item in evidence:
            hit = next((candidate for candidate in hits if _overlaps(candidate, item)), None)
            if hit is None:
                break
            if not any(
                seen["start"] == hit["start"] and seen["end"] == hit["end"]
                for seen in selected
            ):
                selected.append(hit)
        else:
            return annotation, selected
    return None, []


class _ReplayPolicy:
    def __init__(self, actions: list[dict], question_set_hash: str, retriever: dict):
        self.actions = iter(actions)
        self.config = {
            "backend": "qasper_label_replay",
            "model": "none",
            "seed": 0,
            "question_set_hash": question_set_hash,
            "retriever": retriever,
            "note": "Actions are derived from QASPER human labels; no teacher model was run.",
        }

    def act(self, observation: str) -> str:
        return json.dumps(next(self.actions), ensure_ascii=False)


def _answer_actions(question: dict, hits: list[dict], top_k: int):
    annotation, selected = _covered_annotation(question, hits)
    if annotation is None:
        return None, "gold_evidence_not_fully_retrieved", None
    doc_id = question["target_doc_ids"][0]
    actions = [{
        "action": "search_paper",
        "arguments": {"doc_id": doc_id, "query": question["source_question"], "top_k": top_k},
    }]
    evidence = []
    for hit in selected:
        actions.append({
            "action": "passage",
            "arguments": {
                "doc_id": doc_id,
                "start": hit["start"],
                "length": hit["end"] - hit["start"],
            },
        })
        evidence.append({"doc_id": doc_id, "start": hit["start"], "end": hit["end"]})
    actions.append({
        "action": "submit",
        "answer": {
            "claims": [{"text": annotation["answer_text"], "evidence": evidence}],
            "recommendation": "No project recommendation is warranted by this question alone.",
            "limitations": [],
        },
    })
    return actions, None, annotation


def _abstention_actions(question: dict, top_k: int):
    doc_id = question["target_doc_ids"][0]
    return [{
        "action": "search_paper",
        "arguments": {"doc_id": doc_id, "query": question["source_question"], "top_k": top_k},
    }, {
        "action": "submit",
        "answer": {
            "claims": [],
            "recommendation": "No project recommendation is warranted by this question alone.",
            "limitations": ["The paper snapshot does not contain enough evidence to answer."],
        },
    }]


def build_qasper_demonstrations(
    question_set_path: Path,
    snapshot: Path,
    output: Path,
    paper_reranker=None,
    top_k: int = 5,
    max_steps: int = 8,
) -> dict:
    """Export assistant-token SFT conversations after replaying every cited span."""
    if output.exists():
        raise FileExistsError(output)
    if not 1 <= top_k <= 10:
        raise ValueError("top_k must be 1..10")
    question_set = json.loads(question_set_path.read_text())
    manifest, _ = load_snapshot(snapshot)
    if question_set.get("corpus_hash") != manifest["corpus_hash"]:
        raise ValueError("QASPER question set and snapshot corpus hashes differ")
    if question_set.get("source_split") != "train":
        raise ValueError("Only the QASPER train split may be exported for training")

    question_set_hash = configuration_hash(question_set)
    tools = ResearchTools(snapshot / "corpus", paper_reranker=paper_reranker)
    retriever = (
        paper_reranker.config if paper_reranker is not None
        else {"kind": "lexical_paragraph", "version": "v1"}
    )
    planned = []
    exclusions = []
    for question in question_set.get("questions", []):
        if question.get("split") != "train" or question.get("usage") != "training_source":
            exclusions.append({"question_id": question.get("id"), "reason": "not_training_source"})
            continue
        targets = question.get("target_doc_ids", [])
        if len(targets) != 1:
            exclusions.append({"question_id": question.get("id"), "reason": "not_one_known_paper"})
            continue
        hits = tools.search_paper(targets[0], question["source_question"], top_k=top_k)
        if question.get("expected_answerability") == "insufficient":
            actions, annotation = _abstention_actions(question, top_k), None
        elif question.get("expected_answerability") == "sufficient":
            actions, reason, annotation = _answer_actions(question, hits, top_k)
            if actions is None:
                exclusions.append({"question_id": question["id"], "reason": reason})
                continue
        else:
            exclusions.append({"question_id": question["id"], "reason": "unknown_answerability"})
            continue
        if len(actions) > max_steps:
            exclusions.append({"question_id": question["id"], "reason": "action_budget_exceeded"})
            continue
        planned.append((question, actions, annotation))

    output.mkdir(parents=True)
    conversations = []
    runs = []
    for question, actions, annotation in planned:
        run = run_question(
            snapshot,
            question,
            _ReplayPolicy(actions, question_set_hash, retriever),
            output / "runs" / f"{question['id']}.json",
            max_steps=max_steps,
            server_hardware="none; deterministic CPU replay",
            paper_reranker=paper_reranker,
        )
        validate_replay(run)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for step in run["trajectory"]:
            messages.extend([
                {"role": "user", "content": step["observation"]},
                {"role": "assistant", "content": step["raw_action"]},
            ])
        conversations.append({
            "id": question["id"],
            "messages": messages,
            "metadata": {
                "source_dataset": question_set["source_dataset"],
                "source_revision": question_set["source_revision"],
                "source_split": "train",
                "source_question_id": question["source_question_id"],
                "source_paper_id": question["source_paper_id"],
                "expected_answerability": question["expected_answerability"],
                "answer_annotation_id": annotation.get("annotation_id") if annotation else None,
                "derivation": "human_label_plus_replay_verified_tool_path",
                "question_set_hash": question_set_hash,
            },
        })
        runs.append(run)

    (output / "conversations.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in conversations)
    )
    report = {
        "schema_version": GENERATOR_VERSION,
        "question_set_id": question_set.get("question_set_id"),
        "question_set_hash": question_set_hash,
        "source_dataset": question_set["source_dataset"],
        "source_revision": question_set["source_revision"],
        "source_split": "train",
        "corpus_hash": manifest["corpus_hash"],
        "retriever": retriever,
        "top_k": top_k,
        "max_steps": max_steps,
        "candidate_count": len(question_set.get("questions", [])),
        "exported_count": len(runs),
        "answerable_count": sum(bool(run["submission"]["claims"]) for run in runs),
        "abstention_count": sum(not run["submission"]["claims"] for run in runs),
        "excluded_count": len(exclusions),
        "exclusions": exclusions,
        "assistant_turn_count": sum(len(run["trajectory"]) for run in runs),
        "run_artifact_hash": content_hash(output / "runs"),
        "mechanical_validation": "passed",
        "semantic_supervision": "QASPER human answer and evidence annotations",
        "teacher_model": None,
        "training_ready": False,
        "remaining_gates": [
            "manual sample audit",
            "lock paper-disjoint QASPER validation/test evaluation",
            "implement and verify assistant-token-only SFT conversion",
        ],
        "decision_rule": (
            "Use only exported examples: every answerable citation was observed through the "
            "runtime tools and every abstention comes from a QASPER unanswerable annotation."
        ),
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report

"""Offline checks for the QASPER-to-research-agent conversion boundary."""

import json

import pytest

from src.research.agent import load_snapshot, run_question
from src.research.qasper import build_qasper_snapshot


def annotation(
    annotation_id: str,
    *,
    evidence: list[str] | None = None,
    extractive: list[str] | None = None,
    free_form: str = "",
    yes_no=None,
    unanswerable: bool = False,
) -> dict:
    return {
        "annotation_id": annotation_id,
        "worker_id": "worker",
        "answer": {
            "unanswerable": unanswerable,
            "extractive_spans": extractive or [],
            "yes_no": yes_no,
            "free_form_answer": free_form,
            "evidence": evidence or [],
            "highlighted_evidence": [],
        },
    }


def paper() -> dict:
    method = "The model retrieves two passages before producing an answer."
    result = "Evidence selection improves by five points on the development set."
    return {
        "id": "2101.12345",
        "title": "A paper about retrieval",
        "abstract": "We investigate retrieval for scientific question answering.",
        # QASPER's Arrow representation uses parallel lists for nested sequences.
        "full_text": {
            "section_name": ["Method", "Results"],
            "paragraphs": [[method], [result]],
        },
        "qas": {
            "question": [
                "How many passages does the model retrieve?",
                "Does the paper prove that retrieval eliminates hallucinations?",
                "What value is shown in Figure 2?",
            ],
            "question_id": ["q1", "q2", "q3"],
            "nlp_background": ["five", "five", "five"],
            "topic_background": ["familiar", "familiar", "familiar"],
            "paper_read": ["yes", "yes", "yes"],
            "search_query": ["", "", ""],
            "question_writer": ["w1", "w1", "w1"],
            "answers": [
                [annotation("a1", evidence=[method], extractive=["two passages"])],
                [annotation("a2", unanswerable=True)],
                [
                    annotation(
                        "a3",
                        evidence=["FLOAT SELECTED: Figure 2"],
                        free_form="42",
                    )
                ],
            ],
        },
    }


def test_conversion_preserves_split_answers_and_exact_evidence(tmp_path):
    output = tmp_path / "qasper"
    manifest, question_set = build_qasper_snapshot([paper()], output, num_questions=None, seed=7)

    loaded_manifest, documents = load_snapshot(output)
    assert loaded_manifest["corpus_hash"] == manifest["corpus_hash"]
    assert manifest["source_split"] == "train"
    assert manifest["source_question_count"] == 3
    assert manifest["eligible_text_question_count"] == 2
    assert manifest["excluded_nontext_question_count"] == 1
    assert {item["source_question_id"] for item in question_set["questions"]} == {"q1", "q2"}

    answerable = next(
        item for item in question_set["questions"] if item["source_question_id"] == "q1"
    )
    evidence = answerable["gold_evidence"][0]
    document = documents[evidence["doc_id"]]
    assert answerable["answer_types"] == ["extractive"]
    assert answerable["split"] == "train"
    assert answerable["usage"] == "training_source"
    assert answerable["source_question"] == "How many passages does the model retrieve?"
    assert document["doc_id"] in answerable["question"]
    assert document["title"] in answerable["retrieval_query"]
    assert document["text"][evidence["start"] : evidence["end"]] == evidence["text"]

    unanswerable = next(
        item for item in question_set["questions"] if item["source_question_id"] == "q2"
    )
    assert unanswerable["expected_answerability"] == "insufficient"
    assert unanswerable["answer_annotations"][0]["answer_text"] == "Unanswerable"
    assert not unanswerable["gold_evidence"]


def test_selection_is_reproducible_and_snapshots_are_immutable(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _, questions_a = build_qasper_snapshot([paper()], first, num_questions=1, seed=19)
    _, questions_b = build_qasper_snapshot([paper()], second, num_questions=1, seed=19)
    assert questions_a["questions"][0]["id"] == questions_b["questions"][0]["id"]
    assert (
        json.loads((first / "raw" / "source_identity.json").read_text())["selected_question_ids"]
        == json.loads((second / "raw" / "source_identity.json").read_text())[
            "selected_question_ids"
        ]
    )
    with pytest.raises(FileExistsError):
        build_qasper_snapshot([paper()], first)


def test_snapshot_runs_through_production_agent_boundary(tmp_path):
    class AbstainingPolicy:
        config = {"model": "fixture", "revision": "fixture"}

        def act(self, _observation):
            return json.dumps(
                {
                    "action": "submit",
                    "answer": {
                        "claims": [],
                        "recommendation": "The supplied paper does not answer this question.",
                        "limitations": ["No supporting evidence was found."],
                    },
                }
            )

    snapshot = tmp_path / "snapshot"
    _, questions = build_qasper_snapshot([paper()], snapshot, num_questions=None)
    question = next(q for q in questions["questions"] if q["source_question_id"] == "q2")
    result = run_question(snapshot, question, AbstainingPolicy(), tmp_path / "run.json")
    assert result["status"] == "submitted"
    assert result["snapshot_manifest"]["retrieved_at"]
    assert result["checks"]["claim_count"] == 0


def test_unmatched_gold_evidence_is_excluded_instead_of_silently_rewritten(tmp_path):
    row = paper()
    row["qas"]["question"] = ["What was reported?"]
    row["qas"]["question_id"] = ["missing"]
    for field in (
        "nlp_background",
        "topic_background",
        "paper_read",
        "search_query",
        "question_writer",
    ):
        row["qas"][field] = [row["qas"][field][0]]
    row["qas"]["answers"] = [
        [annotation("bad", evidence=["This paragraph is absent."], free_form="A result")]
    ]
    with pytest.raises(ValueError, match="Only 0 text-only questions"):
        build_qasper_snapshot([row], tmp_path / "bad", num_questions=1)
    assert (tmp_path / "bad" / "manifest.failed.json").exists()

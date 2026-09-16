"""Offline checks for converting QASPER into the code-execution benchmark schema."""

import pytest

from src.research.agent import load_snapshot
from src.research.benchmark import validate_benchmark
from src.research.qasper import build_qasper_code_exec_benchmark


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
    return {
        "id": "2101.12345",
        "title": "A paper about retrieval",
        "abstract": "We investigate retrieval for scientific question answering.",
        "full_text": {
            "section_name": ["Method"],
            "paragraphs": [[method]],
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


def test_converted_benchmark_passes_the_shared_validator(tmp_path):
    output = tmp_path / "qasper-code-exec"
    manifest, benchmark = build_qasper_code_exec_benchmark(
        [paper()], output, source_split="test", num_questions=None, seed=1
    )
    stored_manifest, _ = load_snapshot(output)
    validate_benchmark(benchmark, stored_manifest)
    assert stored_manifest == manifest


def test_figure_evidence_question_is_excluded_but_others_survive(tmp_path):
    output = tmp_path / "qasper-code-exec"
    _, benchmark = build_qasper_code_exec_benchmark(
        [paper()], output, source_split="test", num_questions=None, seed=1
    )
    ids = {q["id"] for q in benchmark["questions"]}
    assert len(benchmark["questions"]) == 2
    assert all("q3" not in item for item in ids)


def test_unanswerable_question_has_no_required_documents(tmp_path):
    output = tmp_path / "qasper-code-exec"
    _, benchmark = build_qasper_code_exec_benchmark(
        [paper()], output, source_split="test", num_questions=None, seed=1
    )
    unanswerable = [q for q in benchmark["questions"]
                    if q["expected_answerability"] == "insufficient"]
    assert len(unanswerable) == 1
    assert unanswerable[0]["required_doc_ids"] == []
    assert unanswerable[0]["expected_citations"] == []
    assert unanswerable[0]["minimum_distinct_sources"] == 0


def test_answerable_question_targets_the_known_paper(tmp_path):
    output = tmp_path / "qasper-code-exec"
    _, benchmark = build_qasper_code_exec_benchmark(
        [paper()], output, source_split="test", num_questions=None, seed=1
    )
    answerable = [q for q in benchmark["questions"]
                  if q["expected_answerability"] == "sufficient"][0]
    assert answerable["required_doc_ids"] == ["qasper_2101_12345"]
    assert answerable["minimum_distinct_sources"] == 1
    assert "two passages" in answerable["answer"]


def test_question_text_names_the_known_paper(tmp_path):
    """QASPER questions use bare pronouns ("they", "this paper") that only make
    sense given the specific paper the annotator was looking at. Using the raw
    source_question alone over a multi-paper corpus makes the question
    unanswerable-by-construction — it must carry the paper's identity."""
    output = tmp_path / "qasper-code-exec"
    _, benchmark = build_qasper_code_exec_benchmark(
        [paper()], output, source_split="test", num_questions=None, seed=1
    )
    answerable = [q for q in benchmark["questions"]
                  if q["expected_answerability"] == "sufficient"][0]
    assert "qasper_2101_12345" in answerable["question"]
    assert "A paper about retrieval" in answerable["question"]


def test_reserved_doc_ids_match_every_paper_in_the_corpus(tmp_path):
    output = tmp_path / "qasper-code-exec"
    manifest, benchmark = build_qasper_code_exec_benchmark(
        [paper()], output, source_split="test", num_questions=None, seed=1
    )
    corpus_files = list((output / "corpus").glob("*.json"))
    assert {p.stem for p in corpus_files} == set(benchmark["reserved_doc_ids"])


def test_num_questions_caps_selection_deterministically(tmp_path):
    first = tmp_path / "a"
    second = tmp_path / "b"
    _, bench_a = build_qasper_code_exec_benchmark(
        [paper()], first, source_split="test", num_questions=1, seed=5
    )
    _, bench_b = build_qasper_code_exec_benchmark(
        [paper()], second, source_split="test", num_questions=1, seed=5
    )
    assert len(bench_a["questions"]) == 1
    assert bench_a["questions"] == bench_b["questions"]


def test_requesting_more_questions_than_eligible_raises(tmp_path):
    output = tmp_path / "qasper-code-exec"
    with pytest.raises(ValueError, match="eligible questions"):
        build_qasper_code_exec_benchmark(
            [paper()], output, source_split="test", num_questions=5, seed=1
        )

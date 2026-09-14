"""Convert QASPER papers and annotations into a frozen research-agent snapshot."""

from __future__ import annotations

import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from src.eval.artifacts import content_hash
from src.research.papers import content_hash_all

QASPER_DATASET = "allenai/qasper"
QASPER_CONFIG = "qasper"
QASPER_REVISION = "13b496d2a5359329b110e3419628de3cf791843b"
CONVERTER_VERSION = "qasper-research-v2"
FLOAT_MARKER = "FLOAT SELECTED"


def _records(value) -> list[dict]:
    """Accept both Arrow's dict-of-lists and Python's list-of-dicts sequence shapes."""
    if value is None:
        return []
    if isinstance(value, list):
        if any(not isinstance(item, dict) for item in value):
            raise ValueError("Expected a sequence of objects")
        return value
    if not isinstance(value, dict):
        raise ValueError("Expected a sequence encoded as a list or dict")
    lengths = {len(items) for items in value.values() if isinstance(items, list)}
    if not lengths:
        return []
    if len(lengths) != 1 or any(not isinstance(items, list) for items in value.values()):
        raise ValueError("Parallel sequence fields have inconsistent shapes")
    size = lengths.pop()
    return [{key: items[index] for key, items in value.items()} for index in range(size)]


def _doc_id(paper_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", paper_id).strip("_")
    if not safe:
        raise ValueError("QASPER paper has an empty or unusable ID")
    return f"qasper_{safe}"


def _answer(annotation: dict) -> dict:
    answer = annotation.get("answer")
    if not isinstance(answer, dict):
        raise ValueError("QASPER answer annotation is missing its answer object")
    unanswerable = answer.get("unanswerable") is True
    extractive = [str(item) for item in (answer.get("extractive_spans") or [])]
    free_form = answer.get("free_form_answer")
    yes_no = answer.get("yes_no")
    if unanswerable:
        answer_type, answer_text = "unanswerable", "Unanswerable"
    elif extractive:
        answer_type, answer_text = "extractive", ", ".join(extractive)
    elif isinstance(free_form, str) and free_form.strip():
        answer_type, answer_text = "abstractive", free_form
    elif yes_no is not None:
        answer_type, answer_text = "boolean", "Yes" if yes_no else "No"
    else:
        raise ValueError("Answerable QASPER annotation has no answer value")
    evidence = [str(item) for item in (answer.get("evidence") or [])]
    return {
        "annotation_id": str(annotation.get("annotation_id") or ""),
        "worker_id": str(annotation.get("worker_id") or ""),
        "answer_type": answer_type,
        "answer_text": answer_text,
        "unanswerable": unanswerable,
        "extractive_spans": extractive,
        "free_form_answer": free_form or "",
        "yes_no": yes_no,
        "evidence_text": evidence,
        "highlighted_evidence": [str(item) for item in (answer.get("highlighted_evidence") or [])],
        "requires_nontext_evidence": any(FLOAT_MARKER in item for item in evidence),
    }


def _paper_document(row: dict, source_split: str) -> tuple[dict, dict[str, list[dict]]]:
    paper_id = str(row.get("id") or "")
    title = str(row.get("title") or "").strip()
    abstract = str(row.get("abstract") or "").strip()
    if not title or not abstract:
        raise ValueError(f"QASPER paper {paper_id!r} is missing title or abstract")

    blocks = [("Title", title), ("Abstract", abstract)]
    for section in _records(row.get("full_text")):
        name = str(section.get("section_name") or "Untitled section")
        paragraphs = section.get("paragraphs") or []
        if not isinstance(paragraphs, list):
            raise ValueError(f"QASPER paper {paper_id!r} has malformed paragraphs")
        blocks.extend((name, str(paragraph)) for paragraph in paragraphs if str(paragraph))

    text = ""
    sections: list[dict] = []
    paragraph_locations: dict[str, list[dict]] = {}
    current_section = None
    section_start = 0
    for section_name, paragraph in blocks:
        if text:
            text += "\n\n"
        start = len(text)
        text += paragraph
        end = len(text)
        paragraph_locations.setdefault(paragraph, []).append(
            {"start": start, "end": end, "section": section_name}
        )
        if current_section != section_name:
            if current_section is not None:
                sections[-1]["end"] = start - 2
            sections.append({"section": section_name, "start": start, "end": end})
            current_section = section_name
            section_start = start
        else:
            sections[-1]["end"] = end
    if not text or not sections:
        raise ValueError(f"QASPER paper {paper_id!r} has no usable text")
    sections[-1]["start"] = section_start

    arxiv_like = bool(re.fullmatch(r"\d{4}\.\d{4,5}(?:v\d+)?", paper_id))
    source_url = (
        f"https://arxiv.org/abs/{paper_id}"
        if arxiv_like
        else f"https://huggingface.co/datasets/{QASPER_DATASET}"
    )
    document = {
        "doc_id": _doc_id(paper_id),
        "title": title,
        "text": text,
        "sections": sections,
        "metadata": {
            "source_dataset": QASPER_DATASET,
            "source_split": source_split,
            "qasper_paper_id": paper_id,
            "arxiv_id": paper_id if arxiv_like else None,
            "source_url": source_url,
            "submitted": "unknown",
            "coverage": "qasper_s2orc_text",
            "source_kind": "research_paper",
            "parser_version": CONVERTER_VERSION,
            "license": "CC BY 4.0",
        },
    }
    return document, paragraph_locations


def _convert_question(
    qa: dict,
    document: dict,
    paragraph_locations: dict[str, list[dict]],
    source_split: str,
) -> dict:
    source_id = str(qa.get("question_id") or "")
    source_question = str(qa.get("question") or "").strip()
    if not source_id or not source_question:
        raise ValueError("QASPER question is missing its ID or text")
    converted_annotations = []
    unmatched_text_evidence = []
    ambiguous_text_evidence = []
    for raw_annotation in _records(qa.get("answers")):
        annotation = _answer(raw_annotation)
        mapped = []
        for evidence_text in annotation.pop("evidence_text"):
            if FLOAT_MARKER in evidence_text:
                continue
            locations = paragraph_locations.get(evidence_text, [])
            if not locations:
                unmatched_text_evidence.append(evidence_text)
                continue
            if len(locations) > 1:
                ambiguous_text_evidence.append(evidence_text)
            location = locations[0]
            mapped.append(
                {
                    "doc_id": document["doc_id"],
                    **location,
                    "text": evidence_text,
                    "match_count": len(locations),
                }
            )
        annotation["evidence"] = mapped
        converted_annotations.append(annotation)
    if not converted_annotations:
        raise ValueError(f"QASPER question {source_id!r} has no annotations")

    answer_types = sorted({item["answer_type"] for item in converted_annotations})
    answerability = {not item["unanswerable"] for item in converted_annotations}
    if answerability == {True}:
        expected_answerability = "sufficient"
    elif answerability == {False}:
        expected_answerability = "insufficient"
    else:
        expected_answerability = "annotator_disagreement"
    requires_nontext = any(item["requires_nontext_evidence"] for item in converted_annotations)
    conversion_issues = []
    if requires_nontext:
        conversion_issues.append("requires_figure_or_table_evidence")
    if unmatched_text_evidence:
        conversion_issues.append("unmatched_text_evidence")

    gold_evidence = []
    seen_spans = set()
    for annotation in converted_annotations:
        for item in annotation["evidence"]:
            key = (item["doc_id"], item["start"], item["end"])
            if key not in seen_spans:
                seen_spans.add(key)
                gold_evidence.append(item)
    return {
        "id": f"qasper_{source_split}_{source_id}",
        "question": (
            f'Use the known paper "{document["title"]}" '
            f'(doc_id: "{document["doc_id"]}") to answer: {source_question}'
        ),
        "source_question": source_question,
        "retrieval_query": f"{document['title']} {source_question}",
        "target_doc_ids": [document["doc_id"]],
        "task_scope": "known_paper_evidence_qa",
        "split": source_split,
        "usage": "training_source" if source_split == "train" else "reserved_evaluation",
        "source_dataset": QASPER_DATASET,
        "source_question_id": source_id,
        "source_paper_id": document["metadata"]["qasper_paper_id"],
        "answer_types": answer_types,
        "expected_answerability": expected_answerability,
        "answer_annotations": converted_annotations,
        "gold_evidence": gold_evidence,
        "requires_nontext_evidence": requires_nontext,
        "conversion_issues": conversion_issues,
        "unmatched_text_evidence_count": len(unmatched_text_evidence),
        "ambiguous_text_evidence_count": len(ambiguous_text_evidence),
        "source_metadata": {
            "nlp_background": qa.get("nlp_background"),
            "topic_background": qa.get("topic_background"),
            "paper_read": qa.get("paper_read"),
            "search_query": qa.get("search_query"),
            "question_writer": qa.get("question_writer"),
        },
    }


def build_qasper_snapshot(
    rows: Iterable[dict],
    output: Path,
    source_split: str = "train",
    revision: str = QASPER_REVISION,
    num_questions: int | None = 40,
    seed: int = 42,
) -> tuple[dict, dict]:
    """Write an immutable snapshot and deterministic text-only question selection."""
    if source_split not in {"train", "validation", "test"}:
        raise ValueError("source_split must be train, validation, or test")
    if num_questions is not None and num_questions < 1:
        raise ValueError("num_questions must be positive or None")
    output.mkdir(parents=True, exist_ok=False)
    corpus_dir = output / "corpus"
    raw_dir = output / "raw"
    corpus_dir.mkdir()
    raw_dir.mkdir()

    papers = []
    converted_questions = []
    paper_ids = set()
    question_ids = set()
    try:
        for row in rows:
            document, locations = _paper_document(row, source_split)
            paper_id = document["metadata"]["qasper_paper_id"]
            if paper_id in paper_ids:
                raise ValueError(f"Duplicate QASPER paper ID: {paper_id}")
            paper_ids.add(paper_id)
            path = corpus_dir / f"{document['doc_id']}.json"
            path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n")
            papers.append(
                {
                    "doc_id": document["doc_id"],
                    "qasper_paper_id": paper_id,
                    "title": document["title"],
                    "source_url": document["metadata"]["source_url"],
                    "coverage": document["metadata"]["coverage"],
                    "sha256": content_hash(path),
                }
            )
            for qa in _records(row.get("qas")):
                converted = _convert_question(qa, document, locations, source_split)
                if converted["source_question_id"] in question_ids:
                    raise ValueError(
                        f"Duplicate QASPER question ID: {converted['source_question_id']}"
                    )
                question_ids.add(converted["source_question_id"])
                converted_questions.append(converted)

        if not papers:
            raise ValueError("QASPER input contained no papers")
        eligible = [
            question for question in converted_questions if not question["conversion_issues"]
        ]
        eligible.sort(key=lambda question: question["id"])
        rng = random.Random(seed)
        rng.shuffle(eligible)
        if num_questions is not None:
            if len(eligible) < num_questions:
                raise ValueError(
                    f"Only {len(eligible)} text-only questions passed conversion; "
                    f"cannot select {num_questions}"
                )
            eligible = eligible[:num_questions]
        eligible.sort(key=lambda question: question["id"])

        source_identity = {
            "source_dataset": QASPER_DATASET,
            "source_config": QASPER_CONFIG,
            "source_revision": revision,
            "source_split": source_split,
            "converter_version": CONVERTER_VERSION,
            "seed": seed,
            "requested_question_count": num_questions,
            "paper_ids": sorted(paper_ids),
            "selected_question_ids": [item["source_question_id"] for item in eligible],
        }
        (raw_dir / "source_identity.json").write_text(json.dumps(source_identity, indent=2) + "\n")
        corpus_hash = content_hash(corpus_dir)
        question_set = {
            "schema_version": "research-questions-v1",
            "question_set_id": f"qasper-{source_split}-text-pilot-v1",
            "domain": "Question answering and evidence selection over NLP research papers",
            "corpus_hash": corpus_hash,
            "source_dataset": QASPER_DATASET,
            "source_revision": revision,
            "source_split": source_split,
            "selection_seed": seed,
            "status": (
                "QASPER dataset annotations converted for development/training. "
                "These are targets, not model-generated tool trajectories."
            ),
            "questions": eligible,
        }
        (output / "questions.json").write_text(
            json.dumps(question_set, ensure_ascii=False, indent=2) + "\n"
        )
        excluded_nontext = sum(q["requires_nontext_evidence"] for q in converted_questions)
        excluded_unmatched = sum(
            "unmatched_text_evidence" in q["conversion_issues"] for q in converted_questions
        )
        manifest = {
            "schema_version": "research-snapshot-v1",
            "parser_version": CONVERTER_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_dataset": QASPER_DATASET,
            "source_config": QASPER_CONFIG,
            "source_revision": revision,
            "source_split": source_split,
            "papers": papers,
            "paper_count": len(papers),
            "source_question_count": len(converted_questions),
            "eligible_text_question_count": len(
                [q for q in converted_questions if not q["conversion_issues"]]
            ),
            "selected_question_count": len(eligible),
            "excluded_nontext_question_count": excluded_nontext,
            "excluded_unmatched_evidence_question_count": excluded_unmatched,
            "corpus_hash": corpus_hash,
            "raw_sources_hash": content_hash_all(raw_dir),
            "status": "complete",
            "coverage_note": (
                "QASPER text extracted from S2ORC. Pilot questions requiring figures/tables or "
                "with unmatched text evidence are excluded; validation and test are not loaded "
                "when source_split is train."
            ),
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        )
        return manifest, question_set
    except Exception:
        # Keep failed builds for diagnosis, consistent with other immutable research snapshots.
        failure = {
            "schema_version": "research-snapshot-v1",
            "parser_version": CONVERTER_VERSION,
            "source_dataset": QASPER_DATASET,
            "source_revision": revision,
            "source_split": source_split,
            "status": "failed",
        }
        (output / "manifest.failed.json").write_text(json.dumps(failure, indent=2) + "\n")
        raise

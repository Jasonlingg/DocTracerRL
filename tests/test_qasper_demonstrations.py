"""Verify QASPER label conversion never invents an unobserved tool trajectory."""

import json

from src.research.papers import build_snapshot
from src.research.qasper_demonstrations import build_qasper_demonstrations


def test_qasper_demonstrations_replay_evidence_and_abstention(tmp_path):
    metadata = '''<meta property="og:url" content="https://arxiv.org/abs/2503.09516v1">
    <meta name="citation_title" content="A retrieval study">
    <meta name="citation_abstract" content="We study search with a fixed corpus.">
    <meta name="citation_date" content="2025/03/12">'''
    html = "<article><h2>Methods</h2><p>We study search with a fixed corpus.</p></article>"
    snapshot = tmp_path / "snapshot"
    manifest = build_snapshot(
        ["2503.09516v1"],
        snapshot,
        downloader=lambda url: (html if "/html/" in url else metadata).encode(),
        delay=0,
    )
    doc_path = next((snapshot / "corpus").glob("*.json"))
    doc = json.loads(doc_path.read_text())
    start = doc["text"].index("We study search with a fixed corpus.")
    end = start + len("We study search with a fixed corpus.")
    base = {
        "target_doc_ids": [doc["doc_id"]],
        "task_scope": "known_paper_evidence_qa",
        "split": "train",
        "usage": "training_source",
        "source_dataset": "allenai/qasper",
        "source_paper_id": "2503.09516",
        "requires_nontext_evidence": False,
    }
    answerable = {
        **base,
        "id": "answerable",
        "question": f'Use the known paper (doc_id: "{doc["doc_id"]}") to answer: What is studied?',
        "source_question": "What search uses a fixed corpus?",
        "source_question_id": "source-a",
        "expected_answerability": "sufficient",
        "answer_annotations": [{
            "annotation_id": "annotation-a",
            "answer_type": "extractive",
            "answer_text": "Search is studied with a fixed corpus.",
            "unanswerable": False,
            "evidence": [{"doc_id": doc["doc_id"], "start": start, "end": end}],
        }],
        "gold_evidence": [{"doc_id": doc["doc_id"], "start": start, "end": end}],
    }
    unanswerable = {
        **base,
        "id": "unanswerable",
        "question": f'Use the known paper (doc_id: "{doc["doc_id"]}") to answer: Who judged it?',
        "source_question": "Who judged search?",
        "source_question_id": "source-b",
        "expected_answerability": "insufficient",
        "answer_annotations": [{
            "annotation_id": "annotation-b",
            "answer_type": "unanswerable",
            "answer_text": "Unanswerable",
            "unanswerable": True,
            "evidence": [],
        }],
        "gold_evidence": [],
    }
    question_set = {
        "question_set_id": "fixture",
        "corpus_hash": manifest["corpus_hash"],
        "source_dataset": "allenai/qasper",
        "source_revision": "fixture-revision",
        "source_split": "train",
        "questions": [answerable, unanswerable],
    }
    question_path = tmp_path / "questions.json"
    question_path.write_text(json.dumps(question_set))

    output = tmp_path / "demonstrations"
    report = build_qasper_demonstrations(question_path, snapshot, output, top_k=3)

    assert report["exported_count"] == 2
    assert report["answerable_count"] == 1
    assert report["abstention_count"] == 1
    conversation_lines = (output / "conversations.jsonl").read_text().splitlines()
    conversations = [json.loads(line) for line in conversation_lines]
    answer_actions = [
        json.loads(message["content"])["action"]
        for message in conversations[0]["messages"]
        if message["role"] == "assistant"
    ]
    assert answer_actions == ["search_paper", "passage", "submit"]
    abstention_run = json.loads((output / "runs" / "unanswerable.json").read_text())
    assert abstention_run["status"] == "submitted"
    assert abstention_run["submission"]["claims"] == []
    assert report["training_ready"] is False

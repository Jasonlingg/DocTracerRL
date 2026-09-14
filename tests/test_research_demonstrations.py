"""Prevent contaminated or ungrounded demonstrations from being exported."""

import copy
import json
from pathlib import Path

import pytest

from src.research.demonstrations import validate_batch, validate_replay


@pytest.fixture
def inputs():
    batch = json.loads(Path("data/research/demonstrations_v1.json").read_text())
    sources = json.loads(Path("data/research/development_sources_v1.json").read_text())
    manifest = {"corpus_hash": batch["corpus_hash"], "papers": [
        {"doc_id": "arxiv_" + p["arxiv_id"].replace(".", "_")} for p in sources["papers"]
    ]}
    benchmark = json.loads(Path("data/research/benchmark_pilot_v1.json").read_text())
    return batch, manifest, [benchmark]


def test_authored_batch_passes_split_and_review_checks(inputs):
    validate_batch(*inputs)


def test_other_version_of_reserved_paper_is_rejected(inputs):
    batch, manifest, benchmarks = inputs
    manifest["papers"].append({"doc_id": "arxiv_2310_11511v99"})
    with pytest.raises(ValueError, match="Reserved paper families"):
        validate_batch(batch, manifest, benchmarks)


def test_renaming_a_held_out_question_does_not_hide_reuse(inputs):
    batch, manifest, benchmarks = inputs
    batch["examples"][0]["question"] = benchmarks[0]["questions"][0]["question"]
    with pytest.raises(ValueError, match="question reuse"):
        validate_batch(batch, manifest, benchmarks)


def test_unreviewed_claim_cannot_be_exported(inputs):
    batch, manifest, benchmarks = inputs
    batch["examples"][0]["review"]["claim_reviews"][0]["support"] = None
    with pytest.raises(ValueError, match="supported label"):
        validate_batch(batch, manifest, benchmarks)


def replay_fixture():
    evidence = {"doc_id": "p", "start": 0, "end": 12}
    return {
        "status": "submitted", "question": {"id": "test"},
        "checks": {"claims_with_valid_source_spans": 1, "claim_count": 1,
                   "invalid_quote_count": 0},
        "trajectory": [
            {"action": {"action": "search_papers", "arguments": {"query": "finding"}},
             "output": json.dumps([{**evidence, "quote": "A real claim"}])},
            {"action": {"action": "submit", "answer": {"claims": [
                {"text": "A finding", "evidence": [evidence]},
            ]}}},
        ],
    }


def test_valid_offsets_alone_do_not_allow_unseen_evidence():
    run = replay_fixture()
    validate_replay(run)
    tampered = copy.deepcopy(run)
    tampered["trajectory"][-1]["action"]["answer"]["claims"][0]["evidence"][0]["end"] = 13
    with pytest.raises(ValueError, match="not observed"):
        validate_replay(tampered)


def test_authored_action_cannot_guess_a_paper_id():
    run = replay_fixture()
    run["trajectory"].insert(0, {
        "action": {"action": "paper", "arguments": {"doc_id": "p"}},
        "output": json.dumps({"doc_id": "p"}),
    })
    with pytest.raises(ValueError, match="before discovery"):
        validate_replay(run)

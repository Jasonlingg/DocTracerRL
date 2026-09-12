"""Offline tests of ingestion, evidence integrity, and the multi-turn research path."""

import json
import subprocess

import pytest

from src.env.repl import DockerREPL
from src.research.agent import (
    SYSTEM_PROMPT,
    EndpointPolicy,
    check_submission,
    load_snapshot,
    materialize_evidence,
    normalize_submission_action,
    run_question,
)
from src.research.papers import build_snapshot, discover

META = '''<meta property="og:url" content="https://arxiv.org/abs/2503.09516v1">
<meta name="citation_title" content="A retrieval study">
<meta name="citation_abstract" content="We study search. Evidence has limitations.">
<meta name="citation_date" content="2025/03/12">
<meta name="citation_online_date" content="2025/03/12">'''
HTML = '<article><h2>Methods</h2>' + ''.join(
    f'<p>Search experiment {i} uses a fixed corpus.</p>' for i in range(6)
) + '</article>'


@pytest.fixture
def snapshot(tmp_path):
    path = tmp_path / "snapshot"
    build_snapshot(["2503.09516v1"], path,
                   downloader=lambda u: (HTML if "/html/" in u else META).encode(), delay=0)
    return path


def evidence(doc, quote="We study search."):
    start = doc["text"].index(quote)
    return {"doc_id": doc["doc_id"], "start": start, "end": start + len(quote), "quote": quote}


def submission(item):
    return {"claims": [{"text": "The authors study search.", "evidence": [item]}],
            "recommendation": "Inference: test retrieval on our own questions.",
            "limitations": ["This fixture establishes no real model performance."]}


def test_snapshot_retains_version_sections_and_detects_tampering(snapshot):
    manifest, docs = load_snapshot(snapshot)
    doc = next(iter(docs.values()))
    assert manifest["papers"][0]["arxiv_id"] == "2503.09516v1"
    assert doc["metadata"]["coverage"] == "html_paragraphs"
    section = doc["sections"][1]
    assert section["section"] == "Methods"
    assert doc["text"][section["start"]:section["end"]].startswith("Search experiment 0")
    path = next((snapshot / "corpus").glob("*.json"))
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_snapshot(snapshot)


def test_missing_html_is_explicit_abstract_only_and_snapshot_is_immutable(tmp_path):
    def download(url):
        if "/html/" in url:
            raise OSError("No HTML available")
        return META.encode()

    path = tmp_path / "abstract"
    result = build_snapshot(["2503.09516"], path, downloader=download, delay=0)
    assert result["papers"][0]["coverage"] == "abstract_only"
    assert "No HTML" in result["papers"][0]["html_error"]
    with pytest.raises(FileExistsError):
        build_snapshot(["2503.09516"], path, downloader=download, delay=0)


def test_wrong_version_is_recorded_as_failure(tmp_path):
    result = build_snapshot(["2503.09516v2"], tmp_path / "wrong",
                            downloader=lambda _: META.encode(), delay=0)
    assert result["status"] == "partial"
    assert not result["papers"]
    assert "received 2503.09516v1" in result["failures"][0]["error"]


@pytest.mark.parametrize("mutation", [
    {"doc_id": "invented"}, {"quote": "invented quote"}, {"start": -1},
    {"start": True}, {"end": 100000}, {"quote": ""},
])
def test_fabricated_or_invalid_evidence_is_not_credited(snapshot, mutation):
    _, docs = load_snapshot(snapshot)
    item = evidence(next(iter(docs.values())))
    result = check_submission(submission({**item, **mutation}), docs)
    assert result["claims_with_valid_source_spans"] == 0
    assert result["invalid_quote_count"] == 1
    assert result["reward"] is None


def test_valid_quote_does_not_prove_a_claim(snapshot):
    _, docs = load_snapshot(snapshot)
    answer = submission(evidence(next(iter(docs.values()))))
    answer["claims"][0]["text"] = "This method eliminates all hallucinations."
    result = check_submission(answer, docs)
    assert result["claims_with_valid_source_spans"] == 1
    assert result["semantic_support"] == "not_reviewed"
    assert result["claims"][0]["evidence"][0]["supports_claim"] is None


def test_empty_claims_do_not_receive_perfect_score(snapshot):
    _, docs = load_snapshot(snapshot)
    result = check_submission({"claims": [], "recommendation": "Insufficient evidence",
                               "limitations": ["No supporting source"]}, docs)
    assert result["claim_count"] == 0
    assert result["reward"] is None


def test_protocol_requires_printed_tools_and_normalizes_only_valid_bare_submission(snapshot):
    _, docs = load_snapshot(snapshot)
    doc = next(iter(docs.values()))
    bare = json.dumps(submission(evidence(doc)))
    not_submission = '{"ordinary": "code-like data"}'
    assert "MUST be Python code containing print(...)" in SYSTEM_PROMPT
    assert normalize_submission_action(bare) == "SUBMIT: " + bare
    assert normalize_submission_action(not_submission) == not_submission


def test_snapshot_materializes_omitted_quote_and_rejects_a_wrong_model_quote(snapshot):
    _, docs = load_snapshot(snapshot)
    doc = next(iter(docs.values()))
    item = evidence(doc)
    without_quote = {key: value for key, value in item.items() if key != "quote"}
    resolved = materialize_evidence(submission(without_quote), docs)
    source = resolved["claims"][0]["evidence"][0]
    assert source["quote"] == item["quote"]
    assert source["quote_origin"] == "snapshot_materialized"
    checked = check_submission(resolved, docs)
    assert checked["claims_with_valid_source_spans"] == 1
    assert checked["claims"][0]["evidence"][0]["exact_quote_valid"] is None

    wrong = submission({**item, "quote": "not from this source"})
    checked = check_submission(wrong, docs)
    assert checked["invalid_quote_count"] == 1


def test_multiturn_search_read_submit_and_review(snapshot, tmp_path):
    _, docs = load_snapshot(snapshot)
    doc = next(iter(docs.values()))

    class ScriptedPolicy:
        config = {"backend": "scripted_test", "model": "none", "seed": 0}

        def __init__(self):
            self.step = 0

        def act(self, observation):
            self.step += 1
            if self.step == 1:
                return 'found = search_papers("search", top_k=2); print(found)'
            if self.step == 2:
                assert doc["doc_id"] in observation
                return 'selected = found[0]["doc_id"]; print(passage(selected, 0, 16))'
            assert "We study search." in observation
            return "SUBMIT: " + json.dumps(submission(evidence(doc)))

    output = tmp_path / "run.json"
    result = run_question(snapshot, {"id": "test", "question": "What was studied?"},
                          ScriptedPolicy(), output, max_steps=3, use_docker=False)
    assert result["status"] == "submitted"
    assert len(result["trajectory"]) == 3
    assert result["checks"]["claims_with_valid_source_spans"] == 1
    assert "NOT been reviewed" in output.with_suffix(".md").read_text()
    with pytest.raises(FileExistsError):
        run_question(snapshot, result["question"], ScriptedPolicy(), output, use_docker=False)


def test_policy_failure_still_saves_artifacts(snapshot, tmp_path):
    class BrokenPolicy:
        config = {"backend": "scripted_test"}

        def act(self, observation):
            raise RuntimeError("server offline")

    output = tmp_path / "failure.json"
    result = run_question(snapshot, {"id": "test", "question": "What?"},
                          BrokenPolicy(), output, use_docker=False)
    assert result["status"] == "error"
    assert json.loads(output.read_text())["submission"] is None
    assert "server offline" in output.with_suffix(".md").read_text()


def test_custom_corpus_is_mounted_readonly_into_docker(monkeypatch, tmp_path):
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, "container", "")

    monkeypatch.setattr(subprocess, "run", run)
    repl = DockerREPL(corpus_path=str(tmp_path))
    repl.start_session()
    try:
        create = calls[0]
        assert create[create.index("--mount") + 1] == (
            f"type=bind,src={tmp_path},dst=/workspace/data/corpus,readonly"
        )
    finally:
        repl.kill_session()


def test_discovery_persists_date_bounds_and_rejects_api_errors():
    urls = []

    def download(url):
        urls.append(url)
        return b'''<feed xmlns="http://www.w3.org/2005/Atom"><entry>
        <id>http://arxiv.org/abs/2503.09516v1</id><title>Search</title>
        <published>2025-03-12T00:00:00Z</published></entry></feed>'''

    result = discover('all:"search"', "2025-03-01", "2025-03-31", downloader=download)
    assert "submittedDate" in urls[0] and "sortOrder=descending" in urls[0]
    assert result["papers"][0]["arxiv_id"] == "2503.09516v1"
    with pytest.raises(ValueError, match="since"):
        discover("search", "2025-04-01", "2025-03-01", downloader=download)


def test_endpoint_records_explicit_decoding_and_rejects_truncation(monkeypatch):
    import io

    requests = []

    def respond(request, **kwargs):
        requests.append(json.loads(request.data))
        return io.BytesIO(json.dumps({"choices": [{"finish_reason": "length",
                                                    "message": {"content": "partial"}}]}).encode())

    monkeypatch.setattr("src.research.agent.urlopen", respond)
    policy = EndpointPolicy("http://localhost:8000/v1", "Qwen/Qwen3-8B", "abc", seed=7)
    with pytest.raises(RuntimeError, match="truncated"):
        policy.act("Question")
    assert requests[0]["seed"] == 7
    assert requests[0]["chat_template_kwargs"] == {"enable_thinking": False}


def test_question_set_is_development_only():
    from pathlib import Path

    data = json.loads(Path("data/research/questions.json").read_text())
    assert len(data["questions"]) == 20
    assert len({q["id"] for q in data["questions"]}) == 20
    assert all(q["split"] == "development" and q["review_status"] == "unreviewed"
               for q in data["questions"])

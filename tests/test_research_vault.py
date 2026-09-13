"""Exercise folder isolation, source attribution, PDF extraction, and export round trips."""

import json

import pytest

from src.research.agent import run_question
from src.research.explainer import run_explanation
from src.research.tools_runtime import ResearchTools
from src.research.vault import export_explanation, import_vault


@pytest.fixture
def collection(tmp_path):
    vault = tmp_path / "My Vault"
    source = vault / "Research" / "My note.md"
    source.parent.mkdir(parents=True)
    source.write_text(
        "# My research process\n\nI check the original passage before trusting a claim.\n"
    )
    snapshot = tmp_path / "snapshot"
    return vault, source, snapshot


def test_import_preserves_revision_and_excludes_generated_hidden_and_symlink_notes(collection):
    vault, source, snapshot = collection
    (source.parent / "AI.md").write_text("---\ngenerated_by: rlm-explorer\n---\nInvented claim")
    (source.parent / ".private.md").write_text("Hidden note")
    (source.parent / "link.md").symlink_to(source)
    first = import_vault(vault, "Research", snapshot)
    assert len(first["papers"]) == 1
    assert len(first["skipped"]) == 2
    tools = ResearchTools(snapshot / "corpus")
    hit = tools.search_papers("original passage", 1)[0]
    assert hit["source_kind"] == "personal_note"
    assert hit["source_path"] == "Research/My note.md"
    assert tools.passage(hit["doc_id"], hit["start"], hit["end"] - hit["start"])["quote"] == (
        hit["quote"]
    )
    source.write_text("# Changed\nA new personal opinion.")
    second = import_vault(vault, "Research", snapshot.with_name("next"))
    assert first["papers"][0]["doc_id"] == second["papers"][0]["doc_id"]
    assert first["corpus_hash"] != second["corpus_hash"]
    assert "original passage" in tools.documents[hit["doc_id"]]["text"]
    with pytest.raises(FileExistsError):
        import_vault(vault, "Research", snapshot)
    with pytest.raises(ValueError, match="outside the vault"):
        import_vault(vault, "Research", vault / "snapshot")
    with pytest.raises(ValueError, match="relative path"):
        import_vault(vault, "../", snapshot.with_name("escape"))


def test_pdf_extracts_real_page_text_and_reports_empty_pages(collection):
    pypdf = pytest.importorskip("pypdf")
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    vault, source, snapshot = collection
    writer = pypdf.PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                             NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    page[NameObject("/Resources")] = DictionaryObject({
        NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})
    })
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 20 200 Td (Evidence is on page one.) Tj ET")
    page[NameObject("/Contents")] = stream
    writer.add_blank_page(width=300, height=300)
    with (source.parent / "Study.pdf").open("wb") as target:
        writer.write(target)
    result = import_vault(vault, "Research", snapshot)
    pdf = next(p for p in result["papers"] if p["source_kind"] == "pdf_document")
    assert pdf["empty_pages"] == [2]
    hit = ResearchTools(snapshot / "corpus").search_papers("Evidence page one", 1)[0]
    assert hit["pages"] == [1]
    assert "Evidence is on page one." in hit["quote"]
    # A fully image-only/empty PDF must not silently become valid evidence.
    blank = pypdf.PdfWriter()
    blank.add_blank_page(width=300, height=300)
    with (source.parent / "Scanned.pdf").open("wb") as target:
        blank.write(target)
    result = import_vault(vault, "Research", snapshot.with_name("with-blank"))
    assert result["status"] == "partial"
    assert "OCR" in result["failures"][0]["error"]


def test_scripted_pipeline_exports_links_and_refuses_tampered_evidence(collection, tmp_path):
    vault, source, snapshot = collection
    import_vault(vault, "Research", snapshot)
    tools = ResearchTools(snapshot / "corpus")
    hit = tools.search_papers("original passage", 1)[0]

    class Retriever:
        config = {"backend": "scripted_test"}
        step = 0

        def act(self, observation):
            self.step += 1
            if self.step == 1:
                return json.dumps({"action": "search_papers", "arguments": {"query": "passage"}})
            assert "personal_note" in observation
            return json.dumps({"action": "submit", "answer": {
                "claims": [{"text": "The personal note recommends checking original passages.",
                            "evidence": [{k: hit[k] for k in ("doc_id", "start", "end")}]}],
                "recommendation": "Inference: follow the note's checking process.",
                "limitations": ["This is personal commentary."]}})

    class Explainer:
        config = {"backend": "scripted_test"}

        def explain(self, packet):
            assert packet["evidence"][0]["source_kind"] == "personal_note"
            return json.dumps({"answer": "The note recommends checking passages [E1].",
                               "claims": [{"text": "The note recommends checking passages.",
                                           "evidence_ids": ["E1"]}],
                               "limitations": ["Personal commentary, not a paper finding."]})

    run = tmp_path / "retriever.json"
    result = run_question(snapshot, {"id": "demo", "question": "How should I check claims?"},
                          Retriever(), run, max_steps=2)
    assert result["status"] == "submitted"
    explanation = tmp_path / "explanation.json"
    result = run_explanation(run, Explainer(), explanation)
    note = export_explanation(explanation, snapshot, vault, "Answers/Claim check.md")
    text = note.read_text()
    assert "generated_by: rlm-explorer" in text
    assert "../Research/My%20note.md" in text
    assert "[E1](#E1)" in text
    with pytest.raises(FileExistsError):
        export_explanation(explanation, snapshot, vault, "Answers/Claim check.md")
    source.write_text("Edited source")
    note2 = export_explanation(explanation, snapshot, vault, "Answers/Changed.md")
    assert "changed or moved" in note2.read_text()
    next_import = import_vault(vault, ".", tmp_path / "refresh")
    assert len(next_import["papers"]) == 1  # Both exported AI notes excluded.
    result["evidence_packet"]["evidence"][0]["quote"] = "A forged quote"
    explanation.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="does not match"):
        export_explanation(explanation, snapshot, vault, "Answers/Forged.md")
    assert not (vault / "Answers/Forged.md").exists()

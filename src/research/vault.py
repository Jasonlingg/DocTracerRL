"""Opt-in folder import and append-only Obsidian note export."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urlencode

import yaml

from src.eval.artifacts import content_hash
from src.research.agent import load_snapshot
from src.research.explainer import check_explanation

IMPORT_VERSION = "vault-import-v1"
GENERATED_BY = "rlm-explorer"


def _inside(root: Path, relative: str) -> Path:
    part = Path(relative)
    if part.is_absolute() or ".." in part.parts:
        raise ValueError("Use a relative path inside the vault")
    path = root / part
    if not path.resolve().is_relative_to(root):
        raise ValueError("Path escapes the vault")
    for parent in [path, *path.parents]:
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError("Symlink paths are not imported or exported")
    return path


def _markdown(raw: bytes) -> tuple[str, list[dict]]:
    text = raw.decode("utf-8-sig")
    if text.startswith("---\n"):
        match = re.match(r"\A---\n(.*?)\n---(?:\n|$)", text, re.S)
        if match:
            metadata = yaml.safe_load(match[1]) or {}
            if isinstance(metadata, dict) and metadata.get("generated_by") == GENERATED_BY:
                return "", []
    # Offsets refer to the exact normalized snapshot text, including any frontmatter.
    sections = []
    start, heading = 0, "Note"
    for match in re.finditer(r"^#{1,6} (.+)$", text, re.M):
        if match.start() > start:
            sections.append({"section": heading, "start": start, "end": match.start()})
        start, heading = match.start(), match[1]
    if len(text) > start:
        sections.append({"section": heading, "start": start, "end": len(text)})
    return text, sections


def _pdf(raw: bytes) -> tuple[str, list[dict], list[int]]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(raw))
    text, sections, empty_pages = "", [], []
    for number, page in enumerate(reader.pages, 1):
        page_text = page.extract_text() or ""
        if not page_text.strip():
            empty_pages.append(number)
            continue
        if text:
            text += "\n\n"
        start = len(text)
        text += page_text
        sections.append({"section": f"Page {number}", "page": number,
                         "start": start, "end": len(text)})
    return text, sections, empty_pages


def import_vault(vault: Path, collection: str, output: Path) -> dict:
    """Freeze only the selected collection; never edit or follow links in the source vault."""
    vault = vault.expanduser().resolve(strict=True)
    selected = _inside(vault, collection)
    if not selected.is_dir():
        raise ValueError("Collection must be an existing directory inside the vault")
    if output.resolve().is_relative_to(vault):
        raise ValueError("Keep frozen snapshots outside the vault to avoid recursive imports")
    paths, skipped = [], []
    for path in sorted(selected.rglob("*")):
        relative = path.relative_to(vault).as_posix()
        if any(part.startswith(".") for part in path.relative_to(selected).parts):
            continue
        if path.suffix.lower() not in {".md", ".pdf"}:
            continue
        try:
            _inside(vault, relative)
        except ValueError:
            skipped.append({"path": relative, "reason": "symlink"})
            continue
        if path.is_file():
            paths.append(path)
    if not paths:
        raise ValueError("No Markdown or PDF files found in the selected collection")
    if any(path.suffix.lower() == ".pdf" for path in paths):
        try:
            import pypdf
        except ImportError as exc:
            raise ValueError("PDF import needs pypdf; install the project vault extra") from exc
        pdf_version = pypdf.__version__
    else:
        pdf_version = None
    output.mkdir(parents=True, exist_ok=False)
    (output / "corpus").mkdir()
    (output / "raw").mkdir()
    manifest = {
        "schema_version": "research-snapshot-v1", "parser_version": IMPORT_VERSION,
        "pypdf_version": pdf_version, "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "vault_root": str(vault), "collection": collection, "papers": [],
        "failures": [], "skipped": skipped,
        "coverage_note": "Selected local collection. Notes are personal commentary, not paper "
        "findings. PDF text extraction omits images and may distort tables/equations; no OCR.",
    }
    for path in paths:
        relative = path.relative_to(vault).as_posix()
        try:
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            doc_id = "vault_" + hashlib.sha256(relative.encode()).hexdigest()[:24]
            empty_pages = []
            if path.suffix.lower() == ".md":
                text, sections = _markdown(raw)
                if not text and not sections:
                    manifest["skipped"].append({"path": relative,
                                                "reason": "generated or empty note"})
                    continue
                kind, coverage = "personal_note", "markdown_text"
            else:
                text, sections, empty_pages = _pdf(raw)
                kind, coverage = "pdf_document", "pdf_text_no_ocr"
            if not text.strip():
                raise ValueError("No extractable text; scanned PDFs require OCR")
            source_url = "obsidian://open?" + urlencode({"vault": vault.name, "file": relative})
            metadata = {
                "source_url": source_url, "source_path": relative, "source_kind": kind,
                "source_sha256": digest, "submitted": "unknown", "coverage": coverage,
                "parser_version": IMPORT_VERSION, "empty_pages": empty_pages,
                "authority_note": "File type does not establish authorship or factual accuracy.",
            }
            doc = {"doc_id": doc_id, "title": path.stem, "text": text,
                   "sections": sections, "metadata": metadata}
            corpus_path = output / "corpus" / f"{doc_id}.json"
            corpus_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n")
            (output / "raw" / f"{doc_id}{path.suffix.lower()}").write_bytes(raw)
            manifest["papers"].append({"doc_id": doc_id, "title": path.stem,
                                       **metadata, "sha256": content_hash(corpus_path)})
        except Exception as exc:
            manifest["failures"].append({"path": relative,
                                         "error": f"{type(exc).__name__}: {exc}"})
    manifest["corpus_hash"] = content_hash(output / "corpus")
    complete = bool(manifest["papers"]) and not manifest["failures"]
    manifest["status"] = "complete" if complete else "partial"
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def export_explanation(artifact: Path, snapshot: Path, vault: Path, note: str) -> Path:
    """Recheck exact evidence against the snapshot and create a new, marked Markdown note."""
    vault = vault.expanduser().resolve(strict=True)
    target = _inside(vault, note)
    if target.suffix.lower() != ".md":
        raise ValueError("Export note must end in .md")
    if target.exists():
        raise FileExistsError(target)
    manifest, docs = load_snapshot(snapshot)
    if manifest.get("vault_root") != str(vault):
        raise ValueError("Snapshot belongs to a different vault")
    result = json.loads(artifact.read_text())
    if result.get("status") != "submitted":
        raise ValueError("Only submitted explanations can be exported")
    packet, answer = result["evidence_packet"], result["explanation"]
    if packet.get("corpus_hash") != manifest["corpus_hash"]:
        raise ValueError("Explanation and snapshot corpus hashes differ")
    check_explanation(answer, packet)
    evidence = packet["evidence"]
    ids = [item["evidence_id"] for item in evidence]
    if len(set(ids)) != len(ids) or any(not re.fullmatch(r"E\d+", item) for item in ids):
        raise ValueError("Invalid or duplicate evidence IDs")
    if set(re.findall(r"\[(E\d+)\]", answer["answer"])) - set(ids):
        raise ValueError("Answer references unknown evidence")
    lines = ["---", f"generated_by: {GENERATED_BY}", "review_status: unreviewed", "---", "",
             "# Research explanation", "", str(packet.get("question", "")), "",
             "> AI-generated draft. Check the evidence before relying on the explanation.", "",
             answer["answer"], "", "## Claims", ""]
    for claim in answer["claims"]:
        lines += [claim["text"] + " " + " ".join(f"[{i}]" for i in claim["evidence_ids"]), ""]
    lines += ["## Sources", ""]
    for item in evidence:
        doc = docs.get(item.get("doc_id"))
        start, end = item.get("start"), item.get("end")
        if not (doc and type(start) is int and type(end) is int
                and 0 <= start < end <= len(doc["text"])
                and doc["text"][start:end] == item.get("quote")):
            raise ValueError("Evidence does not match the frozen snapshot")
        metadata = doc["metadata"]
        source = _inside(vault, metadata["source_path"])
        # A Markdown relative link works without a plugin; paths are URL-encoded.
        link = quote(Path(os.path.relpath(source, target.parent)).as_posix(), safe="/")
        pages = [s["page"] for s in doc["sections"]
                 if "page" in s and s["start"] < end and s["end"] > start]
        if pages:
            link += f"#page={pages[0]}"
        changed = (not source.is_file()
                   or hashlib.sha256(source.read_bytes()).hexdigest() != metadata["source_sha256"])
        lines += [f"### {item['evidence_id']}", "",
                  f"[{doc['title']}]({link}) — {metadata['source_kind']}", "",
                  f"Snapshot offsets: {start}–{end}." + (f" PDF pages: {pages}." if pages else ""),
                  ""]
        if changed:
            lines += ["Source has changed or moved since import; the quotation below is from "
                      "the frozen snapshot.", ""]
        lines += ["> " + line for line in item["quote"].splitlines()] + [""]
    lines += ["## Limitations", "", *[f"- {s}" for s in answer["limitations"]], "",
              f"Snapshot: `{manifest['corpus_hash']}`", ""]
    # Turn the explainer's [E1] references into links to headings in this note.
    rendered = re.sub(r"\[(E\d+)\](?!\()", r"[\1](#\1)", "\n".join(lines))
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x") as stream:
        stream.write(rendered)
    return target

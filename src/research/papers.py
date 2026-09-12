"""Download versioned arXiv sources and preserve a reproducible text snapshot."""

from __future__ import annotations

import json
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from src.eval.artifacts import content_hash

ARXIV_ID = re.compile(r"\d{4}\.\d{4,5}(?:v\d+)?")
PARSER_VERSION = "arxiv-paragraphs-v1"


def fetch(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "DocTracerRL-research/0.1"})
    with urlopen(request, timeout=30) as response:
        return response.read()


class MetadataParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.meta: dict[str, list[str]] = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "meta":
            key = attrs.get("name", attrs.get("property", ""))
            self.meta.setdefault(key, []).append(attrs.get("content", ""))


def parse_metadata(html: str) -> dict:
    parser = MetadataParser()
    parser.feed(html)
    meta = parser.meta
    canonical = meta.get("og:url", [""])[0]
    version = canonical.rsplit("/", 1)[-1]
    if not ARXIV_ID.fullmatch(version) or "v" not in version:
        raise ValueError("arXiv page did not identify a pinned paper version")
    title = meta.get("citation_title", [""])[0]
    abstract = meta.get("citation_abstract", [""])[0]
    if not title or not abstract:
        raise ValueError("Missing paper title or abstract")
    return {
        "arxiv_id": version,
        "title": title,
        "authors": meta.get("citation_author", []),
        "submitted": meta.get("citation_date", [""])[0].replace("/", "-"),
        "updated": meta.get("citation_online_date", [""])[0].replace("/", "-"),
        "source_url": f"https://arxiv.org/abs/{version}",
        "abstract": abstract,
    }


class PaperTextParser(HTMLParser):
    """Extract HTML paragraphs/headings, not a faithful PDF/table/math transcription."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.in_article = False
        self.tag: str | None = None
        self.parts: list[str] = []
        self.section = "Preamble"
        self.paragraphs: list[dict] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag == "article":
            self.in_article = True
        if not self.in_article:
            return
        if tag in {"script", "style"}:
            self.skip += 1
        if not self.skip and tag in {"p", "h1", "h2", "h3", "h4"}:
            self.tag = tag
            self.parts = []
        if tag == "br" and self.tag:
            self.parts.append(" ")

    def handle_data(self, data):
        if self.tag and not self.skip:
            self.parts.append(data)

    def handle_endtag(self, tag):
        if tag in {"script", "style"} and self.skip:
            self.skip -= 1
        if tag == self.tag:
            text = " ".join("".join(self.parts).split())
            if text:
                if tag.startswith("h"):
                    self.section = text
                else:
                    self.paragraphs.append({"section": self.section, "text": text})
            self.tag = None
        if tag == "article":
            self.in_article = False


def make_document(metadata: dict, paragraphs: list[dict], coverage: str) -> dict:
    blocks = [{"section": "Abstract", "text": metadata["abstract"]}] + paragraphs
    text = ""
    sections = []
    for block in blocks:
        if text:
            text += "\n\n"
        start = len(text)
        text += block["text"]
        sections.append({"section": block["section"], "start": start, "end": len(text)})
    return {
        "doc_id": "arxiv_" + metadata["arxiv_id"].replace(".", "_"),
        "title": metadata["title"], "text": text, "sections": sections,
        "metadata": {**metadata, "coverage": coverage, "parser_version": PARSER_VERSION},
    }


def build_snapshot(ids: list[str], output: Path, downloader=fetch, delay: float = 3.1) -> dict:
    if not ids or any(not ARXIV_ID.fullmatch(i) for i in ids):
        raise ValueError("Supply modern arXiv IDs such as 2503.09516v1")
    # A snapshot is immutable: a new directory for every refresh, including failed builds.
    output.mkdir(parents=True, exist_ok=False)
    corpus = output / "corpus"
    raw = output / "raw"
    corpus.mkdir()
    raw.mkdir()
    manifest = {
        "schema_version": "research-snapshot-v1", "parser_version": PARSER_VERSION,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "requested_ids": ids, "papers": [], "failures": [],
        "coverage_note": "Selected papers, not an exhaustive or latest-literature search. "
        "HTML extraction omits tables/figures and may distort equations.",
    }
    for index, paper_id in enumerate(ids):
        if index:
            time.sleep(delay)
        try:
            page = downloader(f"https://arxiv.org/abs/{paper_id}")
            metadata = parse_metadata(page.decode("utf-8"))
            version = metadata["arxiv_id"]
            if paper_id.split("v")[0] != version.split("v")[0]:
                raise ValueError(f"Requested {paper_id}, received a different paper: {version}")
            if "v" in paper_id and paper_id != version:
                raise ValueError(f"Requested {paper_id}, received {version}")
            (raw / f"{version}.abs.html").write_bytes(page)
            paragraphs = []
            coverage = "abstract_only"
            html_error = None
            time.sleep(delay)
            try:
                html = downloader(f"https://arxiv.org/html/{version}")
                (raw / f"{version}.html").write_bytes(html)
                parser = PaperTextParser()
                parser.feed(html.decode("utf-8"))
                # Do not promote an abstract or an error page to full paper evidence.
                if len(parser.paragraphs) >= 5:
                    paragraphs = parser.paragraphs
                    coverage = "html_paragraphs"
                else:
                    html_error = "Insufficient article paragraphs; kept abstract only"
            except Exception as exc:
                html_error = f"{type(exc).__name__}: {exc}"
            doc = make_document(metadata, paragraphs, coverage)
            path = corpus / f"{doc['doc_id']}.json"
            path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n")
            manifest["papers"].append({
                "doc_id": doc["doc_id"], "arxiv_id": version, "title": doc["title"],
                "source_url": metadata["source_url"], "coverage": coverage,
                "html_error": html_error, "sha256": content_hash(path),
            })
        except Exception as exc:
            manifest["failures"].append({"id": paper_id, "error": f"{type(exc).__name__}: {exc}"})
    manifest["corpus_hash"] = content_hash(corpus)
    manifest["raw_sources_hash"] = content_hash_all(raw)
    manifest["status"] = "complete" if not manifest["failures"] else "partial"
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def content_hash_all(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    for item in sorted(path.iterdir()):
        digest.update(item.name.encode() + b"\0" + item.read_bytes() + b"\0")
    return digest.hexdigest()


def discover(query: str, since: str, until: str, limit: int = 20, downloader=fetch) -> dict:
    """Bounded live discovery, kept separate from frozen evaluation snapshots."""
    start = datetime.strptime(since, "%Y-%m-%d")
    end = datetime.strptime(until, "%Y-%m-%d")
    if start > end or not 1 <= limit <= 100:
        raise ValueError("Require since <= until and limit between 1 and 100")
    search = f"({query}) AND submittedDate:[{start:%Y%m%d}0000 TO {end:%Y%m%d}2359]"
    url = "https://export.arxiv.org/api/query?" + urlencode({
        "search_query": search, "sortBy": "submittedDate", "sortOrder": "descending",
        "start": 0, "max_results": limit,
    })
    root = ET.fromstring(downloader(url))
    ns = {"a": "http://www.w3.org/2005/Atom"}
    papers = []
    for entry in root.findall("a:entry", ns):
        identifier = entry.findtext("a:id", "", ns).rsplit("/", 1)[-1]
        if not ARXIV_ID.fullmatch(identifier):
            raise ValueError("arXiv returned an error or unsupported paper identifier")
        papers.append({
            "arxiv_id": identifier,
            "title": " ".join(entry.findtext("a:title", "", ns).split()),
            "abstract": " ".join(entry.findtext("a:summary", "", ns).split()),
            "submitted": entry.findtext("a:published", "", ns),
            "updated": entry.findtext("a:updated", "", ns),
            "source_url": f"https://arxiv.org/abs/{identifier}",
        })
    return {"query_url": url, "since": since, "until": until, "limit": limit,
            "retrieved_at": datetime.now(timezone.utc).isoformat(), "papers": papers,
            "coverage_note": "Bounded arXiv search; not an exhaustive literature review."}

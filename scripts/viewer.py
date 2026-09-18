"""Local web demo and trajectory viewer for Envoy policies.

Serves a visual UI. Supports:
  - Browsing past runs (GET /api/runs)
  - Live step streaming via WebSocket (ws://.../ws/live)
  - Runtime policy selection, including OpenAI-compatible model servers
  - Discovered benchmark/vault corpora and custom questions
  - Pre-recorded replays (GET /api/replays)
  - BYOK (bring your own API key) or free demo tier

Usage:
    python scripts/viewer.py              # default port 8000
    python scripts/viewer.py --port 3000
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import threading
import time
from pathlib import Path

import typer
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from src.env.corpus import Corpus
from src.eval.live_harness import StepEvent, run_parallel
from src.policies.registry import (
    build_policies as build_registered_policies,
)
from src.policies.registry import (
    policy_catalog,
    requires_anthropic_key,
)

# An exported-but-EMPTY key shadows .env: load_dotenv() defaults to
# override=False and treats "" as already-set, so the blank value wins and
# every downstream key check fails with a confusing "missing API key".
# Drop empties first, so .env fills them without stomping real values.
for _k in ("ANTHROPIC_API_KEY", "DEMO_API_KEY"):
    if os.environ.get(_k, None) == "":
        del os.environ[_k]
load_dotenv()

ROOT_DIR = Path(__file__).parent.parent
OUT_DIR = ROOT_DIR / "out"
TEMPLATE_DIR = Path(__file__).parent / "templates"

# True when serving on loopback only; enables the ANTHROPIC_API_KEY fallback.
_LOCAL_ONLY = True
REPLAY_DIR = Path(__file__).parent / "replays"

app = FastAPI(title="Envoy — Trajectory Viewer")
cli = typer.Typer()

# ---------------------------------------------------------------------------
# Rate limiting (in-memory, resets on restart — fine for a demo)
# ---------------------------------------------------------------------------
_rate_limits: dict[str, list[float]] = {}  # ip -> list of timestamps
BYOK_LIMIT = 5  # runs/day with own key
FREE_LIMIT = 2  # runs/day with demo key
_race_tokens: dict[str, tuple[float, bool]] = {}  # token -> (created_at, is_free)


def _check_rate(ip: str, is_free: bool) -> bool:
    """Return True if the IP is within its daily rate limit."""
    if _LOCAL_ONLY:
        return True
    limit = FREE_LIMIT if is_free else BYOK_LIMIT
    now = time.time()
    day_ago = now - 86400
    timestamps = _rate_limits.get(ip, [])
    timestamps = [t for t in timestamps if t > day_ago]
    _rate_limits[ip] = timestamps
    return len(timestamps) < limit


def _record_use(ip: str) -> None:
    if _LOCAL_ONLY:
        return
    _rate_limits.setdefault(ip, []).append(time.time())


# ---------------------------------------------------------------------------
# Corpus cache — loaded once on first live eval
# ---------------------------------------------------------------------------
_corpus_cache: dict[str, Corpus] = {}


def get_corpus(corpus_path: str) -> Corpus:
    if corpus_path not in _corpus_cache:
        c = Corpus(corpus_path=corpus_path)
        c.load()
        _corpus_cache[corpus_path] = c
    return _corpus_cache[corpus_path]


def _read_questions(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        data = data.get("questions", [])
    return data if isinstance(data, list) else []


def _corpus_sources() -> dict[str, dict]:
    """Discover safe, server-side corpora available to the local demo."""
    sources: dict[str, dict] = {}

    def add(source_id: str, label: str, corpus: Path, questions: list[Path]) -> None:
        if corpus.is_dir():
            sources[source_id] = {
                "id": source_id,
                "label": label,
                "corpus": corpus,
                "question_files": [path for path in questions if path.exists()],
            }

    add(
        "synthetic",
        "Synthetic sanity corpus",
        ROOT_DIR / "data/corpus",
        [
            ROOT_DIR / "data/questions/eval_set.json",
            ROOT_DIR / "data/questions/hard_eval_set.json",
        ],
    )
    add(
        "musique",
        "MuSiQue",
        ROOT_DIR / "data/musique/corpus",
        [ROOT_DIR / "data/musique/questions/eval_set.json"],
    )

    research_dir = OUT_DIR / "research"
    if research_dir.exists():
        for snapshot in sorted(research_dir.iterdir()):
            if not snapshot.is_dir():
                continue
            add(
                f"research--{snapshot.name}",
                snapshot.name,
                snapshot / "corpus",
                [snapshot / "benchmark.json", snapshot / "questions.json"],
            )
    return sources


def build_policies(
    corpus: Corpus,
    names: list[str],
    api_key: str | None = None,
) -> dict[str, object]:
    return build_registered_policies(corpus, names, api_key=api_key)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((TEMPLATE_DIR / "viewer.html").read_text())


@app.get("/api/policies")
def list_policies() -> JSONResponse:
    return JSONResponse(policy_catalog(allow_anthropic_byok=True))


@app.get("/api/corpora")
def list_corpora() -> JSONResponse:
    result = []
    for source in _corpus_sources().values():
        questions = [
            question for path in source["question_files"] for question in _read_questions(path)
        ]
        result.append(
            {
                "id": source["id"],
                "label": source["label"],
                "question_count": len(questions),
            }
        )
    return JSONResponse(result)


@app.get("/api/runs")
def list_runs() -> JSONResponse:
    runs = sorted(OUT_DIR.glob("run_*.json"), reverse=True)
    return JSONResponse(
        [
            {"name": r.stem, "path": r.name, "size_kb": round(r.stat().st_size / 1024, 1)}
            for r in runs
            if not r.name.endswith(".manifest.json")
        ]
    )


@app.get("/api/runs/{filename}")
def get_run(filename: str) -> JSONResponse:
    path = OUT_DIR / filename
    if not path.exists() or path.suffix != ".json":
        return JSONResponse({"error": "not found"}, status_code=404)
    with open(path) as f:
        return JSONResponse(json.load(f))


# Question cache
_question_cache: dict[str, dict] = {}


def _load_questions(corpus_name: str) -> list[dict]:
    """Load and cache all questions for a corpus."""
    source = _corpus_sources().get(corpus_name)
    if source is None:
        return []
    questions = [
        question for path in source["question_files"] for question in _read_questions(path)
    ]
    for q in questions:
        _question_cache[f"{corpus_name}:{q['id']}"] = q
    return questions


@app.get("/api/questions/{corpus_name}")
def list_questions(corpus_name: str) -> JSONResponse:
    """List available questions for a corpus."""
    questions = _load_questions(corpus_name)
    return JSONResponse([{"id": q["id"], "question": q["question"]} for q in questions])


# ---------------------------------------------------------------------------
# Rate-limit token endpoint
# ---------------------------------------------------------------------------


@app.post("/api/race/start")
async def race_start(request: Request) -> JSONResponse:
    """Acquire a single-use race token (rate-limited)."""
    body = await request.json()
    has_own_key = bool(body.get("api_key"))
    ip = request.client.host if request.client else "unknown"
    is_free = not has_own_key

    if not _check_rate(ip, is_free):
        limit = FREE_LIMIT if is_free else BYOK_LIMIT
        return JSONResponse(
            {"error": f"Rate limit exceeded ({limit} runs/day). Try again tomorrow."},
            status_code=429,
        )

    _record_use(ip)
    token = secrets.token_urlsafe(16)
    _race_tokens[token] = (time.time(), is_free)
    return JSONResponse({"token": token})


# ---------------------------------------------------------------------------
# Replays
# ---------------------------------------------------------------------------


@app.get("/api/replays")
def list_replays() -> JSONResponse:
    """List available pre-recorded replays."""
    if not REPLAY_DIR.exists():
        return JSONResponse([])
    replays = sorted(REPLAY_DIR.glob("*.json"))
    result = []
    for r in replays:
        with open(r) as f:
            data = json.load(f)
        meta = data.get("meta", {})
        result.append(
            {
                "name": r.stem,
                "question": meta.get("question", r.stem),
                "question_id": meta.get("question_id", ""),
            }
        )
    return JSONResponse(result)


@app.get("/api/replays/{name}")
def get_replay(name: str) -> JSONResponse:
    """Return a pre-recorded replay."""
    path = REPLAY_DIR / f"{name}.json"
    if not path.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    with open(path) as f:
        return JSONResponse(json.load(f))


# ---------------------------------------------------------------------------
# Live eval WebSocket
# ---------------------------------------------------------------------------


@app.websocket("/ws/live")
async def live_eval(ws: WebSocket) -> None:
    """Stream a parallel eval over WebSocket.

    Client sends: {"token": "...", "api_key": "...", "corpus": "musique",
                   "question": {...}, "policies": [...], "max_steps": 10}
    Server streams: StepEvent dicts as they happen.
    """
    await ws.accept()
    try:
        msg = await ws.receive_json()

        # Validate race token
        token = msg.get("token")
        if not token or token not in _race_tokens:
            await ws.send_json(
                {
                    "type": "error",
                    "message": "Invalid or missing race token. Call POST /api/race/start first.",
                }
            )
            return
        created_at, _is_free = _race_tokens.pop(token)
        if time.time() - created_at > 120:
            await ws.send_json(
                {
                    "type": "error",
                    "message": "Race token expired. Please try again.",
                }
            )
            return

        policy_names = msg.get("policies", ["claude_policy"])
        if not isinstance(policy_names, list) or not policy_names:
            await ws.send_json({"type": "error", "message": "Select at least one policy."})
            return

        # Resolve an Anthropic key only when the selected policy needs one.
        needs_anthropic = requires_anthropic_key(policy_names)
        api_key = msg.get("api_key") or None
        if needs_anthropic and not api_key:
            api_key = os.environ.get("DEMO_API_KEY")
        if needs_anthropic and not api_key and _LOCAL_ONLY:
            # Local dev convenience: use the developer's own key from .env.
            # Gated on binding to loopback so a network-exposed instance can
            # never spend a personal key on behalf of an anonymous visitor.
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        if needs_anthropic and not api_key:
            await ws.send_json(
                {
                    "type": "error",
                    "message": "Please provide an Anthropic API key.",
                }
            )
            return

        corpus_name = msg.get("corpus", "musique")
        source = _corpus_sources().get(corpus_name)
        if source is None:
            await ws.send_json({"type": "error", "message": "Unknown corpus selection."})
            return
        corpus_path = str(source["corpus"])

        # Look up full question from server-side cache
        q_from_client = msg["question"]
        cache_key = f"{corpus_name}:{q_from_client['id']}"
        if cache_key not in _question_cache:
            _load_questions(corpus_name)
        question = _question_cache.get(cache_key)
        if question is None:
            question = {
                "id": str(q_from_client.get("id", "custom")),
                "question": str(q_from_client["question"]),
                "answer": "",
                "expected_citations": [],
                "scored": False,
            }
        max_steps = msg.get("max_steps", 10)

        corpus = get_corpus(corpus_path)
        policies = build_policies(corpus, policy_names, api_key=api_key)
        if not policies:
            await ws.send_json({"type": "error", "message": "Unknown policy selection."})
            return

        loop = asyncio.get_event_loop()

        def on_step(event: StepEvent) -> None:
            asyncio.run_coroutine_threadsafe(
                ws.send_json({"type": "step", **event.to_dict()}),
                loop,
            )

        def run() -> None:
            run_parallel(
                corpus,
                question,
                policies,
                max_steps,
                corpus_path,
                on_step,
                include_preamble=False,
            )
            asyncio.run_coroutine_threadsafe(
                ws.send_json({"type": "complete"}),
                loop,
            )

        thread = threading.Thread(target=run, daemon=True)
        thread.start()

        while thread.is_alive():
            await asyncio.sleep(0.5)
        thread.join(timeout=5)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


@cli.command()
def main(
    port: int = typer.Option(8000, help="Port to serve on"),
    host: str = typer.Option(
        "127.0.0.1", help="Bind address. Use 0.0.0.0 to expose on your network."
    ),
) -> None:
    """Launch the trajectory viewer."""
    global _LOCAL_ONLY
    _LOCAL_ONLY = host in ("127.0.0.1", "localhost", "::1")
    if not _LOCAL_ONLY:
        typer.echo(
            "WARNING: binding to a non-loopback address. The ANTHROPIC_API_KEY\n"
            "         fallback is disabled; visitors must supply their own key."
        )
    typer.echo(f"Envoy Viewer -> http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    cli()

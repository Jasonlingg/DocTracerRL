"""Web viewer for eval trajectories — with live parallel streaming.

Serves a visual UI. Supports:
  - Browsing past runs (GET /api/runs)
  - Live parallel eval via WebSocket (ws://.../ws/live)
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
import time
import threading
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import typer
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse

from src.env.corpus import Corpus
from src.eval.live_harness import StepEvent, run_parallel
from src.policies.claude_policy import ClaudePolicy
from src.policies.naive_rag import NaiveRAGPolicy
from src.policies.qwen_base_policy import QwenBasePolicy
from src.policies.qwen_sft_policy import QwenSFTPolicy
from src.policies.grpo_policy import GRPOPolicy
from src.policies.single_shot import SingleShotPolicy
from src.policies.sparse_rag import SparseRAGPolicy
from src.policies.stuffing import ContextStuffingPolicy

OUT_DIR = Path(__file__).parent.parent / "out"
TEMPLATE_DIR = Path(__file__).parent / "templates"
REPLAY_DIR = Path(__file__).parent / "replays"

app = FastAPI(title="RLM Explorer — Trajectory Viewer")
cli = typer.Typer()

# ---------------------------------------------------------------------------
# Rate limiting (in-memory, resets on restart — fine for a demo)
# ---------------------------------------------------------------------------
_rate_limits: dict[str, list[float]] = {}  # ip -> list of timestamps
BYOK_LIMIT = 5       # runs/day with own key
FREE_LIMIT = 2        # runs/day with demo key
_race_tokens: dict[str, tuple[float, bool]] = {}  # token -> (created_at, is_free)


def _check_rate(ip: str, is_free: bool) -> bool:
    """Return True if the IP is within its daily rate limit."""
    limit = FREE_LIMIT if is_free else BYOK_LIMIT
    now = time.time()
    day_ago = now - 86400
    timestamps = _rate_limits.get(ip, [])
    timestamps = [t for t in timestamps if t > day_ago]
    _rate_limits[ip] = timestamps
    return len(timestamps) < limit


def _record_use(ip: str) -> None:
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


def build_policies(
    corpus: Corpus, names: list[str], api_key: str | None = None,
) -> dict[str, object]:
    factories = {
        "claude_policy": lambda: ClaudePolicy(api_key=api_key),
        "naive_rag": lambda: NaiveRAGPolicy(corpus=corpus, api_key=api_key),
        "sparse_rag": lambda: SparseRAGPolicy(corpus=corpus, api_key=api_key),
        "context_stuffing": lambda: ContextStuffingPolicy(corpus=corpus, api_key=api_key),
        "single_shot": lambda: SingleShotPolicy(corpus=corpus, api_key=api_key),
        "qwen_base_policy": lambda: QwenBasePolicy(),
        "qwen_sft_policy": lambda: QwenSFTPolicy(),
        "grpo_policy": lambda: GRPOPolicy(),
    }
    return {n: factories[n]() for n in names if n in factories}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((TEMPLATE_DIR / "viewer.html").read_text())


@app.get("/api/runs")
def list_runs() -> JSONResponse:
    runs = sorted(OUT_DIR.glob("run_*.json"), reverse=True)
    return JSONResponse([
        {"name": r.stem, "path": r.name, "size_kb": round(r.stat().st_size / 1024, 1)}
        for r in runs
    ])


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
    base = Path("data") / (corpus_name if corpus_name != "synthetic" else "") / "questions"
    if corpus_name == "synthetic":
        base = Path("data/questions")
    candidates = [base / "eval_set.json", base / "hard_eval_set.json"]
    questions = []
    for p in candidates:
        if p.exists():
            with open(p) as f:
                questions.extend(json.load(f))
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
        result.append({
            "name": r.stem,
            "question": meta.get("question", r.stem),
            "question_id": meta.get("question_id", ""),
        })
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
            await ws.send_json({"type": "error", "message": "Invalid or missing race token. Call POST /api/race/start first."})
            return
        created_at, is_free = _race_tokens.pop(token)
        if time.time() - created_at > 120:
            await ws.send_json({"type": "error", "message": "Race token expired. Please try again."})
            return

        # Resolve API key: user-provided or demo fallback
        api_key = msg.get("api_key") or None
        if not api_key:
            api_key = os.environ.get("DEMO_API_KEY")
        if not api_key:
            await ws.send_json({"type": "error", "message": "Please provide an Anthropic API key."})
            return

        corpus_name = msg.get("corpus", "musique")
        corpus_path = f"data/{corpus_name}/corpus" if corpus_name != "synthetic" else "data/corpus"

        # Look up full question from server-side cache
        q_from_client = msg["question"]
        cache_key = f"{corpus_name}:{q_from_client['id']}"
        if cache_key not in _question_cache:
            _load_questions(corpus_name)
        question = _question_cache.get(cache_key, q_from_client)
        policy_names = msg.get("policies", ["claude_policy", "naive_rag", "context_stuffing", "single_shot"])
        max_steps = msg.get("max_steps", 10)

        corpus = get_corpus(corpus_path)
        policies = build_policies(corpus, policy_names, api_key=api_key)

        loop = asyncio.get_event_loop()

        def on_step(event: StepEvent) -> None:
            asyncio.run_coroutine_threadsafe(
                ws.send_json({"type": "step", **event.to_dict()}), loop,
            )

        def run() -> None:
            run_parallel(corpus, question, policies, max_steps, corpus_path, on_step)
            asyncio.run_coroutine_threadsafe(
                ws.send_json({"type": "complete"}), loop,
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
def main(port: int = typer.Option(8000, help="Port to serve on")) -> None:
    """Launch the trajectory viewer."""
    typer.echo(f"RLM Explorer Viewer -> http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")


if __name__ == "__main__":
    cli()

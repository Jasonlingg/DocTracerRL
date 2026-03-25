"""Local web viewer for eval trajectories — with live parallel streaming.

Serves a visual UI on localhost. Supports:
  - Browsing past runs (GET /api/runs)
  - Live parallel eval via WebSocket (ws://localhost:8000/ws/live)

Usage:
    python scripts/viewer.py              # default port 8000
    python scripts/viewer.py --port 3000
"""

from __future__ import annotations

import asyncio
import json

from dotenv import load_dotenv
load_dotenv()
import threading
from pathlib import Path

import typer
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from src.env.corpus import Corpus
from src.eval.live_harness import StepEvent, run_parallel
from src.policies.claude_policy import ClaudePolicy
from src.policies.naive_rag import NaiveRAGPolicy
from src.policies.single_shot import SingleShotPolicy
from src.policies.stuffing import ContextStuffingPolicy

OUT_DIR = Path(__file__).parent.parent / "out"
TEMPLATE_DIR = Path(__file__).parent / "templates"

app = FastAPI(title="RLM Explorer — Trajectory Viewer")
cli = typer.Typer()

# Corpus cache — loaded once on first live eval
_corpus_cache: dict[str, Corpus] = {}


def get_corpus(corpus_path: str) -> Corpus:
    if corpus_path not in _corpus_cache:
        c = Corpus(corpus_path=corpus_path)
        c.load()
        _corpus_cache[corpus_path] = c
    return _corpus_cache[corpus_path]


def build_policies(corpus: Corpus, names: list[str]) -> dict[str, object]:
    factories = {
        "claude_policy": lambda: ClaudePolicy(),
        "naive_rag": lambda: NaiveRAGPolicy(corpus=corpus),
        "context_stuffing": lambda: ContextStuffingPolicy(corpus=corpus),
        "single_shot": lambda: SingleShotPolicy(corpus=corpus),
    }
    return {n: factories[n]() for n in names if n in factories}


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


# Question cache — keyed by (corpus_name, question_id) → full question dict
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


@app.websocket("/ws/live")
async def live_eval(ws: WebSocket) -> None:
    """Stream a parallel eval over WebSocket.

    Client sends: {"corpus": "musique", "question": {...}, "policies": [...], "max_steps": 10}
    Server streams: StepEvent dicts as they happen, one per policy per step.
    """
    await ws.accept()
    try:
        msg = await ws.receive_json()
        corpus_name = msg.get("corpus", "musique")
        corpus_path = f"data/{corpus_name}/corpus" if corpus_name != "synthetic" else "data/corpus"

        # Look up full question (with answer + citations) from server-side cache
        q_from_client = msg["question"]
        cache_key = f"{corpus_name}:{q_from_client['id']}"
        if cache_key not in _question_cache:
            _load_questions(corpus_name)
        question = _question_cache.get(cache_key, q_from_client)
        policy_names = msg.get("policies", ["claude_policy", "naive_rag", "context_stuffing", "single_shot"])
        max_steps = msg.get("max_steps", 10)

        corpus = get_corpus(corpus_path)
        policies = build_policies(corpus, policy_names)

        loop = asyncio.get_event_loop()

        def on_step(event: StepEvent) -> None:
            asyncio.run_coroutine_threadsafe(
                ws.send_json({"type": "step", **event.to_dict()}), loop,
            )

        # Run parallel eval in a background thread to avoid blocking the event loop
        def run() -> None:
            run_parallel(corpus, question, policies, max_steps, corpus_path, on_step)
            asyncio.run_coroutine_threadsafe(
                ws.send_json({"type": "complete"}), loop,
            )

        thread = threading.Thread(target=run, daemon=True)
        thread.start()

        # Keep connection alive until eval completes or client disconnects
        while thread.is_alive():
            await asyncio.sleep(0.5)
        thread.join(timeout=5)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        await ws.send_json({"type": "error", "message": str(e)})


@cli.command()
def main(port: int = typer.Option(8000, help="Port to serve on")) -> None:
    """Launch the trajectory viewer on localhost."""
    typer.echo(f"RLM Explorer Viewer -> http://localhost:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    cli()

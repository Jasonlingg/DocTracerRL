# Envoy — Implementation Plan

## Context

Building a Gym-compatible RL environment for training LLMs to explore document collections via code execution in a persistent REPL. The repo is empty (LICENSE + .gitattributes only). The user provided a comprehensive build spec — this plan follows it directly across 8 sequential steps.

## Implementation Steps

### Step 1: Scaffold
Create full directory structure and config files.

**Files to create:**
- `pyproject.toml` — dependencies (anthropic, pydantic, docker, sentence-transformers, faiss-cpu, rich, typer, loguru, etc.)
- `configs/default.yaml` — model, env, corpus, eval settings
- `Dockerfile` — python:3.11-slim, pandas/numpy/regex/tabulate, non-root user, `CMD python3 -i`
- `docker-compose.yaml` — sandbox service, corpus mount, network_mode none, mem_limit 512m
- `.gitignore` — data/corpus/, __pycache__, .env, etc.
- `CLAUDE.md` — project description, structure, dev commands, code style
- All `__init__.py` files: `src/env/`, `src/policies/`, `src/eval/`
- Empty `data/questions/`, `data/corpus/`, `scripts/`, `tests/` dirs

**Verify:** `pip install -e ".[dev]"` succeeds

---

### Step 2: Synthetic Corpus + Indexing

**Files to create:**
- `scripts/setup_corpus.py` — generate 20-30 cross-referencing business docs (financial reports, contracts, compliance reports) as JSON in `data/corpus/`
- `src/env/corpus.py` — `Corpus` class with `load()`, `search(query, top_k)`, `read(doc_id)`, `list_documents()`. Uses sentence-transformers (all-MiniLM-L6-v2) + FAISS internally.

**Key detail:** Documents must cross-reference each other for multi-hop questions.

**Verify:** `python scripts/setup_corpus.py` creates files; `Corpus.search("revenue")` returns relevant chunks.

---

### Step 3: Persistent REPL

**Files to create:**
- `src/env/repl.py` — `PersistentREPL` class: `start_session()`, `execute(code)`, `kill_session()`
  - Docker mode: create container, cumulative script approach (append code to /tmp/step.py, run full script each step)
  - Local fallback: subprocess.Popen with `python3 -u -i` via stdin/stdout pipes
- `src/env/tools.py` — tool preamble string with `search()`, `read()`, `extract()`, `aggregate()`, `_memo` cache

**Key detail:** State persistence via cumulative script — each step appends to and re-runs the full script.

**Verify:** Execute `x = 42` in step 1, `print(x)` in step 2 returns `42`.

---

### Step 4: Gym Environment (CORE)

**Files to create:**
- `src/env/document_env.py` — `DocumentExplorationEnv` with `reset()`, `step(action)`, `compute_reward()`, `get_trajectory()`, `close()`
  - Action parsing: "SUBMIT:" prefix → parse answer + citations; else → execute as code
  - Reward = 0 during exploration, verifiable score on submission
- `src/env/reward.py` — `compute_reward()`, `score_answer()` (token overlap F1), `score_citations()` (precision/recall), `efficiency_bonus()`
  - Formula: 0.5 * answer + 0.25 * citation_P + 0.25 * citation_R + efficiency_bonus

**Verify:** Full reset → step(code) → step(code) → step(SUBMIT) loop produces reward > 0.

---

### Step 5: Policies

**Files to create:**
- `src/policies/claude_policy.py` — `ClaudePolicy` with `act(observation)`, `reset()`. Uses anthropic SDK, maintains conversation history. System prompt instructs code-writing exploration.
- `src/policies/naive_rag.py` — `NaiveRAGPolicy`: retrieve top-k, call Claude with chunks, SUBMIT in 1 step
- `src/policies/stuffing.py` — `ContextStuffingPolicy`: concatenate all docs, call Claude, SUBMIT in 1 step
- `src/policies/single_shot.py` — `SingleShotPolicy`: minimal retrieval (top_k=3), SUBMIT in 1 step

**Verify:** Each policy produces a trajectory with a reward when run through the environment.

---

### Step 6: Eval Harness + Question Set

**Files to create:**
- `data/questions/eval_set.json` — 15-20 questions referencing the synthetic corpus. Types: cross_document_aggregation, cross_document_comparison, multi_hop_reasoning, single_document_extraction, contradiction_detection
- `src/eval/harness.py` — `run_eval(config)` → runs each policy × each question through env, collects `EvalResult`
- `src/eval/scorer.py` — scoring functions (may be thin wrapper around reward.py)
- `src/eval/report.py` — markdown results table with rich output
- `scripts/run_eval.py` — Typer CLI with `--policy`, `--question`, `--verbose` flags

**Verify:** `python scripts/run_eval.py` produces a results table.

---

### Step 7: Tests

**Files to create:**
- `tests/test_corpus.py` — load, search, read
- `tests/test_repl.py` — state persistence, timeout, local fallback, cleanup
- `tests/test_env.py` — reset, step, submit, reward, max steps
- `tests/test_reward.py` — score_answer, score_citations with known inputs

**Verify:** `pytest tests/ -v` passes.

---

### Step 8: README

**File to create:**
- `README.md` — hook paragraph, core loop explanation, architecture, quick start, results table, future work, license

**Verify:** A stranger can clone, follow README, reproduce results.

---

## Execution Notes

- Each step is self-contained and builds on previous steps
- Files must stay under 200 lines — split aggressively
- Type hints on all functions, Pydantic models for data structures
- No LangChain, no OpenAI — Claude-native only (anthropic SDK, claude-sonnet-4-20250514)
- loguru for logging, rich for CLI output

# RLM Explorer — Status

*Last updated: 2026-03-19*

## What's Working

### Environment (core product)
- **Gym-compatible loop**: `reset()` → `step(code)` → `step(SUBMIT)` → reward
- **Persistent REPL**: Cumulative script execution via local subprocess (Docker mode implemented but not yet active — needs `docker build -t rlm-sandbox .`)
- **REPL output isolation**: Only the current step's output is returned to the agent (see fix below)
- **REPL rollback**: SyntaxErrors auto-rollback the cumulative script so one bad step doesn't poison the rest
- **Corpus**: 43 cross-referencing synthetic business documents (~180 chunks), with distractor entities for disambiguation
- **FAISS + sentence-transformers**: Embedding-based search over corpus chunks
- **Tool preamble**: `search()`, `read()`, `extract()`, `search_within()`, `verify()`, `list_docs()` injected into REPL
- **Verifiable reward**: `0.5 × answer_F1 + 0.25 × citation_precision + 0.25 × citation_recall + efficiency_bonus`

### Policies
- **Claude policy** (reference): Haiku explores iteratively via code, multi-hop capable
- **Naive RAG baseline**: Top-k retrieve → answer in 1 step
- **Context stuffing baseline**: Concatenate all docs → answer in 1 step
- **Single-shot baseline**: Minimal retrieval → answer in 1 step

### Eval
- **18 easy questions** across 5 types: aggregation, comparison, multi-hop, extraction, contradiction
- **12 hard multi-hop questions** (2-5 hops): hidden bridge, disambiguation, fan-out aggregation, codename bridging, parallel comparison
- **Eval harness**: Runs policies × questions, collects trajectories + rewards
- **Transcript output**: JSON transcripts saved to `out/` with full trajectories
- **`--hard` flag**: Use hard multi-hop question set
- **Rich CLI**: Tables for per-question and per-policy results

### Full Hard Eval Results (12 questions, 2026-03-18)

| Policy | Avg Reward | Avg Answer | Avg Steps |
|--------|-----------|-----------:|----------:|
| context_stuffing | **0.877** | 0.592 | 1.0 |
| single_shot | 0.569 | 0.332 | 1.1 |
| naive_rag | 0.503 | 0.268 | 1.3 |
| claude_policy | 0.349 | 0.259 | 8.0 |

**Problem: Context stuffing dominates. Iterative agent is dead last.**

- Context stuffing sees all 43 docs (~86K chars) at once → answers correctly in 1 step
- Claude policy (Haiku) timed out on 5/12, early-quit on 1, errored on 1
- Only 5/12 questions where claude_policy actually produced an answer
- When claude_policy *does* answer (h03, h04, h09), it often beats baselines — the exploration works, it just fails too often

Full results: [out/hard_eval_results.md](out/hard_eval_results.md)

---

## Key Bugs Fixed

### 6. REPL Output Flooding (2026-03-18) — CRITICAL FIX

**Problem**: The iterative agent (claude_policy) was performing *worse* than naive RAG baselines on hard multi-hop questions (0.161 avg vs 0.399 for naive_rag). The agent would time out after 10 steps, get stuck in loops reading the same document repeatedly, and hallucinate after SyntaxErrors.

**Root Cause**: The REPL uses a *cumulative script* architecture — every step appends code to a growing script that gets re-executed from scratch as a fresh subprocess. This means ALL `print()` output from ALL previous steps gets re-emitted every time. After reading a 4000-char document in step 2, steps 3-10 would all see that same 4000 chars of text prepended to their actual output. By step 5+, the 8000-char output cap (`MAX_OUTPUT_CHARS`) was entirely filled with stale output from earlier steps, and the agent never saw its current step's results.

**Fix** (`src/env/repl.py`):
1. Added a `STEP_MARKER` sentinel string (`___STEP_OUTPUT_MARKER___`) that gets printed immediately before each new step's code via `print("___STEP_OUTPUT_MARKER___")`
2. After execution, `_extract_step_output()` splits on the **last** marker occurrence and returns only the output after it
3. This means the agent now sees ONLY the output from its current step — clean, focused observations
4. Added smart truncation: if a single step's output exceeds `MAX_OUTPUT_CHARS`, keeps the first and last halves with a `[N chars truncated]` gap

**Impact**: h01 went from 0.000 (timeout after 10 steps, agent stuck re-reading board minutes) to **0.993** reward in just 2 steps. The agent successfully followed the multi-hop chain: search → find codename registry reference → look up partner → find compliance report.

### 5. Agent Prompt Improvements (2026-03-18)

**Changes** (`src/policies/claude_policy.py`):
- Added explicit multi-hop exploration STRATEGY section to the system prompt
- Key instructions: break questions into sub-questions, follow codename/vendor code/client ID references to other documents, use `search_within()` for long docs, never re-read documents, track discoveries in `known_facts = {}`
- Examples now demonstrate the discovery chain pattern (search → find reference → follow it → track facts → submit) instead of naive read-everything approach

### 4. Environment Step Counter (2026-03-18)

**Changes** (`src/env/document_env.py`):
- Each observation now includes `[Step N/M]` footer so the agent knows its budget
- When 3 or fewer steps remain, adds urgency: `[Step 8/10 — 2 steps remaining. Submit soon!]`
- Prevents the agent from over-exploring and timing out without submitting

### 3. Haiku XML generation
Haiku was generating `<function_calls>` XML instead of Python. Fixed with `_clean_action()` that strips XML, markdown fences, and prose.

### 2. Prose contamination
Haiku mixed English prose between code lines. Fixed by filtering all non-code lines and extracting SUBMIT lines first.

### 1. Cumulative REPL poisoning
One SyntaxError on step 1 poisoned all 10 subsequent steps. Fixed with auto-rollback on SyntaxError.

---

## Corpus Design (Phase 1-2, completed)

### Distractor Documents (8 new files)
- **Sterling Energy**: 2nd Houston energy company — defeats "Houston energy = Crestline" shortcut
- **NorthStar Logistics**: 2nd logistics company (Atlanta) — competes with Horizon (Memphis)
- **Parkview Consulting**: 2nd DC consulting firm — competes with Atlas Consulting
- **Codename Registry**: Maps project codenames (Phoenix, Sentinel, etc.) to real entities with decoy entries
- **Client Directory**: Maps client codes (4491, 4493, etc.) to companies with two Houston energy companies
- **Contracts**: Meridian now has TWO logistics providers (Horizon=Memphis, NorthStar=Portland)

### Document Expansion
- Financial reports: ~450 → ~1800 chars (quarterly breakdowns, segments, management commentary)
- Bridge documents: ~700 → 3500-5000 chars (answer leaks removed, padding added, bridge entities buried deeper)
- Total: 43 documents, ~180 chunks (was 28 docs, ~75 chunks)

### Hard Question Design (MuSiQue methodology)
- No single chunk contains the full answer
- No single document suffices
- Competing candidate entities for every answer type
- Skipping any hop makes the question unanswerable

---

## Completed

- [x] **Gym-compatible environment**: reset(), step(), reward() loop with persistent REPL
- [x] **Tool preamble**: search(), read(), extract(), search_within(), verify(), list_docs()
- [x] **Synthetic corpus**: 43 cross-referencing business docs with distractor entities (Phase 1-2)
- [x] **Hard question set**: 12 multi-hop questions (2-5 hops) using MuSiQue anti-shortcut methodology
- [x] **4 policies**: claude_policy, naive_rag, context_stuffing, single_shot
- [x] **Eval harness**: Full pipeline — run policies × questions, collect trajectories, compute rewards
- [x] **REPL output isolation**: Step markers fix so agent sees only current step's output
- [x] **REPL rollback**: SyntaxError auto-rollback prevents poisoned cumulative scripts
- [x] **Agent prompt**: Multi-hop strategy, discovery chain examples, step counter with urgency
- [x] **Full hard benchmark**: 12 questions × 4 policies completed (run_20260318_170023)
- [x] **`/results` skill**: Custom skill to summarize eval runs into markdown

---

## What's Next

### Immediate — Fix the gap (context stuffing shouldn't win)

The core problem: **corpus is too small** (43 docs, ~86K chars total). Everything fits in a single context window, so context stuffing trivially wins. The iterative agent's advantage only emerges when the corpus is too large to fit.

- [x] **Scale up corpus via MuSiQue**: `scripts/setup_musique.py` downloads the MuSiQue multi-hop QA benchmark (Wikipedia passages), converts to our corpus format. Default: 200 questions → ~2000+ documents, ~1M+ chars. Use `--num-questions 500` for larger corpora that exceed context window limits.
  - Run: `pip install -e ".[musique]" && python scripts/setup_musique.py`
  - Eval: `python scripts/run_eval.py --musique`
- [x] **Docker data mounting**: Corpus data is now baked into the Docker image via `COPY data/ /workspace/data/`. No volume mount needed. Select corpus at runtime with `CORPUS_DIR` env var.
- [ ] **Cap `read()` output**: Force agent toward `search_within()` instead of reading entire documents. Agent burns steps in pagination loops with manual `text[:3000]` slicing
- [ ] **Upgrade agent model**: Haiku generates prose (SyntaxError) and gets stuck in loops. Try Sonnet — more expensive but should reduce failure rate from 7/12 to ~2-3/12

### Short-term — Improve agent reliability

- [ ] **Fix prose generation**: Haiku's first step often generates English text instead of Python. Better `_clean_action()` filtering or few-shot prompt adjustment
- [ ] **Fix pagination loops**: Agent manually slices `read()` output instead of using `search_within()`. Cap read output or add stronger prompt guidance
- [x] **Docker sandbox**: Data baked into image via `docker build -t rlm-sandbox .`. `CORPUS_DIR` env var selects corpus.
- [ ] **Re-run MuSiQue benchmark**: After corpus scaling + agent fixes, target: claude_policy > context_stuffing on MuSiQue

### Medium-term — RL training

- [ ] **Open-weight model**: Plug in Llama/Qwen as the policy instead of Claude
- [ ] **GRPO training**: Use the environment's reward signal on Prime Intellect to train exploration behavior
- [ ] **Dense/hybrid search**: Add FAISS embedding search to REPL tools (deferred — loading sentence-transformers in subprocess too expensive)

### Long-term

- [ ] **Multi-turn GRPO**: Train models that learn when to explore vs. when to submit
- [ ] **Curriculum learning**: Start with single-doc extraction, progress to multi-hop
- [ ] **Transfer to real corpora**: Validate that RL-trained exploration transfers to unseen document collections

# RLM Explorer — Results & Findings

*Last updated: 2026-03-19*

## 1. Corpus Scaling Experiment

### Problem
The synthetic corpus (43 docs, ~86K chars) was too small. Context stuffing fit everything in one context window and dominated the leaderboard (0.877 avg reward). The iterative agent's exploration advantage was invisible.

### Solution
Integrated **MuSiQue** (Multi-hop Questions via Single-hop Question Composition) — a multi-hop QA benchmark with Wikipedia passages. `scripts/setup_musique.py` downloads the validation split and converts to our corpus format.

- **Before:** 43 docs, ~86K chars, ~180 chunks
- **After:** 2940 docs, ~5094 chunks, corpus far exceeds any context window

### Result: Leaderboard Flipped

| Policy | Synthetic (43 docs) | MuSiQue (2940 docs) |
|--------|:------------------:|:-------------------:|
| context_stuffing | **0.877** (1st) | 0.160 (last) |
| naive_rag | 0.503 | 0.448 |
| single_shot | 0.569 | 0.282 |
| claude_policy (iterative) | 0.349 (last) | **0.568** (1st) |

Context stuffing can't fit ~2940 docs, generates prose instead of code (SyntaxError), and scores 0 on every question. The iterative agent now leads — when it explores properly, it gets perfect 1.0+ scores.

### Key Finding
**The iterative exploration advantage only emerges when the corpus exceeds context window limits.** This validates the core thesis of the environment: iterative tool-using agents beat single-pass approaches on large corpora.

---

## 2. Agent Behavior Analysis

### Tool Usage (claude_policy, Haiku)

Across 5 MuSiQue questions, the agent's tool usage was heavily skewed:

| Tool | Calls | Notes |
|------|------:|-------|
| `search()` | ~41 | Heavily used, sometimes redundantly |
| `read()` | ~20 | Over-used on same docs repeatedly |
| `search_within()` | ~8 | Under-used — only on timeout question |
| `extract()` | ~7 | Rarely used |
| `verify()` | ~4 | Used when prompted by examples |
| `list_docs()` | ~1 | Almost never used |
| `aggregate()` | 0 | Never used |

**The agent defaults to search() + read() and ignores the specialized tools.** This is expected behavior for a base model without RL training — it hasn't learned that `search_within()` is more efficient than re-reading entire documents.

### Failure Modes Observed

| Mode | Frequency | Description |
|------|-----------|-------------|
| **Timeout** | 1/5 | Hit max steps without submitting. Agent re-read the same document 6 times instead of using `search_within()`. |
| **Early quit** | 1/5 | Submitted on step 1 without exploring. Gave up immediately on a hard multi-hop chain. |
| **Multi-step dump** | Common | Generated code for ALL steps at once (6-7K chars), referencing variables from output it hasn't seen yet. |
| **Prose generation** | Occasional | Haiku generates English text instead of Python, causing SyntaxError. |
| **Search loop** | 1/5 | Repeated the same search query across multiple steps when it didn't find results. |

### Successful Patterns

| Pattern | Example | Reward |
|---------|---------|--------|
| **Clean multi-hop chain** | search → find doc → extract answer (m0001, 3 steps) | 1.140 |
| **Iterative discovery** | search "Happy Pills" → find performer → search "Turn Me On" → find songwriter (m0002, 8 steps) | 1.040 |
| **Verify before read** | Used `verify()` to check doc relevance before committing to `read()` (m0003) | 0.460 |

### Takeaway for RL Training
These failure modes (re-reading, early quit, search loops, tool underuse) are exactly the behaviors RL should optimize away. The environment provides clear signal: the agent gets 0 reward for timeout/early quit, and 1.0+ for successful multi-hop exploration. **The gap between failure (0.0) and success (1.14) is large enough for GRPO to learn from.**

---

## 3. Baseline Analysis on MuSiQue

### Context Stuffing — Broken on Large Corpora
- Corpus too large to fit (2940 docs > 100K char limit)
- After fix: falls back to top-20 embedding retrieval + stuff, making it functionally equivalent to a richer naive RAG
- Generates prose instead of code when overwhelmed — the `_clean_action()` pipeline catches some but not all

### Naive RAG — Retrieval Quality Bottleneck
- FAISS embedding search over 5094 chunks works, but top-5 retrieval often misses the right passages for multi-hop questions
- After fix: retrieves top-10, fetches full docs instead of just chunks, deduplicates
- Still limited: single-pass retrieval can't solve 2+ hop questions by design

### Single Shot — Minimal Retrieval
- Top-3 retrieval is too narrow for MuSiQue's multi-hop questions
- Occasionally gets lucky when the question happens to match a single-doc answer

---

## 4. Docker + Data Pipeline

### Architecture
- Corpus data baked into Docker image at build time (`COPY data/ /workspace/data/`)
- `CORPUS_DIR` env var selects active corpus at runtime
- No volume mount needed — self-contained, reproducible sandbox
- Container per episode: create → run steps → destroy

### Performance
- Container startup: ~200ms
- Per-step execution: ~200-300ms (write script + execute)
- Total episode (10 steps): ~15-155s depending on LLM response time
- FAISS index build: ~13s for 2940 docs / 5094 chunks

---

## 5. RL Training Readiness Assessment

### What's Validated
- Environment loop works: reset → step(code) → step(SUBMIT) → reward
- Reward signal is meaningful: 0.0 for failure, 1.0+ for perfect multi-hop answers
- Corpus is large enough that exploration is necessary (context stuffing fails)
- Agent behavior has clear room for improvement (tool underuse, timeout, loops)
- Docker sandbox provides safe, reproducible execution

### Reward Design

Current: `0.5 × answer_F1 + 0.25 × citation_P + 0.25 × citation_R + efficiency_bonus`

Based on literature review (Search-R1, R1-Searcher, HiPRAG, RAG-RL, DeepSeek-R1):

- **Answer F1** — good primary signal. Gives partial credit, helps exploration.
- **Citation P/R** — good secondary signal. Outcome-based, prevents hallucinated sources.
- **Efficiency bonus** — dangerous. Incentivizes skipping retrieval to guess. Should be removed or replaced with search-decision quality metric.
- **Outcome-only is the right starting point.** DeepSeek-R1 and Search-R1 showed that strategies (self-verification, backtracking, tool selection) emerge from pure outcome reward + GRPO without process supervision.

Recommended: `0.7 × answer_F1 + 0.15 × citation_P + 0.15 × citation_R`

### Key Papers for Training Phase

| Paper | Relevance |
|-------|-----------|
| **Search-R1** (2025) | Baseline approach: GRPO + outcome F1 + retrieved token masking. +41% over RAG on MuSiQue. |
| **R1-Searcher** (2025) | Two-stage curriculum: first learn tool use, then optimize answers. |
| **HiPRAG** (2025) | Hierarchical process rewards reduce over-search 27% → 2.3%. |
| **RAG-RL** (2025) | GRPO + curriculum learning (easy → hard). Process rewards 18x more data-efficient. |
| **Search-P1** (2026) | Path-centric reward shaping. +7.7 pts over Search-R1. |
| **DeepSeek-R1** (2025) | Pure outcome RL. Reasoning strategies emerge without supervision. |

### Critical Implementation Detail: Retrieved Token Masking
Search-R1 showed that **masking retrieved tokens in the policy gradient** is critical. Without it, the model learns to echo `read()` output verbatim to inflate F1 scores. Only LLM-generated tokens (search queries, reasoning, answer formulation) should contribute to the gradient.

---

## 6. Next Steps

### Immediate
- [ ] Run full MuSiQue eval (200 questions × 4 policies) for complete baseline numbers
- [ ] Cap `read()` output to encourage `search_within()` usage (observe natural behavior first)
- [ ] Try Sonnet as agent model to reduce SyntaxError/prose generation rate

### Training Phase
- [ ] Plug in open-weight model (Qwen2.5-7B) as policy
- [ ] Implement GRPO training on Prime Intellect using the environment's reward signal
- [ ] Implement retrieved token masking (Search-R1 approach)
- [ ] Start with outcome-only reward, add process rewards only if agent plateaus
- [ ] Curriculum: start with 2-hop questions, progress to 4-5 hop

### Evaluation
- [ ] Compare RL-trained agent vs Claude Haiku baseline on MuSiQue
- [ ] Measure tool usage distribution shift (does RL agent learn to use `search_within`?)
- [ ] Transfer test: run RL-trained agent on the synthetic corpus (unseen domain)

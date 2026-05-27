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

---

## Phase 3 Dev Baseline (MuSiQue dev, 100 questions)

**Run date:** 2026-05-03  
**Split:** `data/musique/questions/dev_set.json` (100 questions, `validation[:100]`)  
**Corpus:** MuSiQue (10,340 docs)  
**Transcript:** `out/run_20260503_165729.json`  
**Model:** All policies use `claude-haiku-4-5-20251001`

### Corrected results (efficiency bonus removed)

Reward = `0.5 × answer_F1 + 0.25 × citation_precision + 0.25 × citation_recall`. Max = 1.0.

The efficiency bonus was confirmed to invert the ranking: one-shot policies (1.6 avg steps) received +0.168 efficiency bonus vs claude_policy's +0.022 (8.9 avg steps), masking the 4× answer quality advantage of iterative exploration. Removed from `reward.py`.

| Policy | Corrected Reward | Answer F1 | Cit Prec | Cit Recall | Avg Steps |
|---|---|---|---|---|---|
| **claude_policy** | **0.179** | **0.157** | 0.222 | 0.179 | 8.9 |
| context_stuffing | 0.176 | 0.050 | 0.343 | 0.262 | 1.6 |
| naive_rag | 0.147 | 0.038 | 0.292 | 0.220 | 1.6 |
| sparse_rag | 0.141 | 0.033 | 0.295 | 0.203 | 1.7 |
| single_shot | 0.089 | 0.027 | 0.172 | 0.130 | 1.8 |

### Key observations

1. **claude_policy leads on answer F1** (0.157 vs 0.027–0.050). Iterative exploration produces ~4× better answers than one-shot retrieval.
2. **context_stuffing nearly ties claude_policy** on total reward (0.176 vs 0.179) due to citation precision from top-20 dense retrieval, but its answer F1 (0.050) is 3× worse.
3. **Absolute F1 is low across all policies** — MuSiQue is hard by design. A Qwen-trained model needs to beat ~0.157 F1 to show improvement over the Claude reference.
4. **Hop-stratified analysis pending** (Task 3.3). The gap between claude_policy and RAG baselines should grow with hop count.

---

---

## Phase 4: SFT + GRPO Training Log

*Last updated: 2026-05-27*

### Reward function (updated)
`reward = 0.1 + 0.9 × (0.5 × answer_F1 + 0.25 × cit_P + 0.25 × cit_R)` if SUBMIT was called, else 0.0.
Format bonus of 0.1 added so GRPO gets gradient signal even on wrong answers.

---

### 4.1 SFT Data Collection
- **Source:** 500 train questions → claude_policy → filtered reward ≥ 0.5
- **Final dataset:** `data/sft/qwen_traj_full.jsonl` — 114 conversations, 716 training pairs
- **Quality fixes:** removed 33 duplicates, 19 dev leakage, 10 prose-contaminated, 5 over-length
- **Bias note:** 89% 2-hop questions (Claude fails 4-hop → structural imitation ceiling)

---

### 4.2 Model Size Decision Log

| Model | Outcome | Reason |
|---|---|---|
| Qwen2.5-1.5B-Instruct | ❌ Abandoned | Syntax errors on ~90% of GRPO rollouts at temp=0.8. No valid Python → no reward variance → no gradient signal. |
| Qwen2.5-3B-Instruct | ⏭ Skipped | Literature (Search-R1, CoSearch, HiPRAG) validates 3B as minimum but results not consistently strong. Went straight to 7B. |
| **Qwen2.5-7B-Instruct** | ✅ Current | Validated by literature. Sanity check: immediately writes `search()` call, not hallucinated prose. |

**Key evidence for 1.5B failure:** Step 0 GRPO rewards were `[0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0]` — 7/8 rollouts timed out with SyntaxErrors, never submitted. Claude Haiku on same env: 8.9 avg steps, valid Python throughout. Confirmed env is healthy; 1.5B is the bottleneck.

---

### 4.3 SFT Results

#### Qwen2.5-1.5B SFT (smoke, 50 dev questions)
| Metric | Value |
|---|---|
| Avg reward | 0.113 |
| Answer F1 | 0.033 |
| Cit Prec/Recall | 0.000 / 0.000 |
| Avg steps | 1.9 |

Model learned format (gets 0.100 format bonus) but submits immediately without searching. Confirms SFT imitation ≠ exploration behavior.

#### Qwen2.5-7B SFT (2 epochs, 114 conversations)
- **Checkpoint:** `checkpoints/sft_qwen_7b/final`
- **Sanity check:** outputs `search("Apex Corp CEO birthplace")` — Python code, not hallucinated prose ✓
- **Dev eval:** pending

---

### 4.4 GRPO Training

#### GRPO Design Choices (informed by DAPO/DR-GRPO/NotebookLM research)
| Choice | Value | Reason |
|---|---|---|
| group_size | 8 | More rollouts → more reward variance → better gradient signal |
| beta (KL) | 0.0 | SFT ref too weak to anchor to usefully; saves memory |
| normalization | fixed 256 tokens | DR-GRPO style; removes length bias |
| temperature | 0.8 | Higher temp → more diverse rollouts |
| Skip condition | uniform rewards | Zero std → zero advantage → skip gradient step |

#### Qwen2.5-1.5B GRPO — FAILED
- Killed at step 0. Syntax errors on 7/8 rollouts per group. No useful gradient.

#### Qwen2.5-7B GRPO — IN PROGRESS
- **Started:** 2026-05-27
- **Checkpoint path:** `checkpoints/grpo_qwen_7b/`
- **Target:** 300 steps
- Results: TBD

---

## 6. Next Steps

### Immediate
- [ ] Monitor 7B GRPO — watch `recent_avg` for upward trend past 0.113 by step 50
- [ ] Eval 7B GRPO checkpoint on 50 dev questions at step 50, 100, 300
- [ ] Eval 7B SFT checkpoint on 50 dev questions (need base comparison)

### Final Eval (Phase 6)
- [ ] Run all policies on test split (200 questions)
- [ ] Hop-stratified analysis: 2-hop / 3-hop / 4-hop per policy
- [ ] Three headline numbers: `sft−base`, `grpo−sft`, `grpo−rag`

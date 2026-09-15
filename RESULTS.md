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

#### Qwen2.5-7B GRPO — one surviving post-training adapter, never evaluated until Phase 5
- **Started:** 2026-05-27
- **Checkpoint path:** `checkpoints/grpo_qwen_7b/`
- **Published checkpoints:** `jasonlingg/doctracerrl-grpo-qwen2.5-7b-50steps`, `jasonlingg/doctracerrl-grpo-qwen2.5-7b`
- Results: see Phase 5 — the belief that this run "flatlined" was never checked against a measured baseline until 2026-09-15.

---

## Phase 5: Checkpoint Evaluation (finally run, 2026-09-15)

The three checkpoints above sat unevaluated for three months. `docs/EVAL_RUNBOOK.md` had the
runbook the whole time; nobody ran it. This phase runs it and documents a bug the first attempt
uncovered along the way.

### 5.1 First attempt (n=5 smoke) — confounded by a REPL bug, not a training result

All four policies — including the **untrained base model** — scored exactly **0.000** on the first
5-question smoke run. Inspecting raw trajectories showed why: `PersistentREPL` executes each step
as `python3 script.py`, not an interactive REPL. A bare `search(...)` call with no `print()`
wrapper produces zero visible output. The model called `search()`, saw nothing, and blindly
retried query variations for up to 10 steps — in the base model too, so this was not a training
difference. It was a structural bug making the comparison meaningless. This is a sibling of the
March 2026 "REPL Output Flooding" and "prose contamination" bugs already logged in `STATUS.md`,
just never caught for this exact case (bare expression statements with no `print()`).

### 5.2 Fix: auto-print the trailing expression

Added `_auto_print_trailing_expression()` to `src/env/repl.py`. It parses each step's code with
`ast`, and if the last statement is a bare expression — not an assignment, not already
`print(...)` — rewrites it into a `print()` call before execution, matching what an interactive
session already does for a trailing expression. The system prompt already says "Use print() to
see output"; the model just doesn't reliably follow it, so the harness now makes that failure
mode structurally impossible instead of relying on the model remembering.

The first implementation had its own bug, caught by `test_timeout` regressing: naive line-based
slicing corrupted code where multiple statements share one physical line via `;`
(`import time; time.sleep(10)` lost the `import` entirely, since both statements' source
overlapped on line 1). Fixed by rebuilding the whole step via `ast.unparse()` on the modified
tree instead of slicing source lines. Regression tests cover both the REPL fix and single-document
vault search; the full suite passes (149 passed, 1 skipped).

### 5.3 Re-run at n=5 — real signal, one-question small-sample noise

With the fix, rewards moved to a real range (0.083–0.110) instead of flat 0.000. The apparent
"base beats trained" result on this tiny sample traced to a single question (`dev_0001`) where
all four policies retrieved the identical correct document but extracted the answer differently:
base cleanly pulled "Dane County"; SFT copied the whole document title verbatim ("York, Dane
County, Wisconsin" — verbose, not wrong, but F1-costly); both GRPO checkpoints mis-parsed the
compound place name and answered "York County" — genuinely wrong, dropping "Dane" entirely. One
question dominated a 5-question average. Not conclusive at n=5; ran the real 50-question split.

### 5.4 Full 50-question result — the actual headline

One SSH disconnect killed the 4th policy mid-run (`last -x` confirmed "gone - no logout"); base,
SFT, and GRPO-50 completed clean, and GRPO-full was rerun alone inside `tmux` for disconnect
resilience, then merged via `scripts/summarize_eval.py`.

| Policy | Outcome | Answer F1 | Cit Precision | Cit Recall | Avg Steps | Submit |
|---|---:|---:|---:|---:|---:|---:|
| base (untrained) | 0.158 | 0.127 | 0.350 | 0.215 | 8.0 | 98% |
| **SFT** | **0.176** | **0.140** | 0.400 | 0.238 | 7.7 | 98% |
| GRPO repository labeled 50 steps | 0.172 | 0.146 | 0.350 | 0.207 | 8.2 | 96% |
| Same GRPO adapter via the other Hub ID | 0.172 | 0.146 | 0.350 | 0.207 | 8.2 | 98% |

**SFT beats base**: +0.018 outcome, +0.013 answer F1 — small but real, on the real split, with the
confound removed. GRPO does not measurably beat SFT (-0.003, noise-level on n=50).

**Resolved checkpoint anomaly:** the two published GRPO repositories contain byte-identical files.
Their `adapter_model.safetensors` objects have the same SHA-256 hash
(`1f44ac5cb716f2947fb23af7426865a4d8af31074b8df13934b1a7b4813c748d`). They are aliases of one
adapter, so the table contains two evaluations of the same policy, not evidence about progression
from step 50 to a later checkpoint. Submission rate is the sole aggregate discrepancy between the
runs (96% versus 98%); the outcome components and average steps match.

The shared GRPO adapter is not an accidental copy of SFT: all 392 tensors differ from the SFT
adapter, with relative L2 delta 0.001369 (about 0.137%). Neither Hub repository includes trainer
state or a recorded global step. The surviving run record says training stopped at step 50, making
"one post-GRPO checkpoint, likely step 50" the strongest supported description. The exact number
of optimizer updates cannot be recovered from the published artifacts.

### 5.5 Reading the result

Per `docs/EVAL_RUNBOOK.md`'s own decision rule: this is closer to the "flat" branch than the
"SFT > base and GRPO > SFT" branch — SFT shows a small real gain, GRPO shows none beyond SFT. This
is a **measured** result, not an impression, for the first time in three months. It also does not
resolve which underlying protocol (code-execution vs. structured JSON tool-calling) is the right
one going forward — that decision was made independently in this session for portfolio/career
reasons, not because this eval favored one over the other.

---

## 6. Next Steps

### Immediate
- [x] Eval 7B GRPO checkpoint on 50 dev questions — done 2026-09-15, see Phase 5
- [x] Eval 7B SFT checkpoint on 50 dev questions (need base comparison) — done 2026-09-15, see Phase 5
- [x] Confirm whether GRPO-50 and GRPO-full are distinct — no; both Hub IDs contain the same adapter

### Final Eval (Phase 6)
- [ ] Run all policies on test split (200 questions)
- [ ] Hop-stratified analysis: 2-hop / 3-hop / 4-hop per policy
- [ ] Three headline numbers: `sft−base`, `grpo−sft`, `grpo−rag`

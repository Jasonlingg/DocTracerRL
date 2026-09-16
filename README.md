# Envoy *(GitHub repo still named `DocTracerRL` — see [Naming](#naming) below)*

A Gym-compatible RL environment for training language models to actively **explore document
collections via code execution** in a persistent REPL, rather than passively consuming retrieved
context — plus a real product built on top of it: a small-model research agent that investigates
an Obsidian vault and AI papers, and reports what it found with exact source citations.

> Inspired by: https://arxiv.org/pdf/2512.24601

## Use case: a dispatched subagent, not a chat interface

The trained policy (currently **Qwen Envoy**, a Qwen2.5/Qwen3 checkpoint) is not meant to be
talked to directly. It's meant to be **called as a subagent by a larger assistant** — Claude, GPT,
or another host — over MCP: the host decides when research is needed, dispatches Qwen Envoy with a
question, the subagent spends several REPL turns searching/reading/extracting across the vault or
paper corpus, and returns a short answer plus exact cited evidence. The host then explains that
evidence to the user in conversation. This is why the reward function scores citation
precision/recall as heavily as answer correctness: a subagent that guesses without evidence is
useless to the assistant that called it. See
[`src/research/knowledge_tool.py`](src/research/knowledge_tool.py) (`query_papers()`) for the
current integration point, and
[`docs/CODE_EXECUTION_SECOND_BRAIN.md`](docs/CODE_EXECUTION_SECOND_BRAIN.md) for the full product
framing.

## Current direction (as of 2026-09-15)

The project has two layers now, and this README was out of date on both:

1. **The environment** — `env.reset()` → `env.step(code)` → `env.step(SUBMIT)` → reward. This is
   the reusable core: a persistent Python REPL, a document corpus, and a verifiable reward. It
   hasn't changed in kind, but the execution backend has (see [Architecture](#architecture)).
2. **The product** — a **weekly AI research radar**: a Qwen-powered agent that searches a frozen
   snapshot of an Obsidian vault and/or AI papers, investigates across turns using
   `search()` / `read()` / `extract()`, and submits an answer with citable evidence. See
   [`docs/CODE_EXECUTION_SECOND_BRAIN.md`](docs/CODE_EXECUTION_SECOND_BRAIN.md) for the product
   goal and [`docs/WEEKLY_RESEARCH_RADAR.md`](docs/WEEKLY_RESEARCH_RADAR.md) for the scope
   contract. The environment's MuSiQue benchmark remains the labeled dataset for measuring whether
   training actually improves this exploration skill; the vault/papers are the target domain.

The synthetic 43-document business corpus and its hard question set (below) were the original
proving ground and still exist as a fast local sanity check, but MuSiQue and the AI-paper pilot are
now the datasets that matter for training claims.

## What's actually been measured (not claimed)

Two real, dated findings supersede everything this README used to say about training status:

- **SFT beats base; GRPO does not (yet) beat SFT — measured, not assumed.** On 50 real MuSiQue dev
  questions, untrained Qwen2.5-7B scores **0.158** outcome reward, the SFT checkpoint scores
  **0.176**, and the published GRPO checkpoint scores **0.172** (noise-level vs. SFT, n=50). This
  is the first time these three checkpoints were actually evaluated against each other — they sat
  unevaluated for three months after a June training run, during which the team's working belief
  was that GRPO had "flatlined." Full writeup: [`RESULTS.md`](RESULTS.md) §Phase 5.
- **The GRPO training loop had a real bug, now fixed.** The custom GRPO update was computing its
  PPO importance ratio from `generate()`'s `scores`, which are distorted by every logits warper
  (temperature, top-k) rather than reflecting actual policy change — measured on a sanity model,
  top_k=50 alone shrank the ratio to 0.037. This silently killed gradient signal on
  negative-advantage samples. Fixed via per-token ratios computed from a fresh forward pass
  (commit `3f57a12`), with regression tests in `tests/test_grpo_loss.py`.
- **Base Qwen3-8B, evaluated for the first time on the AI-paper pilot, retrieves well but doesn't
  reason carefully.** 0.445 avg outcome reward, 0.85/0.95 citation precision/recall — retrieval is
  not the bottleneck. The gap is three distinct, buildable failure modes: it fails an abstention
  trap outright (confidently answers a question it should refuse), silently drops an explicit
  "keep these separate" framing instruction, and once misdescribed a correctly-cited paper's actual
  mechanism. Getting this measurement required fixing a Qwen3-specific bug first: `qwen_common.py`
  never disabled the model's native `<think>` mode, so it burned its whole token budget reasoning
  and never reached executable code. Full writeup:
  [`docs/QWEN3_BASELINE_PILOT.md`](docs/QWEN3_BASELINE_PILOT.md).

## The Core Loop

```
env.reset()          →  Agent receives question + tool descriptions
env.step(code)       →  Agent writes Python, observes stdout/stderr
env.step(code)       →  Agent refines search, cross-references docs
env.step(code)       →  Agent computes aggregations, verifies findings
env.step(SUBMIT)     →  Agent submits answer + citations → receives reward
```

Each episode reconstructs Python state by replaying successful actions. Variables and helper
functions are available across steps; failed actions are removed from subsequent replay. The
local/GPU execution backend now assigns one long-lived Python worker per episode, so an action
executes exactly once and state persists naturally (the Docker backend still uses cumulative
replay and is not yet at parity). The reward signal measures answer token-overlap F1 and citation
precision/recall.

## Agent Tools

The REPL comes pre-loaded with tools the agent can call via Python code:

| Tool | Description |
|------|-------------|
| `search(query, top_k=5)` | Document-level keyword search with TF-IDF scoring. Returns `[{"doc_id", "title", "chunk", "score"}]` |
| `search(query, method="chunk")` | Chunk-level search over 500-char overlapping windows — finds facts buried in long documents |
| `read(doc_id)` | Read full document text by ID |
| `extract(doc_id, pattern)` | Regex extraction from a document |
| `search_within(doc_id, query)` | Search inside a specific document — returns the most relevant 500-char windows ranked by score |
| `verify(doc_id, claim)` | Quick check if a claim's keywords appear in a document. Returns `{found, match_ratio, excerpt}` |
| `list_docs()` | List all documents in the corpus with titles and character counts |

The agent learns to compose these tools across steps, tracking discoveries in a Python dict
(`known_facts = {}`) that persists across steps.

## Architecture

```
src/
├── env/
│   ├── document_env.py   # Gym-compatible environment: reset(), step(), reward()
│   ├── repl.py           # Persistent REPL: one long-lived worker/episode (local), Docker sandbox (not yet at parity)
│   ├── corpus.py         # Load docs, chunk, embed, FAISS index
│   ├── reward.py         # Verifiable reward: answer F1, citation P/R (current version: outcome-v1)
│   └── tools.py          # Tool preamble: search(), read(), extract(), search_within(), verify()
├── policies/
│   ├── claude_policy.py  # Reference policy: Claude explores iteratively
│   ├── qwen_common.py    # Shared Qwen2.5/Qwen3 policy plumbing (chat template, thinking-mode handling)
│   ├── naive_rag.py      # Baseline: top-k retrieve → answer in 1 step
│   ├── stuffing.py       # Baseline: concatenate all docs → answer in 1 step
│   └── single_shot.py    # Baseline: minimal retrieval → answer in 1 step
├── research/              # Vault/paper ingestion + the paused JSON-action research agent
│   └── knowledge_tool.py  # query_papers() wrapper for main-agent integration
└── eval/
    ├── harness.py        # Run policies through env, collect trajectories
    ├── scorer.py         # Scoring functions
    └── report.py         # Results tables
```

## Quick Start

```bash
# Clone and install (repo is still hosted as DocTracerRL until the GitHub rename lands)
git clone https://github.com/Jasonlingg/DocTracerRL.git
cd DocTracerRL
pip install -e ".[dev]"

# GPU-only: GRPO training stack (verifiers, vllm, etc.) — do NOT install on CPU
# pip install -e ".[training]"

# Generate the synthetic sanity-check corpus (43 cross-referencing business documents)
python scripts/setup_corpus.py

# Run tests
pytest tests/ -v

# Run Claude policy on one question
python scripts/run_eval.py --policy claude_policy --question q01 --verbose

# Run full evaluation (all policies, all 18 easy questions)
python scripts/run_eval.py

# Run hard multi-hop evaluation (12 questions, 2-5 hops)
python scripts/run_eval.py --hard --max-steps 15

# Freeze an Obsidian vault or paper folder into a queryable corpus
python scripts/research_vault.py import --help

# Reproduce the Qwen3-8B AI-paper baseline pilot (GPU required)
CHECKPOINT_PATH=jasonlingg/doctracerrl-sft-qwen2.5-7b ./scripts/run_ai_paper_code_eval.sh
```

See [GPU readiness and fixes](docs/GPU_TRAINING_READINESS.md) before launching training, and
[`docs/EVAL_RUNBOOK.md`](docs/EVAL_RUNBOOK.md) before running a base/SFT/GRPO comparison.

## Reward Signal

The reward is designed for GRPO training:

```
reward = 0.8 × answer_F1 + 0.1 × citation_precision + 0.1 × citation_recall
```

Current reward version: `outcome-v1`. Exploration actions receive zero reward; printing answer
words or taking extra steps earns no bonus. Historical results use earlier reward versions and are
not directly comparable — see `RESULTS.md` for the version each number was measured under.

## Question Types

### Easy Set (18 questions)
Cross-document aggregation, cross-document comparison, multi-hop reasoning, single-document
extraction, contradiction detection.

### Hard Set (12 questions, 2-5 hops)
Designed with [MuSiQue](https://arxiv.org/abs/2108.00573) anti-shortcut methodology — no single
chunk or document can answer any question, and competing distractor entities exist for every
answer type: hidden bridge, disambiguation, fan-out aggregation, parallel comparison, codename
bridging.

## Docs Map

- [`docs/CODE_EXECUTION_SECOND_BRAIN.md`](docs/CODE_EXECUTION_SECOND_BRAIN.md) — active product
  goal and architecture (start here for the current direction)
- [`docs/WEEKLY_RESEARCH_RADAR.md`](docs/WEEKLY_RESEARCH_RADAR.md) — product scope contract and
  success gates
- [`docs/QWEN3_BASELINE_PILOT.md`](docs/QWEN3_BASELINE_PILOT.md) — Qwen3-8B baseline findings on
  the AI-paper pilot
- [`docs/EVAL_RUNBOOK.md`](docs/EVAL_RUNBOOK.md) — how to run a base/SFT/GRPO comparison correctly
- [`docs/GPU_TRAINING_READINESS.md`](docs/GPU_TRAINING_READINESS.md) — GPU setup and known fixes
- [`docs/OBSIDIAN_WORKFLOW.md`](docs/OBSIDIAN_WORKFLOW.md) — importing a real vault
- [`RESULTS.md`](RESULTS.md) — dated experiment log, including Phase 5 (the first real
  base/SFT/GRPO comparison)
- [`STATUS.md`](STATUS.md) — environment build log and open bugs

## Naming

The project is renamed to **Envoy** — a dispatched agent that goes, investigates, and returns with
evidence, matching the actual MCP-subagent architecture (see
[Use case](#use-case-a-dispatched-subagent-not-a-chat-interface) above). The specific trained
policy keeps its model name as a prefix — **Qwen Envoy** for the current Qwen2.5/Qwen3 checkpoint
— since the environment's core principle is that policies are swappable (`CLAUDE.md`): a future
Llama- or other-model-backed policy would be "Llama Envoy," not a different project.

Applied: `pyproject.toml` (`name = "envoy"`), `CLAUDE.md`, `AGENTS.md`, `PLAN.md`, and the
`Envoy`/`envoy` strings baked into scripts and tests (CLI titles, the vault's
`generated_by` marker, the paper-fetch User-Agent).

Not yet applied: the GitHub repo itself is still `Jasonlingg/DocTracerRL` — renaming that is a
separate, confirmed step since it changes a shared remote resource GitHub URLs and any external
links depend on.

## Built With

- [Claude API](https://docs.anthropic.com) — Reference policy
- [Qwen2.5 / Qwen3](https://huggingface.co/Qwen) — Trained policy (SFT + GRPO)
- [sentence-transformers](https://sbert.net) — Document embeddings (all-MiniLM-L6-v2)
- [FAISS](https://github.com/facebookresearch/faiss) — Vector search
- [Docker](https://www.docker.com) — Sandboxed code execution (not yet at parity with the local backend)

## Development Stack

- [Claude Code](https://claude.ai/claude-code) — AI-assisted development, debugging, and
  implementation
- [Google NotebookLM](https://notebooklm.google.com) — Research synthesis and project
  documentation

## License

MIT

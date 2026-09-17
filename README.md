# Qwen Envoy

Qwen Envoy is an experimental research worker that answers questions by writing Python against a
document collection. It can search, read, extract, compare, and verify evidence across several
turns before returning an answer with source IDs.

The intended deployment is an agent-as-tool system. A larger assistant decides when research is
needed, sends a bounded question to Envoy, and receives an evidence packet it can explain or use in
a broader task. The first product target is a weekly AI-research workflow over papers and an
Obsidian vault.

This repository contains the working code-execution environment, data and training pipelines,
Qwen and Claude policies, reproducible evaluation harnesses, and the vault importer. The scheduled
weekly pipeline and MCP server are still planned work.

The environment was inspired by [arXiv:2512.24601](https://arxiv.org/abs/2512.24601).

## What the agent does

Each episode gives the model a question and a persistent Python REPL with document tools already
loaded:

```text
Question
   |
   v
Qwen writes Python  ──>  search / read / extract / verify
   ^                              |
   |                              v
   +──────────── stdout and errors
   |
   v
SUBMIT: <answer> CITATIONS: [<document IDs>]
```

The same Python process lives for the whole episode, so variables and intermediate results survive
between actions. This lets the policy do more than issue isolated tool calls: it can filter search
results, inspect multiple documents, run regular expressions, aggregate metadata, and revise a
query based on earlier output.

An episode looks like this:

```text
# action 1
hits = search("retrieval token masking ablation", top_k=5)
print([(h["doc_id"], h["title"]) for h in hits])

# action 2, after seeing stdout
text = read(hits[0]["doc_id"])
print(extract(hits[0]["doc_id"], r"(?i).{0,180}mask.{0,240}"))

# final action
SUBMIT: The paper excludes retrieved tokens from the policy loss ... CITATIONS: ["paper_id"]
```

## Project status

| Component | Status |
| --- | --- |
| Persistent local/GPU Python worker | Working; one isolated process per episode |
| Search, read, passage, extraction, and verification tools | Working |
| Reproducible transcripts and experiment manifests | Working |
| Markdown/PDF and Obsidian snapshot import | Working |
| Qwen2.5-7B SFT and GRPO comparison on MuSiQue | Complete |
| Qwen3-8B baseline on AI-paper and QASPER tasks | Complete |
| Qwen3-8B QLoRA on QASPER code trajectories | Training; behavioral result pending |
| Weekly paper discovery and ranking | Designed, not complete |
| MCP server for host assistants | Interface drafted, server not complete |

## Results

### Historical MuSiQue experiment

The original model experiment used 50 held-out MuSiQue development questions. All three policies
used the same corpus, prompt, step budget, decoding settings, and reward implementation.

| Policy | Outcome reward | Answer F1 | Citation precision | Citation recall |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-7B base | 0.158 | 0.127 | 0.350 | 0.215 |
| Qwen2.5-7B + SFT | **0.176** | 0.140 | **0.400** | **0.238** |
| Qwen2.5-7B + SFT + GRPO | 0.172 | **0.146** | 0.350 | 0.207 |

SFT improved the measured task. The surviving GRPO artifact did not beat SFT, and the two
published GRPO model IDs turned out to contain the same adapter. See [RESULTS.md](RESULTS.md) for
the full protocol and per-run manifests.

### Qwen3-8B research baseline

Base Qwen3-8B was evaluated before domain training. These datasets differ from MuSiQue and from
each other, so their reward values should not be compared across rows.

| Evaluation | Questions | Outcome reward | Citation precision | Citation recall |
| --- | ---: | ---: | ---: | ---: |
| Frozen AI-paper pilot | 10 | 0.445 | 0.850 | 0.950 |
| QASPER test conversion | 20 | 0.355 | 0.800 | 0.950 |

The model usually found the right paper but was less reliable at using it. It guessed on
unanswerable questions, sometimes answered the general topic instead of the requested fact, and
occasionally mischaracterized a correctly cited passage. That diagnosis led to the current
QASPER-grounded SFT experiment. See
[docs/QWEN3_BASELINE_PILOT.md](docs/QWEN3_BASELINE_PILOT.md).

The Qwen3 training result is deliberately omitted until the adapter and base model have been run
through the same locked 40-question evaluation. Training loss and token accuracy alone are not
evidence that research behavior improved.

## Training approach

The current run distills QASPER-grounded research trajectories into Qwen3-8B with rank-4 QLoRA:

1. QASPER supplies paper questions, evidence, and expert answerability labels.
2. A stronger teacher produces multi-step Python trajectories against the frozen corpus.
3. The pipeline replays trajectories and checks that actions execute and cited documents exist;
   QASPER's labels provide the answerability target.
4. Each assistant action becomes a next-action example containing its complete preceding history.
5. Loss is applied only to the action tokens, using the same non-thinking Qwen3 chat prefix used at
   inference.
6. Base and trained Qwen are compared with identical tools, prompts, questions, decoding, and
   step limits.

The per-action representation fixed a real failure in the first Qwen3 run: intermediate code
actions had been formatted differently during training and inference, while the inference prefix
appeared only before `SUBMIT` in the training data. The model learned that accidental correlation
and submitted too early. The diagnosis and token-level checks are documented in
[docs/QWEN3_SFT_PREFIX_DIAGNOSIS.md](docs/QWEN3_SFT_PREFIX_DIAGNOSIS.md).

The project also found and fixed an independent GRPO bug. PPO ratios were computed from warped
generation scores instead of fresh policy logits, which suppressed gradients for some samples.
Regression coverage lives in `tests/test_grpo_loss.py`.

## Evaluation

Every evaluation writes full trajectories plus a manifest containing the checkpoint, question
IDs, corpus hash, prompt settings, decoding settings, reward version, hardware, seed, and source
commit. Results with different protocol hashes are not treated as checkpoint comparisons.

The Qwen3 abstention experiment uses 40 held-out QASPER test questions: 20 answerable and 20
unanswerable, with no paper overlap with training. Its pre-registered decision rule requires:

- a statistically significant increase in abstention recall;
- no more than a 0.15 absolute increase in false abstentions on answerable questions; and
- identical base and adapter inference conditions.

See [docs/SFT_ABSTENTION_PREREGISTRATION.md](docs/SFT_ABSTENTION_PREREGISTRATION.md) and
`scripts/run_qwen3_qasper_abstention_eval.sh`.

## Repository map

| Path | Purpose |
| --- | --- |
| `src/env/` | Gym-style document environment, persistent REPL, corpus, tools, and reward |
| `src/policies/` | Qwen base/SFT/GRPO policies, Claude reference policy, and RAG baselines |
| `src/eval/` | Evaluation harness, manifests, scoring, and abstention metrics |
| `src/research/` | Paper/vault ingestion and the earlier structured research-agent path |
| `scripts/train_sft.py` | Qwen3 QLoRA training with per-action target masking |
| `scripts/train_grpo_custom.py` | Custom GRPO loop for the historical MuSiQue experiment |
| `scripts/run_eval.py` | Shared policy evaluation entry point |
| `scripts/research_vault.py` | Immutable Markdown/PDF/Obsidian snapshot import and export |
| `scripts/viewer.py` | Browser viewer for saved trajectories |

The local worker executes each action once. The optional Docker backend still rebuilds state by
replaying successful actions and is not considered behaviorally equivalent yet.

Core stack: Python, PyTorch, Hugging Face Transformers, TRL, PEFT/QLoRA, bitsandbytes, FAISS,
sentence-transformers, Docker, and pytest.

## Getting started

```bash
git clone https://github.com/Jasonlingg/DocTracerRL.git
cd DocTracerRL
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Build the small synthetic corpus used by local tests and examples.
python scripts/setup_corpus.py

# Run the test suite.
pytest -q
```

Import a folder from an Obsidian vault into an immutable corpus snapshot:

```bash
pip install -e ".[vault]"
python scripts/research_vault.py import \
  --vault /path/to/MyVault \
  --collection "AI Research" \
  --output out/research/my-vault-snapshot
```

Run base Qwen3-8B on the frozen AI-paper pilot (GPU required):

```bash
pip install -e ".[training]"
BASE_MODEL_PATH=Qwen/Qwen3-8B python scripts/run_eval.py \
  --policy qwen_base_policy \
  --questions data/research/code_exec_pilot_v1.json \
  --corpus out/research/starter-2026-09-12/corpus \
  --max-steps 10 \
  --seed 42 \
  --require-evidence \
  --no-vector-index \
  --run-label qwen3_base \
  --output out/research/qwen3-baseline/base.json
```

Compare a Qwen3 adapter with base Qwen on the locked abstention evaluation:

```bash
CHECKPOINT_PATH=/path/to/adapter/final \
  ./scripts/run_qwen3_qasper_abstention_eval.sh
```

See [docs/GPU_TRAINING_READINESS.md](docs/GPU_TRAINING_READINESS.md) before starting a GPU run and
[docs/EVAL_RUNBOOK.md](docs/EVAL_RUNBOOK.md) before comparing checkpoints.

## Agent tools

| Tool | Description |
| --- | --- |
| `search(query, top_k=5)` | Document-level TF-IDF search |
| `search(query, method="chunk")` | Search overlapping windows inside the corpus |
| `read(doc_id)` | Read a complete document |
| `passage(doc_id, start, length)` | Return an exact passage with stable character offsets |
| `extract(doc_id, pattern)` | Run a regular expression over one document |
| `aggregate(doc_ids, field)` | Collect a metadata field across documents |
| `search_within(doc_id, query)` | Rank passages inside one document |
| `verify(doc_id, claim)` | Check claim keywords and return a matching excerpt |
| `list_docs()` | List document IDs, titles, and lengths |

## Current limitations

- The MCP server and unattended weekly scheduler are not implemented yet.
- Citation existence is checked mechanically; whether a passage supports a claim still needs a
  semantic metric or human review.
- The current research training set is small. Its value must be established on held-out papers.
- Qwen inference and training require a CUDA GPU; the vault importer and environment tests run on
  CPU.
- Docker execution does not yet match the persistent local worker exactly.

## Documentation

- [Code-execution second brain](docs/CODE_EXECUTION_SECOND_BRAIN.md): product boundary and active
  model architecture
- [Weekly research radar](docs/WEEKLY_RESEARCH_RADAR.md): product scope and remaining components
- [Qwen3 baseline pilot](docs/QWEN3_BASELINE_PILOT.md): measured research-agent failures
- [Qwen3 SFT prefix diagnosis](docs/QWEN3_SFT_PREFIX_DIAGNOSIS.md): failed run analysis and repair
- [Evaluation runbook](docs/EVAL_RUNBOOK.md): reproducible checkpoint comparison
- [Obsidian workflow](docs/OBSIDIAN_WORKFLOW.md): snapshot import and cited-note export
- [Results](RESULTS.md): dated experiment history

## License

MIT

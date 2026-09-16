# Envoy

This is a Gym-compatible RL environment for training LLMs to explore document collections via code execution in a persistent REPL. The agent receives a multi-document question, writes Python code to search/read/compute across a corpus, and submits a final answer for a verifiable reward. The trained policy (Qwen Envoy) is designed to run as a subagent dispatched over MCP by a larger assistant, not as a standalone chat interface — see [README.md](README.md) for the current product framing.

## Repo Structure

```
src/
├── env/
│   ├── document_env.py   # Gym-compatible environment: reset(), step(), reward()
│   ├── repl.py           # Persistent REPL: start_session(), execute(), kill_session()
│   ├── corpus.py         # Load docs, chunk, embed, FAISS index
│   ├── reward.py         # Verifiable reward: answer accuracy, citation P/R, efficiency
│   └── tools.py          # Tool preamble injected into REPL: search(), read(), extract()
├── policies/
│   ├── claude_policy.py  # Reference policy: Claude API as the agent brain
│   ├── naive_rag.py      # Baseline: top-k retrieve → answer in 1 step
│   ├── stuffing.py       # Baseline: concatenate docs → answer in 1 step
│   └── single_shot.py    # Baseline: minimal retrieval → answer in 1 step
└── eval/
    ├── harness.py        # Run policies through env, collect trajectories + rewards
    ├── scorer.py         # Scoring functions used by reward.py
    └── report.py         # Markdown results table
```

## Dev Commands

```bash
pip install -e ".[dev]"
pytest
python scripts/setup_corpus.py
python scripts/run_eval.py
```

## Code Style

- Python 3.11+, type hints on all functions
- Pydantic models for configs and data structures
- loguru for logging, rich for CLI output
- No LangChain. No OpenAI. Claude-native only.
- anthropic SDK for LLM calls, model = claude-sonnet-4-20250514
- Keep files under 200 lines. Split aggressively.
- sentence-transformers (all-MiniLM-L6-v2) for embeddings, faiss-cpu for vector search

## Core Principle

The environment is the product. Policies are swappable.

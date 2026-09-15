# RLM Explorer agent guide

Read [the Astra improvement brief](docs/GPT_ASTRA_IMPROVEMENT_BRIEF.md) before changing
the training or reward code. It records the current evidence, research, and order of work.

The user chose the executable-code track on September 15, 2026: Qwen writes Python against
`search()`, `read()`, and `extract()` to explore a personal Obsidian snapshot. Read
[the code-execution second-brain plan](docs/CODE_EXECUTION_SECOND_BRAIN.md). The JSON-action
research agent and QASPER routing/reranker track are paused unless their data is retargeted to
code-execution trajectories.

- Preserve the user's uncommitted changes in `scripts/train_grpo_custom.py`,
  `tests/test_grpo_loss.py`, and `docs/QWEN_TRAINING_RESEARCH_AND_IMPROVEMENTS.md`.
- Treat `src/env/reward.py` as the active reward definition. Historical reward formulas in
  `README.md`, `RESULTS.md`, `STATUS.md`, and `PLAN.md` are not authoritative until reconciled.
- That reward applies to MuSiQue, not research synthesis. Research quote-integrity checks are
  not semantic support judgments or training rewards. The 20 starter questions are unreviewed
  development prompts, not gold labels or a held-out test set.
- Keep the task multi-step and tool-mediated. Do not collapse it into a one-shot classifier.
- Make each experiment reproducible: record checkpoint identifier, question split and IDs,
  corpus revision, policy decoding settings, reward version, hardware, and seed.
- Prefer small, isolated changes followed by the narrowest meaningful verification. Do not run
  another costly training job until a baseline over the personal-vault task has named questions,
  reviewed outputs, and a decision rule.

The user wants an evidence-based improvement, not a speculative rewrite. State the hypothesis,
the expected signal, and the decision rule before each material experiment.

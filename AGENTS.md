# RLM Explorer agent guide

Read [the Astra improvement brief](docs/GPT_ASTRA_IMPROVEMENT_BRIEF.md) before changing
the training or reward code. It records the current evidence, research, and order of work.

The user approved a focused AI-paper research assistant on September 12, 2026. The immediate
objective is a working retrieval baseline and reviewed examples of its failures, before choosing
a training method. Read [the research assistant runbook](docs/RESEARCH_ASSISTANT.md).
Keep the existing MuSiQue checkpoint comparison as an outstanding legacy experiment; do not
claim those checkpoints improve performance without the common-split evaluation.

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
  a costly GRPO job until the 5-question checkpoint smoke evaluation and its diagnostics pass.

The user wants an evidence-based improvement, not a speculative rewrite. State the hypothesis,
the expected signal, and the decision rule before each material experiment.

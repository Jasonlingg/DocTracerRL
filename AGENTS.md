# RLM Explorer agent guide

Read [the Astra improvement brief](docs/GPT_ASTRA_IMPROVEMENT_BRIEF.md) before changing
the training or reward code. It records the current evidence, research, and order of work.

The immediate objective is to establish whether the existing public Qwen checkpoints improve
over the base model on the same MuSiQue split. Do that before redesigning the environment or
starting another expensive training run.

- Preserve the user's uncommitted changes in `scripts/train_grpo_custom.py`,
  `tests/test_grpo_loss.py`, and `docs/QWEN_TRAINING_RESEARCH_AND_IMPROVEMENTS.md`.
- Treat `src/env/reward.py` as the active reward definition. Historical reward formulas in
  `README.md`, `RESULTS.md`, `STATUS.md`, and `PLAN.md` are not authoritative until reconciled.
- Keep the task multi-step and tool-mediated. Do not collapse it into a one-shot classifier.
- Make each experiment reproducible: record checkpoint identifier, question split and IDs,
  corpus revision, policy decoding settings, reward version, hardware, and seed.
- Prefer small, isolated changes followed by the narrowest meaningful verification. Do not run
  a costly GRPO job until the 5-question checkpoint smoke evaluation and its diagnostics pass.

The user wants an evidence-based improvement, not a speculative rewrite. State the hypothesis,
the expected signal, and the decision rule before each material experiment.

# GPT-6 Astra improvement brief

> September 12 direction update: the user approved a focused AI-paper research assistant.
> [The research runbook](RESEARCH_ASSISTANT.md) records the current milestone: establish a
> useful retrieval baseline, review failures, then choose an intervention. The MuSiQue
> evidence and training cautions below remain relevant to the legacy experiment; its
> checkpoint comparison is still unexecuted and is not evidence of research-agent quality.

> September 10 update: the subsequent audit reproduced correctness failures and
> the user authorized repairing them before checkpoint evaluation. Read the
> [readiness report and completed repair pass](GPU_TRAINING_READINESS.md) first.
> The current suite has 72 passing tests. New runs use terminal-only `outcome-v1`;
> use `requirements-pod.lock` and the updated evaluation runbook. Historical
> observations and the original execution order below are superseded where they
> conflict with that repair report.

## Mission

RLM Explorer trains an open-weight model to answer multi-hop questions by writing Python in a
persistent REPL. The research question is whether reinforcement learning teaches a Qwen 7B model
to explore documents better than its base model and an SFT warm-start. The project is not ready to
claim that result: the public SFT and GRPO checkpoints have not been evaluated on a common split.

The next action is measurement, then diagnosis, then a single controlled intervention. Do not
start a new long training run or change multiple variables at once.

## Repository map

| Area | Files | Role |
| --- | --- | --- |
| Environment | `src/env/document_env.py`, `repl.py`, `tools.py`, `reward.py`, `corpus.py` | Persistent Python execution, retrieval tools, rewards, and corpus loading. |
| Policies | `src/policies/qwen_*.py`, `qwen_common.py` | Same Qwen base with no adapter, SFT adapter, or GRPO adapter. |
| Training | `scripts/train_sft.py`, `collect_sft_data.py`, `train_grpo_custom.py` | LoRA SFT and a custom multi-turn GRPO implementation. |
| Evaluation | `scripts/run_eval.py`, `eval_checkpoints.sh`, `summarize_eval.py`, `analyze_run.py` | Runs policies and persists transcripts for summary and hop analysis. |
| Evidence | `RESULTS.md`, `docs/EVAL_RUNBOOK.md`, `docs/QWEN_TRAINING_RESEARCH_AND_IMPROVEMENTS.md` | Historical results, the unexecuted checkpoint runbook, and current optimizer analysis. |

`docs/EVAL_RUNBOOK.md` is the operational source for checkpoint evaluation. The three public
adapters are `jasonlingg/doctracerrl-sft-qwen2.5-7b`,
`jasonlingg/doctracerrl-grpo-qwen2.5-7b-50steps`, and
`jasonlingg/doctracerrl-grpo-qwen2.5-7b`.

## What is established

- MuSiQue is a reasonable task choice: it was designed to require connected 2--4-hop reasoning
  and makes disconnected shortcuts substantially less effective. [MuSiQue](https://aclanthology.org/2022.tacl-1.31/)
- Search-R1 is close prior art: it trains multi-turn retrieval through outcome reward and masks
  retrieved context from the policy-gradient loss. This project already computes loss only on
  assistant action tokens, which is the right boundary. [Search-R1](https://arxiv.org/abs/2503.09516)
- Multi-turn credit assignment is a real problem. GTPO reports a modest average improvement over
  GRPO by assigning turn-level returns and adding execution-derived shaping. Treat it as a later
  ablation, not a substitute for a trustworthy GRPO baseline. [GTPO](https://arxiv.org/abs/2511.14846)
- The current working tree contains an uncommitted GRPO correction: clean old-policy log-probs,
  per-token PPO ratios, dropout-disabled updates, and truncation diagnostics. Preserve it and
  validate it rather than replacing it.
- The current test command produced **34 passes and 18 errors**. Every error attempts to download
  `sentence-transformers/all-MiniLM-L6-v2`; sandbox DNS fails and the Hugging Face client later
  raises `RuntimeError: Cannot send a request, as the client has been closed.` This is a
  reproducibility/test-isolation gap, not evidence that the environment tests found a logic bug.

## Material inconsistencies to resolve before interpreting results

1. **Training and evaluation do not report the same return.** During GRPO rollouts,
   `_collect_rollout()` accumulates retrieval-hit shaping rewards and the final submission reward.
   `src/eval/harness.py::run_single()` overwrites `reward` each turn, so its saved transcript keeps
   only the final submission reward. The checkpoint leaderboard therefore cannot say whether it is
   measuring the training objective. Give outcome metrics and shaping return separate names and
   persist both.
2. **Reward documentation is stale.** The active code uses
   `0.8 * answer_F1 + 0.1 * citation_precision + 0.1 * citation_recall`, plus a gated exploration
   bonus. `README.md`, `RESULTS.md`, `STATUS.md`, `PLAN.md`, and analysis helpers still describe
   earlier formulas, including a removed efficiency bonus and a format bonus. Version the reward
   and rewrite reports from structured metrics instead of subtracting assumptions after the fact.
3. **The highest-priority experiment is absent.** `RESULTS.md` says the 7B SFT evaluation is
   pending and GRPO results are TBD. `docs/EVAL_RUNBOOK.md` says the three public checkpoints were
   pushed but never evaluated. Do not infer a failure from the live training counter.
4. **Unit tests require an online model download.** Unit tests for corpus, environment, verifier,
   and sparse RAG cannot run in a clean offline checkout. Add a test-only deterministic embedder
   or fixture that avoids model download; retain one opt-in integration test for the actual
   sentence-transformer model.
5. **The experiment metadata is incomplete.** The run script does not save seed, decoding
   parameters, model revision, corpus hash, reward version, or the selected question IDs next to
   its summary. Add a machine-readable manifest before any result is presented as a comparison.

## Execution order

### 1. Establish a checkpoint baseline

On a GPU machine with the pinned `requirements-pod.txt`, rebuild the deterministic MuSiQue corpus
and run:

```bash
./scripts/eval_checkpoints.sh 5
./scripts/eval_checkpoints.sh 50
```

The smoke run must load all four policies: base, SFT, 50-step GRPO, and later GRPO. Before the
50-question run, confirm the same question IDs, corpus, decoding parameters, and max-step limit
are used for each policy. Store each policy's transcript under a distinct, explicit name so the
two `grpo_policy` runs cannot be accidentally combined.

Report answer F1, citation precision/recall, submission rate, step count, outcome reward, and
training-return-with-shaping separately. Stratify every metric by 2-, 3-, and 4-hop questions.
The primary deltas are `SFT - base`, `GRPO-50 - SFT`, and `GRPO - SFT`, with bootstrap confidence
intervals over common question IDs.

Decision rule:

- If SFT exceeds base and either GRPO checkpoint exceeds SFT on answer F1 with a meaningful
  interval, preserve the setup and extend training only after examining trajectories.
- If SFT improves but GRPO does not, investigate group reward variance, actual optimizer updates,
  ratio/clip diagnostics, and shaping leakage before changing data or model.
- If neither SFT nor GRPO improves, inspect data quality and train/eval format alignment before
  changing the algorithm.

### 2. Make evaluation trustworthy

Implement this as a focused change after the smoke run:

1. Add an `ExperimentManifest` beside each transcript with the checkpoint, base model revision,
   commit SHA, split, exact IDs, corpus checksum, reward version, seed, generation configuration,
   and hardware.
2. Record `outcome_reward`, `shaping_reward`, and `episode_return` separately. Keep historical
   `reward` only as a compatibility field until consumers are migrated.
3. Update `summarize_eval.py` and `analyze_run.py` to derive displays from the stored structured
   fields, not hard-coded formulas.
4. Add a paired-bootstrap report for all checkpoint deltas and a per-hop table.
5. Make offline unit tests deterministic by injecting an embedder or using a tiny precomputed
   embedding fixture. Add a clearly named integration test that downloads the real model only
   when explicitly requested.

### 3. Diagnose GRPO only with recorded signals

Keep the current custom trainer as the first baseline, with the working-tree PPO fix intact.
Persist the following per optimization step: fraction of uniform-reward groups, number of
nonzero-advantage groups, mean and standard deviation of terminal and shaping rewards, ratio at
the first update, clip fraction by PPO epoch, KL, gradient norm, action-token count, truncation
rate, submission rate, and syntax-error rate. Save sample trajectories for the best, median, and
worst rollouts in each checkpoint.

The important control is a tiny overfit run on a fixed handful of questions. It should show reward
and submission improvement before spending on a broad MuSiQue run. If it cannot overfit after the
ratio identity check passes, treat reward/data alignment as the first suspect.

### 4. Change one learning variable at a time

The current SFT data has 114 trajectories and is dominated by 2-hop examples. If the fixed
optimizer still produces no improvement, collect a larger teacher dataset with explicit target
counts for 2-, 3-, and 4-hop questions, reject malformed or non-submitting trajectories, and
measure base versus SFT before attempting GRPO again. Do not claim a universal required data size;
measure the curve at several sizes.

Only after that baseline should you run these ablations, one per experiment:

1. Planning text that is stripped before code execution, to test whether it reduces syntax errors
   or improves discovery. Do not assume a benefit without a controlled comparison.
2. Capped `read()` output plus evidence windows, to reduce repeated full-document reads while
   preserving an agent's ability to inspect a source.
3. Turn-level credit assignment, inspired by GTPO, using signals observable to the environment
   rather than answer-token overlap alone. Compare it to the current outcome-only GRPO baseline.

## GPT-6 Astra operating prompt

Use the following as Astra's task instruction after it has read this repository and this brief:

> You are the research engineer for RLM Explorer. First establish a reproducible common-split
> evaluation of base, SFT, GRPO-50, and GRPO checkpoints; do not change the training recipe before
> that evidence exists. Preserve uncommitted optimizer changes. Separate terminal outcome metrics
> from reward shaping, record experiment manifests, and make each conclusion traceable to a saved
> transcript. When an intervention is justified, state its hypothesis, control, metric, cost, and
> stop condition. Keep the environment multi-turn and executable-code based. Prefer a small,
> reviewable implementation with targeted tests over a speculative rewrite.

## GPT-6 Astra configuration

Use `gpt-6-astra` with the Responses API for tool-heavy work. Start at `high` reasoning effort
for the repository audit and experimental design; use `medium` for routine focused changes. GPT-6
Astra supports MCP, web search, file search, hosted shell, and code tools. Its official guidance
recommends explicit instructions for autonomy, testing scope, and subagent use when those choices
matter. [OpenAI model guidance](https://developers.openai.com/api/docs/guides/latest-model)

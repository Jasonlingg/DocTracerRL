# GPU readiness and portfolio plan

Assessment date: 2026-09-10. Repository HEAD: `3f57a12`, with existing uncommitted trainer, test, and research-document changes. This assessment did not alter application or training code.

## Repair pass completed: 2026-09-10

The assessment below describes the pre-fix state. The user subsequently authorized
step 1, and the following repairs are now implemented in the working tree:

- **Dependencies:** compatible direct Transformers/Datasets pins, plus a resolved
  `requirements-pod.lock` for Linux x86_64/Python 3.11/CUDA 12.1. The Docker client
  is included. SFT now passes its requested sequence length to `SFTConfig.max_length`.
- **REPL:** isolate the current step's stdout and stderr before truncating; roll
  failed and timed-out actions out of cumulative history based on process status,
  rather than matching the word SyntaxError. Docker script transport uses stdin,
  and an in-container timeout bounds the remote Python process.
- **GRPO:** bound context before generation, preserve the system/question prefix,
  and store those exact tokens for old/new/reference scoring. Use explicit
  unfiltered sampling and the same temperature in all log-probability passes.
  Keep the existing dropout and optimizer diagnostics. Abort infrastructure failures
  instead of inserting fake zero rewards; reject invalid reward groups.
- **Rewards:** `outcome-v1` is terminal-only, with unchanged 0.8/0.1/0.1 outcome
  weights and no retrieval-hit or extra-step bonus. New artifacts distinguish
  outcome, shaping, and episode return. Analyzer and viewer no longer invent
  historical reward formulas. The legacy broken scorer import is removed.
- **Evaluation:** exact output files and unique run IDs prevent checkpoint merges.
  Each transcript has a manifest with data hashes, selected IDs, checkpoint name,
  seed, decoding metadata, reward version, and git state. Errors are explicit and
  cause CLI failure after diagnostic artifacts are saved. Summaries withhold
  checkpoint deltas when recorded protocols differ or execution errors occur.

Verification:

- Full suite: **72 passed, 3 existing SWIG deprecation warnings, 23.14 seconds**,
  with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`. Existing corpus integration tests
  used the local model cache; this is not proof that a clean machine needs no downloads.
- Added 20 regression tests. A tiny randomly initialized transformer verifies
  actual generation scores versus stored/update probabilities at temperatures
  0.7, 1.0, and 1.3, including long history and finite backward gradients.
- Linux/Python 3.11 dependency resolution succeeded and the full resolved package
  list is saved. An isolated macOS/Python 3.11 environment successfully imported
  Transformers 4.56.2, TRL 1.3.0, PEFT 0.13.2, Accelerate 1.4.0, and Datasets 4.7.0,
  and constructed assistant-only SFT configuration with max length 8,192. That
  import check used CPU torch 2.14.0; it is not CUDA torch 2.4.0 runtime validation.
- Shell syntax, viewer JavaScript syntax, new helper/test lint, and diff whitespace
  checks passed. Docker transport regression uses mocked Docker calls with real
  local Python execution; actual Docker isolation was not exercised.

Docker was not required or started for this repair pass. It is an optional backend
for executing the model's document-exploration code, separate from GPU training.
See [Docker's role in this project](DOCKER.md) for the rationale, local fallback
behavior, and remaining container validation.

**Next gates:** validate the exact restored corpus and pinned CUDA stack; run the
five-question checkpoint evaluation and a tiny training backward/save/reload smoke
test before substantial GPU training. No GPU run or deployment was performed.
The REPL still replays previous successful actions and cannot roll back external
side effects. Context budgeting can discard old evidence; remote adapter names
must be pinned/archived by revision before publishing reproducible results.

## Decision

Do not launch a substantial training run yet. Fix environment correctness, dependency resolution, and evaluation identity first; then run a small GPU forward/backward/save/reload smoke test. Local tests pass, but neither CUDA execution nor peak GPU memory has been validated in this assessment.

This document refines the earlier `GPT_ASTRA_IMPROVEMENT_BRIEF.md`: passing tests does not establish that the environment is healthy, and there is no measured basis here for a GPU cost estimate or an RL improvement claim.

## Evidence gathered

- `pytest -q --tb=short`, with network access for model downloads: **52 passed, 3 warnings, 19.16 seconds**. This validates the local installed environment, not a fresh installation of `requirements-pod.txt`.
- Local corpus: **10,340 JSON documents**. Question splits: train 500, dev 100, test 200; a separate eval file contains 200. No exact question-string overlap was found between train/dev/test. Expected citation filenames exist; paragraph completeness, aliases, and semantic leakage remain unverified.
- SFT data: **114 conversations, 716 assistant turns**. Matching their questions to train gives 105 two-hop, 6 three-hop, and 3 four-hop conversations. Earlier distribution claims should be updated.
- Local `out/run_*.json` artifacts contain Claude/RAG policies, but no Qwen policy results were found. This does not establish what may have been evaluated on another machine.
- The configured replay directory, `scripts/replays`, has no replay artifacts. The viewer already has useful live/replay infrastructure.

## Blockers, in repair order

### 1. Make fresh GPU installation reproducible

`requirements-pod.txt` pins `trl==1.3.0`, `transformers==4.45.2`, and `datasets==3.1.0`. Published TRL 1.3.0 metadata requires `transformers>=4.56.2` and `datasets>=4.7.0`: these constraints cannot resolve together.

Choose and validate a compatible Linux/CUDA stack. Consider separate dependency sets for custom GRPO/evaluation and TRL-based SFT; the custom trainer does not need TRL. Check the SFT APIs against the chosen version rather than blindly upgrading two pins. Record Python, CUDA, torch, transformers, PEFT, TRL, and bitsandbytes versions in each run.

Source: [published TRL 1.3.0 metadata](https://pypi.org/pypi/trl/1.3.0/json).

### 2. Repair REPL execution semantics before trusting trajectories

In `src/env/repl.py`, both implementations truncate cumulative process output to 8,000 characters before extracting the current step. A direct LocalREPL probe printing 9,000 characters, followed by `print("CURRENT_STEP_SENTINEL")`, lost the sentinel.

Only SyntaxError causes rollback. A direct probe executing `raise ValueError("old failure")`, then `print("RECOVERED")`, re-raised the old exception and never executed the recovery action.

Extract current-step output before truncation and define recovery from runtime errors. Add regression tests for both probes. Longer term, use a persistent isolated worker that executes each action once: cumulative replay currently repeats earlier actions and does not preserve process-local caches between calls. Public live arbitrary-code execution requires real isolation; LocalREPL is a host process.

### 3. Align generated trajectories with the policy loss

In `scripts/train_grpo_custom.py`, `_collect_rollout` generates with full history, while log-probability calculation uses `_prepare_ctx` with a 2,048-token context cap. The head-preserving truncation change helps preserve instructions but can still drop the question or evidence. Old/new ratio identity alone does not establish correctness when both probabilities use different context from the sampler.

Use the same explicitly constructed context for generation and scoring, and store it with each action. Make sampling settings explicit and account for any sampling-distribution transformations. Add a long-history regression test. The current disabled-adapter KL reference is the frozen base model; decide and document whether base or frozen SFT is the intended reference.

Primary background: [TRL GRPO documentation](https://huggingface.co/docs/trl/grpo_trainer), including its discussion of rollout/scoring distribution mismatch and importance sampling correction.

### 4. Remove accidental reward signals and distinguish metrics

`DocumentEnv._retrieval_hit_reward` checks answer-word overlap against arbitrary observation text. A synthetic episode with gold answer `yes` received 0.02 from the environment's own “If yes” step warning, with no retrieved evidence. This is a demonstrated mechanism, not a claim that an actual training example has that answer. Repeated or fabricated output can likewise be mistaken for retrieval.

Start with an outcome-only baseline, or derive evidence rewards from verified, unique source accesses. Log terminal answer/citation scores separately from shaped episode return. Audit active reward code, analyzer, summarizer, and viewer: they currently use different reward formulas. The viewer still displays 0.5/0.25/0.25 answer/citation weights while the active terminal outcome uses 0.8/0.1/0.1.

### 5. Give every evaluated checkpoint its own identity

`scripts/eval_checkpoints.sh` labels both GRPO checkpoints `grpo_policy`; the summarizer groups by policy and can merge them. Selecting the newest four files also risks collecting unrelated runs.

Write explicit output paths and immutable run/checkpoint IDs. Record model/adapter revision, data hashes, prompt, context budget, decoding settings, reward version, and git state. Record execution failures separately from incorrect answers. Report per-question outcomes so comparisons can be paired.

### 6. Reproduce the corpus before training elsewhere

The default `setup_musique.py` path processes only 200 validation examples. It is not sufficient evidence that a fresh GPU checkout recreates the corpus used by the checked-in training/test questions. Repeated article filenames are overwritten rather than paragraph-merged; title slug collisions are another possible loss mechanism.

Provide the exact corpus/split construction recipe or a versioned data snapshot, and validate supporting paragraph content as well as file existence. Keep held-out test data out of tuning decisions.

## First GPU run: acceptance criteria

1. A clean Linux GPU environment installs successfully and loads the intended model/adapter and corpus.
2. A tiny run executes real multi-turn actions, includes an error-recovery case and a long-history case, and obtains at least one nonzero-advantage group. Choose probe questions to exercise these paths; do not treat smoke performance as an evaluation result.
3. Forward/backward/update completes with finite loss, KL, gradients, and parameters. Verify identical-policy likelihood ratios on the actual stored rollout contexts.
4. Save and reload the adapter; verify deterministic generation on a fixed probe with fixed decoding settings.
5. Record peak VRAM, seconds per episode/update, token counts, submission rate, truncation rate, and errors. Use measurements to choose GPU size and estimate cost.

Only then run a modest training pilot. At the current 75 steps × 4 questions × 8 rollouts, the configured run can collect 2,400 episodes and up to 24,000 generation calls; three PPO epochs imply 225 optimizer updates. Do not estimate cost from the outer step count alone. Current adapter saves are not complete optimizer/RNG resume checkpoints.

## Evaluation that can support a résumé claim

- Compare base Qwen, SFT Qwen, and GRPO Qwen on the same questions, tools, action budget, context handling, and decoding protocol. Add a same-model single-pass/RAG baseline if claiming the benefit of iterative exploration; a Claude-vs-Qwen comparison confounds training and model choice.
- Use a small development pilot, then the dev split for checkpoint selection. Reserve the test split for the final selected comparison.
- Report answer F1, citation precision/recall, submission rate, steps, generated tokens, latency, and execution failures. Break down by hop count. Report paired uncertainty on the held-out comparison rather than only one average shaped reward.
- Save per-question transcripts, manifests, training curves, and the exact reproduction commands. Do not claim an RL gain until measured.
- Check whether longer-hop SFT coverage and successful recovery trajectories improve development results; the current SFT mix is heavily two-hop. More data alone is not yet the demonstrated bottleneck.

## Demo and portfolio priorities

Reuse the existing viewer. Its live frontend is currently hardcoded to Claude/RAG policies, not the three Qwen training stages. The live backend also expects an Anthropic key. Loading multiple 7B policies concurrently needs a deliberate GPU memory plan.

Build a replay-first demo from genuine recorded Qwen trajectories. Show the same question across base/SFT/GRPO, with code actions, the source passages read, final answer/citations, and steps/tokens/latency. Label replay and curated examples clearly. Include a two-hop example, a harder bridge question, and an instructive failure. Place aggregate held-out results alongside curated examples so the presentation does not imply cherry-picked runs represent average performance.

For public replay hosting, use an explicit container bind address: the viewer defaults to 127.0.0.1, which is not exposed simply by Docker's EXPOSE instruction. Keep arbitrary execution out of the public replay service.

The strongest portfolio story is an engineered and measured loop: document environment → executable exploration → SFT → RL → held-out evaluation → inspectable trajectories. Add a concise README with an architecture diagram, one-command replay instructions, a short demo recording, reproduction commands, and measured limitations.

An honest current résumé formulation is: “Built a multi-turn document-exploration environment with executable Python actions, answer/citation rewards, and LoRA SFT/custom GRPO tooling for Qwen2.5-7B over a 10,340-document corpus.” After evaluation, replace process-heavy wording with the measured held-out gain and its token/latency tradeoff.

## Implementation sequence for the next agent

Preserve the existing dirty changes. First repair dependencies, REPL semantics, context consistency, reward observability, and checkpoint identity, with targeted regression tests. Next reproduce data and run the GPU smoke test. Then evaluate base/SFT and run a modest GRPO pilot. Finally build the replay presentation around real artifacts and publish only supported performance claims. No GPU purchase, deployment, or training launch was performed in this assessment.

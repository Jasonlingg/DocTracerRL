# Runbook: evaluate the June checkpoints

## September 15 update: run complete, see RESULTS.md Phase 5

This runbook was finally executed. First attempt (n=5) was confounded by a REPL bug — bare
`search(...)` calls with no `print()` produced zero output, so even the untrained base model
scored 0.000. Fixed in `src/env/repl.py` (`_auto_print_trailing_expression`). The real,
unconfounded 50-question result: **SFT beats base** (+0.018 outcome, +0.013 answer F1), **GRPO
does not measurably beat SFT** (-0.003, noise-level). Closer to the "flat" branch below than the
"training was working" branch. The apparent GRPO checkpoint comparison is resolved: both Hub IDs
contain the exact same adapter weight object. Full numbers and methodology are in `RESULTS.md`
Phase 5.

## September 10 correctness update

The dependency conflict, REPL recovery/output handling, rollout/scoring context
mismatch, stdout-based shaping, and checkpoint-summary identity have been repaired.
New runs use `outcome-v1` (0.8 answer F1 + 0.1 citation P + 0.1 citation R, no bonuses).
Historical rewards below use older formulas and must not be treated as a common baseline.
See [the readiness assessment](GPU_TRAINING_READINESS.md) for verification and remaining gates.

## Why

Three checkpoints were trained on 2026-06-15 and pushed to HF. **None were ever evaluated.**
`RESULTS.md` §4.3 records the SFT as "Dev eval: pending", §4.4 records GRPO as "IN PROGRESS,
Results: TBD", and all three Immediate next-steps boxes are still unchecked.

The belief that GRPO flatlined — which drove shelving this project and every design decision
in slmforge — rests on watching a live reward counter, not on a measured result. This run
replaces that impression with three numbers.

Two reasons to doubt the impression:

- The run stopped at **50 of 300 steps**. Literature GRPO runs go hundreds to thousands.
- The trainer **skips the update when a group's rewards are uniform** (zero variance → zero
  advantage). At ~0.113 avg reward most rollouts were failing to submit, so many of those 50
  groups were likely all-zero and skipped outright. The effective number of gradient updates
  may be far below 50 — undertrained, not failed.

## Checkpoints

| HF repo | Policy | Env var |
|---|---|---|
| `jasonlingg/doctracerrl-sft-qwen2.5-7b` | `qwen_sft_policy` | `CHECKPOINT_PATH` |
| `jasonlingg/doctracerrl-grpo-qwen2.5-7b-50steps` | `grpo_policy` | `CHECKPOINT_PATH` |
| `jasonlingg/doctracerrl-grpo-qwen2.5-7b` | `grpo_policy` | `CHECKPOINT_PATH` |
| — (none) | `qwen_base_policy` | — |

All public, LoRA r=16 / alpha=32 / dropout=0.05 on all seven projection modules, base
`Qwen/Qwen2.5-7B-Instruct`. The `base_model_name_or_path` inside `adapter_config.json` points
at `/workspace/models/qwen2.5-7b` (a dead pod path) — harmless, the policies override it with
`BASE_MODEL_PATH` or the `Qwen/Qwen2.5-7B-Instruct` default.

## Pod setup

Use a Linux x86_64/Python 3.11 CUDA environment. Model weights alone do not establish
peak VRAM requirements: measure the five-question smoke evaluation before selecting
a larger run. Training also needs a separate backward/save/reload smoke test.

```bash
git clone <this repo> && cd rlm-explorer
pip install -r requirements-pod.lock
pip install -e . --no-deps

# Restore the exact corpus used to construct the committed question splits.
# The default setup_musique.py invocation alone has NOT been validated to do this.
```

## Run

```bash
./scripts/eval_checkpoints.sh 5     # smoke test — inspect failures before expanding
./scripts/eval_checkpoints.sh 50    # the real run
```

Smoke first, always. A failed adapter load or an OOM costs the same to discover on 5 questions
as on 50, and the whole point of this run is that nobody checked cheaply last time.

Docker is optional for these training/evaluation scripts. In automatic mode,
`PersistentREPL` uses Docker only if the engine is reachable and `rlm-sandbox`
exists; otherwise it executes generated Python locally with the parent process's
access. Use a disposable, restricted environment for local execution. Ownership
of a checkpoint does not guarantee safe generated code. See [Docker's role and
validation status](DOCKER.md) for the backend choices and what was actually tested.

The runner writes four explicitly named transcripts plus manifests in a unique
`out/eval_*` directory, and summarizes only those exact files. Any execution error
or incomplete evaluation returns a nonzero status after saving diagnostic results.
The manifests include data hashes, selected IDs, decoding metadata, seed, reward
version, and git state. A requested remote adapter name is recorded; pin or archive
the adapter revision before publishing a reproducible result.

## Cost

No GPU runtime or cost was measured in this repair pass. Estimate the larger run
from observed seconds per question and the actual machine's hourly price.

## Reading the result

Compare against the documented dev baseline (100 MuSiQue dev questions, `RESULTS.md` Phase 3):

| Policy | Reward | Answer F1 |
|---|---|---|
| claude_policy | 0.179 | 0.157 |
| context_stuffing | 0.176 | 0.050 |
| naive_rag | 0.147 | 0.038 |

The three numbers that matter: **`sft − base`**, **`grpo50 − sft`**, **`grpo − sft`**.

- **SFT > base and GRPO > SFT** → training was working; the project was shelved prematurely,
  and slmforge's premise needs revisiting.
- **Flat** → first real evidence of failure. Reportable, honest, and a much better basis for
  what to change than an impression was.

Either way, record the numbers in `RESULTS.md` and tick the boxes. The failure mode being
corrected here is not "GRPO didn't work" — it is "we stopped measuring."

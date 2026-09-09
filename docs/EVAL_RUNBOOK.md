# Runbook: evaluate the June checkpoints

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

RTX 4090 (24GB, ~$0.34/hr) is enough — 7B in bf16 is ~16GB. Allow **~60GB disk**: base model
~15GB, MuSiQue corpus, FAISS index.

```bash
git clone <this repo> && cd rlm-explorer
pip install -r requirements-pod.txt      # pinned; do NOT resolve fresh
pip install -e .

# Question splits are committed. The corpus is not — rebuild it (deterministic:
# examples[skip : skip+n] over the answerable subset, no shuffle).
python scripts/setup_musique.py
```

## Run

```bash
./scripts/eval_checkpoints.sh 5     # smoke test — catches load/OOM errors for pennies
./scripts/eval_checkpoints.sh 50    # the real run
```

Smoke first, always. A failed adapter load or an OOM costs the same to discover on 5 questions
as on 50, and the whole point of this run is that nobody checked cheaply last time.

Docker isn't available inside most pods, so `PersistentREPL` falls back to `LocalREPL`. Fine
here — we are running our own checkpoints, not defending against an adversarial agent.

## Cost

~30-60 min per policy for 50 questions (up to 10 steps/question, ~1024 tokens/step).
Four policies ≈ 2-4 hours ≈ **$1-1.50** on a 4090.

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

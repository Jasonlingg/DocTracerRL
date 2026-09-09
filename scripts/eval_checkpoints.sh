#!/usr/bin/env bash
# Evaluate the DocTracerRL checkpoints that were trained 2026-06-15 but never measured.
#
# Answers the question three months of design decisions were built on without evidence:
# did SFT beat the untrained base, and did GRPO beat SFT?
#
# Usage:
#   ./scripts/eval_checkpoints.sh 5     # smoke test first — catches load errors for pennies
#   ./scripts/eval_checkpoints.sh 50    # the real run
#
# Requires a GPU (7B in bf16 ≈ 16GB VRAM) and ~40GB disk for the base model.

set -euo pipefail

N="${1:-5}"
SPLIT="${SPLIT:-dev}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="out/eval_${STAMP}"
mkdir -p "$LOGDIR"

BASE_SFT="jasonlingg/doctracerrl-sft-qwen2.5-7b"
BASE_GRPO_50="jasonlingg/doctracerrl-grpo-qwen2.5-7b-50steps"
BASE_GRPO="jasonlingg/doctracerrl-grpo-qwen2.5-7b"

echo "=============================================="
echo " questions : $N (split=$SPLIT)"
echo " logs      : $LOGDIR"
echo "=============================================="

run() {
  local name="$1" policy="$2" ckpt="${3:-}"
  echo
  echo "----- $name -----"
  local start=$SECONDS
  if [ -n "$ckpt" ]; then
    CHECKPOINT_PATH="$ckpt" python scripts/run_eval.py \
      --musique --split "$SPLIT" -t "$N" -p "$policy" 2>&1 | tee "$LOGDIR/${name}.log"
  else
    python scripts/run_eval.py \
      --musique --split "$SPLIT" -t "$N" -p "$policy" 2>&1 | tee "$LOGDIR/${name}.log"
  fi
  echo "[$name done in $((SECONDS - start))s]" | tee -a "$LOGDIR/${name}.log"
}

# Untrained base first. If this errors, nothing downstream is worth running,
# and it's also the baseline every other number is meaningless without.
run "1_base"     qwen_base_policy
run "2_sft"      qwen_sft_policy  "$BASE_SFT"
run "3_grpo50"   grpo_policy      "$BASE_GRPO_50"
run "4_grpo"     grpo_policy      "$BASE_GRPO"

echo
echo "=============================================="
echo " RESULTS"
echo "=============================================="
# Summarize from the saved JSON transcripts, not console scraping. The summarizer
# subtracts the efficiency bonus (RESULTS.md:159) and is validated to reproduce
# the documented Phase 3 baseline exactly.
python scripts/summarize_eval.py $(ls -t out/run_*.json | head -4) | tee "$LOGDIR/SUMMARY.txt"

echo
echo "Summary saved to $LOGDIR/SUMMARY.txt"
echo "Record these in RESULTS.md and tick the boxes in Next Steps."

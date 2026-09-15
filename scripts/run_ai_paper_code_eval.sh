#!/usr/bin/env bash
# Run the same frozen ten-question AI-paper evaluation for base and SFT Qwen.
set -euo pipefail

cd "$(dirname "$0")/.."

: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to the SFT adapter directory or Hub ID}"

pilot_python="${RESEARCH_PYTHON:-python3}"
pilot_snapshot="${RESEARCH_SNAPSHOT:-out/research/starter-2026-09-12}"
pilot_questions="${RESEARCH_QUESTIONS:-data/research/code_exec_pilot_v1.json}"
pilot_output="${RESEARCH_OUTPUT:-out/research/code-exec-pilot-$(date -u +%Y%m%dT%H%M%SZ)}"

if [[ -e "$pilot_output" ]]; then
  echo "Output already exists: $pilot_output" >&2
  exit 1
fi

if [[ ! -f "$pilot_snapshot/manifest.json" ]]; then
  "$pilot_python" scripts/research.py snapshot \
    --sources data/research/sources.json \
    --output "$pilot_snapshot"
fi

"$pilot_python" scripts/research_benchmark.py validate \
  --benchmark "$pilot_questions" \
  --snapshot "$pilot_snapshot"

mkdir -p "$pilot_output"

"$pilot_python" scripts/run_eval.py \
  --policy qwen_base_policy \
  --questions "$pilot_questions" \
  --corpus "$pilot_snapshot/corpus" \
  --max-steps 10 \
  --seed 42 \
  --require-evidence \
  --no-vector-index \
  --run-label base \
  --output "$pilot_output/base.json"

"$pilot_python" scripts/run_eval.py \
  --policy qwen_sft_policy \
  --questions "$pilot_questions" \
  --corpus "$pilot_snapshot/corpus" \
  --max-steps 10 \
  --seed 42 \
  --require-evidence \
  --no-vector-index \
  --run-label sft \
  --output "$pilot_output/sft.json"

"$pilot_python" scripts/review_code_exec_pilot.py prepare \
  --questions "$pilot_questions" \
  --corpus "$pilot_snapshot/corpus" \
  --run "$pilot_output/base.json" \
  --run "$pilot_output/sft.json" \
  --output "$pilot_output/blind-review" \
  --seed 42

echo "Evaluation complete: $pilot_output"
echo "Review blind-review/review.md, then enter pass/partial/fail in review.json."
echo "Keep blind-review/blind-key.json closed until the review is finished."

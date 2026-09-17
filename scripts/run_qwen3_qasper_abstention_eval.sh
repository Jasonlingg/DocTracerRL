#!/usr/bin/env bash
# Compare base Qwen3-8B with one QASPER SFT adapter under the locked protocol.
set -euo pipefail

cd "$(dirname "$0")/.."

: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to the SFT adapter directory}"

eval_python="${RESEARCH_PYTHON:-python3}"
base_model="${BASE_MODEL_PATH:-Qwen/Qwen3-8B}"
benchmark_dir="${QASPER_EVAL_DIR:-out/research/qasper-test-abstention-v1}"
output_dir="${QASPER_EVAL_OUTPUT:-out/research/qwen3-qasper-abstention-$(date -u +%Y%m%dT%H%M%SZ)}"
candidate_checkpoint="$CHECKPOINT_PATH"

if [[ -e "$output_dir" ]]; then
  echo "Output already exists: $output_dir" >&2
  exit 1
fi

"$eval_python" scripts/research_benchmark.py validate \
  --benchmark "$benchmark_dir/benchmark.json" \
  --snapshot "$benchmark_dir"

mkdir -p "$output_dir"
exec > >(tee "$output_dir/run.log") 2>&1

common_args=(
  --questions "$benchmark_dir/benchmark.json"
  --corpus "$benchmark_dir/corpus"
  --max-steps 10
  --seed 42
  --no-vector-index
  --question-only-observation
)

env -u CHECKPOINT_PATH BASE_MODEL_PATH="$base_model" \
  "$eval_python" scripts/run_eval.py \
  --policy qwen_base_policy \
  --run-label qwen3_base_qasper_abstention_v2 \
  --output "$output_dir/base.json" \
  "${common_args[@]}"

BASE_MODEL_PATH="$base_model" CHECKPOINT_PATH="$candidate_checkpoint" \
  "$eval_python" scripts/run_eval.py \
  --policy qwen_sft_policy \
  --run-label qwen3_prefix_aligned_sft_qasper_abstention_v1 \
  --output "$output_dir/candidate.json" \
  "${common_args[@]}"

"$eval_python" scripts/score_abstention_eval.py \
  --benchmark "$benchmark_dir/benchmark.json" \
  --base "$output_dir/base.json" \
  --candidate "$output_dir/candidate.json" \
  --output "$output_dir/abstention-score.json"

echo "Evaluation complete: $output_dir"

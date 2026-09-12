#!/usr/bin/env bash
# Run the locked eight-question pilot against an already-running model server.
set -euo pipefail

cd "$(dirname "$0")/.."
: "${RESEARCH_MODEL_REVISION:?Set the exact model commit served by the endpoint}"
: "${RESEARCH_SERVER_HARDWARE:?Record GPU, dtype, server version, and context limit}"

research_python="${RESEARCH_PYTHON:-python3}"
research_model="${RESEARCH_MODEL:-Qwen/Qwen3-8B}"
research_endpoint="${RESEARCH_ENDPOINT:-http://localhost:8000/v1}"
research_snapshot="${RESEARCH_SNAPSHOT:-out/research/starter-2026-09-12}"
research_benchmark="${RESEARCH_BENCHMARK:-data/research/benchmark_pilot_v1.json}"
research_output="${RESEARCH_OUTPUT:-out/research/pilot-$(date -u +%Y%m%dT%H%M%SZ)}"

if [[ -e "$research_output" ]]; then
  echo "Output already exists: $research_output" >&2
  exit 1
fi

"$research_python" scripts/research_benchmark.py validate \
  --benchmark "$research_benchmark" --snapshot "$research_snapshot"
mkdir -p "$research_output/runs"

for research_question in pilot_01 pilot_02 pilot_03 pilot_04 pilot_05 pilot_06 pilot_07 pilot_08; do
  echo "Running $research_question"
  if ! "$research_python" scripts/research.py ask \
    --snapshot "$research_snapshot" \
    --questions "$research_benchmark" \
    --question-id "$research_question" \
    --endpoint "$research_endpoint" \
    --model "$research_model" \
    --revision "$RESEARCH_MODEL_REVISION" \
    --server-hardware "$RESEARCH_SERVER_HARDWARE" \
    --max-steps 10 --max-tokens 1800 --seed 42 \
    --output "$research_output/runs/$research_question.json"; then
    research_status=$("$research_python" -c \
      'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
      "$research_output/runs/$research_question.json")
    echo "$research_question ended with status $research_status"
    if [[ "$research_status" == "error" ]]; then
      echo "Stopping the batch after a server or runner error."
      break
    fi
  fi
done

"$research_python" scripts/research_benchmark.py score \
  --benchmark "$research_benchmark" --runs "$research_output/runs" \
  --output "$research_output/automatic-score.json"
"$research_python" scripts/research_benchmark.py review-template \
  --benchmark "$research_benchmark" --runs "$research_output/runs" \
  --output "$research_output/human-review.json"

echo "Pilot complete: $research_output"
echo "Fill human-review.json, then rerun the scorer with --reviews."

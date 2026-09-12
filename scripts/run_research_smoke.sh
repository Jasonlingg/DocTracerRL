#!/usr/bin/env bash
# Run three fixed development questions against an already-running model server.
set -euo pipefail

cd "$(dirname "$0")/.."
: "${RESEARCH_MODEL_REVISION:?Set the exact model commit served by the endpoint}"
: "${RESEARCH_SERVER_HARDWARE:?Record GPU, dtype, server version, and context limit}"

research_python="${RESEARCH_PYTHON:-python3}"
research_model="${RESEARCH_MODEL:-Qwen/Qwen3-8B}"
research_endpoint="${RESEARCH_ENDPOINT:-http://localhost:8000/v1}"
research_snapshot="${RESEARCH_SNAPSHOT:-out/research/starter-2026-09-12}"
research_output="${RESEARCH_OUTPUT:-out/research/smoke-$(date -u +%Y%m%dT%H%M%SZ)}"

if [[ -e "$research_output" ]]; then
  echo "Output already exists: $research_output" >&2
  exit 1
fi

# Fail before making model calls if the snapshot is unavailable.
"$research_python" - "$research_snapshot" <<'PY'
import sys
from pathlib import Path
from src.research.agent import load_snapshot
manifest, docs = load_snapshot(Path(sys.argv[1]))
print(f"Snapshot verified: {len(docs)} papers; {manifest['corpus_hash']}")
PY
mkdir -p "$research_output"

for research_question in research_01 research_04 research_11; do
  echo "Running $research_question"
  "$research_python" scripts/research.py ask \
    --snapshot "$research_snapshot" \
    --question-id "$research_question" \
    --endpoint "$research_endpoint" \
    --model "$research_model" \
    --revision "$RESEARCH_MODEL_REVISION" \
    --server-hardware "$RESEARCH_SERVER_HARDWARE" \
    --max-steps 10 --max-tokens 1800 --seed 42 \
    --output "$research_output/$research_question.json"
done

echo "Three submissions saved in $research_output. Inspect the JSON traces and Markdown reviews."
echo "Submission is not a quality pass. Review claim support and coverage before expanding."

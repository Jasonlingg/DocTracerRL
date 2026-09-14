#!/usr/bin/env bash
# Run five fixed known-paper QASPER questions against an already-running model server.
set -euo pipefail

cd "$(dirname "$0")/.."
: "${RESEARCH_MODEL_REVISION:?Set the exact model commit served by the endpoint}"
: "${RESEARCH_SERVER_HARDWARE:?Record GPU, dtype, server version, and context limit}"

research_python="${RESEARCH_PYTHON:-python3}"
research_model="${RESEARCH_MODEL:-Qwen/Qwen3-8B}"
research_endpoint="${RESEARCH_ENDPOINT:-http://localhost:8000/v1}"
research_snapshot="${RESEARCH_SNAPSHOT:-out/research/qasper-train-pilot-v2}"
research_questions="${RESEARCH_QUESTIONS:-$research_snapshot/questions.json}"
research_plan="${RESEARCH_PLAN:-data/research/qasper_smoke_v1.json}"
research_output="${RESEARCH_OUTPUT:-out/research/qasper-smoke-$(date -u +%Y%m%dT%H%M%SZ)}"

if [[ -e "$research_output" ]]; then
  echo "Output already exists: $research_output" >&2
  exit 1
fi

"$research_python" - "$research_snapshot" "$research_questions" "$research_plan" <<'PY'
import json
import sys
from pathlib import Path
from src.research.agent import load_snapshot

snapshot, questions_path, plan_path = map(Path, sys.argv[1:])
manifest, docs = load_snapshot(snapshot)
questions = json.loads(questions_path.read_text())
plan = json.loads(plan_path.read_text())
if manifest["corpus_hash"] != plan["snapshot_corpus_hash"]:
    raise ValueError("Smoke plan and QASPER snapshot hashes differ")
available = {question["id"] for question in questions["questions"]}
missing = set(plan["question_ids"]) - available
if missing:
    raise ValueError(f"Smoke questions missing from snapshot: {sorted(missing)}")
print(f"Snapshot verified: {len(docs)} papers; {manifest['corpus_hash']}")
print("Decision rule:", plan["decision_rule"])
PY

mkdir -p "$research_output/runs"
cp "$research_plan" "$research_output/experiment-plan.json"

while IFS= read -r research_question; do
  echo "Running $research_question"
  "$research_python" scripts/research.py ask \
    --snapshot "$research_snapshot" \
    --questions "$research_questions" \
    --question-id "$research_question" \
    --endpoint "$research_endpoint" \
    --model "$research_model" \
    --revision "$RESEARCH_MODEL_REVISION" \
    --server-hardware "$RESEARCH_SERVER_HARDWARE" \
    --max-steps 10 --max-tokens 1800 --seed 42 \
    --output "$research_output/runs/$research_question.json" || true
done < <("$research_python" -c \
  'import json,sys; print("\n".join(json.load(open(sys.argv[1]))["question_ids"]))' \
  "$research_plan")

"$research_python" - "$research_output" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
runs = [json.loads(path.read_text()) for path in sorted((output / "runs").glob("*.json"))]
summary = {
    "run_count": len(runs),
    "submitted_count": sum(run["status"] == "submitted" for run in runs),
    "statuses": {run["question"]["id"]: run["status"] for run in runs},
    "valid_source_claims": sum(
        (run.get("checks") or {}).get("claims_with_valid_source_spans", 0) for run in runs
    ),
    "claim_count": sum((run.get("checks") or {}).get("claim_count", 0) for run in runs),
    "note": "Gold-evidence overlap and answer correctness still require scoring/review.",
}
(output / "automatic-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

echo "QASPER smoke artifacts saved in $research_output"

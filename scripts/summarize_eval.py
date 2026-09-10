"""Summarize explicit checkpoint transcripts without merging distinct runs.

Usage: python scripts/summarize_eval.py out/eval_*/1_base.json ...
Historical artifacts keep their original reward semantics and separate file identity.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

from src.eval.artifacts import comparison_key, outcome_reward

# Compatibility for existing imports; this does not rescore historical answers.
corrected_reward = outcome_reward


def load(paths: list[Path]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for path in paths:
        if path.name.endswith(".manifest.json"):
            continue
        rows = json.loads(path.read_text())
        if not isinstance(rows, list):
            raise ValueError(f"Expected a transcript list: {path}")
        for row in rows:
            groups[comparison_key(row, str(path.resolve()))].append(row)
    for key, rows in groups.items():
        ids = [r["question_id"] for r in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate question IDs in {key}; pass each transcript only once")
    return groups


def submitted(row: dict) -> bool:
    return bool((row.get("predicted_answer") or "").strip())


def paired_delta(a: list[dict], b: list[dict]) -> float:
    """Require a shared recorded protocol before reporting an outcome delta."""
    a_ids = {r["question_id"] for r in a}
    b_ids = {r["question_id"] for r in b}
    protocols = {r.get("comparison_id") for r in a + b}
    versions = {r.get("reward_version") for r in a + b}
    if a_ids != b_ids or len(protocols) != 1 or None in protocols or len(versions) != 1:
        raise ValueError("Question IDs, protocol, and reward version must match")
    if any(r.get("status") == "error" for r in a + b):
        raise ValueError("Execution errors must be resolved before comparing checkpoints")
    return mean(outcome_reward(r) for r in a) - mean(outcome_reward(r) for r in b)


def main() -> None:
    paths = [Path(a) for a in sys.argv[1:]] or sorted(Path("out").glob("run_*.json"))
    groups = load(paths)
    if not groups:
        raise SystemExit("No transcripts found")
    print("Outcome = recorded terminal outcome; legacy = saved reward minus saved bonus.")
    print("Different reward versions are not interchangeable. Each run stays separate.\n")
    labels: dict[str, list[list[dict]]] = defaultdict(list)
    for key, rows in sorted(groups.items()):
        labels[rows[0].get("run_label", rows[0]["policy"])].append(rows)
        print(f"{key}\n  checkpoint={rows[0].get('checkpoint_id') or '(base/unspecified)'}")
        print(
            f"  n={len(rows)} outcome={mean(outcome_reward(r) for r in rows):.3f}"
            f" answerF1={mean(r['answer_score'] for r in rows):.3f}"
            f" citP={mean(r['citation_precision'] for r in rows):.3f}"
            f" citR={mean(r['citation_recall'] for r in rows):.3f}"
            f" steps={mean(r['steps'] for r in rows):.1f}"
            f" submit={100 * mean(submitted(r) for r in rows):.0f}%"
            f" errors={sum(r.get('status') == 'error' for r in rows)}"
            f" version={rows[0].get('reward_version', 'legacy/unversioned')}"
        )
    for a, b in [("2_sft", "1_base"), ("3_grpo50", "2_sft"), ("4_grpo", "2_sft")]:
        if len(labels[a]) == len(labels[b]) == 1:
            try:
                delta = paired_delta(labels[a][0], labels[b][0])
                print(f"\n{a} - {b}: {delta:+.3f} outcome (descriptive; no significance claim)")
            except ValueError as error:
                print(f"\n{a} - {b}: comparison withheld: {error}")


if __name__ == "__main__":
    main()

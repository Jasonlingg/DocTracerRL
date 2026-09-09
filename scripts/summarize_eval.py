"""Summarize eval transcripts into the comparison table this project never produced.

Reads out/run_*.json transcripts, groups by policy, and reports the three deltas
that decide whether DocTracerRL's training actually did anything:

    sft - base      did supervised warm-start help?
    grpo50 - sft    did 50 steps of GRPO add anything on top?
    grpo - sft      same, for the other GRPO checkpoint

Usage:
    python scripts/summarize_eval.py                 # newest transcripts
    python scripts/summarize_eval.py out/run_*.json  # explicit files
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

# Documented dev baseline — RESULTS.md, Phase 3, 100 MuSiQue dev questions.
REFERENCE = {
    "claude_policy": 0.179,
    "context_stuffing": 0.176,
    "naive_rag": 0.147,
    "sparse_rag": 0.141,
    "single_shot": 0.089,
}


def load(paths: list[Path]) -> dict[str, list[dict]]:
    by_policy: dict[str, list[dict]] = defaultdict(list)
    for p in paths:
        try:
            rows = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"  ! skipping {p.name}: {e}")
            continue
        for r in rows:
            by_policy[r["policy"]].append(r)
    return by_policy


def corrected_reward(row: dict) -> float:
    """Reward with the efficiency bonus removed.

    The saved `reward` field is the UNCORRECTED value. RESULTS.md:159 documents
    that the efficiency bonus inverted the policy ranking — one-shot policies
    (1.6 avg steps) banked +0.168 against claude_policy's +0.022 (8.9 steps),
    masking a 4x answer-quality advantage — and it was removed from reward.py.
    Transcripts written before that change still carry it, so subtract it here
    or the comparison reproduces the exact distortion this project already fixed.

        corrected = 0.5*answer_F1 + 0.25*cit_P + 0.25*cit_R
    """
    return row["reward"] - row.get("efficiency_bonus", 0.0)


def submitted(row: dict) -> bool:
    """A row counts as submitted if it produced a non-empty answer.

    The dominant DocTracerRL failure was computing the answer and never issuing
    SUBMIT (69/100 in the Task 3.2 debrief), so this is tracked separately from
    reward — a policy can improve a lot on this axis before reward moves.
    """
    return bool((row.get("predicted_answer") or "").strip())


def main() -> None:
    args = [Path(a) for a in sys.argv[1:]]
    paths = args or sorted(Path("out").glob("run_*.json"))
    if not paths:
        print("No transcripts found in out/")
        raise SystemExit(1)

    print(f"Reading {len(paths)} transcript(s)")
    print("reward = efficiency bonus removed (RESULTS.md:159); raw = as saved\n")
    by_policy = load(paths)
    if not by_policy:
        print("No rows parsed.")
        raise SystemExit(1)

    hdr = (f"{'policy':<22}{'n':>5}{'reward':>9}{'raw':>8}{'answerF1':>10}"
           f"{'citP':>7}{'citR':>7}{'steps':>7}{'submit%':>9}")
    print(hdr)
    print("-" * len(hdr))

    scores: dict[str, float] = {}
    for name, rows in sorted(by_policy.items()):
        r = mean(corrected_reward(x) for x in rows)
        scores[name] = r
        print(
            f"{name:<22}{len(rows):>5}{r:>9.3f}"
            f"{mean(x['reward'] for x in rows):>8.3f}"
            f"{mean(x['answer_score'] for x in rows):>10.3f}"
            f"{mean(x['citation_precision'] for x in rows):>7.3f}"
            f"{mean(x['citation_recall'] for x in rows):>7.3f}"
            f"{mean(x['steps'] for x in rows):>7.1f}"
            f"{100 * mean(submitted(x) for x in rows):>8.0f}%"
        )

    print("\nreference (RESULTS.md Phase 3, 100 dev questions):")
    for k, v in REFERENCE.items():
        got = scores.get(k)
        delta = f"   (this run: {got:.3f}, {got - v:+.3f})" if got is not None else ""
        print(f"  {k:<20}{v:.3f}{delta}")

    print("\nthe three numbers that matter:")
    base = scores.get("qwen_base_policy")
    sft = scores.get("qwen_sft_policy")
    grpo = scores.get("grpo_policy")

    def delta(label: str, a: float | None, b: float | None) -> None:
        if a is None or b is None:
            print(f"  {label:<16} — missing ({'/'.join(n for n, v in [('a', a), ('b', b)] if v is None)})")
        else:
            verdict = "training helped" if a - b > 0.01 else "flat or worse"
            print(f"  {label:<16}{a - b:+.3f}   {verdict}")

    delta("sft - base", sft, base)
    delta("grpo - sft", grpo, sft)

    if grpo is not None and len(by_policy.get("grpo_policy", [])) > 0:
        print(
            "\nnote: both GRPO checkpoints report as 'grpo_policy'. Summarize their\n"
            "      transcripts separately (pass the specific run_*.json files) to tell\n"
            "      the 50-step checkpoint apart from the other one."
        )


if __name__ == "__main__":
    main()

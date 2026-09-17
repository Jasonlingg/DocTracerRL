"""Filter QASPER teacher trajectories and export Qwen chat-template SFT data.

Kept separate from collect_sft_data.py (which targets MuSiQue run_eval output)
because the formats and the filter rules differ: these trajectories are JSONL
with EvalResult field names, and the quality gates are answerability-specific.

Filters, in order:
  1. Drop episodes that never submitted (hit max_steps still searching).
  2. Drop actions containing English prose — the protocol is code-or-SUBMIT only.
  3. "sufficient" questions: drop when answer_score is below --min-answer-score,
     which is a real signal the answer missed or contradicted the gold answer.
  4. "insufficient" questions: answer_score is useless there (a good verbose
     abstention scores low against the terse gold "Unanswerable"), so instead
     require actual abstention language, and reject the hedge-then-guess pattern
     ("does not state X. However, based on ... <confident answer>"), which is an
     abstention in wording only.

Usage:
  python scripts/export_qasper_sft_data.py \
      --trajectories out/research/qasper-teacher-batch-v2/trajectories.jsonl \
      --out-dir data/sft/qasper-v2
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

import typer
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.policies.qwen_common import SYSTEM_PROMPT

console = Console()

ABSTAIN_WORDS = re.compile(
    r"unanswerable|does not (state|specify|explicitly|mention|provide|contain)|"
    r"cannot be (determined|established|answered)|no (explicit )?(information|mention)|"
    r"not (explicitly )?(stated|specified|provided)",
    re.I,
)
HEDGE_THEN_GUESS = re.compile(
    r"(unanswerable|does not|cannot be|no explicit|not stated|not specified)"
    r"[^.]*\.\s*(however|but|although|that said|based on)",
    re.I,
)
PROSE_START = re.compile(
    r"^(I('ll| will| can| need)|Let me|To |Here|Sure|First|Now|Next|Step|The |This )",
    re.I,
)


STOPWORDS = {"that", "this", "with", "from", "they", "have", "which", "their",
             "were", "used", "using", "about", "there", "these", "those"}

# Varied so 40-odd recovered abstentions don't teach one canned sentence. Each
# leads with "Unanswerable" (the gold token) then gives the useful reason.
ABSTENTION_TEMPLATES = [
    "Unanswerable — the paper does not explicitly state this.",
    "Unanswerable. Searching the paper turns up no explicit statement of this.",
    "Unanswerable — this is not specified anywhere in the paper's text.",
    "Unanswerable. The paper does not report this explicitly.",
]


def _has_prose(action: str) -> bool:
    first = action.strip().split("\n")[0].strip()
    return bool(first) and bool(PROSE_START.match(first))


def _gold_is_visible_in_evidence(row: dict, gold: str, threshold: float = 0.6) -> bool:
    """True when the gold answer's distinctive words appear in what the model actually saw.

    Appending a gold answer to a trajectory whose searches never surfaced it would
    teach the model to assert facts its own evidence does not support — precisely
    the hallucination we are trying to train out.
    """
    tokens = {t for t in re.findall(r"[a-z0-9]{4,}", gold.lower()) if t not in STOPWORDS}
    if not tokens:
        return False
    observed = " ".join(s["observation"] for s in row["trajectory"]).lower()
    return sum(1 for t in tokens if t in observed) / len(tokens) >= threshold


def _recover_with_gold(row: dict, question: dict, rng: random.Random) -> bool:
    """Replace a missing/incorrect conclusion with one grounded in QASPER's gold answer.

    Returns False when recovery would be unsafe. Mutates row's trajectory in place.
    """
    trajectory = row.get("trajectory") or []
    if not trajectory:
        return False

    if row.get("expected_answerability") == "insufficient":
        # Concluding "not stated" after fruitless searching is justified by the
        # absence of evidence, so this is always coherent.
        answer = rng.choice(ABSTENTION_TEMPLATES)
        submit = f"SUBMIT: {answer} CITATIONS: []"
    else:
        gold = question.get("answer") or ""
        if not gold or not _gold_is_visible_in_evidence(row, gold):
            return False
        citations = json.dumps(question.get("expected_citations", []))
        submit = f"SUBMIT: {gold} CITATIONS: {citations}"

    last = trajectory[-1]["action"].strip().upper()
    if last.startswith("SUBMIT:"):
        trajectory[-1] = {**trajectory[-1], "action": submit}
    else:
        trajectory.append({"step": len(trajectory) + 1, "action": submit,
                           "observation": "", "reward": 0.0, "done": True})
    return True


def _to_conversation(row: dict, max_chars: int) -> dict | None:
    trajectory = row.get("trajectory") or []
    if not trajectory:
        return None
    if not trajectory[-1]["action"].strip().upper().startswith("SUBMIT:"):
        return None

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {row['question']}"},
    ]
    for step in trajectory:
        action = step["action"].strip()
        if not action.upper().startswith("SUBMIT:") and _has_prose(action):
            return None
        messages.append({"role": "assistant", "content": action})
        if action.upper().startswith("SUBMIT:"):
            break
        messages.append({"role": "user", "content": step["observation"].strip()})

    if sum(len(m["content"]) for m in messages) > max_chars:
        return None
    return {"messages": messages}


def main(
    trajectories: list[Path] = typer.Option(..., "--trajectories"),
    out_dir: Path = typer.Option(..., "--out-dir"),
    benchmark: Path = typer.Option(None, "--benchmark",
        help="Required with --recover-with-gold; supplies QASPER's gold answers"),
    recover_with_gold: bool = typer.Option(False, "--recover-with-gold",
        help="Salvage trajectories with good searches but a missing/wrong conclusion "
             "by grounding the SUBMIT in QASPER's human-annotated gold answer"),
    min_answer_score: float = typer.Option(0.15),
    val_fraction: float = typer.Option(0.2),
    max_chars: int = typer.Option(32000, help="~8k tokens; longer conversations are dropped"),
    seed: int = typer.Option(42),
) -> None:
    rows: list[dict] = []
    for path in trajectories:
        text = path.read_text()
        if path.suffix == ".jsonl":
            rows += [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            rows += json.loads(text)
    console.print(f"Loaded {len(rows)} trajectories")

    questions = {}
    if benchmark:
        questions = {q["id"]: q for q in json.loads(benchmark.read_text())["questions"]}
    if recover_with_gold and not questions:
        raise typer.BadParameter("--recover-with-gold requires --benchmark")

    rng = random.Random(seed)
    kept, stats = [], {"no_submit": 0, "prose_or_long": 0, "low_score": 0,
                       "no_abstention": 0, "hedge_then_guess": 0,
                       "recovered": 0, "unsafe_to_recover": 0}
    for row in rows:
        answerability = row.get("expected_answerability")
        traj = row.get("trajectory") or []
        answer = row.get("predicted_answer", "")

        submitted = bool(traj) and traj[-1]["action"].strip().upper().startswith("SUBMIT:")
        if answerability == "insufficient":
            passes = submitted and bool(ABSTAIN_WORDS.search(answer)) \
                and not HEDGE_THEN_GUESS.search(answer)
        else:
            passes = submitted and row.get("answer_score", 0.0) >= min_answer_score

        if not passes:
            if recover_with_gold and traj and row["question_id"] in questions:
                if _recover_with_gold(row, questions[row["question_id"]], rng):
                    stats["recovered"] += 1
                else:
                    stats["unsafe_to_recover"] += 1
                    continue
            elif not submitted:
                stats["no_submit"] += 1
                continue
            elif answerability == "insufficient":
                key = "hedge_then_guess" if HEDGE_THEN_GUESS.search(answer) else "no_abstention"
                stats[key] += 1
                continue
            else:
                stats["low_score"] += 1
                continue

        conv = _to_conversation(row, max_chars)
        if conv is None:
            stats["prose_or_long"] += 1
            continue
        kept.append((row["question_id"], answerability, conv))

    console.print(f"[yellow]Dropped: {stats}[/yellow]")

    # Deduplicate by question_id (the v1 batch overlaps v2 on some questions).
    seen, deduped = set(), []
    for qid, answerability, conv in kept:
        if qid in seen:
            continue
        seen.add(qid)
        deduped.append((qid, answerability, conv))

    random.Random(seed).shuffle(deduped)
    n_val = max(1, int(len(deduped) * val_fraction))
    val, train = deduped[:n_val], deduped[n_val:]

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, split in (("train", train), ("val", val)):
        path = out_dir / f"{name}.jsonl"
        with path.open("w") as f:
            for _, _, conv in split:
                f.write(json.dumps(conv) + "\n")
        n_insuff = sum(1 for _, a, _ in split if a == "insufficient")
        console.print(f"[green]{name}: {len(split)} examples "
                      f"({n_insuff} insufficient / {len(split) - n_insuff} sufficient) -> {path}[/green]")

    turns = [sum(1 for m in c["messages"] if m["role"] == "assistant") for _, _, c in deduped]
    console.print(f"Avg assistant turns per example: {sum(turns)/len(turns):.1f}")


if __name__ == "__main__":
    typer.run(main)

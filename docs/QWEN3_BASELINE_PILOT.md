# Qwen3-8B baseline on the frozen AI-paper code-execution pilot

Recorded 2026-09-16. This is the first time base Qwen3-8B has been run against the
`data/research/code_exec_pilot_v1.json` pilot, using the code-execution protocol
(`search()`/`read()`/`passage()`/`extract()` in `src/env/`), not the paused JSON-action
protocol in `src/research/`. No training has happened yet — this is the untrained
starting point the next SFT/GRPO attempt needs to beat.

## Setup

- **Model:** `Qwen/Qwen3-8B`, untrained, loaded directly (no adapter)
- **Corpus:** the frozen `out/research/starter-2026-09-12` snapshot, 6 reserved papers
  (`arxiv_2005_11401v4`, `arxiv_2210_03629v3`, `arxiv_2310_11511v1`, `arxiv_2401_15884v3`,
  `arxiv_2403_14403v2`, `arxiv_2503_09516v5`), corpus hash
  `ca28990a801741357e84438b36685c8f521817d941b203bbbd8735aba884d2aa`
- **Questions:** `data/research/code_exec_pilot_v1.json`, 10 questions
- **Decoding:** greedy (temperature 0), seed 42, 1,024 max new tokens per turn,
  max 10 steps
- **Hardware:** 1x NVIDIA L4, RunPod EU-RO-1, on a persistent network volume
  (`rlm-explorer-persistent`) so the environment and model cache survive pod restarts

## A bug had to be fixed first

The first attempt at this run produced garbage: every action was a `<think>...</think>`
reasoning block, never actual code. Root cause — `src/policies/qwen_common.py` builds
each turn with `tokenizer.apply_chat_template(...)` and never passed `enable_thinking=False`.
Qwen3 has a native "think before answering" mode that Qwen2.5 (used everywhere else in
this project) doesn't have. Combined with the unchanged `max_new_tokens=1024` default,
the model routinely burned its entire token budget mid-thought and never reached the
code-writing part of its turn — one question timed out after 10 steps of nothing but
reasoning text, reward 0.000.

**Fix**, in `src/policies/qwen_common.py`:
- Added `enable_thinking=False` to `apply_chat_template(...)`, matching the same setting
  already used by the JSON-protocol `EndpointPolicy` elsewhere in this codebase.
- Kept the effective 1,024-token generation limit used by the concrete Qwen policies and
  centralized it as `DEFAULT_MAX_TOKENS`. The earlier draft incorrectly claimed this run used
  1,536 tokens; changing the limit after the run would make the reproduction command a different
  experiment.

Effect was immediate: actions dropped from 1000+ character reasoning blocks to 50-90
character direct tool calls, and per-question wall time dropped from ~167s to ~30-50s
(roughly 3-4x faster, since no tokens are wasted on an unused reasoning pass).

## Result

| Metric | Value |
|---|---:|
| Avg outcome reward | **0.445** |
| Avg answer accuracy | 0.331 |
| Avg citation precision | 0.850 |
| Avg citation recall | 0.950 |
| Avg steps | 8.9 |
| Avg wall time / question | 48.3s |

For comparison, the same night's Qwen2.5-7B numbers on the (different, MuSiQue) 50-question
dev set: base 0.158, SFT 0.176 (see `RESULTS.md` Phase 5). Untrained Qwen3-8B on this
AI-paper pilot already scores well above either — expected, given it's a newer model
generation, but now measured rather than assumed.

Per-question breakdown:

| Question | Reward | Answer | Cit. Precision | Cit. Recall | Steps |
|---|---:|---:|---:|---:|---:|
| paper_01 | 0.526 | 0.41 | 1.00 | 1.00 | 10 |
| paper_02 | 0.504 | 0.38 | 1.00 | 1.00 | 10 |
| paper_03 | 0.459 | 0.32 | 1.00 | 1.00 | 10 |
| paper_04 | 0.474 | 0.34 | 1.00 | 1.00 | 10 |
| paper_05 | 0.463 | 0.33 | 1.00 | 1.00 | 10 |
| paper_06 | 0.469 | 0.34 | 1.00 | 1.00 | 10 |
| paper_07 | 0.491 | 0.36 | 1.00 | 1.00 | 3 |
| paper_08 | 0.290 | 0.24 | **0.00** | 1.00 | 10 |
| paper_09 | 0.346 | 0.31 | 0.50 | 0.50 | 10 |
| paper_10 | 0.424 | 0.28 | 1.00 | 1.00 | 6 |

## What's actually going wrong — three specific, distinct failure modes

Retrieval is not the bottleneck. 7 of 10 questions hit perfect citation precision and
recall — the model reliably finds and cites the right source. The gap is entirely in what
it does with that source. Three separate, non-overlapping failure patterns, each with a
concrete example from this run:

### 1. Fails abstention entirely (`paper_08`) — the most important one

> **Question:** "What work published after March 2025 proves the best current method for
> evidence-grounded AI-paper research? Use only this snapshot and clearly state what
> cannot be established."
>
> **Expected:** *"This snapshot cannot establish the best current method. It is a selected
> historical collection rather than an exhaustive post-March-2025 search. A bounded, dated
> literature search and common evaluation would be needed before making a current-best claim."*
>
> **Predicted:** *"The best current method for evidence-grounded AI-paper research
> published after March 2025 is the Adaptive-RAG method described in [...]. This method
> offers a robust middle ground [...]"*

This question has zero required documents by design — the correct behavior is refusing
to answer. Qwen3-8B instead confidently named a real, correctly-cited paper as the
answer to a claim that paper doesn't establish. This is the citation-precision-vs-recall
split visible in the table (`citP=0.00, citR=1.00`): it found a real, relevant document
and then used it to support a claim that document doesn't back. A valid citation licensed
an invalid claim.

### 2. Drops the question's explicit epistemic instruction (`paper_01`)

> **Question:** "Why does Search-R1 exclude retrieved passage tokens from its
> policy-gradient loss, and what experiment in the paper supports that choice? **Keep the
> paper's finding separate from advice for our agent.**"
>
> **Expected:** *"[...] Testing the same choice in this project is an inference, not a
> result established for this agent."*
>
> **Predicted:** *"Search-R1 excludes retrieved passage tokens [...] to prevent
> unintended optimization effects and ensure stable training. The experiment [...] is the
> comparison of training dynamics with and without retrieved token loss masking [...]"*

The factual explanation is correct and well-cited. But the question's second half — keep
the paper's finding separate from advice for our own agent — is never addressed at all.
Same pattern recurs across most of the 7 "good citation, capped answer score" questions:
the model answers the "what/why," and silently skips the instruction about how to frame
that answer.

### 3. Cites the right source, misdescribes its content (`paper_03`)

> **Question:** "What actions does Corrective RAG trigger for correct, incorrect, and
> ambiguous retrieval results?"
>
> **Expected:** *"[...] refines relevant documents when retrieval is judged correct,
> replaces incorrect retrieval with web search, and combines refinement with web search
> when results are ambiguous."*
>
> **Predicted:** *"**Correct retrieval results**: The system uses internal knowledge for
> generation [...] **Incorrect retrieval results**: External knowledge is integrated to
> correct the error [...]"*

Citation precision and recall are both 1.00 — it cited the right paper. But "correct
retrieval → refine documents" became "correct retrieval → use internal knowledge," a
real factual swap, not a phrasing difference. A valid citation does not guarantee the
paraphrase built on top of it is accurate.

## External validation: QASPER test split

The three failures above came from a pilot we built ourselves — a real risk, since
hand-designed questions can unconsciously confirm what we already expect. To check that,
the same base Qwen3-8B, same code-execution protocol, was run against 20 real questions
from QASPER's actual test split (`scripts/setup_qasper_code_exec.py`, real annotators,
real papers, independent of this project). See `docs/QASPER_PILOT.md` for the dataset
background.

One conversion bug had to be fixed first: QASPER's raw questions use bare pronouns
("they", "this paper") that only resolve given the specific paper the annotator was
looking at. The first version of the converter used that raw text directly over a
416-paper corpus, making questions unanswerable by construction. Fixed by using the
paper-scoped question text QASPER's own converter already produces
(`src/research/qasper.py::_to_code_exec_question`), matching QASPER's documented design:
it tests within-paper evidence reading, not paper discovery.

| Metric | QASPER (external, n=20) | Our pilot (n=10) |
|---|---:|---:|
| Avg outcome reward | 0.355 | 0.445 |
| Avg answer accuracy | 0.224 | 0.331 |
| Avg citation precision | 0.800 | 0.850 |
| Avg citation recall | 0.950 | 0.950 |

Lower across the board on the benchmark we didn't design ourselves — the expected
direction if our own pilot was even slightly generous. Citation recall held steady, so
retrieval generalizes; answer quality and precision both dropped, so answer-synthesis is
the weaker half either way.

Two of the three original failure modes recurred, plus two new ones specific to QASPER's
citation-level question style:

**Abstention failure, confirmed independently.** `qasper_test_3e4e415e...` is a
QASPER-native unanswerable question — real annotators marked it "insufficient," we did
not design it. *"What language is the model tested on?"* — the paper never says.
Qwen3-8B answered **"English"**, a plausible-default guess, and cited the correct paper
to back the claim anyway. Same shape of failure as `paper_08` in our own pilot, now seen
on a dataset we had no hand in writing — this is the strongest single finding across both
runs.

**Answers the general topic instead of the specific fact.** Asked "what dataset was
used," it answered "EmoInt-2017" — the shared-task name lifted straight from the paper's
own title, not the actual dataset composition the paper describes. Asked "what rule-based
models were evaluated," it answered "rule-based, CRF, and neural network-based models" —
a category description, when the real answer is a specific citation ("Rashel et al.").
Both episodes ended early (3 and 6 of 10 steps) — it settled on a plausible-sounding
answer instead of searching further for the precise fact.

**Submission format breaks down under the full step budget.** One question used all 10
steps and then submitted an empty answer with zero citations — a malformed final SUBMIT
line, not a knowledge gap. Worth fixing as harness/prompt robustness, independent of the
other four.

## Decision rule and next step

Five specific, buildable training targets now, confirmed across two independent
datasets for the most important one:

1. Recognize when to abstain instead of guessing a plausible default (confirmed twice)
2. Answer the specific fact asked, not the general topic (QASPER)
3. Produce a well-formed SUBMIT even after a long, unproductive search (QASPER)
4. Follow every explicit framing instruction in the question, not just the factual half
   (our pilot)
5. Verify a paraphrase against the source before submitting it — a valid citation is not
   proof the paraphrase is accurate (our pilot)

Per the standing project decision rule (`docs/CODE_EXECUTION_SECOND_BRAIN.md`), the next
step is targeted trajectory generation — demonstrations that specifically exercise these
five behaviors in the code-execution protocol — followed by re-running both this pilot
and the QASPER benchmark to check for improvement. Not a broad SFT dump: the failure
modes are narrow enough to target directly, and a generic data pile risks fixing none of
them cleanly.

## Reproduce

```bash
BASE_MODEL_PATH=Qwen/Qwen3-8B python scripts/run_eval.py \
  --policy qwen_base_policy \
  --questions data/research/code_exec_pilot_v1.json \
  --corpus out/research/starter-2026-09-12/corpus \
  --max-steps 10 --seed 42 --require-evidence --no-vector-index \
  --run-label qwen3_base --output out/research/qwen3-baseline/base.json
```

Requires the `enable_thinking=False` fix in `src/policies/qwen_common.py` — without it,
this reproduces the earlier confounded run (avg reward 0.251, non-deterministic step
counts) instead of the result above.

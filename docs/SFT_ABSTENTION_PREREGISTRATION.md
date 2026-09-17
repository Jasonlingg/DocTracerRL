# Pre-registration: does QASPER SFT fix Qwen3-8B's abstention failure?

Written 2026-09-17, **while the SFT run was still training and before any result
existed**. The point of fixing the criteria now is that "was this an improvement?"
is very easy to answer generously after seeing the numbers.

## The claim being tested

Base Qwen3-8B confidently answers questions whose source document does not contain
the answer. LoRA SFT on 189 gold-grounded QASPER-train trajectories (101 of them
abstention examples) should make it decline instead.

## Why reward is not the headline metric

Outcome reward scores answer token-overlap F1 against gold, and gold for an
unanswerable question is the terse string `"Unanswerable"`. A good verbose
abstention scores *low* against that. A model whose abstention genuinely improved
can therefore show **falling** reward. Reward is reported as a diagnostic only;
it cannot settle this question in either direction.

## Metrics (defined in `src/eval/abstention.py`, tested before results existed)

- **`abstention_recall`** — of the 20 genuinely unanswerable questions, the
  fraction declined. **Primary metric.**
- **`false_abstention_rate`** — of the 20 genuinely answerable questions, the
  fraction wrongly declined. **Guardrail.** Training on a 43% abstention mix
  risks "conservative shift", where the model simply refuses more often. A model
  that refuses everything scores a perfect abstention_recall, so the primary
  metric is meaningless without this one.

An answer counts as an abstention only if it declines *and* does not immediately
undercut itself ("does not state X. However, based on the text, X is Y") — that
pattern is an answer wearing an abstention's clothes and is scored as answering.

## Evaluation protocol

- Set: `out/research/qasper-test-abstention-v1` — 40 questions, 20 unanswerable /
  20 answerable, from QASPER's **test** split. Verified zero question-ID and zero
  paper overlap with any training data.
- Base and SFT are both run under the **same** current system prompt. The old
  0.445 baseline was measured under the pre-verification-instruction prompt and
  is **not** a valid comparison point; base must be re-measured.
- Identical decoding, seed, and max_steps for both.

## Success criteria, fixed in advance

- **Success** — `abstention_recall` improves over base at p < 0.05 (Fisher exact,
  two-tailed), **and** `false_abstention_rate` rises by no more than 0.15
  absolute. Both conditions required: recall bought by refusing everything is not
  a fix.
- **Partial** — recall improves directionally but p ≥ 0.05, or it improves while
  breaching the false-abstention guardrail. Reportable as suggestive, not as a
  result.
- **Failure** — no improvement, or a regression.

With n=20 unanswerable, a shift like 4/20 → 12/20 reaches p ≈ 0.02 and is
detectable. The previous eval set had 5 unanswerable questions total, where even
a complete 0/5 → 5/5 fix only reached p = 0.008 and anything partial was
indistinguishable from luck — which is why this set was built.

## Known limits

- n=20 per cell still only resolves fairly large effects. A real but modest
  improvement will land in "Partial" and should be described that way.
- Abstention detection is lexical. It will miss abstentions phrased in
  unanticipated ways, which biases *against* showing improvement rather than
  toward it.
- One eval set, one domain (NLP papers). Nothing here establishes that the
  behaviour transfers to the Obsidian-vault domain.

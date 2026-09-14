# QASPER conversion pilot

## Purpose and decision rule

QASPER supplies human questions, answers, answerability labels, and paragraph evidence over NLP
papers. In this project it teaches the knowledge-query worker to inspect a known paper and return
supported evidence. It does not supply open-ended paper discovery, multi-paper synthesis, or tool
trajectories.

Hypothesis: the QASPER train split can be converted into the existing frozen research corpus and
question schemas without losing its split identity, answer types, unanswerable cases, or exact
text evidence. Expected signal: every selected text-evidence paragraph resolves exactly in the
snapshot and no validation/test paper enters the train snapshot. Accept the conversion if those
checks pass; exclude figure/table tasks and any evidence that cannot be mapped exactly rather than
inventing replacements.

## Frozen source and conversion

- Dataset: `allenai/qasper`, configuration `qasper`
- Dataset revision: `13b496d2a5359329b110e3419628de3cf791843b`
- Official source split: train
- Converter: `qasper-research-v2`
- Selection: 40 eligible questions, seed 42
- Local snapshot: `out/research/qasper-train-pilot-v2` (ignored by Git)

The task prompt names the known paper and its `doc_id`; `source_question` preserves QASPER's
original wording. This distinction matters because questions such as “How big is their dataset?”
make sense with a paper in view but cannot identify one paper among hundreds by themselves.

## Conversion result

The pinned train split contains 888 papers and 2,593 questions. The converter found 2,121 questions
eligible for the strict text-only path, then selected 40 deterministically. The selected set has:

- 33 answerable and 7 unanswerable questions;
- 24 extractive, 6 abstractive, 3 boolean, and 7 unanswerable answer annotations; and
- 46 distinct gold evidence paragraphs, all mapped to exact character spans with no ambiguous
  duplicate match.

Across the full source split, 320 questions mention figure/table evidence and 164 contain at least
one text-evidence string that does not exactly match the constructed corpus. These groups can
overlap. Both are excluded from this pilot and recorded in `manifest.json`; they are not silently
treated as clean text supervision.

The resulting v2 corpus hash is
`2647fb92032a2ac812adc6840fe8e27d9fb05cba0a7670263c144ac2ef6821a0`. Paper titles are part of
the searchable text, and the question record makes the known-paper task explicit.

## What the first retrieval diagnostic showed

Before the known-paper correction, the existing lexical scan searched all 888 papers using only
the original question. It found the target paper in the top five for 8 of 40 questions
(`target_document_recall = 0.20`) and took 245.45 seconds on the local CPU. Qwen was not involved.

The misses show two separate limitations:

1. QASPER assumes the target paper is known, so many original questions contain pronouns or generic
   wording that cannot serve as paper-discovery queries.
2. The current retriever rescans and retokenizes the full corpus for every query, which is too slow
   for a useful second brain at this size.

The 0.20 value must not be reported as QASPER answer performance or model performance. It is a
diagnostic from a mismatched open-corpus routing setup. QASPER remains useful for within-paper
evidence selection and abstention. Open-ended discovery needs separate search-oriented data and an
indexed ranker; connected multi-step behavior still needs MuSiQue/Hotpot-style examples and actual
tool trajectories.

## Next experiment

Generate a handful of known-paper QASPER trajectories with the current base Qwen worker and inspect
whether it uses the supplied document ID, reaches the gold paragraphs, answers correctly, and
abstains on unanswerable questions. Do this before SFT. In parallel, replace the scan-based lexical
retriever with a persisted lexical or hybrid index and evaluate paper discovery on a dataset built
for that task, such as the reserved LitSearch benchmark.

The fixed five-question smoke plan is `data/research/qasper_smoke_v1.json`. It covers two
extractive questions, one abstractive question, one boolean question, and one unanswerable
question. Run it against an existing Qwen server with:

```bash
export RESEARCH_MODEL_REVISION=b968826d9c46dd6066d109eabc6255188de91218
export RESEARCH_SERVER_HARDWARE='1x A40; BF16; vLLM 0.10.2; context 16384; eager'
bash scripts/run_qasper_smoke.sh
```

## Base-Qwen smoke result

The fixed smoke ran on September 14, 2026 with base `Qwen/Qwen3-8B` revision
`b968826d9c46dd6066d109eabc6255188de91218`: one NVIDIA A40, BF16, vLLM 0.10.2,
Transformers 4.57.6, 16,384-token context, eager execution, temperature 0, seed 42, and ten
actions per question. The corpus hash was
`2647fb92032a2ac812adc6840fe8e27d9fb05cba0a7670263c144ac2ef6821a0`. The run artifacts and
serving freeze are under `out/research/qasper-base-qwen3-8b-smoke-v1/`. The RunPod was stopped
after the artifacts were copied back; all four account pods were then reported stopped.

Automatic results:

| Metric | Result |
| --- | ---: |
| Valid submission rate | 80% (4/5) |
| Claims with valid source spans | 100% (4/4) |
| Answerable questions reaching a gold evidence paragraph | 75% (3/4) |
| Unanswerable questions correctly abstained | 0% (0/1) |
| Questions with repeated identical tool actions | 2/5 |

The run met the two numeric continuation thresholds: four submissions and three answerable gold
evidence hits. It did not meet the complete expected signal because the unanswerable case produced
an unsupported answer.

A Codex semantic review, which is transparent but not independent human review, found that the
three completed answerable responses matched their QASPER reference answers: the Ranker/Reasoner
cooperation, the boolean “No” answer about use only with incomplete data, and the seven comparison
methods. The fourth answerable run read consecutive 1,600-character chunks from the beginning,
never reached the two gold paragraphs in the later `Data` section, and then hit the 1,800-token
generation limit. The unanswerable run cited a paragraph about correlation with human judgments
but claimed what questions the judges were asked; the passage did not state those questions, and
the model's own limitation admitted this gap.

This is a useful positive smoke result for QASPER ingestion and known-paper navigation, alongside
two concrete training targets. QASPER trajectory generation can proceed in a small batch, provided
it includes examples that select a relevant section before reading sequentially, stop after enough
evidence, and abstain when related text does not answer the question. The result does not justify
SFT yet, and it does not measure open-ended paper discovery.

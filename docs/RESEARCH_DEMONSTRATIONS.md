# First research demonstration batch

The first batch contains six assistant-authored demonstrations over the separate development
corpus. It contains 32 assistant action turns, 13 factual claims, two comparisons, a failed-search
recovery, and one abstention. These are a small dataset-design trial. They are not enough to
establish a training recipe or a performance improvement.

The tracked source is `data/research/demonstrations_v1.json`. Codex inspected the cited passages,
wrote the actions and answers, and performed a self-review of each claim and answer. No teacher
endpoint was called. An exact revision for interactive authoring is unavailable and is recorded
as such. No Qwen generation, training, or independent human review occurred in this step.

## What the examples teach

| ID | Subject | Behavior |
| --- | --- | --- |
| research_demo_001 | WebGPT | Explain a workflow completely, then separate a proposed experiment from paper findings |
| research_demo_002 | Toolformer | Recover from an empty search; identify a method's limitations |
| research_demo_003 | FActScore and ALCE | Compare two papers; distinguish factual precision, citation recall, and answer completeness |
| research_demo_004 | HAGRID and ALCE | Notice differences in annotation rules across sources |
| research_demo_005 | FreshLLMs | Report a measured tradeoff without assuming it transfers to Qwen |
| research_demo_006 | Missing internal study | Inspect the collection and decline to invent a retention measurement |

The empty search in example 002 is an authored typo, not a naturally sampled model failure.
The internal study name in example 006 is a synthetic missing-source scenario. More realistic
failure diversity will be needed before scaling training.

## Rebuild and inspect

```bash
python3 scripts/research_demonstrations.py \
  --snapshot out/research/ai-agents-development-v1-20260912 \
  --benchmark data/research/benchmark_pilot_v1.json \
  --output out/research/development-demonstrations-v1
```

The output directory must be new. The first local build is already at that location. It contains:

- `runs/`: real search, metadata, and passage observations obtained by replaying the authored actions.
- `conversations.jsonl`: system/user/assistant messages in the runner's inference format.
- `review.json`: materialized quotes and explicit assistant self-review notes for every example.
- `manifest.json`: corpus, batch, exclusion and artifact hashes, counts, provenance, and pending gates.

Training should apply the Qwen chat template and compute loss only on assistant action tokens.
The user messages contain questions and actual tool observations; they are conditioning context.
The exporter does not implement a trainer or certify token masking. Metadata and review notes are
outside the messages and must stay out of the model input.

The builder checks snapshot integrity, schema validity, bounded multi-step execution, discovery
of paper IDs before their use, and observation of each cited span before submission. It rejects
any corpus containing reserved paper families, including other arXiv versions, and flags exact
or substantial lexical reuse of benchmark questions. Repeat `--benchmark` for every future
reserved evaluation manifest. These checks do not detect all semantic paraphrases or prove
entailment; the review notes address semantic support separately.

`agent_research_dev_15` is explicitly excluded from training because it closely repeats the
pilot's GPU-and-latency question. It remains in the old prompt file for historical diagnostics.
Do not use an older generator that ignores split and exclusion metadata to promote data.

## JSON generation experiment

`scripts/research.py ask --structured-output json_schema` now requests the complete action schema
from the serving endpoint. It constrains action names, argument fields and types, and the final
answer structure, including an empty claims list for abstention. The client checks returned JSON
against that schema and fails if the server ignores it. It does not silently repair malformed
answers or fall back to unconstrained generation.

The request uses `response_format` as documented for
[vLLM 0.10.2 structured outputs](https://docs.vllm.ai/en/v0.10.2/features/structured_outputs.html).
The mode and schema hash enter the policy configuration, so the scorer refuses to mix the new
mode with the original unconstrained pilot. The default remains `none` to reproduce that pilot.
Set `RESEARCH_STRUCTURED_OUTPUT=json_schema` when using `run_research_benchmark.sh`.

Hypothesis: constrained generation removes malformed final JSON while leaving evidence-choice
failures visible. Compare base Qwen under `none` and `json_schema` with identical questions,
corpus, prompt, seed, token limit and action budget. The first live check should verify that the
pinned vLLM server accepts this particular union schema and returns schema-valid actions.
Offline tests verify request construction and rejection behavior, not real GPU decoding.

Adopt the mode only after a live run produces no schema violations and review finds no new
answer-quality regression. Report formatting improvements as a harness effect. Keep the old run
and store every new run in its own directory. Constrained generation cannot guarantee valid
document IDs, correct offsets, semantic support, or completion before the token budget expires.

Duplicate-search handling is deliberately a separate proposed experiment. This change does not
alter search, add retries, increase token limits, or force early submission.

## Before scaling or training

Inspect the batch independently and resolve any disputed claims; then collect more diverse
development examples. Lock a larger final evaluation over new reserved papers, with a separate
development split for checkpoint selection. The eight-question pilot has now influenced our
engineering and remains a diagnostic, not a pristine final test.

Keep both base and trained models on the same chosen harness. Reserve harmless formatting and
result-order variations for a robustness test, and measure whole-answer usefulness and support
alongside failure and abstention rates. The manifest currently marks `training_ready: false`:
the batch is reproducible and mechanically checked, but the final benchmark and live harness
comparison are still outstanding.

## Verification

The real 20-paper snapshot replay passed for all six examples. The focused test suite passed
43 tests covering research execution, benchmark scoring, structured-output requests, malformed
responses, reserved paper versions, reused questions and unobserved citations. Ruff, the Git
whitespace check and the benchmark shell syntax check passed. No GPU was started for this work.

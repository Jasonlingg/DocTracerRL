# Qwen3-8B research-agent baseline pilot

This record covers the first complete run of the locked eight-question AI-paper research
benchmark. It is a pipeline pilot and failure diagnosis, not a headline model-quality result.
The benchmark is too small for a general capability claim, and its reserved papers and questions
must remain excluded from training.

## Experiment

Hypothesis: the untrained Qwen3-8B checkpoint can use the validated JSON research tools, but will
not reliably complete multi-paper comparisons, select evidence, or abstain when the frozen corpus
cannot answer a question.

Expected signal: complete submissions with claim-level evidence from the required papers, correct
abstention on the two insufficient-evidence questions, and no repeated or malformed tool actions.

Decision rule: preserve the run as the base checkpoint if all eight artifacts use one locked
configuration. Create targeted training examples only if semantic review exposes repeated,
teachable failures. Do not train until a larger held-out benchmark is locked and the examples pass
contamination review.

The run used:

- Model: `Qwen/Qwen3-8B`
- Model revision: `b968826d9c46dd6066d109eabc6255188de91218`
- Protocol: `research-tools-v2`
- Benchmark: `ai-paper-research-pilot-v1`, questions `pilot_01` through `pilot_08`
- Benchmark hash: `9cfd62dec5d8a8979b965eb1c7662ca84e87dd2e4e5806f7a1dc1c097f7d6695`
- Frozen corpus hash: `ca28990a801741357e84438b36685c8f521817d941b203bbbd8735aba884d2aa`
- Prompt hash: `e9bd3031ad2e97e4d6dacbe35a0c17031d437556649a4e70a1eab58e9624fc65`
- Policy configuration hash: `a249490b45d8bfa993240363c2237c7a59a56abee2cf349f52ded261f8b651e5`
- Decoding: thinking disabled, temperature 0, top-p 1, seed 42, 1,800 tokens per action
- Tool budget: 10 actions per question
- Server: one NVIDIA A40, 46 GiB usable memory, BF16, vLLM 0.10.2, 16,384-token
  context, eager execution, CUDA 12.8
- RunPod pod: `0ubfxj38xc9c56`, Secure Cloud in `EU-SE-1`
- Remote repository base: `6595060c0e34226dcd50a1d599747d994a18e3cf`
- Copied retrieval runtime SHA-256:
  `e64e99b0f4f1911d36b773d755b6538a0cd7d5c7620dfe14762b70ef51fd88be`
- Research reward: none; this is an inference benchmark

The pod used a persistent 50 GB volume mounted at `/workspace`. The complete run directory was
copied back before the pod was stopped. RunPod reported $1.76 in pod charges when checked after the
run; retained disk can continue to add small storage charges.

## Results

All eight run artifacts are present and share the locked configuration. Automatic checks report:

| Metric | Result |
| --- | ---: |
| Submission rate | 50.0% |
| Claims with structurally valid source spans | 100.0% |
| Required-document recall | 33.3% |
| Source-count requirement rate | 62.5% |
| Rejected or error step rate | 4.3% |
| Average trajectory length | 8.75 steps |

The 100% source-valid figure means that every submitted citation points to a real span in the
snapshot. It does not mean that the passage supports the claim.

Codex completed a transparent semantic audit using the benchmark grader notes and original paper
passages. This is not independent human validation. Under the strict rule that the complete
question must be answered or correctly declined, the audit reports:

| Metric | Result |
| --- | ---: |
| Submitted claims judged fully supported | 83.3% (5/6) |
| Submitted claims judged unsupported | 16.7% (1/6) |
| Questions handled as expected | 12.5% (1/8) |
| Recommendations without unsupported premises | 37.5% |
| Questions missing important available evidence | 87.5% |
| Average usefulness | 0.5/2 |

The claim-support rate is conditional on the six claims the model managed to submit. It therefore
looks much stronger than question-level performance and must not be reported alone.

## Failure diagnosis

- `pilot_01` cited the Search-R1 masking rationale but omitted the masking ablation explicitly
  requested by the question.
- `pilot_02` was the only complete, useful answer.
- `pilot_03` described Correct, Incorrect, and Ambiguous thresholds but omitted the corresponding
  refine, web-search, and combined actions.
- `pilot_04` repeated one query nine times and cited Corrective RAG for a claim about Self-RAG.
- `pilot_05` formed the correct abstention in its raw output, but added a disallowed top-level
  field, so the final action was rejected.
- `pilot_06` inspected both required papers but generated beyond the per-action output limit and
  never submitted.
- `pilot_07` found both required papers, then appended a Markdown code fence after its JSON. The
  action was rejected.
- `pilot_08` repeated one weak query eight times and tried to support a best-current-method claim
  with an acknowledgments passage instead of abstaining.

The primary categories are four evidence-selection failures, three output-protocol failures, and
one successful answer. Retrieval alone is not the main bottleneck: several failed runs found the
required papers but did not convert them into a valid, complete answer.

## Decision

Do not rerun this baseline under the same configuration. The output-limit failure is part of the
locked base behavior, and silently replacing only that question would mix configurations.

The pipeline is capable of exposing a meaningful improvement, and the base model has repeated,
teachable weaknesses. Before renting a training GPU:

1. Test a small inference-side fix for duplicate searches and strict JSON submission. This checks
   whether the protocol failures can be removed without changing model weights.
2. Lock a larger 50–75-question benchmark over new reserved papers.
3. Create and review development trajectories that teach query reformulation, evidence-to-claim
   matching, complete comparison answers, and explicit abstention.
4. Train the Qwen3-8B adapter with Unsloth QLoRA only after contamination checks pass.
5. Compare the base and trained checkpoint with identical tools, corpus, decoding, and budgets.

The saved local artifacts are under
`out/research/pilot-base-qwen3-8b-20260913/`. The reviewed score is
`assistant-reviewed-score.json`; `assistant-review.json` contains the per-question judgments.

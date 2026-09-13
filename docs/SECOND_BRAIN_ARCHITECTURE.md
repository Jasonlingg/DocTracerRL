# AI-paper second brain architecture

For a longer plain-language walkthrough with a complete example, see
[how the AI research assistant works](HOW_THE_AI_RESEARCH_ASSISTANT_WORKS.md).

The product goal is a private, inspectable research assistant. A small Qwen agent searches a
curated paper library and produces an evidence packet. A larger model explains that packet in the
user's preferred level of detail. Neither model is allowed to turn a real citation into a guarantee
that the citation supports its prose.

```text
PDFs / arXiv versions / notes
            ↓
versioned ingestion and searchable index
            ↓
Qwen retrieval agent → exact passages and candidate claims
            ↓
validated evidence packet (E1, E2, ...)
            ↓
larger explainer → clear answer with evidence IDs
            ↓
reference validation + human semantic review
```

## Current implementation

The current corpus is a six-paper, versioned arXiv HTML snapshot. Search is a deterministic lexical
baseline. Qwen can request four bounded read-only actions: list papers, search, inspect metadata,
and read a passage. The runner materializes exact passages from submitted source offsets.

`src/research/explainer.py` creates a bounded evidence packet from a submitted Qwen run. Invalid
source spans are not forwarded. Candidate claims are explicitly marked untrusted, because a valid
span establishes provenance but not entailment. A larger OpenAI-compatible endpoint can receive
this packet and must cite only its evidence IDs. The resulting explanation and its complete input
are saved together for review.

This design limits what leaves the second brain: the larger endpoint receives the user's question,
the selected passages, and Qwen's candidate interpretation. It does not receive the full corpus.
For material that cannot leave the machine, use a local explainer as well.

## What scaling requires

The current implementation is not a 10,000-PDF system. Reaching that scale requires:

1. Local PDF and note ingestion with stable document versions, page references, deduplication,
   parsing diagnostics, and explicit table/figure coverage.
2. A persistent hybrid index combining lexical and embedding retrieval instead of scanning every
   JSON document for each query.
3. Incremental updates that preserve old benchmark snapshots while refreshing the live library.
4. Access controls and collection filters for private or project-specific sources.
5. Retrieval evaluation using known relevant passages, followed by claim-support and usefulness
   review of the final explanations.

The 20 existing prompts are unreviewed development questions, not golden trajectories. The locked
eight-question pilot is evaluation data, and its six papers are excluded from future training.
Training examples must come from different papers. Exact references reduce fabricated citations;
they do not make hallucination impossible.

## Training boundary

First measure base Qwen on the locked pilot using the structured action protocol. Then build a
disjoint training corpus and collect reviewed search/read/evidence trajectories. Use Unsloth QLoRA
on RunPod only after those artifacts pass contamination and quality checks. Compare the trained
adapter with the same base checkpoint, tools, question IDs, corpus, decoding settings, and budgets.

The small-model result and the complete-system result are different measurements:

- Qwen is evaluated on finding relevant passages, avoiding unsupported candidate claims, handling
  insufficient evidence, and using tools efficiently.
- The larger explainer is evaluated on claim support, clarity, limitations, and project usefulness
  using the exact same evidence packet for every explainer comparison.

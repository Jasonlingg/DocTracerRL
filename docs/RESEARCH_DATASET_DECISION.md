# Dataset strategy for the AI-paper knowledge query agent

## Decision

MuSiQue is **not the best single training dataset** for the AI-paper research assistant. It is a
strong source of connected multi-hop problems, and it remains useful as a general reasoning control,
but its Wikipedia questions do not teach the model how to interpret research claims, distinguish
experimental evidence from author speculation, find evidence throughout a long paper, or cite a
claim precisely.

The best mature off-the-shelf starting point for the paper-reading part of the system is
**QASPER**. Its questions were written by readers of NLP papers, its answers were written by a
separate group of NLP practitioners, and it includes supporting evidence and unanswerable cases.
The official training split contains 2,593 questions over 888 papers and is licensed CC BY 4.0.[^1]
QASPER still does not provide multi-turn search-tool trajectories or multi-paper literature reviews,
so it cannot be the whole training set.

The recommended approach is a **three-part training mixture**:

1. QASPER training examples for scientific-paper reading, evidence selection, answerability, and
   grounded answers.
2. MuSiQue or HotpotQA examples converted to the same JSON tools for connected multi-step search
   and query refinement.
3. Reviewed teacher trajectories over separate AI-paper collections for the exact
   `search_papers` → `paper`/`passage` → `submit` behavior, multi-paper comparison, citation, and
   abstention that the product needs.

MuSiQue should therefore be retained, but its role changes from “the dataset” to “one component and
one control.” The product pivot was sound. The incomplete part of the pivot was treating a small set
of paper demonstrations as if it could replace the scale and mechanical supervision that MuSiQue
provided.

## The capability being trained

The small Qwen model is a knowledge-query worker used by a larger agent. Its useful output is an
evidence packet, not a polished essay. Given a research question, it must:

1. identify the concepts and constraints that matter;
2. issue one or more searches against the paper library;
3. inspect promising papers and passages;
4. reformulate the query when the first retrieval is incomplete;
5. select evidence that actually supports each claim;
6. stop when the evidence is sufficient, or state that the corpus is insufficient; and
7. return structured claims, paper identifiers, and exact source locations to the larger model.

This separation has close published precedent. The s3 system trains a 7B searcher while keeping the
answer generator frozen. The searcher iteratively retrieves documents, selects useful evidence,
decides whether to continue, and passes the resulting context to the generator. The paper reports
that 2,400 selected training examples outperformed baselines trained on much larger sets across
general and medical QA, including transfer to domains absent from training.[^2] This result does not
guarantee the same outcome here, but it supports the architecture and shows that data usefulness can
matter more than raw volume.

The data must therefore supervise several distinct skills. No available dataset covers all of them:

| Skill | What must be learned | Best available supervision |
|---|---|---|
| Connected search | A later query depends on an earlier finding | MuSiQue; HotpotQA |
| Paper discovery | Match a research need to relevant papers | LitSearch for evaluation; separate LitSearch-style training data |
| Within-paper reading | Find evidence across sections of a long paper | QASPER train |
| Scientific reasoning | Combine technical rationales into an answer | QASPER; QASA as a reserved benchmark |
| Evidence support | Tie a claim to passages that support or refute it | QASPER; SciFact as auxiliary data |
| Abstention | Refuse claims absent from the available sources | QASPER unanswerable; MuSiQue-Full; reviewed local examples |
| Citation generation | Attach sufficiently complete citations to claims | Reviewed local trajectories; LongCite as optional auxiliary data |
| Exact tool protocol | Produce valid actions and recover from weak search results | Reviewed trajectories generated in this repository |
| Multi-paper synthesis | Compare findings without erasing disagreements | Reviewed local trajectories; reserved LitTraceQA evaluation |

## Why MuSiQue remains valuable

MuSiQue was designed specifically to reduce shortcuts in multi-hop QA. It contains approximately
25,000 answerable questions requiring two to four hops, with annotated decompositions and supporting
paragraphs. Its construction filters questions whose later step can be solved without the earlier
bridge, and the paper reports a 30-point F1 drop for a disconnected single-hop model. MuSiQue-Full
adds paired unanswerable questions and grows to approximately 50,000 examples.[^3]

Those properties are unusually useful for a search agent. The decomposition tells us which facts
must be found and in what dependency structure. Supporting paragraphs provide a mechanically
checkable target. Distractors discourage the policy from treating the first plausible passage as
the answer. The CC BY 4.0 license also makes the dataset practical to adapt.[^4]

MuSiQue is also harder to shortcut than HotpotQA. HotpotQA is much larger, with 113,000 questions
and sentence-level supporting facts, but the MuSiQue authors show why some HotpotQA questions can
be answered from the final clue without resolving the intended first hop.[^3] That makes MuSiQue a
good diagnostic for whether a policy performs connected search rather than merely producing a
correct short answer.

Recent agentic-search work nevertheless points to a different role for MuSiQue. Search-R1 trains on
Natural Questions and HotpotQA, then evaluates on seven datasets that include MuSiQue as an unseen
multi-hop test.[^5] s3 follows the same NQ-plus-HotpotQA training setup and evaluates transfer to
MuSiQue and five medical QA datasets.[^2] This protocol makes MuSiQue valuable as a transfer test,
not only as training material.

For this repository, there are two valid uses:

- The existing Qwen2.5 MuSiQue checkpoints should still receive the outstanding common-split
  evaluation. That closes the original experiment and determines whether the earlier GRPO work
  improved anything.
- A new Qwen3 paper-agent experiment may use a subset of MuSiQue training examples converted to
  the paper agent's read-only JSON protocol. Any MuSiQue examples used for training must be kept
  disjoint from the MuSiQue transfer evaluation.

MuSiQue's limitations are equally clear. Its evidence comes from short Wikipedia passages, its
answers are usually brief entities or values, and its questions do not require understanding
methods, baselines, experimental design, limitations, or the difference between reported and
inferred claims. It supplies supporting paragraphs but not claim-level citations in a research
answer. It also supplies decompositions rather than observed search-tool trajectories. Training on
MuSiQue alone can therefore teach the model to solve the harness while leaving the product task
largely unmeasured.

## Why QASPER should be the domain core

QASPER is the closest mature dataset to the daily “help me understand this AI paper” problem. It
contains 5,049 information-seeking questions over 1,585 NLP papers. A question writer saw the title
and abstract and asked something they wanted to learn from the full paper; a different practitioner
then answered the question and selected evidence.[^6] This collection process is closer to a real
reader's information need than datasets in which a question is constructed after the annotator has
already seen the answer sentence.

The supervision is unusually well aligned with our failure modes:

- 51.8% of answers are extractive, 24.2% are abstractive, 13.9% are yes/no, and 10.2% are
  unanswerable.[^7]
- Among answerable questions with text evidence, 55.5% require evidence from multiple paragraphs.
- Evidence may appear anywhere in a paper; no single common section contains most evidence.
- The official split is paper-disjoint, reducing direct paper leakage between training and test.
- The Hugging Face release includes full text, questions, free-form or extractive answers,
  paragraph evidence, highlighted evidence, and figure/table metadata.[^1]

These are the behaviors the current agent needs: search beyond the abstract, combine evidence from
multiple parts of a paper, and recognize that some plausible questions are not answered by the
source. The original QASPER baseline lagged a human lower bound by at least 27 F1 points on full
paper QA and by 32 F1 points on evidence selection, so the task is not mechanically trivial.[^6]

QASPER also has important gaps. Each question is anchored to one known paper. It does not ask the
agent to find that paper among thousands of candidates, compare multiple papers, generate search
queries, or emit our JSON actions. Thirteen percent of its questions require figures or tables,
while the current corpus extractor is paragraph-focused. Those examples should initially be
excluded or tagged rather than silently converted into impossible text-only tasks.[^6]

The practical conversion is to place each target QASPER paper in a small pool of topically related
distractor papers. The gold paper and evidence give a verifier, while a teacher creates the action
sequence that discovers and reads them. This produces a tool-mediated task without pretending that
QASPER itself contains tool traces. It also lets us vary difficulty by increasing the pool size and
including near-neighbor distractors.

## Other datasets and their proper roles

### QASA

QASA contains 1,798 question-answer pairs over AI and ML papers and was designed to include
surface, testing, and deep questions. Its authors report that 39.4% of questions fall in their deep
reasoning category and that answers can use up to nine evidential rationales.[^8] This is stronger
reasoning coverage than QASPER's average question, and the domain is an excellent fit.

The released files are explicitly called test sets, including 1,554 answerable and 244 unanswerable
examples.[^9] Training on them would remove one of the best independent scientific-reading
benchmarks. QASA should be reserved for evaluation unless a future release provides a designated
training split or we create a separate, non-overlapping collection using its methodology.

### PeerQA

PeerQA's questions come from real peer reviews and its answers were annotated by the papers'
authors. It has 579 QA pairs across 208 papers, mostly ML and NLP, with mapped evidence and
answerability labels.[^10] These questions are valuable because reviewers ask about ambiguities,
missing justification, and experimental decisions rather than only asking for a fact already
obvious from an abstract.

Its small size and benchmark role make it more valuable as a difficult held-out evaluation than as
bulk training data. The paper texts also have to be downloaded under their original terms rather
than redistributed with the annotations. Reserve PeerQA to test whether gains extend to natural,
expert-originated questions.

### LitSearch

LitSearch directly measures the first stage of the product: finding relevant research papers from
a realistic description of an information need. It includes 597 expert-reviewed queries over a
corpus of roughly 64,000 recent ML and NLP papers.[^11] The study found a 24.8-point recall@5 gap
between BM25 and its strongest dense retriever and an additional 4.4-point improvement from LLM
reranking.[^11]

That finding matters before policy training. If the retriever never surfaces the target paper, Qwen
cannot recover through better reasoning. LitSearch should be kept intact as the retrieval and query
understanding benchmark. We can generate LitSearch-style training queries from a disjoint paper
collection, but should not train on the 597 official queries.

### ResearchQA and LitTraceQA

Two 2026 releases are even closer to the product, but both should initially be treated as evaluation
resources. ResearchQA reports 6,211 citation-grounded questions over 494 open-access papers across
eight domains, including lookup, comprehension, multi-hop, and adversarial questions. It rewards
grounded refusal when a paper does not support the requested answer.[^12] It is a recent preprint,
its data is generated through an automated pipeline rather than the human process used for QASPER,
and its public dataset describes itself as an evaluation set. It is useful, but not yet the safest
foundation for training.

LitTraceQA asks systems to return paper identifiers, evidence locations, and final answers. Its 55
public development examples include both hidden-source single-paper and multi-paper questions, and
the hidden test evaluates retrieval, evidence, and answer quality separately.[^13] This is almost an
external version of our desired contract. The correct use is a final held-out check, especially for
multi-paper retrieval and tables, figures, equations, algorithms, and citation contexts that our
paragraph-only corpus currently misses.

### ALCE and LongCite

ALCE evaluates long-form answers with citations across ASQA, QAMPARI, and ELI5. It measures answer
quality, citation correctness, and citation completeness; the original study found that even its
best ELI5 system lacked complete citation support about half the time.[^14] ALCE is useful for
evaluation design and for testing the larger answer-writing agent. It is less well matched to the
small query worker, whose main job is gathering evidence rather than writing a long report.

LongCite-45k contains 44,600 synthetic long-context QA instances with sentence-level citations and
an Apache 2.0 release.[^15] It is attractive because of its size, but it teaches citation formatting
over supplied context rather than search, query reformulation, or paper discovery. The collection
is bilingual and includes contexts up to 128,000 words, which is far beyond the intended small
worker's operating window. Use a carefully inspected English subset only if held-out failures show
that citation attachment remains a bottleneck after the agent can already retrieve correct
evidence.

### SciFact

SciFact contains 1,409 expert-written scientific claims verified against 5,183 abstracts, with
supporting or contradicting rationales.[^16] It is useful auxiliary supervision for deciding
whether a passage supports or contradicts an atomic claim. It does not teach open-ended question
answering, multi-step search, or multi-paper synthesis, and its Hugging Face dataset is licensed
CC BY-NC 2.0.[^17] It should remain a small auxiliary or evaluation component, with the license
recorded in any released derivative.

### KIWI

KIWI contains 1,260 interaction turns from 234 sessions in which an expert iteratively asks a model
to revise a long-form scientific answer using relevant papers.[^18] It is well aligned with the
larger orchestrator or explainer, especially for incorporating new information and making precise
edits. It is not the best source for the Qwen query worker because the papers are already supplied
and the interaction centers on writing revisions rather than finding evidence.

## Comparative assessment

The labels below are analytical judgments against this project's target, not published leaderboard
scores. “Train” means an official or clearly usable training source; “reserve” means using it for
training would weaken an independent evaluation.

| Dataset | Connected multi-step search | AI/scientific domain | Evidence or abstention | Native tool traces | Recommended role |
|---|---:|---:|---:|---:|---|
| MuSiQue | Strong | Weak | Strong paragraph supervision | None | General search component and transfer control |
| HotpotQA | Moderate | Weak | Strong supporting facts | None | Scalable general search component |
| QASPER | Moderate within a paper | Strong | Strong, human annotated | None | **Primary domain training source** |
| QASA | Strong within a paper | Strong | Strong rationales | None | Reserve for scientific reasoning evaluation |
| PeerQA | Moderate within a paper | Strong | Strong, expert-originated | None | Reserve for realistic difficult evaluation |
| LitSearch | Retrieval only | Strong | Gold paper IDs | None | Reserve for paper discovery/ranking evaluation |
| ResearchQA | Moderate, single paper | Strong | Strong citation/refusal design | None | Reserve; promising recent evaluation |
| LitTraceQA | Strong, including multi-paper | Strong | Strong and multi-format | None | Reserve for end-to-end external evaluation |
| ALCE | Weak | Weak to moderate | Strong citation metrics | None | Evaluate the answer-writing layer |
| LongCite-45k | Weak | Mixed | Synthetic sentence citations | None | Optional citation SFT auxiliary |
| SciFact | Weak | Strong scientific text | Strong support/refute labels | None | Optional claim-verification auxiliary |
| Local teacher trajectories | Designed by us | Strong | Strong after semantic review | **Yes** | Exact protocol and multi-paper SFT |

The central implication is that dataset choice and trajectory choice are different decisions. A
QASPER row gives a trustworthy question, answer, and evidence target. It does not give a trustworthy
sequence of tool actions. MuSiQue gives an explicit reasoning decomposition but still does not show
how an agent should react to actual search results. Teacher or successful-policy rollouts are needed
to turn both into our runtime protocol.

## Recommended data construction

### Stage 1: a conversion and learning pilot

Build a small, balanced batch before generating hundreds of expensive trajectories:

- 40 QASPER training questions over text-only evidence;
- 40 MuSiQue or HotpotQA training questions with two to four hops;
- 40 reviewed AI-paper questions over the separate development corpus, including comparisons and
  insufficient-evidence cases.

This 120-example batch is a pipeline diagnostic, not the final training set. It should cover valid
search/read/submit behavior, weak first retrieval, repeated-search recovery, two-source comparison,
and abstention. Every generated trajectory must pass schema and source-span checks. Every
scientific claim in the local teacher portion must also receive semantic support review.

The purpose is to answer three cheap questions: can Qwen learn the exact protocol, does it stop
repeating actions, and does the conversion preserve the source dataset's gold evidence? If it
cannot improve on a deliberately small development slice, more data is unlikely to repair a broken
conversion or harness.

### Stage 2: a decision-scale SFT set

If the pilot works, expand to roughly 600–1,000 reviewed or mechanically verifiable trajectories.
The initial target is an engineering estimate, not a claim that this is the optimal data size. A
reasonable first mixture is:

- about half QASPER-derived paper reading tasks;
- about one quarter general multi-hop tasks from MuSiQue or HotpotQA; and
- about one quarter reviewed AI-paper trajectories for multi-paper search, exact citation, and
  abstention.

Keep sampling by capability rather than copying the raw dataset distribution. Include answerable
and unanswerable questions, single- and multi-paper tasks, successful first searches and recovery
after weak searches, extractive and short synthetic answers, and comparisons where the papers do
not support a single winner. Record the source dataset, source IDs, paper IDs, conversion version,
teacher model, teacher prompt, decoding settings, tool version, and review state for each example.

Recent search-agent results caution against assuming that 170,000 generic questions are better
than a smaller selected set. s3 filters out examples already solved by naïve retrieval and trains on
the harder remainder; its reported 2,400-example result supports prioritizing examples where
iterative search adds value.[^2] Apply the same principle here: drop trivial questions whose gold
paper and evidence are already returned by the first search, unless they teach a needed stop
decision.

### Stage 3: generate difficult examples from measured failures

After the first SFT evaluation, add data only for repeated held-out failure categories:

- missed target paper → paper-discovery/query-rewrite examples;
- correct paper but wrong passage → within-paper evidence selection examples;
- correct passage but unsupported claim → support/contradiction and abstention examples;
- correct evidence but invalid JSON → protocol examples;
- correct claims but incomplete comparison → multi-paper coverage examples;
- excessive searching after sufficient evidence → stop-decision examples.

This failure-driven expansion reduces the chance that the model learns superficial formatting while
the actual research behavior remains unchanged.

## Evaluation design

The cleanest experiment is a four-arm comparison using the same Qwen3-8B base, QLoRA settings,
training action-token budget, and inference harness:

1. base model with no training;
2. general multi-hop training only;
3. QASPER-derived training only; and
4. the mixed training set.

This design separates three claims. If the general-only model improves on unseen paper tasks, the
project can credibly claim transferable search behavior. If QASPER-only improves evidence reading
but not multi-step retrieval, it demonstrates the value and limit of domain data. If the mixed
model wins, it supports the product-oriented curriculum.

Use paper-disjoint and task-disjoint evaluation:

- **Protocol:** the frozen eight-question repository pilot, then the planned 50–75 question
  held-out paper set after checking it for contamination.
- **Paper discovery and ranking:** LitSearch, with recall@5 and recall@10.
- **Within-paper evidence and answerability:** QASPER validation/test after training only on its
  train papers.
- **Natural expert questions:** PeerQA.
- **Deep AI-paper reasoning:** QASA.
- **End-to-end paper ID, evidence, and answer:** LitTraceQA validation and hidden test when the
  extractor supports its evidence types.
- **General multi-hop transfer:** an untouched MuSiQue or 2WikiMultiHopQA split.

The core metrics should remain tied to the worker's job:

| Layer | Metrics |
|---|---|
| Paper retrieval | target-paper recall@5/10, all-required-papers recall |
| Evidence retrieval | gold-passage recall, redundant-passage rate |
| Tool policy | valid-action rate, duplicate actions, recovery after a miss, steps to completion |
| Grounding | fully supported claim rate, unsupported claim rate, citation validity |
| Answerability | correct answer/refusal decision, over-refusal rate |
| Utility | blinded human usefulness score and missed-evidence flag |
| Operations | latency, tokens, tool calls, GPU/serving configuration |

Exact source spans prove provenance but do not prove semantic support. Human or independently
validated support labels remain necessary for the repository's main claim. Automatic citation
metrics can scale development checks, but they cannot replace the reviewed held-out result.

The existing runbook's proposed success threshold remains a defensible product decision rule: the
trained model should improve fully supported claims by at least 15 percentage points and reduce the
unsupported-claim rate by at least one third, without lowering answerability accuracy or submission
rate by more than five points. The four-arm study adds attribution: it tells us whether improvement
came from generic search data, scientific-paper data, or their combination.

## Retrieval, ranking, and indexing are separate from policy training

None of these datasets automatically improves the current BM25 index. The Qwen worker chooses and
reformulates queries; the retriever decides which passages those queries return. A poor index or
ranker places a hard ceiling on the agent.

LitSearch is the most relevant independent retrieval test because its queries describe research
needs and its corpus contains ML/NLP papers. BEIR's SciFact and SciDocs tasks can provide smaller
scientific retrieval checks, although SciDocs focuses on citation prediction rather than natural
user queries.[^19] The first retrieval experiment should compare the existing lexical search with a
fixed dense or hybrid baseline on the same frozen corpus. It should precede expensive policy
training if target-paper recall is low.

Knowledge graphs and distributed processing are later scaling choices. A citation graph or typed
evidence graph may help follow related work, authors, methods, datasets, and result tables, but it
will not compensate for weak questions, unsupported answers, or unreviewed trajectories. At the
current corpus size, a reproducible local index is sufficient. Distributed ingestion and serving
become relevant only when the corpus grows enough that indexing, updates, or query latency are an
observed bottleneck.

## Final recommendation for this repository

The next work should be data and evaluation preparation, not another GPU run:

1. Keep the existing MuSiQue checkpoint comparison open and evaluate it separately.
2. Add a QASPER importer that preserves paper IDs, paragraph evidence, answer type, and official
   paper-disjoint splits.
3. Define a conversion manifest from QASPER and MuSiQue into the research agent's read-only JSON
   environment; do not reuse the Python-writing MuSiQue protocol.
4. Build and inspect the 120-example conversion pilot.
5. Lock the larger held-out AI-paper benchmark and reserve QASA, PeerQA, LitSearch, ResearchQA, and
   LitTraceQA from training.
6. Run the four-arm QLoRA comparison only after the conversion, contamination, and baseline checks
   pass.

The resulting project claim can be both meaningful and honest:

> We trained an 8B knowledge-query agent to search iteratively and return evidence packets for a
> larger assistant. General multi-hop data taught connected search, scientific-paper data taught
> evidence reading and abstention, and the mixed model improved supported answers on papers and
> question sets excluded from training.

An even stronger transfer result is possible if the general-only arm improves on paper benchmarks:

> Training on Wikipedia multi-hop search improved retrieval and grounded answers on unseen AI
> papers, showing that the learned search behavior transferred beyond its training domain.

That claim requires the general-only arm and held-out scientific evaluation. It cannot be inferred
from valid JSON, correct source offsets, or performance on examples used to create the training
set.

## Sources

1. Allen Institute for AI. “[QASPER dataset](https://huggingface.co/datasets/allenai/qasper).” Dataset card and current Hub structure, accessed September 14, 2026.
2. Pengcheng Jiang et al. “[s3: You Don't Need That Much Data to Train a Search Agent via RL](https://arxiv.org/abs/2505.14146).” arXiv, revised November 2025.
3. Harsh Trivedi et al. “[MuSiQue: Multihop Questions via Single-hop Question Composition](https://aclanthology.org/2022.tacl-1.31/).” TACL, 2022.
4. Stony Brook NLP. “[MuSiQue repository](https://github.com/stonybrooknlp/musique).” GitHub, accessed September 14, 2026.
5. Bowen Jin et al. “[Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](https://arxiv.org/abs/2503.09516).” COLM, 2025.
6. Pradeep Dasigi et al. “[A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers](https://aclanthology.org/2021.naacl-main.365/).” NAACL, 2021.
7. Pradeep Dasigi et al. “[QASPER paper, dataset analysis](https://aclanthology.org/2021.naacl-main.365.pdf).” NAACL, 2021.
8. Yoonjoo Lee et al. “[QASA: Advanced Question Answering on Scientific Articles](https://proceedings.mlr.press/v202/lee23n.html).” ICML, 2023.
9. LG AI Research. “[QASA repository](https://github.com/lgresearch/QASA).” GitHub, accessed September 14, 2026.
10. Tim Baumgärtner, Ted Briscoe, and Iryna Gurevych. “[PeerQA: A Scientific Question Answering Dataset from Peer Reviews](https://aclanthology.org/2025.naacl-long.22/).” NAACL, 2025; [official repository](https://github.com/UKPLab/PeerQA).
11. Anirudh Ajith et al. “[LitSearch: A Retrieval Benchmark for Scientific Literature Search](https://aclanthology.org/2024.emnlp-main.840/).” EMNLP, 2024.
12. Saba Imran and Debanjum Singh Solanky. “[ResearchQA: Benchmarking Citation-Grounded Question-Answering on Scientific Papers](https://arxiv.org/abs/2607.11074).” arXiv, July 2026.
13. Xuye Liu et al. “[LitTraceQA: A Benchmark for Multi-Stage Grounding and Verification in Scientific Question Answering](https://arxiv.org/abs/2608.07370).” arXiv, August 2026; [dataset](https://huggingface.co/datasets/LitTraceQA/LitTraceQA).
14. Tianyu Gao et al. “[Enabling Large Language Models to Generate Text with Citations](https://arxiv.org/abs/2305.14627).” EMNLP, 2023.
15. Jiajie Zhang et al. “[LongCite: Enabling LLMs to Generate Fine-grained Citations in Long-context QA](https://arxiv.org/abs/2409.02897).” arXiv, 2024; [LongCite-45k dataset](https://huggingface.co/datasets/zai-org/LongCite-45k).
16. David Wadden et al. “[Fact or Fiction: Verifying Scientific Claims](https://aclanthology.org/2020.emnlp-main.609/).” EMNLP, 2020.
17. Allen Institute for AI. “[SciFact dataset](https://huggingface.co/datasets/allenai/scifact).” Dataset card, accessed September 14, 2026.
18. Fangyuan Xu et al. “[KIWI: A Dataset of Knowledge-Intensive Writing Instructions for Answering Research Questions](https://aclanthology.org/2024.findings-acl.770/).” Findings of ACL, 2024.
19. Nandan Thakur et al. “[BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models](https://arxiv.org/abs/2104.08663).” NeurIPS Datasets and Benchmarks, 2021; [dataset inventory](https://github.com/beir-cellar/beir/wiki/Datasets-available).
20. Zhilin Yang et al. “[HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering](https://aclanthology.org/D18-1259/).” EMNLP, 2018.

[^1]: Allen Institute for AI, “[QASPER dataset](https://huggingface.co/datasets/allenai/qasper),” dataset card and current Hub structure, accessed September 14, 2026.
[^2]: Pengcheng Jiang et al., “[s3: You Don't Need That Much Data to Train a Search Agent via RL](https://arxiv.org/abs/2505.14146),” arXiv, revised November 2025.
[^3]: Harsh Trivedi et al., “[MuSiQue: Multihop Questions via Single-hop Question Composition](https://aclanthology.org/2022.tacl-1.31/),” TACL, 2022.
[^4]: Stony Brook NLP, “[MuSiQue repository](https://github.com/stonybrooknlp/musique),” accessed September 14, 2026.
[^5]: Bowen Jin et al., “[Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](https://arxiv.org/abs/2503.09516),” COLM, 2025.
[^6]: Pradeep Dasigi et al., “[A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers](https://aclanthology.org/2021.naacl-main.365/),” NAACL, 2021.
[^7]: Dasigi et al., “[QASPER paper, dataset analysis](https://aclanthology.org/2021.naacl-main.365.pdf),” 2021, pp. 4601–4603.
[^8]: Yoonjoo Lee et al., “[QASA: Advanced Question Answering on Scientific Articles](https://proceedings.mlr.press/v202/lee23n.html),” ICML, 2023.
[^9]: LG AI Research, “[QASA repository](https://github.com/lgresearch/QASA),” accessed September 14, 2026.
[^10]: Tim Baumgärtner, Ted Briscoe, and Iryna Gurevych, “[PeerQA: A Scientific Question Answering Dataset from Peer Reviews](https://aclanthology.org/2025.naacl-long.22/),” NAACL, 2025; [official repository](https://github.com/UKPLab/PeerQA).
[^11]: Anirudh Ajith et al., “[LitSearch: A Retrieval Benchmark for Scientific Literature Search](https://aclanthology.org/2024.emnlp-main.840/),” EMNLP, 2024.
[^12]: Saba Imran and Debanjum Singh Solanky, “[ResearchQA: Benchmarking Citation-Grounded Question-Answering on Scientific Papers](https://arxiv.org/abs/2607.11074),” arXiv, July 2026.
[^13]: Xuye Liu et al., “[LitTraceQA: A Benchmark for Multi-Stage Grounding and Verification in Scientific Question Answering](https://arxiv.org/abs/2608.07370),” arXiv, August 2026; [dataset](https://huggingface.co/datasets/LitTraceQA/LitTraceQA).
[^14]: Tianyu Gao et al., “[Enabling Large Language Models to Generate Text with Citations](https://arxiv.org/abs/2305.14627),” EMNLP, 2023.
[^15]: Jiajie Zhang et al., “[LongCite: Enabling LLMs to Generate Fine-grained Citations in Long-context QA](https://arxiv.org/abs/2409.02897),” arXiv, 2024; [dataset](https://huggingface.co/datasets/zai-org/LongCite-45k).
[^16]: David Wadden et al., “[Fact or Fiction: Verifying Scientific Claims](https://aclanthology.org/2020.emnlp-main.609/),” EMNLP, 2020.
[^17]: Allen Institute for AI, “[SciFact dataset](https://huggingface.co/datasets/allenai/scifact),” dataset card, accessed September 14, 2026.
[^18]: Fangyuan Xu et al., “[KIWI: A Dataset of Knowledge-Intensive Writing Instructions for Answering Research Questions](https://aclanthology.org/2024.findings-acl.770/),” Findings of ACL, 2024.
[^19]: Nandan Thakur et al., “[BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models](https://arxiv.org/abs/2104.08663),” NeurIPS Datasets and Benchmarks, 2021; [dataset inventory](https://github.com/beir-cellar/beir/wiki/Datasets-available).

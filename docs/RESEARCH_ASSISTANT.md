# AI-paper research assistant

For the motivation, proposed teacher-to-Qwen distillation workflow, data requirements, and
open decisions, see [the research pivot notes](RESEARCH_PIVOT_NOTES.md).
For the concrete GPU setup and first live result, see
[the first smoke-run record](RESEARCH_SMOKE_RUN.md).
For the complete locked Qwen3-8B pilot and its failure diagnosis, see
[the baseline pilot record](RESEARCH_BASELINE_PILOT.md).
For the first authored development examples and the isolated JSON-generation experiment, see
[the demonstration batch](RESEARCH_DEMONSTRATIONS.md).
For the retrieval-agent and larger-explainer boundary, see
[the second-brain architecture](SECOND_BRAIN_ARCHITECTURE.md).
For the evidence-based training-data choice and proposed general/domain mixture, see
[the research dataset decision](RESEARCH_DATASET_DECISION.md).
For the concrete QASPER conversion result and its boundary, see
[the QASPER pilot record](QASPER_PILOT.md).
For a plain-language walkthrough of the complete product and training loop, see
[how the AI research assistant works](HOW_THE_AI_RESEARCH_ASSISTANT_WORKS.md).
For importing a personal collection and saving answers as Obsidian notes, see
[the folder-based Obsidian workflow](OBSIDIAN_WORKFLOW.md).

The user-approved objective is: given an AI project and its constraints, find relevant research,
explain what the papers demonstrate, and make the evidence inspectable. Start with retrieval and
tool-using agents. A useful baseline and reviewed failures come before a preference model, SFT,
or RL. The existing MuSiQue training and rewards remain a separate experimental track.

## Prepare the QASPER domain-data pilot

QASPER is the primary source for teaching within-paper evidence reading and answerability. The
converter pins the Hugging Face dataset revision, preserves official split/paper/question IDs and
all answer annotations, and maps human text-evidence paragraphs to exact snapshot offsets. It does
not turn those annotations into invented tool trajectories. Questions requiring figures/tables or
whose text evidence cannot be mapped exactly are excluded from the text-only pilot and counted in
the manifest.

Build the first 40-question pilot from the official train split. Each converted task supplies the
known paper title and document ID because QASPER's questions were written with that paper already
in view. The searchable snapshot still contains all 888 train papers, which lets the same agent
tools run without a second corpus format:

```bash
python scripts/setup_qasper.py \
  --split train --num-questions 40 --seed 42 \
  --output out/research/qasper-train-pilot-v2
```

This creates `manifest.json`, `questions.json`, and the compatible `corpus/` directory. Validation
and test are reserved for later paper-disjoint evaluation. The immediate hypothesis is that the
existing lexical retriever can surface the target paper for a useful fraction of these natural
scientific questions. The expected signal is 100% exact mapping for every selected text-evidence
paragraph, correct answer-type preservation, and split isolation. Accept the conversion only if
those checks pass. Open-ended paper discovery is measured separately; QASPER does not supervise it.

Run the CPU-only routing diagnostic with:

```bash
python scripts/research_retrieval_check.py \
  --snapshot out/research/qasper-train-pilot-v2 \
  --questions out/research/qasper-train-pilot-v2/questions.json \
  --top-k 5 \
  --output out/research/qasper-train-pilot-v2-retrieval-top5.json
```

That command is only a routing diagnostic for title-plus-question queries. It is not the primary
QASPER score. Evidence selection, answer accuracy, and answerability on known papers are the skills
QASPER can evaluate.

## What works now

- Six real starter papers: RAG, ReAct, Self-RAG, Corrective RAG, Adaptive-RAG, and Search-R1.
  `data/research/sources.json` pins the versions downloaded on September 12, 2026.
- Immutable snapshots contain original downloaded HTML, normalized paragraph text, section
  offsets, paper URLs/versions/dates, extraction coverage, and checksums. A failed download is
  recorded; missing full text is explicitly marked `abstract_only`.
- A CPU-only passage search command, plus a multi-turn model-driven search/read/submit runner.
  No embedding download is needed for the lexical baseline.
- A chat-completions client for a Qwen server such as vLLM. The research prompt and submission
  schema are separate from the MuSiQue policy prompts.
- Each model run saves actions, observations, failure status, question/snapshot identities,
  decoding settings, operator-supplied model revision/hardware, and a Markdown review sheet.
- Submission checks materialize exact text from valid frozen-snapshot spans and flag missing
  sources, invalid offsets, or incorrect model-supplied quotations. They do **not** judge whether
  a passage supports its associated claim. No research training reward is assigned. Factual
  claims in recommendations also need human review.

## Inspect the downloaded papers now

The initial download is at `out/research/starter-2026-09-12/` in this workspace. `out/` is ignored
by Git; keep the entire snapshot when transferring to a pod or sharing an experiment.

```bash
python scripts/research.py inspect \
  --snapshot out/research/starter-2026-09-12 \
  --query 'retrieved token masking' --top-k 3
```

To reproduce from the pinned source list into a new directory:

```bash
python scripts/research.py snapshot --output out/research/starter-rebuilt
```

This downloads public sources with spacing between requests. It never overwrites an existing
snapshot. Save/copy a snapshot for exact replay: upstream HTML or parser changes can change text
offsets even when a paper version stays the same. A corpus hash mismatch stops a model run.

The starter collection is historical and deliberately small. It is **not** a latest-research
index. HTML extraction covers paragraphs; figures, tables, formulas, and PDF layout need further
work. Open the original source to verify details absent or distorted in the extracted text.

## Discover recent candidates

```bash
python scripts/research.py discover \
  --query 'all:"retrieval augmented" OR all:"search agent"' \
  --since 2026-08-01 --until 2026-09-12 --limit 20 \
  --output out/research/candidates.json
```

This records the exact query, date range, retrieval time, and bounded arXiv results. Review the
candidates, then pass the candidate JSON as `snapshot --sources ...` to freeze them. Narrow the
list first if needed. Live discovery does not silently modify a benchmark corpus. A date-sorted
search is not proof of comprehensive coverage, and paper dates are not guarantees of quality.

During the September 12 check, the metadata API first timed out and later returned HTTP 429.
The command's XML parsing is tested offline, but live discovery has not passed yet. It fails
explicitly without presenting a cached collection as fresh. Respect the rate limit; try later.
Direct downloads of all six selected paper pages succeeded.

API reference: [arXiv API user manual](https://info.arxiv.org/help/api/user-manual.html).

## Run the agent when a model server is available

Use a server exposing `/v1/chat/completions`, with a pinned model revision and recorded serving
configuration. The first training comparison uses the same Qwen3-8B checkpoint before and after
SFT. This repository does not install or launch vLLM as part of the command. Keep serving
dependencies separate from the pinned legacy training environment.

```bash
python scripts/research.py ask \
  --snapshot out/research/starter-2026-09-12 \
  --question-id research_01 \
  --endpoint http://localhost:8000/v1 \
  --model Qwen/Qwen3-8B \
  --revision REPLACE_WITH_SERVED_MODEL_COMMIT \
  --server-hardware 'REPLACE_WITH_GPU_COUNT_MODEL_AND_DTYPE' \
  --max-steps 10 --max-tokens 1800 \
  --output out/research/qwen3-research-01.json
```

Or replace `--question-id` with `--question 'your project question'`. If the server needs a key,
set `RESEARCH_API_KEY` in the environment; the run artifact does not record it. The client uses
vLLM-style extra parameters to disable thinking and explicitly set sampling controls; other
servers may reject these. See [vLLM serving documentation](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/).
Model revision and server hardware are declarations by the operator, not independently
attested by the server. Record server context limit, quantization, and software versions with
the hardware description for a comparison. Token limits are per action; context accumulates
across turns. If the server's context limit is reached, the run records the failure rather than
silently dropping evidence.

The research model emits one validated JSON action per turn. The application exposes five
read-only operations—list papers, search across papers, search within one known paper, inspect
paper metadata, and read a bounded passage—and rejects unknown actions or extra arguments. The
within-paper search can use a version-pinned cross encoder and records that retriever in the run
artifact. Qwen does not write or execute Python.
The older MuSiQue RLM environment still uses its separate Python REPL and Docker sandbox.

The JSON contains the full trajectory; the adjacent `.md` file contains the answer, quotations,
source links, and review fields. `submitted` means syntactically valid submission, not a correct
answer. Empty answers, invalid quotations, and semantic support are reported separately.

## Questions and the first experiment

`data/research/questions.json` contains 20 **assistant-proposed, unreviewed development questions**.
They cover weak retrieval, multi-hop questions, citation support, costs, uncertainty, freshness,
and project decisions. They have no fabricated gold answers. Review their usefulness with the
user before treating them as a representative workload. They must not later become the held-out
test set after being used to guide training.

Hypothesis: a newer small model follows this evidence-gathering workflow more reliably than the
current Qwen2.5-7B. First smoke-test 3 questions (`research_01`, `research_04`, `research_11`) to
check tool use, evidence access, and coverage disclosure. Stop and repair protocol failures
before a broader comparison. Then run the same 20 prompts with each candidate under the same
snapshot, tools, prompt, non-thinking mode, and step/output budgets.

Expected signal: more useful answers with fewer unsupported claims, not merely more citations
or higher submission rates. Review answers blind to the model name where practical:

1. For every factual claim, including the recommendation: supported, partially supported,
   unsupported, or contradicted by the cited evidence. Check the original paper when necessary.
2. Missing relevant evidence and whether the question was actually answered.
3. Practical usefulness: 0 = unusable, 1 = partially useful, 2 = actionable with caveats.
4. Appropriate uncertainty, source coverage, and date disclosure.
5. Tool failures, repeated searches, steps, and measured serving cost/latency.

Decision rule: proceed with a candidate only if paired review shows better usefulness without
increasing unsupported claims, within an agreed serving budget. If the tradeoff is ambiguous,
inspect failures instead of declaring a winner from 20 examples. Do not optimize for abstaining
on everything. This is exploratory model selection, not a statistically established result.

Categorize repeated failures before selecting a fix: retrieval, context handling, tool protocol,
evidence interpretation, or presentation. Test retrieval/prompt fixes before spending on training.
Only then create reviewed training examples and separate held-out papers/questions. If training
is justified, compare the **same model before and after training** with identical tools and data.
The eventual claim must be measured improvement in evidence-supported, useful answers.

## Separate development corpus

`data/research/development_sources_v1.json` selects 20 version-pinned papers about research
agents, tool use, attribution, factuality, retrieval, and agent evaluation. They are disjoint from
the six papers reserved by the pilot benchmark. The frozen local copy is expected at
`out/research/ai-agents-development-v1-20260912`; like every directory under `out/`, it must be
copied explicitly when moving the experiment to another machine or pod.

`data/research/development_questions_v1.json` contains 15 unreviewed development prompts. Fourteen
have expected papers for a retrieval-routing check and one tests abstention when the corpus does
not establish a deployment guarantee. These prompts and papers may guide SFT data creation, so
they are not a held-out benchmark and cannot support a before/after training claim.

Run the CPU-only lexical retrieval check with:

```bash
python scripts/research_retrieval_check.py \
  --snapshot out/research/ai-agents-development-v1-20260912 \
  --questions data/research/development_questions_v1.json \
  --top-k 5 \
  --output out/research/development-retrieval-top5-v2.json
```

The resulting target-document recall measures whether the fixed diagnostic queries surface the
intended papers. It does not measure whether passages are relevant, whether evidence supports a
claim, or whether Qwen can formulate those queries itself.

The September 13 retrieval check exposed repeated passages from one paper crowding other sources
out of the top five results. Returning each paper's best matching passage raised target-document
recall from 0.7692 to 0.9231 and the all-targets-found rate from 0.5714 to 0.8571, while the
any-target-found rate remained 1.0. The two remaining misses are both two-paper comparisons; this
supports teaching the agent to issue separate searches for each side rather than treating the
hand-written diagnostic queries as a solved research benchmark.

## Locked benchmark pilot before training

`data/research/benchmark_pilot_v1.json` is a separate eight-question evaluation pilot. Its purpose
is to verify the baseline, review, and scoring workflow before constructing the larger final test.
All six papers in its frozen snapshot are reserved from training. Do not put those documents,
questions, or paraphrases of their grader notes into SFT, preference, or RL examples.

The pilot deliberately includes answerable questions, comparisons requiring two papers, and two
questions where the correct behavior includes refusing an unsupported guarantee. Validate its
identity against the snapshot before running anything:

```bash
python scripts/research_benchmark.py validate \
  --benchmark data/research/benchmark_pilot_v1.json \
  --snapshot out/research/starter-2026-09-12
```

With a model server available, `scripts/run_research_benchmark.sh` runs all eight questions under
fixed settings, saves an automatic score, and creates `human-review.json`. Automatic metrics cover
submission, exact source provenance, expected-document recall, tool use, and execution errors.
They cannot establish that evidence supports a claim. Complete every claim-level support label,
answerability judgment, recommendation-faithfulness judgment, missed-evidence flag, and usefulness
score before asking the scorer for human metrics:

```bash
python scripts/research_benchmark.py score \
  --benchmark data/research/benchmark_pilot_v1.json \
  --runs out/research/PILOT_DIR/runs \
  --reviews out/research/PILOT_DIR/human-review.json \
  --output out/research/PILOT_DIR/reviewed-score.json
```

This pilot is frozen early enough to compare base Qwen3-8B with an SFT adapter, but eight questions
are too few for a headline claim. Before training, expand the final benchmark to 50–75 questions
over new reserved papers and lock it. The training hypothesis is that reviewed search/read/answer
trajectories teach transferable evidence use rather than citation formatting. Count SFT as useful
only if it improves fully supported claims by at least 15 percentage points and cuts the unsupported
claim rate by at least one third on the held-out set, without reducing answerability accuracy or
submission rate by more than 5 percentage points. Report uncertainty intervals on the paired
question-level differences; revise the threshold before seeing trained-model results if the base
rate makes it statistically unrealistic.

The first training implementation should use Unsloth QLoRA on RunPod and preserve the exact Qwen
chat template at inference. Do not rent the training GPU until the expanded benchmark is locked,
the base checkpoint has been scored, and reviewed training examples pass contamination checks.

## Validation and remaining work

Offline tests cover snapshots, provenance checks, multi-turn replay, failed runs, endpoint
request controls, benchmark validation/scoring, and Docker mount arguments. A scripted policy
tests the plumbing and establishes no model capability.

```bash
python -m pytest tests/test_research.py tests/test_research_benchmark.py tests/test_repl.py -q
```

September 12 validation: the complete suite passed **99 tests** with
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q`, using the already cached embedding
model. Without offline mode, 18 legacy fixtures attempted Hugging Face network requests and
failed during setup; no research test failed. Ruff passed for the added research code.
The real downloaded corpus also passed a scripted three-step search/read/submit run; inspect
`out/research/scripted-smoke.json` and its `.md` review sheet. That run explicitly labels its
backend as scripted and is not evidence of model answer quality.

The first live Qwen3-8B run submitted an answer and used a valid snapshot span, but it repeated
actions, hit a Python syntax failure, covered only one of the requested comparison approaches,
and has not received semantic human review. That is a protocol/provenance signal, not evidence of
research quality. That run used the superseded `research-evidence-v1` Python-action protocol; the
`research-tools-v2` protocol removed that syntax failure mode. `research-tools-v3` added bounded
within-paper search after the QASPER smoke exposed expensive linear reading. The current
`research-tools-v4` also rejects an identical tool action instead of executing it again and
duplicating a large observation in the serving context. The disposable RunPod GPU was stopped
after the run.

Still needed: run and review the eight-question base pilot; expand and lock the final held-out
paper set; create reviewed training trajectories from disjoint papers; then compare base and SFT
on identical runs. No research fine-tuning job has been started.

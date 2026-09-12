# Research assistant pivot: purpose, data, and training plan

Conversation notes — September 12, 2026.

This document records the agreed direction and the proposed training approach. The
[research assistant runbook](RESEARCH_ASSISTANT.md) covers implementation status and commands.
The direction is approved; the teacher model, final student model, dataset size, and later
training algorithm are still decisions to make from evidence.

## The problem we want to solve

When someone is building an AI project, they want to know what relevant recent research says,
which ideas might help under their constraints, and whether the assistant's claims are actually
supported by the papers it cites.

The intended product is:

> Given my AI project and its constraints, find relevant research, explain what the papers
> demonstrate, and show the evidence behind each recommendation.

Start with **retrieval and tool-using AI agents**. This is a focused domain that can help with
our own project. A representative question is: “My document agent makes unsupported claims.
Which approaches could help, what evidence do they have, and what should I test first?”

A useful answer should compare sources, distinguish reported findings from the assistant's
inference, identify limitations, and propose an experiment suited to the project. The first
version is not expected to autonomously resolve every scientific disagreement or comprehensively
review all AI research.

## Why we are changing direction

The MuSiQue project studies whether a model can improve at multi-step document exploration.
The user wants that capability connected to a practical problem they care about.

We can reuse the search/read/action loop, execution infrastructure, experiment logging, and
parts of the training machinery. We need domain-specific paper ingestion, research questions,
reviewed demonstrations, and evaluation of evidence support and usefulness.

MuSiQue remains an earlier experimental track. Its checkpoints have not established an
improvement on a common evaluation split, and they are not evidence of research-assistant
quality. Its short-answer overlap and citation-ID reward is not suitable as the main measure
of an open-ended research explanation.

## Freshness and training serve different purposes

New papers enter through search and refreshed document collections. We do not need to retrain
the model every time a paper appears.

Training is intended to improve how the agent searches, inspects evidence, decides what to
read next, compares findings, and answers with appropriate uncertainty. Whether those habits
transfer to new papers must be tested.

Every evaluation uses a frozen, dated corpus with pinned paper versions. Everyday discovery
can refresh the collection, but must state its coverage and must not silently alter an
evaluation snapshot. “Recent results from this search” is a narrower claim than “the latest
research across the entire field.”

## The proposed first training approach: trajectory distillation

Use a stronger model as a **teacher** to help teach a smaller Qwen **student**. This is behavioral
or trajectory distillation: the student learns from the teacher's observable tool actions and
answers. We do not need the teacher's private internal reasoning or access to its weights.

The proposed pipeline is:

1. Give the teacher a realistic project question, project constraints, and paper-search tools.
2. Record its searches, tool observations, passages read, and final answer with supporting evidence.
3. Check the trajectory and review the answer. Repair or reject unsupported demonstrations.
4. Fine-tune Qwen on accepted demonstrations using supervised fine-tuning (SFT), initially with
   LoRA adapters.
5. Compare the same Qwen model before and after training on separate questions and papers.

One example would look like this:

```text
Question: My agent retrieves relevant-looking documents but still gives unsupported answers.
          What should I try under a small compute budget?

Teacher action: Search for approaches to weak retrieval and evidence checking.
Tool result:   Candidate paper passages with versions, source URLs, and offsets.
Teacher action: Read methods and limitations in one paper.
Tool result:   The requested evidence.
Teacher action: Inspect an alternative paper and compare the experimental settings.
Tool result:   Further evidence and limits on what can be concluded.
Teacher answer: Supported findings, a clearly labeled recommendation, and remaining uncertainty.
```

This is an illustration of the desired format, not an existing reviewed training example.

In SFT, the question and tool observations provide context; the model learns to produce the
assistant's actions and final answer. Retrieved paper text should not itself be a prediction
target. The research-specific data conversion, masking, and context handling still need to be
implemented and validated before using the existing trainer for this task.

The teacher can also draft questions. **Generating questions alone is synthetic data generation;
training the student to imitate the teacher's demonstrated behavior is the distillation step.**

## Data: what we have and what “enough” means

As of this note, we have six downloaded starter papers and 20 unreviewed development questions.
We have **no reviewed research-agent training demonstrations yet**. Papers are source material;
questions are prompts; neither alone is the proposed SFT dataset.

The initial proposal is **100–300 checked demonstrations as a pilot**, not a proven requirement
or a guarantee of improvement. Collecting that amount is feasible in principle, but the teacher
generation cost and human review time have not been measured. We should time a small batch
before committing to a larger collection.

Include different tasks and outcomes: finding evidence, comparing methods, following a second
search, rejecting a misleading comparison, recovering from weak retrieval, and saying what the
available evidence cannot establish. Merely paraphrasing the same question many times does not
create equivalent diversity.

Measure performance at increasing dataset sizes. Expand when additional examples help or when
review reveals missing task coverage. If gains plateau, inspect data quality, retrieval, context,
and model limitations before assuming more examples will solve the problem.

The existing MuSiQue trajectories may inform tool-format training, but are not a substitute for
reviewed research investigations. The six starter papers are enough for initial infrastructure
tests, not a sufficient basis for broad claims about generalization across AI literature.

## How demonstrations will be checked

Automatic checks can establish whether tool calls execute, paper IDs exist, offsets are valid,
and quoted text matches the saved source. They cannot establish that a quotation supports a
claim or that advice is useful.

Review must also check:

- Whether each factual claim follows from the cited evidence, including claims in recommendations.
- Whether the answer exaggerates findings or compares incompatible experimental settings.
- Whether the recommendation addresses the user's project and constraints.
- Whether missing evidence, date coverage, and extraction limitations are disclosed.

The teacher can hallucinate. Its outputs are candidates, not automatic ground truth. A second
model may help flag problems, but should not be the sole authority for accepting examples.
Preserve teacher identity, prompt version, decoding settings, corpus identity, and review status
with every accepted trajectory.

## Evaluation and the order of work

First get real baseline answers and review failures. A retrieval or prompting fix may address
the problem before training is needed.

For model selection, keep Qwen2.5-7B as the current baseline and test Qwen3-8B as the practical
first alternative. Qwen3.5-9B is another candidate if its integration cost is justified; no switch
has been made. The small-model size range is a plausible starting point, not an established
capability result. A stronger teacher has not been selected.

Use the same papers, tools, questions, and comparable execution budgets for candidate models.
Our first smoke questions are `research_01`, `research_04`, and `research_11`. Expand to the 20
development prompts after tool protocol and evidence access work.

Before generating training data, reserve separate test papers and questions. Keep versions of
the same paper and close question duplicates in the same split. Do not use held-out test answers
to guide training-example creation or tuning. The 20 current development questions cannot later
be relabeled as an untouched final test set.

For an SFT experiment:

- **Hypothesis:** reviewed teacher trajectories improve useful, evidence-supported answers.
- **Expected signal:** better usefulness with fewer unsupported claims on unseen questions/papers.
- **Control:** the same student model before training, with identical tools and evaluation data.
- **Decision rule:** continue if paired review shows improved usefulness without increasing
  unsupported claims, within the measured cost budget. Inspect mixed or ambiguous results rather
  than declaring success from submission rates or valid quotes alone.

Count missed evidence and inadequate answers too, so refusing to answer everything does not
appear successful. Record uncertainty in small-sample results. Any later claim of training gains
must separate those gains from improvements caused by changing the base model or retrieval setup.

## Where preference models, DPO, and RL fit

We are not currently building a preference model. A preference-based reward model would learn
to score outputs from judgments about which answer or trajectory is better. DPO is an alternative
that trains on preferred/rejected pairs without a separate learned reward model.

After SFT, either preference training or reinforcement learning could be worth testing if the
remaining failures justify it. GRPO is available in the earlier project, but using it for this
task requires a trustworthy research-specific reward. Rewarding exact quotation alone could
encourage copying without useful reasoning. Optimizing a judge that prefers polished prose
could reward unsupported answers.

The current proposal is therefore **baseline → reviewed teacher demonstrations → SFT → held-out
evaluation**, with further training conditional on the evidence. Running RL is not itself the
project's success criterion.

## Current implementation and next milestone

Implemented: versioned paper snapshots, lexical passage search, a multi-turn research runner,
exact-quote checks, review artifacts, and the 20 draft questions. The complete suite passed 93
tests in offline mode on September 12. A scripted run passed against real paper text.

Still unverified: real Qwen research-answer quality, live model-server integration, and live
recent-paper discovery after arXiv returned a rate limit. Full table/PDF extraction, reviewed
training examples, the research SFT pipeline, and training gains are not complete. No GPU was
provisioned and no research training job has run.

The next milestone is to obtain and review real baseline answers, identify repeated failures,
and create a small checked teacher batch if training is justified. This will make the data
collection cost and training plan concrete.

The result we ultimately want to earn is:

> I trained a small research agent to give more useful, evidence-supported answers about AI
> papers, and demonstrated the improvement on held-out questions and papers.

That result has not yet been established. Transfer to other document-heavy domains is a future
hypothesis requiring its own evaluation, not a claim supported by this prototype.

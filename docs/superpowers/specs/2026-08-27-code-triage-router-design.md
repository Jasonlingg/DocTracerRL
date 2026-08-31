# Code Triage Router — Design Spec

## One-line pitch

Train a small open model (Qwen2.5-7B-Instruct, LoRA) with GRPO to triage AI-generated code diffs — deciding whether to ship, verify (tests/lint), escalate to a stronger model, or flag a human — using verifiable, cost-aware rewards. The task is a demonstration instance of a general recipe: taking a hard, ambiguous classification/routing decision and making it RL-trainable via reward design, not a one-off code-review tool.

## Motivation

This project fills a specific gap in the author's track record: prior work (Preference Model — designed RL *environments* for other models to act in; Assort Health — built the *pipeline* around an entity-resolution decision) never included personally training an RL *policy* end-to-end and showing it learned something. This project closes that gap while reusing and extending both threads: RL-environment design (Preference Model) and cost-aware routing on a real judgment call (Assort Health).

A prior project (DocTracerRL/rlm-explorer) attempted GRPO on multi-hop QA and stalled — flat reward curves with 21 SFT examples, sparse/fuzzy trajectory-level reward (F1 across a whole multi-step episode), and a task that was likely beyond a 7B LoRA model's raw capability ceiling. This project is deliberately designed to avoid both failure modes (see "Why this avoids the prior failure mode" below).

## Why this task, not another one

Considered and rejected during brainstorming:
- **Binary accept/escalate on synthetic garbled voice entities** — thin decision space (1 bit), required building a synthetic TTS→telephony→STT pipeline from scratch, thinner research backing.
- **Chess via GRPO** — well-documented research finding that general-purpose LLMs plateau far below expert play because pretraining gives them little real chess competence; RL cannot instill capability that pretraining never provided. Same failure shape as DocTracerRL, just relocated.
- **Negotiation / Werewolf social deduction** — genuinely research-backed (RLVR negotiation papers, ICML Werewolf paper) and high clickbait ceiling, but orphaned from the author's resume narrative and a larger build (multi-agent loops) for a ~1 month timeline.

Code triage was selected because:
1. It is something the author does mentally, every day, when using AI coding assistants — labeled data (git history: was a commit followed by a fix shortly after?) is free and self-renewing, and the tool remains useful after the project ends.
2. The underlying judgment ("does this diff look right given the task and cheap mechanical signals") is within a 7B instruct model's existing general-reasoning competence — unlike chess, this is not asking the model to exercise a narrow skill pretraining never gave it.
3. It extends to a richer, multi-way action space (see below) rather than a binary decision, without requiring capability the base model lacks.
4. It is directly relevant to a voice-AI/healthcare-agent employer (Assort Health) as well as to general "RL for agent infrastructure" roles.

## Why this avoids the prior project's failure mode

| DocTracerRL failure cause | How this project is designed differently |
|---|---|
| Reward was sparse/fuzzy (F1 across a 10-step trajectory) — hard credit assignment | Reward is computed per single-turn decision, binary-gated on mechanical, unambiguous signals (test pass/fail, lint pass/fail) |
| SFT warm-start was 21 examples — model had near-zero competence, so GRPO groups were mostly all-zero-reward (no gradient) | SFT warm-start target is ≥300 examples; a cheap GRPO diagnostic (15-20 steps) is run and inspected *before* committing to the full run, specifically to catch a repeat of the all-zero-reward cold start early |
| Task (multi-hop QA synthesis) may have exceeded the base model's raw capability ceiling | Task (triage judgment from evidence) is within general-reasoning competence a 7B instruct model already has, evidenced by the routing/classification literature below |
| No clear "why did it fail" diagnosis available afterward | A known research finding (xRouter: small open models often collapse to simple strategies rather than sophisticated orchestration) gives an interpretable, citable explanation if the result is weak, rather than an ambiguous null result |

## Research grounding

- **"Cognitive Overhang" (2026 industry framing)** — the motivating premise for why routing/triage is worth doing at all, pre-empting the natural objection "why not just always use the best model for everything." Named as *"the central operating dynamic of the 2026 AI economy... the widening delta between the maximum processing capacity of a frontier model and the actual reasoning requirements of the median production task"* ([Frontier Models Are Incredible, That's Why You Should Almost Never Use Them](https://medium.com/analysts-corner/frontier-models-are-incredible-454320562bad)), with the same source explicitly recommending *"hybrid, intelligent orchestration"* over blanket frontier-model use. Backed by concrete, sourced numbers: a ~50-250x cost gap between frontier ($5/$25 per million input/output tokens for Claude Opus 5) and budget-tier models (~$0.10/million), and a ~10x throughput gap (frontier models run 65-170 tokens/sec vs. 1,000+ tokens/sec for fast/budget tiers) — both of which compound badly across the many sequential tool-calls a coding agent makes per task, which is exactly the workload this project's router sits inside of.
- **[xRouter (Salesforce AI Research, arXiv 2510.08439)](https://arxiv.org/abs/2510.08439)** — the direct methodological anchor. Trains Qwen2.5-7B-Instruct with a GRPO-style algorithm (DAPO) and reward `R_final = R_binary × (K − λC)`: binary task success gates everything, cost only matters when the answer was right. Achieves near-GPT-5 accuracy at ~1/8th cost on Olympiad Bench. Documents a key limitation directly relevant here: *"complex orchestration behaviors... do not naturally emerge from standard RL training"* for small open models, and that Qwen2.5 trains more effectively as a router than newer Qwen3 variants (which bias toward internal reasoning over tool use). Open training code and reward implementation available at [SalesforceAIResearch/xRouter](https://github.com/SalesforceAIResearch/xRouter) (Apache 2.0), built on `verl`.
- **[Rewards as Labels: Revisiting RLVR from a Classification Perspective (Zhai et al., Feb 2026, arXiv 2602.05630)](https://arxiv.org/pdf/2602.05630)** — formalizes RLVR as classification: verifiable outcome rewards partition rollouts into correct/incorrect sets, functioning as binary labels. This is the theoretical grounding for the project's core claim — that RL is a general recipe for training hard classification/routing decisions, not something specific to this task.
- **[TruthRL (Wei et al., 2025, arXiv 2509.25760)](https://arxiv.org/pdf/2509.25760)** and multi-reward GRPO literature — precedent for ternary/multi-way reward shaping (correct / abstain-or-escalate / wrong) with asymmetric penalties, which this project's cost-aware multi-action reward extends.
- **[SWE-PRBench](https://www.researchgate.net/publication/403262187_SWE-PRBench_Benchmarking_AI_Code_Review_Quality_Against_Pull_Request_Feedback)** (350 human-annotated PRs) and **[SWE-Review-Bench](https://arxiv.org/pdf/2607.06065)** (built on SWE-bench Verified, multi-quality candidate PRs with executable tests) — establish that frontier models catch only 15-31% of human-flagged issues on diff-only review, giving a documented, citable context for the eval (not the primary claim to beat, but the honest backdrop).
- **Learned routing vs. fixed thresholds** — well-established that trained routers beat static confidence-threshold cascades ([RouteLLM](https://arxiv.org/pdf/2410.13284), routing survey [arXiv 2603.04445](https://arxiv.org/html/2603.04445v2)), with a documented caveat that learned routers can underperform under distribution shift — motivating the entity-disjoint train/test split used here.
- **Known counter-example, explicitly out of scope for comparison:** general-purpose LLM chess via GRPO plateaus far below expert level because pretraining does not give the model real chess competence ([arXiv 2507.00726](https://arxiv.org/html/2507.00726v2)). Cited here only to justify why code triage (general reasoning) was chosen over a narrow-skill domain (chess).

## What is NOT proven, stated honestly

- No published work has shown RL specifically (vs. supervised fine-tuning) winning on a *single-shot* routing/triage decision at small scale — most production routers are supervised classifiers. This project's SFT-vs-GRPO comparison is the genuinely open part of the experiment, not a guaranteed win.
- Whether a 3-7B model can reliably distinguish "task-diff mismatch" from "looks fine but is subtly wrong" using only diff + mechanical signals (no code execution beyond existing tests) is untested at this scale specifically.

## Why this is not redundant with existing AI code review tools

A natural objection: CodeRabbit and Greptile already exist, so why build this? They are structurally a different layer, not a competitor. Both are *PR-level reviewers* — they read a diff (Greptile with a whole-codebase graph index for cross-file context, CodeRabbit with 40+ bundled linters/SAST scanners underneath its LLM layer) and generate natural-language review comments, calling a capable model on every PR. This project's router does not read code and explain what is wrong with it; it is the cheaper decision layer that would sit *in front of* a reviewer like that, deciding whether a diff needs that level of scrutiny at all, or can ship as-is, or just needs tests/lint run first.

The more precise, differentiated use case is **in-session triage during agentic coding** — deciding whether to trust a tool-call's output before an agent's next step, inside an active Claude Code session — not after-the-fact PR review once a commit already exists. PR-review products are not built for that moment; they operate on finished diffs, not mid-loop agent output. This is also the moment the author personally experiences daily, which is what motivated the project (see Motivation).

## The decision (task definition)

Given `(task_description, diff, mechanical_signals)` where `mechanical_signals` = lint pass/fail, type-check pass/fail, existing test suite pass/fail — output one action:

| Action | Meaning | Relative cost |
|---|---|---|
| `SHIP` | Trust it, ship as-is | ~0 |
| `RUN_TESTS` | Run the test suite before deciding (if not already run) | low |
| `LINT_TYPECHECK` | Run static analysis before deciding | low |
| `ESCALATE_STRONG_MODEL` | Send to a stronger model to redo/review | high |

v1 scope is these four actions. `ASK_CLARIFY` (task was ambiguous) and `FLAG_HUMAN` (high-stakes change) are explicitly deferred to a v2 stretch goal — they require fuzzier ground truth (ambiguity, "high-stakes" labeling) that is harder to verify mechanically, and adding them to v1 risks repeating DocTracerRL's mistake of stacking multiple unverified risks at once.

## Reward design

```
reward = task_success_indicator × (K − λ · cost(action)) − format_penalty
```

- `task_success_indicator` ∈ {0, 1}: did the final resolved code (after whatever action was taken) actually pass tests / match the task? Binary-gated exactly as in xRouter — a wrong `SHIP` is a hard failure regardless of how cheap it was.
- `cost(action)`: fixed, ordered cost per action (SHIP cheapest, ESCALATE most expensive), following xRouter's `λC` term.
- `format_penalty`: applied if output is unparseable (lesson carried over from DocTracerRL's reward-banking incident — never let a malformed action be free).

## Data pipeline

1. Generate ≥500 `(task, diff)` pairs, primarily by running a cheap coding model against SWE-bench-style tasks (controllable difficulty, executable tests already exist, no dependency on having enough personal git history). The author's own git history (commit followed by a same-day fix commit as a free negative label; no follow-up as a free positive label) is used as a secondary, smaller supplementary set to ground the eval in real daily-use examples, not as the primary source.
2. Label each pair mechanically: run existing tests + lint + type-check, record pass/fail. No model-as-judge labeling for ground truth.
3. Split train/eval by task/repo (not by individual diff) to avoid leakage, mirroring the entity-disjoint-split lesson from routing literature.
4. **Shortcut-resistance requirement (mandatory, not optional):** at least 20% of examples must have mechanical signals that do *not* trivially predict the correct label — i.e., cases where `test_pass=True` but the diff still does not solve the stated task (the SWE-PRBench "silent failure" category — verified by the author reading the diff against the task, not by an LLM judge), and cases where `test_pass=False` for a reason unrelated to the diff's correctness (e.g. a flaky or pre-existing failing test) yet the diff itself was fine. Rationale: RLVR is documented to make models "systematically abandon [genuine understanding] and instead enumerate instance-level [shortcut] labels" when a cheap correlated signal is available (arXiv 2604.15149, "LLMs Gaming Verifiers") — without this requirement, a model could achieve strong eval accuracy by reading only `signals.test_pass` and ignoring the diff's actual content, never developing real code comprehension. This is checked the same way class balance is checked (Task 7 in the implementation plan) — as an explicit gate before full-scale data generation, not an afterthought.
4. Generate ≥300 SFT reasoning traces (Claude-labeled hindsight-optimal action + short justification) for warm-start, avoiding the 21-example mistake from DocTracerRL.

## Training

- Base model: Qwen2.5-7B-Instruct (matches xRouter exactly, de-risking "does this model work as a router at all").
- SFT warm-start with `assistant_only_loss=True` (lesson carried over from the prior project).
- GRPO via TRL's `GRPOTrainer` (not a hand-rolled loop — the custom PPO-clip/KL loop from DocTracerRL was never fully verified end-to-end; TRL's implementation is proven and removes one full axis of infrastructure risk).
- Infra: existing RunPod + pinned-`requirements.txt` workflow (proven in prior project; no new infra risk introduced).
- **Mandatory gate:** a 15-20 step, ≤$5 diagnostic run before committing to the full budget. If reward is flat (repeating the all-zero-advantage-group failure), stop and diagnose before spending further — this check did not exist early enough in the prior project.

## Evaluation

Five-policy comparison on held-out, task-disjoint data: always-SHIP, always-ESCALATE, tuned lint/test-pass threshold rule, SFT-only, GRPO. Primary metric: accuracy (task success under the chosen action) vs. cost, plotted as a frontier. Secondary, contextual (not the primary claim): compare against SWE-PRBench's documented 15-31% frontier-model diff-review catch rate, to honestly frame what kind of gap this is and is not closing.

**Shortcut-detection check (required before trusting the primary eval result):** on the held-out shortcut-resistance subset (spec §Data pipeline step 4), score GRPO's accuracy in isolation. If accuracy on this subset is near chance while overall accuracy looks strong, this is direct evidence the model is exploiting the `test_pass` signal rather than reading the diff — the isomorphic-perturbation-testing pattern from the reward-hacking literature (arXiv 2604.15149), adapted here as: does the model's decision change appropriately when the diff's actual content changes but mechanical signals stay fixed? If it does not, this is reported honestly as a shortcut-learning finding, not papered over by the aggregate accuracy number.

## Success criteria (see SMART goals for schedule)

- **Must-have:** GRPO (or SFT, reported honestly either way) beats the tuned threshold baseline on the accuracy-cost frontier on held-out data.
- **Stretch:** GRPO measurably beats SFT-only, evidencing that the binary-gated cost-aware reward adds something supervised learning on hindsight labels does not.
- **Failure is still reportable:** if GRPO collapses to a simple strategy (e.g., always escalates or always ships once tests pass), this replicates xRouter's own documented small-model finding and is written up as such — not as an unexplained null result.

## Deliverables

1. Training + eval code, in a new repository (not inside rlm-explorer — see Approach C from brainstorming: TRL's proven loop, fresh dependency set, no coupling to the unfinished DocTracerRL project).
2. Results write-up: five-policy frontier table/plot, honest statement of what was and wasn't beaten, explicit framing as "one instance of a general RL-for-classification recipe" (citing the Rewards-as-Labels paper) rather than a claim to have solved code review.
3. A Claude Code skill (e.g. `/ship-check`) wrapping the best-performing checkpoint, usable on a real diff. MCP server wrapper is an explicit stretch goal, not required for v1.

## Demo & write-up assets

Decided during scoping to make the result legible and shareable, not just correct. These are deliverables, not afterthoughts — the logging requirements below must be in place *before* the full GRPO run and final eval (goals 5-6), since transcripts cannot be reconstructed after the fact.

- **Lead with the failure rate, not the model.** Open the write-up with SWE-PRBench's documented number (frontier models catch only 15-31% of human-flagged issues on diff-only review) before introducing the router — sets up the problem as bigger than the reader expects.
- **Name the comparison models concretely** in the write-up (e.g. "catches what Claude Opus / GPT-5 misses reviewing its own generated code"), not "a frontier model," so the claim is specific and checkable.
- **Frame the premise as the problem**, not just the fix: AI-written code is routinely reviewed only by more AI (often the same model), with no independent check — state this plainly as the setup being indicted.
- **Terminal recording (~15 seconds):** a real diff, the skill flags it live, its reasoning printed — captured once the Claude Code skill (goal 7) exists. This is the single highest-priority asset; a claim people can watch happen beats a claim they have to trust.
- **Logging requirement (applies to goals 5 and 6):** save full transcripts, not just pass/fail, for every eval example during the final five-policy eval — enables mining a "gallery of catches" (2-3 vivid, individually-explainable real cases) after the fact without re-running training or eval.
- **Fixed test case(s), frozen before training starts:** pick 1-2 representative examples specifically to re-run through every checkpoint (always-SHIP baseline, SFT-only, GRPO) for a before/after transcript showing the judgment visibly improving across training stages.
- **Mine real embarrassing bugs**, not only synthetic ones, from the author's own git-history supplementary set (data pipeline step 1) as headline examples — "here's a bug that cost real time, that the router would have caught" is a stronger hook than an aggregate statistic.
- **Give the project a short, memorable name** (not "the code triage router") before the write-up is published, so it reads as a named thing rather than a generic description.

## SMART goals / schedule (4 weeks)

1. **Data pipeline** — ≥500 labeled `(task, diff, signals)` triples, class balance checked. *End of week 1.*
2. **Baselines** — always-SHIP / always-ESCALATE / tuned-threshold scored on held-out data. *End of week 1.*
3. **SFT warm-start** — ≥300 reasoning traces, `assistant_only_loss=True`, checkpoint beats always-SHIP baseline. *End of week 2.*
4. **Cheap GRPO diagnostic** — 15-20 steps, ≤$5, explicit go/no-go decision recorded before further spend. *Start of week 3.*
5. **Full GRPO run** — 75-150 steps, ≤$50 total budget. *End of week 3.*
6. **Final eval** — five-policy frontier table; one true, specific sentence written about the result either way. *End of week 3.*
7. **Ship the artifact** — Claude Code skill, used by the author on at least one real PR. *End of week 4.*
8. **Write-up** — README/post citing xRouter, Rewards-as-Labels, and TruthRL; states the result honestly; frames the recipe as generalizable beyond code triage. *End of week 4.*

## Risks

- **Base rate of usable diffs may be skewed** (e.g., cheap models mostly succeed or mostly fail on the chosen tasks), leaving too little class balance for a real decision boundary — mitigated by explicitly checking class balance in goal 1 and adjusting task difficulty/model choice if skewed.
- **RL may not beat SFT** — the genuinely open part of the experiment; treated as a legitimate reportable outcome, not a failure of the project.
- **TRL version/dependency conflicts** — the prior project lost significant time to unpinned dependency resolution on the training pod; this project starts with a pinned `requirements.txt` from day one rather than discovering the need for one mid-project.
